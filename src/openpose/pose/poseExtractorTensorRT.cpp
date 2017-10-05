#ifdef USE_CAFFE
#include <openpose/core/netCaffe.hpp>
#include <openpose/core/netTensorRT.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/cuda.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/pose/poseExtractorTensorRT.hpp>
#include <openpose/core/FashionTracker.h>

std::shared_ptr<PackagedAsyncTracker> tracker(new FashionTracker());

//#define TIMING_LOGS

typedef std::vector<std::pair<std::string, std::chrono::high_resolution_clock::time_point>> OpTimings;

static OpTimings timings;

inline void timeNow(const std::string& label){
#ifdef TIMING_LOGS
    const auto now = std::chrono::high_resolution_clock::now();
    const auto timing = std::make_pair(label, now);
    timings.push_back(timing);
#endif
}

inline std::string timeDiffToString(const std::chrono::high_resolution_clock::time_point& t1,
                                const std::chrono::high_resolution_clock::time_point& t2 ) {
    return std::to_string((double)std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t2).count() * 1e3) + " ms";
}


namespace op
{
    PoseExtractorTensorRT::PoseExtractorTensorRT(const Point<int>& netInputSize, const Point<int>& netOutputSize, const Point<int>& outputSize, const int scaleNumber,
                                           const PoseModel poseModel, const std::string& modelFolder, const int gpuId, const std::vector<HeatMapType>& heatMapTypes,
                                           const ScaleMode heatMapScale) :
        PoseExtractor{netOutputSize, outputSize, poseModel, heatMapTypes, heatMapScale},
        mResizeScale{mNetOutputSize.x / (float)netInputSize.x},
        spNet{std::make_shared<NetTensorRT>(std::array<int,4>{scaleNumber, 3, (int)netInputSize.y, (int)netInputSize.x},
                                         modelFolder + POSE_PROTOTXT[(int)poseModel], modelFolder + POSE_TRAINED_MODEL[(int)poseModel], gpuId)},
        spResizeAndMergeTensorRT{std::make_shared<ResizeAndMergeCaffe<float>>()},
        spNmsTensorRT{std::make_shared<NmsCaffe<float>>()},
        spBodyPartConnectorTensorRT{std::make_shared<BodyPartConnectorCaffe<float>>()}
    {
        try
        {
            const auto resizeScale = mNetOutputSize.x / (float)netInputSize.x;
            const auto resizeScaleCheck = resizeScale / (mNetOutputSize.y/(float)netInputSize.y);
            if (1+1e-6 < resizeScaleCheck || resizeScaleCheck < 1-1e-6)
                error("Net input and output size must be proportional. resizeScaleCheck = " + std::to_string(resizeScaleCheck), __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PoseExtractorTensorRT::~PoseExtractorTensorRT()
    {
    }

    void PoseExtractorTensorRT::netInitializationOnThread()
    {
        try
        {
            log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
          

            // TensorRT net
            spNet->initializationOnThread();
            spTensorRTNetOutputBlob = ((NetTensorRT*)spNet.get())->getOutputBlob();
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // HeatMaps extractor blob and layer
            spHeatMapsBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            spResizeAndMergeTensorRT->Reshape({spTensorRTNetOutputBlob.get()}, {spHeatMapsBlob.get()}, mResizeScale * POSE_CCN_DECREASE_FACTOR[(int)mPoseModel]);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // Pose extractor blob and layer
            spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            spNmsTensorRT->Reshape({spHeatMapsBlob.get()}, {spPeaksBlob.get()}, POSE_MAX_PEAKS[(int)mPoseModel]);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            // Pose extractor blob and layer
            spPoseBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            spBodyPartConnectorTensorRT->setPoseModel(mPoseModel);
            spBodyPartConnectorTensorRT->Reshape({spHeatMapsBlob.get(), spPeaksBlob.get()}, {spPoseBlob.get()});
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);

            log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractorTensorRT::forwardPass(const Array<float>& inputNetData, const Point<int>& inputDataSize, const std::vector<float>& scaleRatios)
    {
        try
        {
            if (!mFashionDemo)
            {
                tracker->pause();
                // Security checks
                if (inputNetData.empty())
                    error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
                timeNow("Start");
                // 1. TensorRT deep network
                spNet->forwardPass(inputNetData.getConstPtr());
                timeNow("TensorRT forward");
                // 2. Resize heat maps + merge different scales
                spResizeAndMergeTensorRT->setScaleRatios(scaleRatios);
                timeNow("SpResizeAndMergeTensorRT");
                #ifndef CPU_ONLY
                    spResizeAndMergeTensorRT->Forward_gpu({spTensorRTNetOutputBlob.get()}, {spHeatMapsBlob.get()});       // ~5ms
                    timeNow("RaM forward_gpu");
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                    timeNow("CudaCheck");
                #else
                    error("ResizeAndMergeTensorRT CPU version not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
                #endif
                timeNow("Resize heat Maps");
                // 3. Get peaks by Non-Maximum Suppression
                spNmsTensorRT->setThreshold((float)get(PoseProperty::NMSThreshold));
                #ifndef CPU_ONLY
                    spNmsTensorRT->Forward_gpu({spHeatMapsBlob.get()}, {spPeaksBlob.get()});                           // ~2ms
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #else
                    error("NmsTensorRT CPU version not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
                #endif
                timeNow("Peaks by nms");
                // Get scale net to output
                const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
                const Point<int> netSize{intRound(scaleProducerToNetInput*inputDataSize.x), intRound(scaleProducerToNetInput*inputDataSize.y)};
                mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, mOutputSize)};
                timeNow("Scale net to output");
                // 4. Connecting body parts
                spBodyPartConnectorTensorRT->setScaleNetToOutput(mScaleNetToOutput);
                spBodyPartConnectorTensorRT->setInterMinAboveThreshold((int)get(PoseProperty::ConnectInterMinAboveThreshold));
                spBodyPartConnectorTensorRT->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
                spBodyPartConnectorTensorRT->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
                spBodyPartConnectorTensorRT->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));
                // GPU version not implemented yet
                spBodyPartConnectorTensorRT->Forward_cpu({spHeatMapsBlob.get(), spPeaksBlob.get()}, mPoseKeypoints);
                // spBodyPartConnectorTensorRT->Forward_gpu({spHeatMapsBlob.get(), spPeaksBlob.get()}, {spPoseBlob.get()}, mPoseKeypoints);
                timeNow("Connect Body Parts");

    #ifdef TIMING_LOGS 
                const auto totalTimeSec = timeDiffToString(timings.back().second, timings.front().second);
                const auto message = "Pose estimation successfully finished. Total time: " + totalTimeSec + " seconds.";
                op::log(message, op::Priority::High);

                for(OpTimings::iterator timing = timings.begin()+1; timing != timings.end(); ++timing) {
                  const auto log_time = (*timing).first + " - " + timeDiffToString((*timing).second, (*(timing-1)).second);
                  op::log(log_time, op::Priority::High);
                }
    #endif
            }
            else // mFashionDemo == true
            {
                tracker->resume();
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    const float* PoseExtractorTensorRT::getHeatMapCpuConstPtr() const
    {
        try
        {
            checkThread();
            return spHeatMapsBlob->cpu_data();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorTensorRT::getHeatMapGpuConstPtr() const
    {
        try
        {
            checkThread();
            return spHeatMapsBlob->gpu_data();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorTensorRT::getPoseGpuConstPtr() const
    {
        try
        {
            error("GPU pointer for people pose data not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
            checkThread();
            return spPoseBlob->gpu_data();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }
}

#endif



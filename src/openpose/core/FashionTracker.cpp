#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <cstring>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <openpose/core/FashionTracker.h>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;


//#define FASHION_LOG

#ifdef FASHION_LOG
    #define fashion_log(log) std::cout << log << std::endl;
#else
    #define fashion_log(log) /* log */
#endif


#define FASHION_CUDA_CHECK(status)												\
    {																\
	if (status != 0)												\
	{																\
	    std::cout << "Cuda failure: " << cudaGetErrorString(status)	\
		      << " at line " << __LINE__							\
	              << std::endl;									\
	    abort();													\
	}																\
    }

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_C = 3;
static const int INPUT_H = 480;
static const int INPUT_W = 640;
static const int IM_INFO_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 16;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;

const std::string CLASSES[OUTPUT_CLS_SIZE]{"background", "sweater", "hat", "dress", "bag", "jacket-coat", "shoe", "pants", "suit", "skirt", "sunglasses", "romper", "top-shirt", "jumpsuit", "shorts", "swimwear"};

const char* INPUT_BLOB_NAME0 = "data";
const char* INPUT_BLOB_NAME1 = "im_info";
const char* OUTPUT_BLOB_NAME0 = "bbox_pred";
const char* OUTPUT_BLOB_NAME1 = "cls_prob";
const char* OUTPUT_BLOB_NAME2 = "rois";
const char* OUTPUT_BLOB_NAME3 = "count";


const int poolingH = 7;
const int poolingW = 7;
const int featureStride = 16;
const int preNmsTop = 6000;
const int nmsMaxOut = 300;
const int anchorsRatioCount = 3;
const int anchorsScaleCount = 3;
const float iouThreshold = 0.7f;
const float minBoxSize = 16;
const float spatialScale = 0.0625f;
const float anchorsRatios[anchorsRatioCount] = { 0.5f, 1.0f, 2.0f };
const float anchorsScales[anchorsScaleCount] = { 8.0f, 16.0f, 32.0f };

// LoggerFashion for GIE info/warning/errors
class LoggerFashion : public ILogger
{
        void log(Severity severity, const char* msg) override
        {
                // suppress info-level messages
                //if (severity != Severity::kINFO)
                        fashion_log(msg);
        }
} gLoggerFashion;

struct BBox
{
	float x1, y1, x2, y2;
};


template<int OutC>
class Reshape : public IPlugin
{
public:
	Reshape() {}
	Reshape(const void* buffer, size_t size)
	{
		assert(size == sizeof(mCopySize));
		mCopySize = *reinterpret_cast<const size_t*>(buffer);
	}

	int getNbOutputs() const override
	{
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		assert(nbInputDims == 1);
		assert(index == 0);
		assert(inputs[index].nbDims == 3);
		assert((inputs[0].d[0])*(inputs[0].d[1]) % OutC == 0);
		return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);
	}

	int initialize() override
	{
		return 0;
	}

	void terminate() override
	{
	}

	size_t getWorkspaceSize(int) const override
	{
		return 0;
	}

	// currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
	int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
	{
		FASHION_CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
		return 0;
	}

	size_t getSerializationSize() override
	{
		return sizeof(mCopySize);
	}

	void serialize(void* buffer) override
	{
		*reinterpret_cast<size_t*>(buffer) = mCopySize;
	}

	void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
	{
		mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
	}

protected:
	size_t mCopySize;
};


// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
	// deserialization plugin implementation
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
                fashion_log("createPlugin1");
		assert(isPlugin(layerName));
		if (!strcmp(layerName, "ReshapeCTo2"))
		{
			assert(mPluginRshp2 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>());
			return mPluginRshp2.get();
		}
		else if (!strcmp(layerName, "ReshapeCTo18"))
		{
			assert(mPluginRshp18 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>());
			return mPluginRshp18.get();
		}
		else if (!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
				(createFasterRCNNPlugin(featureStride, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize, spatialScale,
					DimsHW(poolingH, poolingW), Weights{ nvinfer1::DataType::kFLOAT, anchorsRatios, anchorsRatioCount },
					Weights{ nvinfer1::DataType::kFLOAT, anchorsScales, anchorsScaleCount }), nvPluginDeleter);
			return mPluginRPROI.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
                fashion_log("createPlugin2");
		assert(isPlugin(layerName));
		if (!strcmp(layerName, "ReshapeCTo2"))
		{
			assert(mPluginRshp2 == nullptr);
			mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>(serialData, serialLength));
			return mPluginRshp2.get();
		}
		else if (!strcmp(layerName, "ReshapeCTo18"))
		{
			assert(mPluginRshp18 == nullptr);
			mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>(serialData, serialLength));
			return mPluginRshp18.get();
		}
		else if (!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createFasterRCNNPlugin(serialData, serialLength), nvPluginDeleter);
			return mPluginRPROI.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	// caffe parser plugin implementation
	bool isPlugin(const char* name) override
	{
		return (!strcmp(name, "ReshapeCTo2")
			|| !strcmp(name, "ReshapeCTo18")
			|| !strcmp(name, "RPROIFused"));
	}

	// the application has to destroy the plugin when it knows it's safe to do so
	void destroyPlugin()
	{
		mPluginRshp2.release();		mPluginRshp2 = nullptr;
		mPluginRshp18.release();	mPluginRshp18 = nullptr;
		mPluginRPROI.release();		mPluginRPROI = nullptr;
	}


	std::unique_ptr<Reshape<2>> mPluginRshp2{ nullptr };
	std::unique_ptr<Reshape<18>> mPluginRshp18{ nullptr };
	void(*nvPluginDeleter)(INvPlugin*) { [](INvPlugin* ptr) {ptr->destroy(); } };
	std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mPluginRPROI{ nullptr, nvPluginDeleter };
};

ICudaEngine* FashionTracker::caffeToGIEModel()
{
        PluginFactory pluginFactory;
        // batch size
        const int maxBatchSize = 1;
        std::vector < std::string > outputs({ OUTPUT_BLOB_NAME0, OUTPUT_BLOB_NAME1, OUTPUT_BLOB_NAME2, OUTPUT_BLOB_NAME3 });
    
	// create the builder
	IBuilder* builder = createInferBuilder(gLoggerFashion);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactory(&pluginFactory);

	fashion_log("Begin parsing model...");
	const IBlobNameToTensor* blobNameToTensor = parser->parse(mCaffeProto.c_str(),
		mCaffeTrainedModel.c_str(),
		*network,
		DataType::kFLOAT);
	fashion_log("End parsing model...");
	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(32 << 20);	// we need about 6MB of scratch space for the plugin layer for batch size 5

	fashion_log("Begin building engine...");
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);
	fashion_log("End building engine...");

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
        pluginFactory.destroyPlugin();
    
        return engine;
}


void bboxTransformInvAndClip(float* rois, float* deltas, float* predBBoxes, float* imInfo,
	const int N, const int nmsMaxOut, const int numCls)
{
	float width, height, ctr_x, ctr_y;
	float dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	float *deltas_offset, *predBBoxes_offset, *imInfo_offset;
	for (int i = 0; i < N * nmsMaxOut; ++i)
	{
		width = rois[i * 4 + 2] - rois[i * 4] + 1;
		height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
		ctr_x = rois[i * 4] + 0.5f * width;
		ctr_y = rois[i * 4 + 1] + 0.5f * height;
		deltas_offset = deltas + i * numCls * 4;
		predBBoxes_offset = predBBoxes + i * numCls * 4;
		imInfo_offset = imInfo + i / nmsMaxOut * 3;
		for (int j = 0; j < numCls; ++j)
		{
			dx = deltas_offset[j * 4];
			dy = deltas_offset[j * 4 + 1];
			dw = deltas_offset[j * 4 + 2];
			dh = deltas_offset[j * 4 + 3];
			pred_ctr_x = dx * width + ctr_x;
			pred_ctr_y = dy * height + ctr_y;
			pred_w = exp(dw) * width;
			pred_h = exp(dh) * height;
			predBBoxes_offset[j * 4] = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 1] = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 2] = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 3] = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
		}
	}
}

std::vector<int> nms(std::vector<std::pair<float, int> >& score_index, float* bbox, const int classNum, const int numClasses, const float nms_threshold)
{
	auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
		if (x1min > x2min) {
			std::swap(x1min, x2min);
			std::swap(x1max, x2max);
		}
		return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
	};
	auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
		float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
		float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
		float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
		float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
		float overlap2D = overlapX * overlapY;
		float u = area1 + area2 - overlap2D;
		return u == 0 ? 0 : overlap2D / u;
	};

	std::vector<int> indices;
	for (auto i : score_index)
	{
		const int idx = i.second;
		bool keep = true;
		for (unsigned k = 0; k < indices.size(); ++k)
		{
			if (keep)
			{
				const int kept_idx = indices[k];
				float overlap = computeIoU(&bbox[(idx*numClasses + classNum) * 4],
					&bbox[(kept_idx*numClasses + classNum) * 4]);
				keep = overlap <= nms_threshold;
			}
			else
				break;
		}
		if (keep) indices.push_back(idx);
	}
	return indices;
}
                    
inline bool file_exists(const std::string& file_path) {
    struct stat buffer;
    return (stat(file_path.c_str(), &buffer) == 0);
}

ICudaEngine* FashionTracker::createEngine()
{
    ICudaEngine *engine;
    
    std::string serializedEnginePath = mCaffeProto + ".bin";
    fashion_log("Serialized engine path: " << serializedEnginePath.c_str());
    
    // create a GIE model from the caffe model and serialize it to a stream
    
    if (file_exists(serializedEnginePath))
    {
        fashion_log("Found serialized TensorRT engine, deserializing...");
        char *gieModelStream{nullptr};
        size_t size{0};
        std::ifstream file(serializedEnginePath, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            gieModelStream = new char[size];
            assert(gieModelStream);
            file.read(gieModelStream, size);
            file.close();
        }
        
        // deserialize the engine
        PluginFactory pluginFactory;
        IRuntime* runtime = createInferRuntime(gLoggerFashion);
        engine = runtime->deserializeCudaEngine(gieModelStream, size, &pluginFactory);
        if (gieModelStream) delete [] gieModelStream;
        runtime->destroy();
        pluginFactory.destroyPlugin();
    }
    else
    {
        engine = caffeToGIEModel();
        if (!engine)
        {
            std::cerr << "Engine could not be created" << std::endl;
            return nullptr;
        }
        else // serialize engine
        {  
            std::ofstream p(serializedEnginePath);
            if (!p)
            {
                std::cerr << "could not serialize engine" << std::endl;
            }
            IHostMemory *ptr = engine->serialize();
            assert(ptr);
            p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
            ptr->destroy();
        }
    }
    return engine;
}
                    
                    
FashionTracker::FashionTracker() : PackagedAsyncTracker(INPUT_W, true),
                    mCaffeProto("models/fashion/deploy.prototxt"),
                    mCaffeTrainedModel("models/fashion/snapshot.caffemodel")
{
    cudaEngine = createEngine();
    cudaContext = cudaEngine->createExecutionContext();
    
    
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 2 inputs and 4 outputs.
    assert(cudaEngine->getNbBindings() == 6);
    
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex0 = cudaEngine->getBindingIndex(INPUT_BLOB_NAME0);
    inputIndex1 = cudaEngine->getBindingIndex(INPUT_BLOB_NAME1);
    outputIndex0 = cudaEngine->getBindingIndex(OUTPUT_BLOB_NAME0);
    outputIndex1 = cudaEngine->getBindingIndex(OUTPUT_BLOB_NAME1);
    outputIndex2 = cudaEngine->getBindingIndex(OUTPUT_BLOB_NAME2);
    outputIndex3 = cudaEngine->getBindingIndex(OUTPUT_BLOB_NAME3);
    
    int batchSize = 1; 
    // create GPU buffers and a stream
    FASHION_CUDA_CHECK(cudaMalloc(&buffers[inputIndex0], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));   // data
    FASHION_CUDA_CHECK(cudaMalloc(&buffers[inputIndex1], batchSize * IM_INFO_SIZE * sizeof(float)));                  // im_info
    FASHION_CUDA_CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float))); // bbox_pred
    FASHION_CUDA_CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float)));  // cls_prob
    FASHION_CUDA_CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float)));                // rois
    FASHION_CUDA_CHECK(cudaMalloc(&buffers[outputIndex3], batchSize * sizeof(float)));                                // count
    
    FASHION_CUDA_CHECK(cudaStreamCreate(&stream));
}

FashionTracker::~FashionTracker()
{
    // release the stream and the buffers
    if (stream)
        cudaStreamDestroy(stream);
    
    FASHION_CUDA_CHECK(cudaFree(buffers[inputIndex0]));
    FASHION_CUDA_CHECK(cudaFree(buffers[inputIndex1]));
    FASHION_CUDA_CHECK(cudaFree(buffers[outputIndex0]));
    FASHION_CUDA_CHECK(cudaFree(buffers[outputIndex1]));
    FASHION_CUDA_CHECK(cudaFree(buffers[outputIndex2]));
    FASHION_CUDA_CHECK(cudaFree(buffers[outputIndex3]));
    
    if (cudaContext)
        cudaContext->destroy();
    if (cudaEngine)
        cudaEngine->destroy();
}

static cv::Mat imgTo4DMat(const cv::Mat &img)
{
    const int sizes[] = { 1, img.channels(), img.rows, img.cols };
    
    cv::Mat big_mat(4, sizes, CV_MAKETYPE(img.depth(), 1), cv::Scalar(0));

    // extract each channels in a single plane and store it in a 3D mat
    
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    std::vector<cv::Range> ranges(4, cv::Range::all());
    ranges[0] = cv::Range(0, 1);

    for (int i = 0; i < img.channels(); ++i) {
        ranges[1] = cv::Range(i, i + 1);

        // this will be 1x1xHxW
        cv::Mat plane4D = big_mat(&ranges[0]);

        // we need a 2D mat so that the copyTo works (copying directly to the plane4D does not work)
        cv::Mat plane2D(plane4D.size[2], plane4D.size[3], big_mat.type(), plane4D.data);

        channels[i].copyTo(plane2D);
    }

    return big_mat;
}
                    
std::list<tf_tracking::Recognition> FashionTracker::getDetections(const cv::Mat &frame) {
    fashion_log("getDetections 0");
    const int N = 1;
    
    cv::Scalar mean(102.9801f, 115.9465f, 122.7717f);
    cv::Mat frameResized, frameMinusMean;
    cv::resize(frame, frameResized, cv::Size(INPUT_W, INPUT_H));
    frameResized.convertTo(frameMinusMean, CV_32FC3);
    frameMinusMean -= mean;
    cv::Mat caffeInput = imgTo4DMat(frameMinusMean);

    float* data = (float*)caffeInput.data;
    fashion_log("getDetections 1");
    
    // host memory for outputs
    float* rois = new float[N * nmsMaxOut * 4];
    float* bboxPreds = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
    float* clsProbs = new float[N * nmsMaxOut * OUTPUT_CLS_SIZE];
   
    float imInfo[3] = {frame.rows, frame.cols, 1}; 
    
    // predicted bounding boxes
    float* predBBoxes = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
    
    // run inference
    doInference(data, imInfo, bboxPreds, clsProbs, rois, N);
    

    fashion_log("getDetections 2");
    
    // unscale back to raw image space
    for (int i = 0; i < N; ++i)
    {
        float * rois_offset = rois + i * nmsMaxOut * 4;
        for (int j = 0; j < nmsMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j)
            rois_offset[j] /= imInfo[i * 3 + 2];
    }
    
    bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, N, nmsMaxOut, OUTPUT_CLS_SIZE);
    
    const float nms_threshold = 0.3f;
    const float score_threshold = 0.9f;
    fashion_log("getDetections 3");
    
    
    std::list<tf_tracking::Recognition> normalized_results;
    
    for (int i = 0; i < N; ++i)
    {
        float *bbox = predBBoxes + i * nmsMaxOut * OUTPUT_BBOX_SIZE;
        float *scores = clsProbs + i * nmsMaxOut * OUTPUT_CLS_SIZE;
        for (int c = 1; c < OUTPUT_CLS_SIZE; ++c) // skip the background
        {
            std::vector<std::pair<float, int> > score_index;
            for (int r = 0; r < nmsMaxOut; ++r)
            {
                if (scores[r*OUTPUT_CLS_SIZE + c] > score_threshold)
                {

                    score_index.push_back(std::make_pair(scores[r*OUTPUT_CLS_SIZE + c], r));
                    std::stable_sort(score_index.begin(), score_index.end(),
                                     [](const std::pair<float, int>& pair1,
                                        const std::pair<float, int>& pair2) {
                                         return pair1.first > pair2.first;
                                     });
                }
            }
            
            // apply NMS algorithm
            std::vector<int> indices = nms(score_index, bbox, c, OUTPUT_CLS_SIZE, nms_threshold);
            // Show results
            for (unsigned k = 0; k < indices.size(); ++k)
            {
                int idx = indices[k];
                std::cout << "Detected " << CLASSES[c] << " with confidence " << scores[idx*OUTPUT_CLS_SIZE + c] * 100.0f << "% " << std::endl;
               
                const float ratioW = (float)frame.rows / (float)frameMinusMean.cols;
                const float ratioH = (float)frame.rows / (float)frameMinusMean.rows; 
                const float x1 = bbox[idx*OUTPUT_BBOX_SIZE + c * 4];
                const float y1 = bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 1];
                const float x2 = bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 2];
                const float y2 = bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 3];
                normalized_results.emplace_back("fashion", CLASSES[c],
                                                scores[idx*OUTPUT_CLS_SIZE + c],
                                                tf_tracking::BoundingBox(
                                                    x1 * ratioW / float(frame.cols),
                                                    y1 * ratioH / float(frame.rows),
                                                    x2 * ratioW / float(frame.cols),
                                                    y2 * ratioH / float(frame.rows)));
            }
        }
    }
    fashion_log("getDetections 4");
    
    
    delete[] rois;
    delete[] bboxPreds;
    delete[] clsProbs;
    delete[] predBBoxes;


    fashion_log("getDetections 5");
    
    return normalized_results;
}

void FashionTracker::doInference(float* inputData, float* inputImInfo, float* outputBboxPred, float* outputClsProb, float *outputRois, int batchSize)
{
    
    fashion_log("FashionTracker.cpp: doInference");
    usleep(500000); 
     
    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    FASHION_CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    FASHION_CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex1], inputImInfo, batchSize * IM_INFO_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
    cudaContext->enqueue(batchSize, buffers, stream, nullptr);
    FASHION_CUDA_CHECK(cudaMemcpyAsync(outputBboxPred, buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    FASHION_CUDA_CHECK(cudaMemcpyAsync(outputClsProb, buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    FASHION_CUDA_CHECK(cudaMemcpyAsync(outputRois, buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    
    
    fashion_log("FashionTracker.cpp: doInference End");
}

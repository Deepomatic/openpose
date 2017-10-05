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

#include <openpose/core/FashionTracker.h>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

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
static const int INPUT_H = 375;
static const int INPUT_W = 500;
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
                        std::cout << msg << std::endl;
        }
} gLoggerFashion;

struct BBox
{
	float x1, y1, x2, y2;
};


static void doInference(IExecutionContext& context, float* inputData, float* inputImInfo, float* outputBboxPred, float* outputClsProb, float *outputRois, int batchSize)
{
        std::cout << "FashionTracker.cpp: doInference" << std::endl;
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly 2 inputs and 4 outputs.
	assert(engine.getNbBindings() == 6);
	void* buffers[6];
        std::cout << "FashionTracker.cpp: doInference 0" << std::endl;

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex0 = engine.getBindingIndex(INPUT_BLOB_NAME0),
		inputIndex1 = engine.getBindingIndex(INPUT_BLOB_NAME1),
		outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
		outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
		outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2),
		outputIndex3 = engine.getBindingIndex(OUTPUT_BLOB_NAME3);
        std::cout << "FashionTracker.cpp: doInference 1" << std::endl;


	// create GPU buffers and a stream
	FASHION_CUDA_CHECK(cudaMalloc(&buffers[inputIndex0], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));   // data
	FASHION_CUDA_CHECK(cudaMalloc(&buffers[inputIndex1], batchSize * IM_INFO_SIZE * sizeof(float)));                  // im_info
	FASHION_CUDA_CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float))); // bbox_pred
	FASHION_CUDA_CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float)));  // cls_prob
	FASHION_CUDA_CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float)));                // rois
	FASHION_CUDA_CHECK(cudaMalloc(&buffers[outputIndex3], batchSize * sizeof(float)));                                // count

	cudaStream_t stream;
	FASHION_CUDA_CHECK(cudaStreamCreate(&stream));
        std::cout << "FashionTracker.cpp: doInference 2" << std::endl;

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	FASHION_CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	FASHION_CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex1], inputImInfo, batchSize * IM_INFO_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	FASHION_CUDA_CHECK(cudaMemcpyAsync(outputBboxPred, buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	FASHION_CUDA_CHECK(cudaMemcpyAsync(outputClsProb, buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	FASHION_CUDA_CHECK(cudaMemcpyAsync(outputRois, buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);


	// release the stream and the buffers
	cudaStreamDestroy(stream);
	FASHION_CUDA_CHECK(cudaFree(buffers[inputIndex0]));
	FASHION_CUDA_CHECK(cudaFree(buffers[inputIndex1]));
	FASHION_CUDA_CHECK(cudaFree(buffers[outputIndex0]));
	FASHION_CUDA_CHECK(cudaFree(buffers[outputIndex1]));
	FASHION_CUDA_CHECK(cudaFree(buffers[outputIndex2]));
	FASHION_CUDA_CHECK(cudaFree(buffers[outputIndex3]));
        std::cout << "FashionTracker.cpp: doInference End" << std::endl;
}

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
                std::cout << "createPlugin1" << std::endl;
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
                std::cout << "createPlugin2" << std::endl;
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

	std::cout << "Begin parsing model..." << std::endl;
	const IBlobNameToTensor* blobNameToTensor = parser->parse(mCaffeProto.c_str(),
		mCaffeTrainedModel.c_str(),
		*network,
		DataType::kFLOAT);
	std::cout << "End parsing model..." << std::endl;
	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(32 << 20);	// we need about 6MB of scratch space for the plugin layer for batch size 5

	std::cout << "Begin building engine..." << std::endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);
	std::cout << "End building engine..." << std::endl;

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
    std::cout << "Serialized engine path: " << serializedEnginePath.c_str() << std::endl;
    
    // create a GIE model from the caffe model and serialize it to a stream
    
    if (file_exists(serializedEnginePath))
    {
        std::cout << "Found serialized TensorRT engine, deserializing..." << std::endl;
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
                    
                    
FashionTracker::FashionTracker() : PackagedAsyncTracker(800, false),
                    mCaffeProto("models/fashion/deploy.prototxt"),
                    mCaffeTrainedModel("models/fashion/snapshot.caffemodel")
{
    cudaEngine = createEngine();
    cudaContext = cudaEngine->createExecutionContext();
}

FashionTracker::~FashionTracker()
{
    if (cudaContext)
        cudaContext->destroy();
    if (cudaEngine)
        cudaEngine->destroy();
}
                    
std::list<tf_tracking::Recognition> FashionTracker::getDetections(const cv::Mat &frame) {
    std::cout << "getDetections 0" << std::endl;
    const int N = 1;
    
    cv::Scalar mean(102.9801f, 115.9465f, 122.7717f);
    cv::Mat frameResized, frameMinusMean;
    cv::resize(frame, frameResized, cv::Size(375, 500));
    frameResized.convertTo(frameMinusMean, CV_32FC3);
    frameMinusMean -= mean;
    float* data = (float*)frameMinusMean.data;
    std::cout << "getDetections 1" << std::endl;
    
    // host memory for outputs
    float* rois = new float[N * nmsMaxOut * 4];
    float* bboxPreds = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
    float* clsProbs = new float[N * nmsMaxOut * OUTPUT_CLS_SIZE];
   
    float imInfo[3] = {frame.rows, frame.cols, 1}; 
    //float imInfo[3] = {frame.cols, frame.rows, 1}; 
    // predicted bounding boxes
    float* predBBoxes = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
    
    // run inference
    doInference(*cudaContext, data, imInfo, bboxPreds, clsProbs, rois, N);
    

    std::cout << "getDetections 2" << std::endl;
    
    // unscale back to raw image space
    for (int i = 0; i < N; ++i)
    {
        float * rois_offset = rois + i * nmsMaxOut * 4;
        for (int j = 0; j < nmsMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j)
            rois_offset[j] /= imInfo[i * 3 + 2];
    }
    
    bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, N, nmsMaxOut, OUTPUT_CLS_SIZE);
    
    const float nms_threshold = 0.3f;
    const float score_threshold = 0.8f;
    std::cout << "getDetections 3" << std::endl;
    
    
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
                
                const float x1 = bbox[idx*OUTPUT_BBOX_SIZE + c * 4];
                const float y1 = bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 1];
                const float x2 = bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 2];
                const float y2 = bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 3];
                normalized_results.emplace_back("fashion", CLASSES[c],
                                                scores[idx*OUTPUT_CLS_SIZE + c],
                                                tf_tracking::BoundingBox(x1 / float(frame.cols), y1 / float(frame.rows), x2 / float(frame.cols), y2 / float(frame.rows)));
            }
        }
    }
    std::cout << "getDetections 4" << std::endl;
    
    
    delete[] rois;
    delete[] bboxPreds;
    delete[] clsProbs;
    delete[] predBBoxes;
    std::cout << "getDetections 5" << std::endl;
    
    return normalized_results;
}

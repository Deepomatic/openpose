#ifndef __FASHION_TRACKER__
#define __FASHION_TRACKER__

#include "tracker/multibox_tracker.h"
#include <thread>
#include <unistd.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <openpose/core/PackagedAsyncTracker.h>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"


class FashionTracker : public PackagedAsyncTracker {
public:
    FashionTracker(); // ctor
    ~FashionTracker();
    
protected:
    
    std::list<tf_tracking::Recognition> getDetections(const cv::Mat &frame);
    
private:
    const std::string mCaffeProto;
    const std::string mCaffeTrainedModel;
    
    // TensorRT stuff
    nvinfer1::ICudaEngine* cudaEngine;
    nvinfer1::IExecutionContext* cudaContext;
    nvinfer1::cudaStream_t* stream;
    nvinfer1::ICudaEngine* caffeToGIEModel();
    nvinfer1::ICudaEngine* createEngine();
    int inputIndex0;
    int inputIndex1;
    int outputIndex0;
    int outputIndex1;
    int outputIndex2;
    int outputIndex3;
};


#endif

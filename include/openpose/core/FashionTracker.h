#ifndef __FASHION_TRACKER__
#define __FASHION_TRACKER__

#include "tracker/multibox_tracker.h"
#include <thread>
#include <unistd.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <openpose/core/PackagedAsyncTracker.h>



class PluginFactory;

class FashionTracker : public PackagedAsyncTracker {
public:
    FashionTracker(); // ctor
    ~FashionTracker();
    
protected:
    
    std::list<tf_tracking::Recognition> getDetections(const cv::Mat &frame);
    
private:
    const std::string mCaffeProto;
    const std::string mCaffeTrainedModel;
    
    
    ICudaEngine* FashionTracker::createEngine();
    ICudaEngine* caffeToGIEModel(PluginFactory *pluginFactory);
    
    // TensorRT stuff
    nvinfer1::ICudaEngine* cudaEngine;
    nvinfer1::IExecutionContext* cudaContext;
    nvinfer1::ICudaEngine* caffeToGIEModel();
    nvinfer1::ICudaEngine* createEngine();
};


#endif

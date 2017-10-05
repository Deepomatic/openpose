#ifndef __PACKAGED_TRACKER__
#define __PACKAGED_TRACKER__

#include "tracker/multibox_tracker.h"
#include <thread>
#include <unistd.h>
#include <opencv2/imgproc/imgproc.hpp>

class PackagedAsyncTracker {
public:
    PackagedAsyncTracker(int width = 800, bool stop = true) {
        _counter = 0;
        _width = width;
        _stop = stop;
        _detection_thread = std::thread(&PackagedAsyncTracker::detection_thread_main, this);
    }

    void onFrame(const cv::Mat &frame) {
        std::lock_guard<std::mutex> lock(_mtx);
        if (_stop) {
            return;
        }
        _last_frame = frame.clone();
        _last_y_frame = _tracker.onFrame(_last_frame, ++_counter);
    }

    void drawDetection(cv::Mat &frame) {
        std::hash<std::string> hasher;
        const auto &tracks = _tracker.getTrackedObjects();
        for(const auto &id_object : tracks) {
            const auto &id = id_object.first;
            const auto &object = id_object.second;

            cv::Scalar color;
            cv::RNG rng(hasher(id));
            color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

            const auto &bbox = object.getLocation();
            const auto &center = bbox.GetCenter();
            cv::Point c(center.x, center.y);

            float xmin = bbox.left_  * frame.cols;
            float xmax = bbox.right_ * frame.cols;
            float ymin = bbox.top_    * frame.rows;
            float ymax = bbox.bottom_ * frame.rows;

            rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), color, 2, 8, 0);

            std::string text = object.getLabel();
            putText(frame, text, cv::Point(xmin + 4, ymin + 20),
                cv::FONT_HERSHEY_PLAIN, 1, color, 1, CV_AA);
        }
    }

    void resume() {
        _stop = false;
    }

    void pause() {
        std::lock_guard<std::mutex> lock(_mtx);
        _stop = true;
        _tracker.clearTracks();
        _last_frame = cv::Mat();
        _last_y_frame = cv::Mat();
    }

    void togglePause() {
        if (_stop) {
            resume();
        }
        else {
            pause();
        }
    }

    void launch_detection_thread() {
        while (true) {
            while (!_stop) {
                cv::Mat frame, y_frame;
                int64_t counter;
                {
                    std::lock_guard<std::mutex> lock(_mtx);
                    frame = _last_frame;
                    y_frame = _last_y_frame;
                    counter = _counter;
                }
                if (frame.empty()) {
                    break;
                }
                cv::resize(frame, frame, cv::Size(_width, frame.rows * _width / frame.cols), 0, 0, cv::INTER_CUBIC);
                std::list<tf_tracking::Recognition> objects = getDetections(frame);
                {
                    std::lock_guard<std::mutex> lock(_mtx);
                    if (!_stop) {
                        _tracker.trackObjects(objects, y_frame, counter);
                    }
                }
            }
            while(_stop) {
                usleep(1000);
            }
        }
    }

protected:
    static void detection_thread_main(PackagedAsyncTracker *ptr) {
        ptr->launch_detection_thread();
    }

    virtual std::list<tf_tracking::Recognition> getDetections(const cv::Mat &frame) { return {};};

    bool _stop;
    int _width;

    tf_tracking::MultiBoxTracker _tracker;
    int64_t _counter;
    cv::Mat _last_frame;
    cv::Mat _last_y_frame;

    std::thread _detection_thread;
    std::mutex _mtx;
};

#endif

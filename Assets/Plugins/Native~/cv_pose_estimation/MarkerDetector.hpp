#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <vector>

// Android logging
#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "UnityCV"
#define LOG_INFO(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOG_DEBUG(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#else
#include <stdio.h>
#define LOG_INFO(...) printf(__VA_ARGS__); printf("\n")
#define LOG_ERROR(...) printf("ERROR: "); printf(__VA_ARGS__); printf("\n")
#define LOG_DEBUG(...) printf("DEBUG: "); printf(__VA_ARGS__); printf("\n")
#endif

using namespace std;

class MarkerDetector
{
    private:
        cv::aruco::DetectorParameters detectorParams;
        cv::aruco::Dictionary dictionary;
        cv::aruco::ArucoDetector detector;
        vector<int> ids;
        vector<vector<cv::Point2f>> corners, rejected;
        cv::Point2i faceAndPointCount;

    public:
        MarkerDetector() {
            detectorParams = cv::aruco::DetectorParameters();
            detectorParams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
            dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
            detector = cv::aruco::ArucoDetector(dictionary, detectorParams);
            faceAndPointCount = cv::Point2i(0, 0);
        }

        void ProcessImage(void* imgData, int height, int width)
        {
            LOG_DEBUG("MarkerDetector::ProcessImage: START - height=%d, width=%d", height, width);
            
            // Wrap the raw memory pointer into cv::Mat
            // CV_8UC1 = 8-bit unsigned, single channel
            cv::Mat img(height, width, CV_8UC1, imgData);
            cv::Mat img_flipped;
            LOG_DEBUG("MarkerDetector::ProcessImage: Created cv::Mat from external memory");
            
            cv::flip(img, img_flipped, 0); // Flip vertically
            LOG_DEBUG("MarkerDetector::ProcessImage: Flipped image");

            // Call detector on the flipped copy
            LOG_DEBUG("MarkerDetector::ProcessImage: Calling detector.detectMarkers()");
            detector.detectMarkers(img_flipped, corners, ids, rejected);
            LOG_INFO("MarkerDetector::ProcessImage: Detected %d markers, %d rejected", (int)ids.size(), (int)rejected.size());

            LOG_DEBUG("MarkerDetector::ProcessImage: Computing face and point count");
            faceAndPointCount = ComputeFaceAndPointCount();
            
            // Debug frames
            static int frameCount = 0;
            if (frameCount++ % 100 == 0) {
                cv::Mat debugImg;
                cv::cvtColor(img_flipped, debugImg, cv::COLOR_GRAY2BGR); // Convert to color for drawings
                cv::aruco::drawDetectedMarkers(debugImg, corners, ids);
                cv::imwrite("/storage/emulated/0/Android/data/com.samples.passthroughcamera_kalman/files/debug.png", debugImg);
            }
        }

        cv::Point2i ComputeFaceAndPointCount() const {
            std::array<int, 6> markersOnFace = {0};
            for (int i = 0; i < static_cast<int>(ids.size()); i++)
            {
                int faceIdx = ids[i] / 4;
                // Ensure tag is within range
                if (faceIdx >= 0 && faceIdx < 6) {
                    markersOnFace[faceIdx]++;
                } else {
                    LOG_ERROR("Unexpected Marker ID: %d", ids[i]);
                }
            }

            int faceCount = 0;
            for (int i = 0; i < 6; i++)
                if (markersOnFace[i] > 0) faceCount++;

            return cv::Point2i(faceCount, static_cast<int>(ids.size()) * 4);
        }

        const vector<int>& getIds() const { return ids; }
        const vector<vector<cv::Point2f>>& getCorners() const { return corners; }
        const vector<vector<cv::Point2f>>& getRejected() const { return rejected; }
        cv::Point2i getFaceAndPointCount() const { return faceAndPointCount; }
};
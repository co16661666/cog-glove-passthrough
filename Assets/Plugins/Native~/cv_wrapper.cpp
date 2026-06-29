#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <vector>
#include <fstream>
#include <iomanip>

#include <cv_pose_estimation/MarkerDetector.hpp>
#include <cv_pose_estimation/PoseSolver.hpp>
#include <cv_pose_estimation/CubePoints.hpp>

#include <esekf/OpenCVKF.hpp>
#include <esekf/RCalibration.hpp>

#include <optional>

// Android logging
// .\adb logcat -s Unity ActivityManager DEBUG UnityCV
// .\adb logcat -v color -s Unity:D ActivityManager:D DEBUG:D UnityCV:D | tee C:\Users\liong\Downloads\log_output2.txt
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

#ifdef _MSC_VER
#define EXPORT_API extern "C" __declspec(dllexport)
#else
#define EXPORT_API extern "C" __attribute__((visibility("default")))
#endif

using namespace std;

struct CVPose
{
    float tx, ty, tz;
    float rx, ry, rz;

    int grasped;
    int poseSuccess;
    unsigned long timestamp;
};

MarkerDetector tags;
std::unordered_map<int, std::vector<cv::Point3f>> cube_points;
PoseSolver* poser = nullptr;
OpenCVKF* kf = nullptr;
float marker_detection_timeout = 0.016 * 12; // Timeout in seconds for marker loss detection
unsigned long last_marker_detection_timestamp = 0;
unsigned long last_image_timestamp = 0;

bool setup_complete = false;

// CSV logging
static std::ofstream g_csv_file;
static bool g_csv_initialized = false;
static const char* g_csv_path = "/storage/emulated/0/Android/data/com.samples.passthroughcamera_kalman/files/positionRecord.csv";

void LogToCsv(unsigned long timestamp, int face_count, int point_count, 
              float raw_tx, float raw_ty, float raw_tz,
              float raw_rx, float raw_ry, float raw_rz,
              float filt_tx, float filt_ty, float filt_tz,
              float filt_rx, float filt_ry, float filt_rz) {
    if (!g_csv_initialized) {
        // Open in write mode (overwrites existing file)
        g_csv_file.open(g_csv_path, std::ios::out);
        if (!g_csv_file.is_open()) {
            LOG_ERROR("Failed to open CSV file: %s", g_csv_path);
            return;
        }

        // Write header
        g_csv_file << "timestamp,face_count,point_count,"
                   << "raw_tx,raw_ty,raw_tz,raw_rx,raw_ry,raw_rz,"
                   << "filt_tx,filt_ty,filt_tz,filt_rx,filt_ry,filt_rz\n";
        
        g_csv_initialized = true;
        LOG_INFO("CSV logging initialized: %s", g_csv_path);
    }

    if (g_csv_file.is_open()) {
        // Write data row with high precision
        g_csv_file << timestamp << ","
                   << face_count << "," << point_count << ","
                   << std::fixed << std::setprecision(6)
                   << raw_tx << "," << raw_ty << "," << raw_tz << ","
                   << raw_rx << "," << raw_ry << "," << raw_rz << ","
                   << filt_tx << "," << filt_ty << "," << filt_tz << ","
                   << filt_rx << "," << filt_ry << "," << filt_rz << "\n";
        
        // Flush periodically (every 30 frames roughly)
        static int flush_counter = 0;
        if (++flush_counter > 30) {
            g_csv_file.flush();
            flush_counter = 0;
        }
    }
}

extern "C" {
    EXPORT_API void ProcessImage(void* img_data, int width, int height, unsigned long timestamp, CameraPose cam_pose, CVPose* out_pose) {
        LOG_INFO("ProcessImage: START - width=%d, height=%d, timestamp=%lu", width, height, timestamp);
        
        if (setup_complete == false || img_data == nullptr || out_pose == nullptr || width <= 0 || height <= 0) {
            LOG_ERROR("ProcessImage: Early return - setup_complete=%d, img_data=%p, out_pose=%p, w=%d, h=%d", 
                     setup_complete, img_data, out_pose, width, height);
            if (out_pose != nullptr) {
                out_pose->poseSuccess = 0;
                out_pose->timestamp = timestamp;
            }
            return;
        }

        // Calculate dynamic dt from timestamp difference
        float dt = 0.033f; // Default fallback
        if (last_image_timestamp > 0) {
            unsigned long elapsed_ns = timestamp - last_image_timestamp;
            dt = (float)elapsed_ns / 1E9f; // Convert to seconds
            LOG_INFO("ProcessImage: Dynamic dt calculated - elapsed_ns=%lu, dt=%.6f seconds", elapsed_ns, dt);
        } else {
            LOG_INFO("ProcessImage: First frame, using default dt=%.6f seconds", dt);
        }
        last_image_timestamp = timestamp;

        // Find ArUco markers
        LOG_INFO("ProcessImage: Calling tags.ProcessImage()");
        tags.ProcessImage(img_data, height, width);

        const vector<int>& ids = tags.getIds();
        const vector<vector<cv::Point2f>>& corners = tags.getCorners();
        const vector<vector<cv::Point2f>>& rejected = tags.getRejected();
        
        LOG_INFO("ProcessImage: MarkerDetector found %d markers", (int)ids.size());

        // Check if markers found
        if (ids.empty() || corners.empty() || corners[0].empty()) {
            LOG_INFO("ProcessImage: No markers detected");
            
            // Calculate elapsed time since last marker detection
            unsigned long elapsed_ns = timestamp - last_marker_detection_timestamp;
            float elapsed_time = elapsed_ns / 1E9f; // Convert to seconds
            
            LOG_INFO("ProcessImage: Time since last detection: %.3f seconds, timeout: %.3f seconds", elapsed_time, marker_detection_timeout);
            
            if (elapsed_time < marker_detection_timeout) {
                // Run predict to continue filter trajectory
                LOG_INFO("ProcessImage: Elapsed time < timeout, running predict only");
                kf->predict(dt);
            } else {
                // Reset filter to base state with high uncertainty
                LOG_INFO("ProcessImage: Elapsed time >= timeout, resetting filter to base state");
                kf->softReset();
            }
            
            // Return current filtered state
            Eigen::Vector3f tvec = kf->getTvec();
            Eigen::Vector3f rvec = kf->getRvec();
            out_pose->tx = tvec(0);
            out_pose->ty = tvec(1);
            out_pose->tz = tvec(2);
            out_pose->rx = rvec(0);
            out_pose->ry = rvec(1);
            out_pose->rz = rvec(2);
            out_pose->grasped = 0;
            out_pose->poseSuccess = 1;
            out_pose->timestamp = timestamp;
            
            LOG_INFO("ProcessImage: No detection - returning filtered state - pos=(%.3f, %.3f, %.3f), rot=(%.3f, %.3f, %.3f)",
                    out_pose->tx, out_pose->ty, out_pose->tz, 
                    out_pose->rx, out_pose->ry, out_pose->rz);
            return;
        }

        last_marker_detection_timestamp = timestamp; // Update last detection time

        // Prepare and match up 3D object point cloud with detected 2D marker points
        std::vector<cv::Point3f> obj_points;
        std::vector<cv::Point2f> img_points;

        LOG_INFO("ProcessImage: Building point correspondences");
        for (int idx = 0; idx < ids.size(); idx++)
        {
            LOG_INFO("ProcessImage: Processing marker ID %d", ids[idx]);
            
            if (cube_points.find(ids[idx]) == cube_points.end()) {
                LOG_ERROR("ProcessImage: Marker ID %d not found in cube_points map!", ids[idx]);
                continue;
            }
            
            for (const cv::Point3f& tag_points : cube_points[ids[idx]])
            {
                obj_points.push_back(tag_points);
            }
            for (const cv::Point2f& corner : corners[idx])
            {
                img_points.push_back(corner);
            }
        }

        LOG_INFO("ProcessImage: Collected %d object points and %d image points", 
                (int)obj_points.size(), (int)img_points.size());

        // Check if we have valid point correspondences after filtering
        if (obj_points.empty() || img_points.empty() || obj_points.size() != img_points.size()) {
            LOG_ERROR("ProcessImage: Invalid point correspondences - obj=%d, img=%d", 
                     (int)obj_points.size(), (int)img_points.size());
            out_pose->poseSuccess = 0;
            out_pose->timestamp = timestamp;
            return;
        }

        if (poser == nullptr) {
            LOG_ERROR("ProcessImage: poser is NULL!");
            out_pose->poseSuccess = 0;
            out_pose->timestamp = timestamp;
            return;
        }

        LOG_INFO("ProcessImage: Calling poser->solvePose()");
        poser->solvePose(obj_points, img_points);
        
        LOG_INFO("ProcessImage: Calling poser->cameraToWorld()");
        poser->cameraToWorld(cam_pose);

        LOG_INFO("ProcessImage: Kalman filter prediction and update");
        // Kalman filtering with ESEKF TODO: Reset filter for large dt
        kf->predict(dt);

        Eigen::Matrix<float, 6, 1> measurement;
        measurement.segment<3>(0) = poser->getTvecWorldEigen(); // Translation
        measurement.segment<3>(3) = poser->getRvecWorldEigen(); // Rotation vector

        // Get calibrated R covariance
        cv::Point2i faceAndPoints = tags.ComputeFaceAndPointCount();
        std::optional<Eigen::Matrix<float, 6, 6>> R = LookupR(faceAndPoints);
        
        bool accepted;
        if (R.has_value()) {
            LOG_DEBUG("ProcessImage: Using calibrated R for face=%d, points=%d", faceAndPoints.x, faceAndPoints.y);
            accepted = kf->update(measurement, &R.value());
        } else {
            LOG_DEBUG("ProcessImage: No calibrated R found for face=%d, points=%d; using default R", faceAndPoints.x, faceAndPoints.y);
            accepted = kf->update(measurement);
        }

        if (accepted) {
            Eigen::Vector3f raw_t = measurement.segment<3>(0);
            Eigen::Vector3f raw_r = measurement.segment<3>(3);
            Eigen::Vector3f filt_t = kf->getTvec();
            Eigen::Vector3f filt_r = kf->getRvec();

            Eigen::Vector3f diff_t = filt_t - raw_t;
            Eigen::Vector3f diff_r = filt_r - raw_r;

            LOG_INFO("=== RAW vs FILTERED ===");
            LOG_INFO("  raw_pos:  (%.4f, %.4f, %.4f)", raw_t(0), raw_t(1), raw_t(2));
            LOG_INFO("  filt_pos: (%.4f, %.4f, %.4f)", filt_t(0), filt_t(1), filt_t(2));
            LOG_INFO("  diff_pos magnitude: %.6f m", diff_t.norm());
            LOG_INFO("  diff_rot magnitude: %.6f rad", diff_r.norm());

            // Log to CSV
            LogToCsv(timestamp, faceAndPoints.x, faceAndPoints.y,
                     raw_t(0), raw_t(1), raw_t(2),
                     raw_r(0), raw_r(1), raw_r(2),
                     filt_t(0), filt_t(1), filt_t(2),
                     filt_r(0), filt_r(1), filt_r(2));
        } else {
            LOG_INFO("ProcessImage: Measurement rejected by innovation gate — holding filtered state");
        }

        // Set pose data
        LOG_DEBUG("ProcessImage: Extracting pose data from filtered state");
        Eigen::Vector3f tvec = kf->getTvec();
        Eigen::Vector3f rvec = kf->getRvec();
        out_pose->tx = tvec(0);
        out_pose->ty = tvec(1);
        out_pose->tz = tvec(2);
        out_pose->rx = rvec(0);
        out_pose->ry = rvec(1);
        out_pose->rz = rvec(2);
        out_pose->grasped = 0;
        out_pose->poseSuccess = 1;
        out_pose->timestamp = timestamp;
        
        LOG_INFO("ProcessImage: SUCCESS - pos=(%.3f, %.3f, %.3f), rot=(%.3f, %.3f, %.3f)",
                out_pose->tx, out_pose->ty, out_pose->tz, 
                out_pose->rx, out_pose->ry, out_pose->rz);
    }
    
    EXPORT_API void RuntimeSetup(float fx, float fy, float cx, float cy, int width, int height, int intrinsics_width=1280, int intrinsics_height=1280)
    {
        LOG_INFO("RuntimeSetup: START - fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f, w=%d, h=%d, intr_w=%d, intr_h=%d",
                fx, fy, cx, cy, width, height, intrinsics_width, intrinsics_height);
        
        // Close previous CSV file and reset for new session
        if (g_csv_file.is_open()) {
            g_csv_file.close();
        }
        g_csv_initialized = false;
        LOG_INFO("RuntimeSetup: CSV file will be overwritten on next frame");
        
        // Define 3D points
        LOG_INFO("RuntimeSetup: Calling getTagPoints3D()");
        std::unordered_map<int, Eigen::MatrixXf> cube_points_eigen = getTagPoints3D();
        LOG_INFO("RuntimeSetup: getTagPoints3D() returned %d tag point sets", (int)cube_points_eigen.size());
        
        LOG_INFO("RuntimeSetup: Converting Eigen to OpenCV points");
        cube_points = eigen2cv_points(cube_points_eigen);
        LOG_INFO("RuntimeSetup: cube_points map now has %d entries", (int)cube_points.size());

        // Scale intrinsics
        float sx = width / (float)intrinsics_width;
        float sy = height / (float)intrinsics_height;

        float fx_scaled = fx * sx;
        float fy_scaled = fy * sy;

        float cx_scaled = cx * sx;
        float cy_scaled = cy * sy;

        LOG_INFO("RuntimeSetup: Scaled intrinsics - fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f (scale: sx=%.3f, sy=%.3f)",
                fx_scaled, fy_scaled, cx_scaled, cy_scaled, sx, sy);

        cv::Mat camera_matrix = (cv::Mat_<float>(3, 3) << fx_scaled, 0, cx_scaled, 0, fy_scaled, cy_scaled, 0, 0, 1);
        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_32F);

        LOG_INFO("RuntimeSetup: Creating PoseSolver");
        poser = new PoseSolver(camera_matrix, dist_coeffs);

        if (poser == nullptr) {
            LOG_ERROR("RuntimeSetup: FAILED to create PoseSolver!");
            return;
        }

        LOG_INFO("RuntimeSetup: Initializing ESEKF");
        // Initialize ESEKF
        Eigen::Matrix<float, 16, 1> x_k = Eigen::Matrix<float, 16, 1>::Zero();
        x_k(9) = 1.0f; // Identity quaternion w
        Eigen::Matrix<float, 15, 1> dx_k = Eigen::Matrix<float, 15, 1>::Zero();
        Eigen::Matrix<float, 15, 15> P_k = Eigen::Matrix<float, 15, 15>::Identity() * 0.5f;
        Eigen::Matrix<float, 6, 6> Q = Eigen::Matrix<float, 6, 6>::Identity();
        // Q.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() * 1E-1f;
        // Q.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() * 8E-2f;
        Q.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() * 1E-1f; // TODO: attempt to tighten Q
        Q.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() * 16E-4f;
        Eigen::Matrix<float, 6, 6> R = Eigen::Matrix<float, 6, 6>::Identity() * 0.1f;
        R.block<3, 3>(0, 0) = Eigen::Vector3f(5E-4,   2E-4,   3E-4).asDiagonal(); // TODO: Tighten value, such a low R could lead to drifting
        R.block<3, 3>(3, 3) = Eigen::Vector3f(6e-3,   1.2e-2,  1.2e-2).asDiagonal();

        kf = new OpenCVKF(x_k, dx_k, P_k, Q, R);

        last_marker_detection_timestamp = 0;
        last_image_timestamp = 0;
        setup_complete = true;
        LOG_INFO("RuntimeSetup: COMPLETE - setup_complete=true, poser=%p", poser);
    }
}

/**
 * Build command
 * 
 * cmake -G Ninja -B build -DCMAKE_TOOLCHAIN_FILE="C:\Program Files\Unity\Hub\Editor\6000.0.61f1\Editor\Data\PlaybackEngines\AndroidPlayer\NDK\build\cmake\android.toolchain.cmake" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 -DANDROID_STL=c++_shared
 * cmake --build build [--config Debug]
 */
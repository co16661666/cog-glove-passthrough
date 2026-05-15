#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
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

struct CameraPose
{
	float tx, ty, tz, rw, rx, ry, rz;
};

class PoseSolver
{
	private:
		cv::Mat tvec;
		cv::Mat rvec;
		cv::Mat last_tvec;
		cv::Mat last_rvec;
		cv::Mat tvec_world;
		cv::Mat rvec_world;
		Eigen::Vector3f tvec_world_eigen;
		Eigen::Vector3f rvec_world_eigen;
		Eigen::Quaternionf rvec_world_quat;

		int last_point_count = -1;
		int timestamp;

		cv::Mat camera_matrix;
		cv::Mat dist_coeffs;
		cv::Point2f intrinsics_resolution;

	public:
		PoseSolver(cv::Mat camera_matrix, cv::Mat dist_coeffs)
		{
			tvec = cv::Mat::zeros(3, 1, CV_64F);
        	rvec = cv::Mat::zeros(3, 1, CV_64F);
			tvec_world = cv::Mat::zeros(3, 1, CV_64F);
        	rvec_world = cv::Mat::zeros(3, 1, CV_64F);
			rvec_world_quat = Eigen::Quaternionf::Identity();
			
			timestamp = 0;

			this->camera_matrix = camera_matrix;
			this->dist_coeffs = dist_coeffs;
		}
		
		void solvePose(const std::vector<cv::Point3f>& obj_points, const std::vector<cv::Point2f>& image_points)
		{
			LOG_DEBUG("PoseSolver::solvePose: START - obj_points=%d, img_points=%d", 
					 (int)obj_points.size(), (int)image_points.size());
					 
			cv::Mat inliers;

			const bool has_prior = !last_rvec.empty() && !last_tvec.empty() && last_point_count == (int)obj_points.size();
			
			if (has_prior) {
				rvec = last_rvec.clone();
				tvec = last_tvec.clone();
			}

			const bool success = cv::solvePnPRansac(
				obj_points, image_points,
				camera_matrix, dist_coeffs,
				rvec, tvec,
				has_prior,
				100, 3.0, 0.95, inliers, // TODO: tune reprojection error
				has_prior ? cv::SOLVEPNP_ITERATIVE : cv::SOLVEPNP_SQPNP
			);

			if (!success)
			{
				LOG_ERROR("PoseSolver::solvePose: PnP solve FAILED!");
				return;
			}
			
			// Check if rvec and tvec are within range and physically possible; check if RANSAC succeeded with no inliers (shouldn't happen)
			if (!cv::checkRange(rvec, true, nullptr, -1e5, 1e5) || tvec.at<double>(2) < 0)
			{
				LOG_ERROR("PoseSolver::solvePose: rvec/tvec out of range or Z negative! rvec_range_ok=%d, z=%.3f",
						 cv::checkRange(rvec, true, nullptr, -1e5, 1e5), tvec.at<double>(2));
				last_rvec.release();
				last_tvec.release();

				// TODO: Reset Kalman Filter maybe
				return;
			}

			LOG_DEBUG("PoseSolver::solvePose: Pose validation passed");
			
			// Default assumes RANSAC failed and using regular solvePnP
			std::vector<cv::Point3f> obj_points_inliers; // 3D object points corresponding to inliers
			std::vector<cv::Point2f> image_points_inliers; // 2D image point corresponding to inliers

			// If there are inliers from RANSAC
			if (!inliers.empty())
			{
				LOG_DEBUG("PoseSolver::solvePose: Using %d inliers from RANSAC", inliers.rows);
				for (int idx = 0; idx < inliers.rows; idx++)
				{
					obj_points_inliers.push_back(obj_points[inliers.at<int>(idx)]);
					image_points_inliers.push_back(image_points[inliers.at<int>(idx)]);
				}
			}
			else
			{
				LOG_DEBUG("PoseSolver::solvePose: No inliers, using all points");
				// If regular solvePnP used, use all points (equivalent to regular solvePnP)
				obj_points_inliers = obj_points;
				image_points_inliers = image_points;
			}

			try
			{
				LOG_DEBUG("PoseSolver::solvePose: Calling solvePnPRefineVVS");
				cv::solvePnPRefineVVS(
					obj_points_inliers, image_points_inliers,
					camera_matrix, dist_coeffs,
					rvec, tvec
				);
				LOG_DEBUG("PoseSolver::solvePose: VVS refinement completed");
			}
			catch (const cv::Exception& e)
			{
				LOG_ERROR("PoseSolver::solvePose: VVS refinement threw exception: %s", e.what());
				return;
			}

			// Store values for the next frame
			rvec.copyTo(last_rvec);
			tvec.copyTo(last_tvec);
			last_point_count = (int)obj_points.size();
		}

		void cameraToWorld(CameraPose cam)
		{
			LOG_DEBUG("PoseSolver::cameraToWorld: START - cam_pos=(%.3f, %.3f, %.3f), cam_rot=(%.3f, %.3f, %.3f, %.3f)",
					 cam.tx, cam.ty, cam.tz, cam.rw, cam.rx, cam.ry, cam.rz);
			
			// Camera position and rotation
			Eigen::Quaternionf cam_rot(-cam.rw, cam.rx, -cam.ry, cam.rz); // Must convert Unity camera pose to OpenCV right-hand coord
			cam_rot.normalize();
			Eigen::Vector3f cam_pos(cam.tx, -cam.ty, cam.tz);
			
			LOG_DEBUG("PoseSolver::cameraToWorld: Camera position normalized");
			
			// Cube position and rotation relative to camera
			// Convert rvec to Eigen
			Eigen::Matrix3d rvec_eigen_double;
			cv::Mat rvec_cv;
			cv::Rodrigues(rvec, rvec_cv); // 3 x 1 -> 3 x 3 matrix
			cv::cv2eigen(rvec_cv, rvec_eigen_double); // cv to eigen
			Eigen::AngleAxisf rvec_eigen(rvec_eigen_double.cast<float>()); // Convert to angle-axis representation
			// Convert tvec to Eigen
			Eigen::Vector3d tvec_eigen_double;
			cv::cv2eigen(tvec, tvec_eigen_double);
			Eigen::Vector3f tvec_eigen = tvec_eigen_double.cast<float>();

			LOG_DEBUG("PoseSolver::cameraToWorld: Converted camera frame pose to Eigen");

			// Convert cube tvec, rvec in camera frame to world frame (rvec_world, tvec_world)
			rvec_world_quat = cam_rot * rvec_eigen;
			rvec_world_quat.normalize();
			Eigen::AngleAxisf rvec_world_aa(rvec_world_quat);
			rvec_world_eigen = rvec_world_aa.axis() * rvec_world_aa.angle();
			tvec_world_eigen = cam_pos + cam_rot * tvec_eigen;

			cv::eigen2cv(rvec_world_eigen, this->rvec_world);
    		cv::eigen2cv(tvec_world_eigen, this->tvec_world);
    		
    		LOG_DEBUG("PoseSolver::cameraToWorld: COMPLETE - world_pos=(%.3f, %.3f, %.3f), world_rot=(%.3f, %.3f, %.3f)",
    				 tvec_world_eigen.x(), tvec_world_eigen.y(), tvec_world_eigen.z(),
    				 rvec_world_eigen.x(), rvec_world_eigen.y(), rvec_world_eigen.z());
		}

		cv::Mat getTvecWorld() { return tvec_world; }
		cv::Mat getRvecWorld() { return rvec_world; }
		Eigen::Vector3f getTvecWorldEigen() { return tvec_world_eigen; }
		Eigen::Vector3f getRvecWorldEigen() { return rvec_world_eigen; } // Convert quaternion to angle-axis vector
		Eigen::Quaternionf getRvecWorldQuat() { return rvec_world_quat; }
	};

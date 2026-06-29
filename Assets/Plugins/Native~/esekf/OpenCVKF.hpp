#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Dense>

class OpenCVKF
{
    private:
        Eigen::Matrix<float, 16,  1> x_k; // State vector
        Eigen::Matrix<float, 15,  1> dx_k; // Error state
        Eigen::Matrix<float, 15, 15> P_k; // Error covariance
        Eigen::Matrix<float,  6,  6> Q; // Covariance of process noise (noisy movement)
        Eigen::Matrix<float,  6,  6> R; // Covariance of measurement noise (noisy sensor)
        // TODO: Validate
        float q_multiplier = 1.0f;          // Dynamically adjusted
        const float q_multiplier_max = 1000.0f; // How much to trust camera during "bursts"
        const float q_decay = 0.95f;        // How fast to return to "smooth" mode
        int frames_since_update = 0;        // Track occlusions

    public:
        OpenCVKF(Eigen::Matrix<float, 16,  1>& x_k, Eigen::Matrix<float, 15,  1>& dx_k, Eigen::Matrix<float, 15, 15>& P_k, Eigen::Matrix<float,  6,  6>& Q, Eigen::Matrix<float,  6,  6>& R)
        {
            this->x_k = x_k;
            this->dx_k = dx_k;
            this->P_k = P_k;
            this->Q = Q;
            this->R = R;
        }

        // PREDICTION STEP
        void predict(float dt)
        {
            frames_since_update++;

            x_k.segment<3>(0) += x_k.segment<3>(3) * dt + 0.5f * x_k.segment<3>(6) * std::pow(dt, 2);
            x_k.segment<3>(3) += x_k.segment<3>(6) * dt;
            // x_k.segment<3>(6) = x_k.segment<3>(6); constant acceleration assumption

            Eigen::Quaternionf q_k(x_k(9), x_k(10), x_k(11), x_k(12));
            Eigen::Vector3f rot_vec = x_k.segment<3>(13) * dt;
            Eigen::Quaternionf wt = rotvec_to_quat(rot_vec);

            Eigen::Quaternionf rotation = wt * q_k;
            rotation.normalize();

            x_k.segment<4>(9) << rotation.w(), rotation.x(), rotation.y(), rotation.z();
            // x_k.segment<3>(13) = x_k.segment<3>(13); constant angular velocity assumption
        
            // ERROR PREDICTION
            // Define F and L matrices
            Eigen::Matrix<float, 15,15> F_k = Eigen::Matrix<float, 15,15>::Identity();
            F_k.block<3, 3>(0, 3)  = Eigen::Matrix3f::Identity() * dt;
            F_k.block<3, 3>(0, 6)  = Eigen::Matrix3f::Identity() * 0.5f * std::pow(dt, 2);
            F_k.block<3, 3>(3, 6)  = Eigen::Matrix3f::Identity() * dt;
            F_k.block<3, 3>(9, 12) = Eigen::Matrix3f::Identity() * dt;

            Eigen::Matrix<float, 15, 6> L_k = Eigen::Matrix<float, 15, 6>::Zero();
            L_k.block<3, 3>(6, 0) = Eigen::Matrix3f::Identity();
            L_k.block<3, 3>(12, 3) = Eigen::Matrix3f::Identity();

            Eigen::Matrix<float,  6,  6> Q_k = Q * dt * q_multiplier; // TODO: Validate dynamic Q scaling
            // if (frames_since_update > 1) { // TODO: Validate occlusion inflation
            //     Q_k *= (1.0f + (0.5f * frames_since_update)); 
            // }
            
            P_k = F_k * P_k * F_k.transpose() + L_k * Q_k * L_k.transpose();
            P_k = 0.5f * (P_k + P_k.transpose()); // Symmetrize

            debugPredict();
        }

        // UPDATE STEP
        bool update(const Eigen::Matrix<float, 6, 1>& y_k, const Eigen::Matrix<float,  6,  6>* R_override = nullptr, float gate_threshold = 20.0f)
        {
            // Define H matrix
            Eigen::Matrix<float, 6, 15> H_k = Eigen::Matrix<float, 6, 15>::Zero();
            H_k.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
            H_k.block<3, 3>(3, 9) = Eigen::Matrix3f::Identity();

            // Check for calibrated R, otherwise use default
            const Eigen::Matrix<float, 6, 6>& R_active = (R_override != nullptr) ? *R_override : this->R;

            // Calculate Kalman gain
            Eigen::Matrix<float, 6, 6> S = H_k * P_k * H_k.transpose() + R_active;
            Eigen::Matrix<float, 15, 6> K_k = P_k * H_k.transpose() * S.inverse();

            // Calculate difference between measured and predicted (residual)
            Eigen::Matrix<float, 6, 1> z_k = Eigen::Matrix<float, 6, 1>::Zero();
            z_k.segment<3>(0) = y_k.segment<3>(0) - x_k.segment<3>(0); // Translation residual

            Eigen::Quaternionf rot_y = rotvec_to_quat(y_k.segment<3>(3));
            Eigen::Quaternionf rot_k(x_k(9), x_k(10), x_k(11), x_k(12));
            Eigen::AngleAxisf rot_residual(rot_y * rot_k.inverse());
            z_k.segment<3>(3) = rot_residual.axis() * rot_residual.angle(); // Rotation residual

            debugUpdate(K_k, z_k, S, R_active);
            
            float mahal = z_k.transpose() * S.inverse() * z_k;
            LOG_DEBUG("OpenCVKF::update: Innovation gate mahalanobis distance = %.2f", mahal);
            if (mahal > 2000.0f) { // TODO: Separate rotation and translation gating
                // Multiply covariance to force the gate wider for the next frame
                // q_multiplier = std::min(q_multiplier_max, mahal / gate_threshold);
                return false; 
            }
            else if (mahal > gate_threshold) {
                LOG_INFO("OpenCVKF::update: Innovation gate increasing Q (mahal=%.2f > %.2f)", mahal, gate_threshold);
                q_multiplier = std::min(q_multiplier_max, mahal / gate_threshold);
            }
            else
            {
                q_multiplier = std::max(1.0f, q_multiplier * q_decay);
            }

            // Update error state
            dx_k = K_k * z_k;

            // Update state error
            P_k = (Eigen::Matrix<float, 15, 15>::Identity() - K_k * H_k) * P_k * (Eigen::Matrix<float, 15, 15>::Identity() - K_k * H_k).transpose() + K_k * R_active * K_k.transpose();
            P_k = 0.5f * (P_k + P_k.transpose()); // Symmetrize

            // Update state vector
            x_k.segment<9>(0) += dx_k.segment<9>(0);

            Eigen::Quaternionf q_k(x_k(9), x_k(10), x_k(11), x_k(12));
            Eigen::Quaternionf dq = rotvec_to_quat(dx_k.segment<3>(9));
            Eigen::Quaternionf updated_q = dq * q_k;
            updated_q.normalize();
            x_k.segment<4>(9) << updated_q.w(), updated_q.x(), updated_q.y(), updated_q.z();

            x_k.segment<3>(13) += dx_k.segment<3>(12);

            dx_k.setZero();

            frames_since_update = 0; // Reset occlusion counter on successful update
            return true;
        }

        void debugPredict() const
        {
            LOG_INFO("=== P DIAGONAL ===");
            LOG_INFO("  P_pos:    (%.6f, %.6f, %.6f)", P_k(0,0), P_k(1,1), P_k(2,2));
            LOG_INFO("  P_vel:    (%.6f, %.6f, %.6f)", P_k(3,3), P_k(4,4), P_k(5,5));
            LOG_INFO("  P_acc:    (%.6f, %.6f, %.6f)", P_k(6,6), P_k(7,7), P_k(8,8));
            LOG_INFO("  P_rot:    (%.6f, %.6f, %.6f)", P_k(9,9), P_k(10,10), P_k(11,11));
            LOG_INFO("  P_angvel: (%.6f, %.6f, %.6f)", P_k(12,12), P_k(13,13), P_k(14,14));
            LOG_INFO("  P_trace: %.6f", P_k.trace());
        }

        void debugUpdate(const Eigen::Matrix<float, 15, 6>& K_k,
                         const Eigen::Matrix<float, 6, 1>& z_k,
                         const Eigen::Matrix<float, 6, 6>& S,
                         const Eigen::Matrix<float, 6, 6>& R_active) const
        {
            // Raw K gain for observed states
            LOG_INFO("=== KALMAN GAIN (observed state rows) ===");
            LOG_INFO("  K_pos:    (%.6f, %.6f, %.6f)", K_k(0,0), K_k(1,1), K_k(2,2));
            LOG_INFO("  K_vel:    (%.6f, %.6f, %.6f)", K_k(3,3), K_k(4,4), K_k(5,5));
            LOG_INFO("  K_rot:    (%.6f, %.6f, %.6f)", K_k(9,3), K_k(10,4), K_k(11,5));

            // Residual — if this is ~0, measurement == prediction
            LOG_INFO("=== RESIDUAL z_k ===");
            LOG_INFO("  z_pos (m):   (%.6f, %.6f, %.6f)", z_k(0), z_k(1), z_k(2));
            LOG_INFO("  z_rot (rad): (%.6f, %.6f, %.6f)", z_k(3), z_k(4), z_k(5));

            // S diagonal — innovation covariance; K = P*H^T * S^-1
            // If S ≈ R, P has collapsed. If S >> R, P is dominating.
            LOG_INFO("=== INNOVATION COV S diag ===");
            LOG_INFO("  S_pos: (%.6f, %.6f, %.6f)", S(0,0), S(1,1), S(2,2));
            LOG_INFO("  S_rot: (%.6f, %.6f, %.6f)", S(3,3), S(4,4), S(5,5));
            LOG_INFO("=== R_active diag ===");
            LOG_INFO("  R_pos: (%.6f, %.6f, %.6f)", R_active(0,0), R_active(1,1), R_active(2,2));
            LOG_INFO("  R_rot: (%.6f, %.6f, %.6f)", R_active(3,3), R_active(4,4), R_active(5,5));
            LOG_INFO("  S/R ratio pos: (%.3f, %.3f, %.3f)",
                S(0,0)/R_active(0,0), S(1,1)/R_active(1,1), S(2,2)/R_active(2,2));

            // Correction actually applied to state
            Eigen::Matrix<float, 15, 1> dx = K_k * z_k;
            LOG_INFO("=== CORRECTION dx ===");
            LOG_INFO("  dx_pos (m):    (%.6f, %.6f, %.6f)", dx(0), dx(1), dx(2));
            LOG_INFO("  dx_vel (m/s):  (%.6f, %.6f, %.6f)", dx(3), dx(4), dx(5));
            LOG_INFO("  dx_rot (rad):  (%.6f, %.6f, %.6f)", dx(9), dx(10), dx(11));
        }

        Eigen::Quaternionf rotvec_to_quat(const Eigen::Vector3f& rot_vec)
        {
            float angle = rot_vec.norm();
            if (angle < 1e-12) {
                return Eigen::Quaternionf::Identity();
            } else {
                return Eigen::Quaternionf(Eigen::AngleAxisf(angle, rot_vec.normalized()));
            }
        }

        // GETTER FUNCTIONS
        Eigen::Vector3f getTvec() const
        {
            return x_k.segment<3>(0);
        }

        Eigen::Vector3f getRvec() const
        {
            Eigen::Quaternionf q(x_k(9), x_k(10), x_k(11), x_k(12));
            Eigen::AngleAxisf angle_axis(q);
            return angle_axis.axis() * angle_axis.angle();
        }

        Eigen::Quaternionf getRvecQuaternion() const
        {
            return Eigen::Quaternionf(x_k(9), x_k(10), x_k(11), x_k(12));
        }

        // RESET METHOD
        void softReset()
        {
            // Zero velocity, acceleration, angular velocity — stop dead-reckoning
            x_k.segment<3>(3).setZero();   // linear velocity
            x_k.segment<3>(6).setZero();   // linear acceleration
            x_k.segment<3>(13).setZero();  // angular velocity
            dx_k.setZero();
            // Large uncertainty — next measurement dominates
            P_k = Eigen::Matrix<float, 15, 15>::Identity() * 0.5f;
        }

        // Full reset: only call when you truly have no prior (first init).
        void hardReset()
        {
            x_k = Eigen::Matrix<float, 16, 1>::Zero();
            x_k(9) = 1.0f;
            dx_k.setZero();
            P_k = Eigen::Matrix<float, 15, 15>::Identity() * 0.5f;
        }
};
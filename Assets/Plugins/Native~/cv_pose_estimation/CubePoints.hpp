#include <iostream>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <Eigen/Dense>

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

const float TAG_AREA = 0.031f;
const float TAG_WIDTH = 0.01323;
const float MARGIN = 0.0045;

// Front View
std::unordered_map<int, Eigen::MatrixXf> template_tag = {
    // TL
    {0, (Eigen::MatrixXf(4, 3) << 
        -TAG_AREA / 2, TAG_AREA / 2, 0,
        -MARGIN / 2, TAG_AREA / 2, 0,
        -MARGIN / 2, MARGIN / 2, 0,
        -TAG_AREA / 2, MARGIN / 2, 0).finished()
    },
    // TR
    {1, (Eigen::MatrixXf(4, 3) << 
        MARGIN / 2, TAG_AREA / 2, 0,
        TAG_AREA / 2, TAG_AREA / 2, 0,
        TAG_AREA / 2, MARGIN / 2, 0,
        MARGIN / 2, MARGIN / 2, 0).finished()
    },
    // BL
    {2, (Eigen::MatrixXf(4, 3) << 
        -TAG_AREA / 2, -MARGIN / 2, 0,
        -MARGIN / 2, -MARGIN / 2, 0,
        -MARGIN / 2, -TAG_AREA / 2, 0,
        -TAG_AREA / 2, -TAG_AREA / 2, 0).finished()
    },
    // BR
    {3, (Eigen::MatrixXf(4, 3) << 
        MARGIN / 2, -MARGIN / 2, 0,
        TAG_AREA / 2, -MARGIN / 2, 0,
        TAG_AREA / 2, -TAG_AREA / 2, 0,
        MARGIN / 2, -TAG_AREA / 2, 0).finished()
    }
};

Eigen::MatrixXf get_rotation_matrix_90(char axis, int num_rotations)
{
    int sin90 = 1;
    Eigen::MatrixXf rot_mat = Eigen::MatrixXf::Identity(3, 3);
    Eigen::MatrixXf final_rot = Eigen::MatrixXf::Identity(3, 3);

    if (num_rotations == 0)
    {
        return final_rot;
    }
    else if (num_rotations > 0)
    {
        sin90 = 1;
    }
    else
    {
        sin90 = -1;
    }

    if (axis == 'x')
    {
        rot_mat = (Eigen::MatrixXf(3, 3) << 
            1, 0, 0,
            0, 0, -sin90,
            0, sin90, 0).finished();
    }
    else if (axis == 'y')
    {
        rot_mat = (Eigen::MatrixXf(3, 3) << 
            0, 0, sin90,
            0, 1, 0,
            -sin90, 0, 0).finished();
    }
    else
    {
        rot_mat = (Eigen::MatrixXf(3, 3) << 
            0, -sin90, 0,
            sin90, 0, 0,
            0, 0, 1).finished();
    }

    for (int i = 0; i < std::abs(num_rotations); i++)
    {
        final_rot *= rot_mat;
    }

    return final_rot;
}

std::unordered_map<int, Eigen::MatrixXf> getTagPoints3D()
{
    LOG_DEBUG("CubePoints::getTagPoints3D: START");
    std::unordered_map<int, Eigen::MatrixXf> tag_points_3D;

    char rot_axis = 'x';
    int rot_amount = 0;
    Eigen::MatrixXf offset = (Eigen::MatrixXf(4, 3) << 
        MARGIN / 2, -MARGIN / 2, 0,
        TAG_AREA / 2, -MARGIN / 2, 0,
        TAG_AREA / 2, -TAG_AREA / 2, 0,
        MARGIN / 2, -TAG_AREA / 2, 0).finished();
    
    for (int i = 0; i < 24; i += 4)
    {
        if (i == 0)
        {
            // Top Face
            rot_axis = 'x';
            rot_amount = -1; // Num 90 deg rotations, negative indicates -90 deg rotation

            offset = (Eigen::MatrixXf(4, 3) << 
                0, MARGIN + TAG_AREA / 2, 0,
                0, MARGIN + TAG_AREA / 2, 0,
                0, MARGIN + TAG_AREA / 2, 0,
                0, MARGIN + TAG_AREA / 2, 0).finished();
        }
        else if (i == 4)
        {
            rot_axis = 'y';
            rot_amount = -2;

            offset = (Eigen::MatrixXf(4, 3) << 
                0, 0, -(MARGIN + TAG_AREA / 2),
                0, 0, -(MARGIN + TAG_AREA / 2),
                0, 0, -(MARGIN + TAG_AREA / 2),
                0, 0, -(MARGIN + TAG_AREA / 2)).finished();
        }
        else if (i == 8)
        {
            rot_axis = 'y';
            rot_amount = -1;

            offset = (Eigen::MatrixXf(4, 3) << 
                -(MARGIN + TAG_AREA / 2), 0, 0,
                -(MARGIN + TAG_AREA / 2), 0, 0,
                -(MARGIN + TAG_AREA / 2), 0, 0,
                -(MARGIN + TAG_AREA / 2), 0, 0).finished();
        }
        else if (i == 12)
        {
            rot_axis = 'y';
            rot_amount = 0;

            offset = (Eigen::MatrixXf(4, 3) << 
                0, 0, MARGIN + TAG_AREA / 2,
                0, 0, MARGIN + TAG_AREA / 2,
                0, 0, MARGIN + TAG_AREA / 2,
                0, 0, MARGIN + TAG_AREA / 2).finished();
        }
        else if (i == 16)
        {
            rot_axis = 'y';
            rot_amount = 1;

            offset = (Eigen::MatrixXf(4, 3) << 
                MARGIN + TAG_AREA / 2, 0, 0,
                MARGIN + TAG_AREA / 2, 0, 0,
                MARGIN + TAG_AREA / 2, 0, 0,
                MARGIN + TAG_AREA / 2, 0, 0).finished();
        }
        else if (i == 20)
        {
            rot_axis = 'x';
            rot_amount = 1;

            offset = (Eigen::MatrixXf(4, 3) << 
                0, -(MARGIN + TAG_AREA / 2), 0,
                0, -(MARGIN + TAG_AREA / 2), 0,
                0, -(MARGIN + TAG_AREA / 2), 0,
                0, -(MARGIN + TAG_AREA / 2), 0).finished();
        }
        
        Eigen::MatrixXf rotation_matrix = get_rotation_matrix_90(rot_axis, rot_amount);
        rotation_matrix.transposeInPlace();

        tag_points_3D[i] = template_tag[i % 4] * rotation_matrix + offset;
        tag_points_3D[i + 1] = template_tag[(i + 1) % 4] * rotation_matrix + offset;
        tag_points_3D[i + 2] = template_tag[(i + 2) % 4] * rotation_matrix + offset;
        tag_points_3D[i + 3] = template_tag[(i + 3) % 4] * rotation_matrix + offset;
    }

    LOG_INFO("CubePoints::getTagPoints3D: COMPLETE - created %d tag point sets", (int)tag_points_3D.size());
    return tag_points_3D;
};

std::unordered_map<int, std::vector<cv::Point3f>> eigen2cv_points(std::unordered_map<int, Eigen::MatrixXf> eigen_points)
{
    LOG_DEBUG("CubePoints::eigen2cv_points: START - converting %d point sets", (int)eigen_points.size());
    std::unordered_map<int, std::vector<cv::Point3f>> cv_map;
    for (const auto& [id, matrix] : eigen_points)
    {
        std::vector<cv::Point3f> points;
        for (int i = 0; i < matrix.rows(); i++)
        {
            points.emplace_back(matrix(i, 0), matrix(i, 1), matrix(i, 2));
        }
        cv_map[id] = points;
        LOG_DEBUG("CubePoints::eigen2cv_points: Set %d has %d points", id, (int)points.size());
    }

    LOG_INFO("CubePoints::eigen2cv_points: COMPLETE - created cv_map with %d entries", (int)cv_map.size());
    return cv_map;
}
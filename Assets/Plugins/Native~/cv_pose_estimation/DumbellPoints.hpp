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

// ---------------------------------------------------------------------------
// Object geometry
//
// Two 50x50x50mm cubes joined along the X axis by a 30mm, un-tagged spacer.
// Axes (all units in meters, object frame centered at the middle of the
// spacer):
//   X : left (-) <-> right (+)   -- the axis the cubes are joined along
//   Y : bottom (-) <-> top (+)
//   Z : back (-) <-> front (+)
//
// Cube A sits on the -X side, Cube B sits on the +X side. Each cube keeps
// tags on every face except the one glued to the spacer (Cube A has no
// "right" tag, Cube B has no "left" tag).
//
// Each face carries a single 40x40mm marker, centered on the 50x50mm face,
// which is exactly consistent with a 5mm margin on every side.
// ---------------------------------------------------------------------------

const float CUBE_SIZE   = 0.050f;   // 50 mm
const float SPACER_SIZE = 0.030f;   // 30 mm
const float MARKER_SIZE = 0.040f;   // 40 mm
const float MARGIN      = 0.005f;   // 5 mm (per side) -- kept for reference/checks

const float HALF_CUBE   = CUBE_SIZE / 2.0f;    // 0.025
const float HALF_MARKER = MARKER_SIZE / 2.0f;  // 0.020

// Sanity check (not enforced at runtime): HALF_CUBE - HALF_MARKER should equal MARGIN.

// Cube centers along X
const float CUBE_A_X = -(SPACER_SIZE / 2.0f + HALF_CUBE);  // -0.040
const float CUBE_B_X =  (SPACER_SIZE / 2.0f + HALF_CUBE);  //  0.040

// Outer (tagged) face planes along X
const float A_OUTER_X = CUBE_A_X - HALF_CUBE;  // -0.065  (Cube A "left" face)
const float B_OUTER_X = CUBE_B_X + HALF_CUBE;  //  0.065  (Cube B "right" face)

// Marker-edge X positions on the faces that run along the join axis
// (top/bottom/front/back of each cube). "NEAR" = edge closest to the spacer,
// "FAR" = edge closest to the outer end of the assembly.
const float A_X_FAR  = CUBE_A_X - HALF_MARKER;  // -0.060  ("left")
const float A_X_NEAR = CUBE_A_X + HALF_MARKER;  // -0.020  ("center_left")
const float B_X_NEAR = CUBE_B_X - HALF_MARKER;  //  0.020  ("center_right")
const float B_X_FAR  = CUBE_B_X + HALF_MARKER;  //  0.060  ("right")

// Marker-edge Y / Z positions (shared by both cubes, since only X shifts)
const float Y_TOP = HALF_MARKER;    //  0.020
const float Y_BOT = -HALF_MARKER;   // -0.020
const float Z_FRONT = HALF_MARKER;  //  0.020
const float Z_BACK  = -HALF_MARKER; // -0.020

// Face plane positions (Y/Z) for top/bottom/back/front faces
const float TOP_Y    =  HALF_CUBE;  //  0.025
const float BOTTOM_Y = -HALF_CUBE;  // -0.025
const float BACK_Z   = -HALF_CUBE;  // -0.025
const float FRONT_Z  =  HALF_CUBE;  //  0.025

// ---------------------------------------------------------------------------
// Corner ordering convention
//
// Each tag's 4 rows are the marker corners in standard ArUco order (as used
// by cv::aruco / solvePnP object points): clockwise, starting at the corner
// you specified as that face's origin, when the face is viewed head-on from
// outside the object.
//
// To resolve "clockwise as viewed from outside" into actual 3D directions,
// each face uses a face-local (right, up) basis:
//   left_plane / right_plane / front_plane / back_plane : up = +Y (world up)
//   top_plane  / bottom_plane                            : up = +Z (front-is-up)
// This second choice (top/bottom "up") can't be derived from your written
// spec -- it's a convention I picked so the code is internally consistent.
// If solvePnP produces a visibly flipped/rotated pose for a top or bottom
// tag, this is the first thing to revisit (rotate that tag's 4 rows by
// two positions, i.e. swap TL<->BR and TR<->BL, to flip the up reference).
//
// Row 0 in every matrix below = the origin corner you specified.
// ---------------------------------------------------------------------------

std::unordered_map<int, Eigen::MatrixXf> getTagPoints3D()
{
    LOG_DEBUG("DumbbellPoints::getTagPoints3D: START");
    std::unordered_map<int, Eigen::MatrixXf> tag_points_3D;

    // ---- Cube A (left cube) ----

    // id 0: left_plane -- origin: left, bottom, back
    tag_points_3D[0] = (Eigen::MatrixXf(4, 3) <<
        A_OUTER_X, Y_BOT,  Z_BACK,
        A_OUTER_X, Y_TOP,  Z_BACK,
        A_OUTER_X, Y_TOP,  Z_FRONT,
        A_OUTER_X, Y_BOT,  Z_FRONT).finished();

    // id 1: top_plane (A) -- origin: left, top, back
    tag_points_3D[1] = (Eigen::MatrixXf(4, 3) <<
        A_X_FAR,  TOP_Y, Z_BACK,
        A_X_NEAR, TOP_Y, Z_BACK,
        A_X_NEAR, TOP_Y, Z_FRONT,
        A_X_FAR,  TOP_Y, Z_FRONT).finished();

    // id 2: bottom_plane (A) -- origin: center_left, bottom, back
    tag_points_3D[2] = (Eigen::MatrixXf(4, 3) <<
        A_X_NEAR, BOTTOM_Y, Z_BACK,
        A_X_FAR,  BOTTOM_Y, Z_BACK,
        A_X_FAR,  BOTTOM_Y, Z_FRONT,
        A_X_NEAR, BOTTOM_Y, Z_FRONT).finished();

    // id 3: back_plane (A) -- origin: center_left, bottom, back
    tag_points_3D[3] = (Eigen::MatrixXf(4, 3) <<
        A_X_NEAR, Y_BOT, BACK_Z,
        A_X_NEAR, Y_TOP, BACK_Z,
        A_X_FAR,  Y_TOP, BACK_Z,
        A_X_FAR,  Y_BOT, BACK_Z).finished();

    // id 4: front_plane (A) -- origin: left, bottom, front
    tag_points_3D[4] = (Eigen::MatrixXf(4, 3) <<
        A_X_FAR,  Y_BOT, FRONT_Z,
        A_X_FAR,  Y_TOP, FRONT_Z,
        A_X_NEAR, Y_TOP, FRONT_Z,
        A_X_NEAR, Y_BOT, FRONT_Z).finished();

    // ---- Cube B (right cube) ----

    // id 5: right_plane -- origin: right, top, back
    tag_points_3D[5] = (Eigen::MatrixXf(4, 3) <<
        B_OUTER_X, Y_TOP, Z_BACK,
        B_OUTER_X, Y_BOT, Z_BACK,
        B_OUTER_X, Y_BOT, Z_FRONT,
        B_OUTER_X, Y_TOP, Z_FRONT).finished();

    // id 6: top_plane (B) -- origin: center_right, top, back
    tag_points_3D[6] = (Eigen::MatrixXf(4, 3) <<
        B_X_NEAR, TOP_Y, Z_BACK,
        B_X_FAR,  TOP_Y, Z_BACK,
        B_X_FAR,  TOP_Y, Z_FRONT,
        B_X_NEAR, TOP_Y, Z_FRONT).finished();

    // id 7: bottom_plane (B) -- origin: center_right, bottom, back
    tag_points_3D[7] = (Eigen::MatrixXf(4, 3) <<
        B_X_NEAR, BOTTOM_Y, Z_BACK,
        B_X_NEAR, BOTTOM_Y, Z_FRONT,
        B_X_FAR,  BOTTOM_Y, Z_FRONT,
        B_X_FAR,  BOTTOM_Y, Z_BACK).finished();

    // id 8: back_plane (B) -- origin: center_right, top, back
    tag_points_3D[8] = (Eigen::MatrixXf(4, 3) <<
        B_X_NEAR, Y_TOP, BACK_Z,
        B_X_NEAR, Y_BOT, BACK_Z,
        B_X_FAR,  Y_BOT, BACK_Z,
        B_X_FAR,  Y_TOP, BACK_Z).finished();

    // id 9: front_plane (B) -- origin: far right, top, front
    tag_points_3D[9] = (Eigen::MatrixXf(4, 3) <<
        B_X_FAR,  Y_TOP, FRONT_Z,
        B_X_FAR,  Y_BOT, FRONT_Z,
        B_X_NEAR, Y_BOT, FRONT_Z,
        B_X_NEAR, Y_TOP, FRONT_Z).finished();

    LOG_INFO("DumbbellPoints::getTagPoints3D: COMPLETE - created %d tag point sets", (int)tag_points_3D.size());
    return tag_points_3D;
};

std::unordered_map<int, std::vector<cv::Point3f>> eigen2cv_points(std::unordered_map<int, Eigen::MatrixXf> eigen_points)
{
    LOG_DEBUG("DumbbellPoints::eigen2cv_points: START - converting %d point sets", (int)eigen_points.size());
    std::unordered_map<int, std::vector<cv::Point3f>> cv_map;
    for (const auto& [id, matrix] : eigen_points)
    {
        std::vector<cv::Point3f> points;
        for (int i = 0; i < matrix.rows(); i++)
        {
            points.emplace_back(matrix(i, 0), matrix(i, 1), matrix(i, 2));
        }
        cv_map[id] = points;
        LOG_DEBUG("DumbbellPoints::eigen2cv_points: Set %d has %d points", id, (int)points.size());
    }

    LOG_INFO("DumbbellPoints::eigen2cv_points: COMPLETE - created cv_map with %d entries", (int)cv_map.size());
    return cv_map;
}
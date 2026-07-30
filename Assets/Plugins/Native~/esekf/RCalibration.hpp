#include <array>
#include <opencv2/core.hpp>

#include <Eigen/Dense>
#include <optional>

static constexpr float R_SCALE_TRANS = 0.5f;
static constexpr float R_SCALE_ROT   = 0.5f;

constexpr int MakeKey(int faceCount, int numPoints)
{
    return faceCount * 1000 + numPoints;
}

struct LookupEntry
{
    int key;
    std::array<double, 6> weights; // {tx, ty, tz, rx, ry, rz}
};

// static constexpr std::array<LookupEntry, 12> R_lookup = {{
//     { MakeKey(1, 16), { 5e-6,   2e-6,   3e-6,   6e-4,   1.2e-3,  1.2e-3 } },
//     { MakeKey(2, 20), { 3e-6,   4e-6,   4e-6,   2.6e-4, 2.4e-4,  2.1e-4 } },
//     { MakeKey(2, 24), { 2e-6,   2e-6,   4e-6,   1.5e-4, 1.1e-4,  1.6e-4 } },
//     { MakeKey(2, 28), { 2e-6,   1e-6,   3e-6,   7.1e-5, 7.4e-5,  1.1e-4 } },
//     { MakeKey(2, 32), { 1e-6,   1e-6,   3e-6,   4.8e-5, 4.3e-5,  9.4e-5 } },
//     { MakeKey(3, 24), { 1e-6,   1e-6,   1e-6,   9.6e-5, 2.4e-4,  2.0e-4 } },
//     { MakeKey(3, 28), { 1e-6,   1e-6,   1e-6,   9.0e-5, 1.9e-4,  1.6e-4 } },
//     { MakeKey(3, 32), { 1e-6,   2e-6,   1e-6,   6.3e-5, 1.1e-4,  1.1e-4 } },
//     { MakeKey(3, 36), { 1e-6,   1e-6,   1e-6,   6.0e-5, 9.0e-5,  8.3e-5 } },
//     { MakeKey(3, 40), { 1e-6,   2e-6,   5e-6,   5.9e-4, 2.3e-3,  2.3e-3 } },
//     { MakeKey(3, 44), { 1e-6,   1e-6,   1e-6,   3.6e-5, 5.1e-5,  4.8e-5 } },
//     { MakeKey(3, 48), { 3e-6,   1.1e-4, 3e-6,   2.3e-4, 3.4e-5,  6.0e-5 } },
// }};

static constexpr std::array<LookupEntry, 15> R_lookup = {{
    { MakeKey(1,  4), { 5e-5,   2e-5,   3e-5,   12e-4,  2.4e-3,  2.4e-3 } },
    { MakeKey(1,  8), { 5e-5,   2e-5,   3e-5,   12e-4,  2.4e-3,  2.4e-3 } },
    { MakeKey(1, 12), { 5e-6,   2e-6,   3e-6,   6e-4,   1.2e-3,  1.2e-3 } },
    { MakeKey(1, 16), { 5e-6,   2e-6,   3e-6,   6e-4,   1.2e-3,  1.2e-3 } },
    { MakeKey(2, 20), { 3e-6,   4e-6,   4e-6,   2.6e-4, 2.4e-4,  2.1e-4 } },
    { MakeKey(2, 24), { 2e-6,   2e-6,   4e-6,   1.5e-4, 1.1e-4,  1.6e-4 } },
    { MakeKey(2, 28), { 2e-6,   1e-6,   3e-6,   7.1e-5, 7.4e-5,  1.1e-4 } },
    { MakeKey(2, 32), { 1e-6,   1e-6,   3e-6,   4.8e-5, 4.3e-5,  9.4e-5 } },
    { MakeKey(3, 24), { 1e-6,   1e-6,   1e-6,   9.6e-5, 2.4e-4,  2.0e-4 } },
    { MakeKey(3, 28), { 1e-6,   1e-6,   1e-6,   9.0e-5, 1.9e-4,  1.6e-4 } },
    { MakeKey(3, 32), { 1e-6,   2e-6,   1e-6,   6.3e-5, 1.1e-4,  1.1e-4 } },
    { MakeKey(3, 36), { 1e-6,   1e-6,   1e-6,   6.0e-5, 9.0e-5,  8.3e-5 } },
    { MakeKey(3, 40), { 1e-6,   2e-6,   5e-6,   5.9e-4, 2.3e-3,  2.3e-3 } },
    { MakeKey(3, 44), { 1e-6,   1e-6,   1e-6,   3.6e-5, 5.1e-5,  4.8e-5 } },
    { MakeKey(3, 48), { 3e-6,   1.1e-4, 3e-6,   2.3e-4, 3.4e-5,  6.0e-5 } },
}};

std::optional<Eigen::Matrix<float, 6, 6>> LookupR(cv::Point2i faceAndPointCount)
{
    int key = MakeKey(faceAndPointCount.x, faceAndPointCount.y);
    for (int i = 0; i < static_cast<int>(R_lookup.size()); i++)
    {
        if (R_lookup[i].key != key) continue;

        const std::array<double, 6>& w = R_lookup[i].weights;
        Eigen::Matrix<float, 6, 6> R = Eigen::Matrix<float, 6, 6>::Zero();
        for (int j = 0; j < 3; j++)
            R(j, j) = static_cast<float>(w[j]) * R_SCALE_TRANS;
        for (int j = 3; j < 6; j++)
            R(j, j) = static_cast<float>(w[j]) * R_SCALE_ROT;

        return R;
    }
    return std::nullopt;
}
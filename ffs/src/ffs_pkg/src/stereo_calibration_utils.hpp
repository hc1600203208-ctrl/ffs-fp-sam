#ifndef FFS_PKG_STEREO_CALIBRATION_UTILS_HPP_
#define FFS_PKG_STEREO_CALIBRATION_UTILS_HPP_

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

struct StereoCalibration {
    cv::Matx33d k_left = cv::Matx33d::eye();
    cv::Matx33d k_right = cv::Matx33d::eye();
    cv::Vec<double, 5> dist_left = cv::Vec<double, 5>::all(0.0);
    cv::Vec<double, 5> dist_right = cv::Vec<double, 5>::all(0.0);
    cv::Matx33d rotation = cv::Matx33d::eye();
    cv::Vec3d translation = cv::Vec3d(0.0, 0.0, 0.0);
    double baseline = 0.0;
};

struct StereoRectificationMaps {
    cv::Size image_size;
    cv::Mat left_map_x;
    cv::Mat left_map_y;
    cv::Mat right_map_x;
    cv::Mat right_map_y;
    cv::Mat left_inverse_map_x;
    cv::Mat left_inverse_map_y;
    cv::Mat r1;
    cv::Mat r2;
    cv::Mat p1;
    cv::Mat p2;
    cv::Mat q;
};

inline bool LoadStereoCalibrationFromTxt(
    const std::string& path,
    StereoCalibration* calibration,
    std::string* error_message = nullptr) {
    if (calibration == nullptr) {
        if (error_message != nullptr) {
            *error_message = "StereoCalibration output pointer is null.";
        }
        return false;
    }

    std::ifstream input(path);
    if (!input.is_open()) {
        if (error_message != nullptr) {
            *error_message = "Failed to open calibration file: " + path;
        }
        return false;
    }

    std::vector<double> values;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        std::istringstream line_stream(line);
        double value = 0.0;
        while (line_stream >> value) {
            values.push_back(value);
        }
    }

    constexpr std::size_t kExpectedValues = 41;
    if (values.size() < kExpectedValues) {
        if (error_message != nullptr) {
            *error_message =
                "Calibration file has insufficient numeric values. Expected at least " +
                std::to_string(kExpectedValues) + ", got " + std::to_string(values.size()) + ".";
        }
        return false;
    }

    StereoCalibration parsed;
    std::size_t idx = 0;

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            parsed.k_left(r, c) = values[idx++];
        }
    }

    parsed.baseline = values[idx++];

    for (int i = 0; i < 5; ++i) {
        parsed.dist_left[i] = values[idx++];
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            parsed.k_right(r, c) = values[idx++];
        }
    }

    for (int i = 0; i < 5; ++i) {
        parsed.dist_right[i] = values[idx++];
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            parsed.rotation(r, c) = values[idx++];
        }
    }

    for (int i = 0; i < 3; ++i) {
        parsed.translation[i] = values[idx++];
    }

    if (parsed.baseline <= 0.0) {
        parsed.baseline = std::abs(parsed.translation[0]);
    }
    if (parsed.baseline <= 0.0) {
        parsed.baseline = cv::norm(parsed.translation);
    }
    if (parsed.baseline <= 0.0) {
        if (error_message != nullptr) {
            *error_message = "Baseline parsed from calibration file is invalid.";
        }
        return false;
    }

    *calibration = parsed;
    return true;
}

inline cv::Mat CameraMatrixToMat(const cv::Matx33d& matrix) {
    return cv::Mat(matrix).clone();
}

inline cv::Mat DistCoeffsToMat(const cv::Vec<double, 5>& coeffs) {
    return cv::Mat(coeffs).clone().reshape(1, 1);
}

inline bool ComputeStereoRectificationMaps(
    const StereoCalibration& calibration,
    const cv::Size& image_size,
    StereoRectificationMaps* maps,
    std::string* error_message = nullptr) {
    if (maps == nullptr) {
        if (error_message != nullptr) {
            *error_message = "StereoRectificationMaps output pointer is null.";
        }
        return false;
    }

    if (image_size.width <= 0 || image_size.height <= 0) {
        if (error_message != nullptr) {
            *error_message = "Image size for stereo rectification must be positive.";
        }
        return false;
    }

    StereoRectificationMaps rectification;
    rectification.image_size = image_size;

    const cv::Mat left_k = CameraMatrixToMat(calibration.k_left);
    const cv::Mat right_k = CameraMatrixToMat(calibration.k_right);
    const cv::Mat left_dist = DistCoeffsToMat(calibration.dist_left);
    const cv::Mat right_dist = DistCoeffsToMat(calibration.dist_right);
    const cv::Mat rotation = CameraMatrixToMat(calibration.rotation);
    const cv::Mat translation = cv::Mat(calibration.translation).clone();

    cv::stereoRectify(
        left_k,
        left_dist,
        right_k,
        right_dist,
        image_size,
        rotation,
        translation,
        rectification.r1,
        rectification.r2,
        rectification.p1,
        rectification.p2,
        rectification.q,
        cv::CALIB_ZERO_DISPARITY,
        0.0,
        image_size);

    cv::initUndistortRectifyMap(
        left_k,
        left_dist,
        rectification.r1,
        rectification.p1,
        image_size,
        CV_32FC1,
        rectification.left_map_x,
        rectification.left_map_y);
    cv::initUndistortRectifyMap(
        right_k,
        right_dist,
        rectification.r2,
        rectification.p2,
        image_size,
        CV_32FC1,
        rectification.right_map_x,
        rectification.right_map_y);

    rectification.left_inverse_map_x = cv::Mat(image_size, CV_32FC1);
    rectification.left_inverse_map_y = cv::Mat(image_size, CV_32FC1);

    std::vector<cv::Point2f> original_points(static_cast<size_t>(image_size.width));
    std::vector<cv::Point2f> rectified_points;
    rectified_points.reserve(original_points.size());

    for (int y = 0; y < image_size.height; ++y) {
        for (int x = 0; x < image_size.width; ++x) {
            original_points[static_cast<size_t>(x)] =
                cv::Point2f(static_cast<float>(x), static_cast<float>(y));
        }

        cv::undistortPoints(
            original_points,
            rectified_points,
            left_k,
            left_dist,
            rectification.r1,
            rectification.p1);

        float* map_x_row = rectification.left_inverse_map_x.ptr<float>(y);
        float* map_y_row = rectification.left_inverse_map_y.ptr<float>(y);
        for (int x = 0; x < image_size.width; ++x) {
            const cv::Point2f& rectified = rectified_points[static_cast<size_t>(x)];
            map_x_row[x] = rectified.x;
            map_y_row[x] = rectified.y;
        }
    }

    *maps = rectification;
    return true;
}

inline cv::Mat AlignRectifiedDepthToOriginalLeft(
    const cv::Mat& rectified_depth,
    const StereoRectificationMaps& maps,
    int interpolation = cv::INTER_NEAREST) {
    if (rectified_depth.empty()) {
        return {};
    }

    if (maps.left_inverse_map_x.empty() || maps.left_inverse_map_y.empty()) {
        return {};
    }

    cv::Mat aligned_depth;
    cv::remap(
        rectified_depth,
        aligned_depth,
        maps.left_inverse_map_x,
        maps.left_inverse_map_y,
        interpolation,
        cv::BORDER_CONSTANT,
        cv::Scalar(0.0f));
    return aligned_depth;
}

#endif  // FFS_PKG_STEREO_CALIBRATION_UTILS_HPP_

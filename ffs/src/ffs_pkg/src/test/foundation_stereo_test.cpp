#include <iostream>
#include <chrono>

#include "../estimator/foundation_stereo_estimator.h"

int main(int argc, char** argv) {

    
    std::string engine_path = "/workspaces/data/deployable_foundationstereo_small_320x736_v1.0_fp32.engine";
    std::string left_image_path = "/workspaces/data/1_left_image.png";
    std::string right_image_path = "/workspaces/data/1_right_image.png";
    std::string output_path = "/workspaces/data/1_disparity_foundationstereo.png";
    

    FoundationStereoEstimator::Config config;
    config.enginePath = engine_path;
    config.inputWidth = 736;
    config.inputHeight = 320;
    

    FoundationStereoEstimator estimator(config);
    if (!estimator.init()) return -1;

    // Load as Grayscale
    cv::Mat imgL = cv::imread(left_image_path, cv::IMREAD_UNCHANGED);
    cv::Mat imgR = cv::imread(right_image_path, cv::IMREAD_UNCHANGED);

    if (imgL.empty() || imgR.empty()) {
        std::cerr << "Cannot load images." << std::endl;
        return -1;
    }

    cv::Mat disp;
    if (estimator.inference(imgL, imgR, disp)) {
        std::cout << "Inference Success." << std::endl;

        // Visualization
        cv::Mat disp_vis;
        double minVal, maxVal;
        cv::minMaxLoc(disp, &minVal, &maxVal);
        std::cout << "Disparity Range: " << minVal << " - " << maxVal << std::endl;

        // Normalize 0-255 for saving
        // Use normalize with MinMax to stretch contrast
        cv::normalize(disp, disp_vis, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        
        cv::Mat color_disp;
        cv::applyColorMap(disp_vis, color_disp, cv::COLORMAP_JET);
        
        cv::imwrite(output_path, color_disp);
        std::cout << "Saved " << output_path << std::endl;
    } else {
        std::cerr << "Inference Failed." << std::endl;
    }

    return 0;
}
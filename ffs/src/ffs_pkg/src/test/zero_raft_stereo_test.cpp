#include <iostream>
#include <opencv2/opencv.hpp>

#include <chrono>

#include "../estimator/zero_raft_stereo_estimator.h"

cv::Mat gray2color(const cv::Mat& input8u) {

    std::vector<cv::Mat> channels;
    channels.push_back(input8u);
    channels.push_back(input8u);
    channels.push_back(input8u);


    // 3. Merge them into one 3-channel image (axis=-1)
    cv::Mat output;
    cv::merge(channels, output);

    return output;
}

int main(int argc, char** argv) {

    std::string engine_path = "/workspaces/data/ZeroRAFTStereo_iter12_fp16.engine";
    std::string left_image_path = "/workspaces/data/1_left_image.png";
    std::string right_image_path = "/workspaces/data/1_right_image.png";
    std::string output_path = "/workspaces/data/1_disparity_gen.png";
    

    // 1. Configuration (matches your ONNX export)
    const int target_height = 480;
    const int target_width = 640;

    // 2. Initialize Engine
    std::cout << "[Main] Initializing Engine..." << std::endl;
    ZeroRAFTStereoEstimator estimator(engine_path, target_height, target_width);

    // 3. Load Images
    cv::Mat leftRaw = cv::imread(left_image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat rightRaw = cv::imread(right_image_path, cv::IMREAD_GRAYSCALE); 

    cv::cvtColor(leftRaw, leftRaw, cv::COLOR_GRAY2RGB);
    cv::cvtColor(rightRaw, rightRaw, cv::COLOR_GRAY2RGB);

    if (leftRaw.empty() || rightRaw.empty()) {
        std::cerr << "[Main] Error: Could not load images." << std::endl;
        return -1;
    }

    // 4. Resize if necessary (The wrapper expects 480x640 input)
    //cv::Mat leftInput, rightInput;
    //cv::resize(leftRaw, leftInput, cv::Size(target_width, target_height));
    //cv::resize(rightRaw, rightInput, cv::Size(target_width, target_height));

    // 5. Inference
    cv::Mat disparity;
    
    // Warmup
    std::cout << "[Main] Warming up..." << std::endl;
    for(int i=0; i<3; ++i) {
        estimator.inference(leftRaw, rightRaw, disparity);
    }

    // Timing
    std::cout << "[Main] Running inference..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    estimator.inference(leftRaw, rightRaw, disparity);
    //disparity = padder.unpad(disparity);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[Main] Inference Time: " << duration << " ms" << std::endl;


    // 6. Visualization
    // Disparity is usually float. Normalize to 0-255 for display.
    cv::Mat dispVis;
    cv::normalize(disparity, dispVis, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    
    // Apply colormap (Jet or Plasma)
    cv::applyColorMap(dispVis, dispVis, cv::COLORMAP_JET);
    
    // Save result
    cv::imwrite(output_path, dispVis);
    
    std::cout << "[Main] Result saved to " << output_path << std::endl;
    
    return 0;
}
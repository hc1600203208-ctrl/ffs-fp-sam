#ifndef DEPTH_PROCESSOR_H
#define DEPTH_PROCESSOR_H

#include <opencv2/opencv.hpp>

class DepthProcessor {

public:
	/*
	 *  深度图去噪：
     *  1. 双边滤波保边平滑，移除深度跳变噪声
     *  2. 孤立点移除：如果某像素周围有效深度点太稀疏，置零
     */
    cv::Mat denoise_depth(const cv::Mat& depth, float zmin) {
        // Assume depth is CV_32FC1 (32-bit float, 1 channel)

        int denoise_kernel_arg = 15;
        
        // 1. Check validity
        cv::Mat valid_mask = depth > zmin; // Returns CV_8UC1 mask (0 or 255)
        if (cv::countNonZero(valid_mask) == 0) {
            return depth.clone();
        }

        // 2. Bilateral Filter: Keep edges, smooth depth noise
        // d=5, sigmaColor=0.05, sigmaSpace=5
        cv::Mat depth_filtered;
        cv::bilateralFilter(depth, depth_filtered, 5, 0.05, 5);

        // 3. Apply filtered result only to valid regions (others become 0.0)
        cv::Mat depth_out = cv::Mat::zeros(depth.size(), depth.type());
        depth_filtered.copyTo(depth_out, valid_mask);

        // 4. Isolated point removal
        int kernel_size = std::max(3, denoise_kernel_arg);
        if (kernel_size % 2 == 0) {
            kernel_size += 1;
        }

        // Update valid mask based on depth_out
        cv::Mat current_valid = depth_out > zmin;

        // Convert mask (0 or 255) to float (0.0 or 1.0) for the blur ratio
        cv::Mat valid_float;
        current_valid.convertTo(valid_float, CV_32F, 1.0 / 255.0);

        // Compute local valid density using a box filter (blur)
        cv::Mat valid_ratio;
        cv::blur(valid_float, valid_ratio, cv::Size(kernel_size, kernel_size));

        // Create mask for pixels where valid ratio is < 0.3
        cv::Mat sparse_mask = valid_ratio < 0.3f; // Returns CV_8UC1 mask
        
        // noise_mask = (depth_out > zmin) & (valid_ratio < 0.3)
        cv::Mat noise_mask;
        cv::bitwise_and(current_valid, sparse_mask, noise_mask);

        // Set noise pixels to 0.0
        depth_out.setTo(0.0f, noise_mask);

        return depth_out;
    }

    /*
     *  移除深度不连续边缘的飞点。
     *  在物体边缘，立体匹配会产生前景/背景之间的错误中间值，
     *  投影到 3D 后形成沿射线方向的杂点。
     *  检测深度梯度相对于深度值的比率，超过阈值的像素置零。
     */
    cv::Mat filter_edge_flying_pixels(const cv::Mat& depth, float zmin) {
        // Assume depth is CV_32FC1 (32-bit float, 1 channel)
        float edge_threshold_arg = 0.05f;
        
        // 1. Validity Check
        cv::Mat valid_mask = depth > zmin;
        if (cv::countNonZero(valid_mask) == 0) {
            return depth.clone();
        }

        // 2. Calculate depth gradients in X and Y directions
        cv::Mat grad_x, grad_y;
        cv::Sobel(depth, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(depth, grad_y, CV_32F, 0, 1, 3);

        // 3. Calculate gradient magnitude: sqrt(grad_x^2 + grad_y^2)
        cv::Mat grad_mag;
        cv::magnitude(grad_x, grad_y, grad_mag);

        // 4. Calculate ratio (gradient magnitude / depth)
        // cv::divide handles element-wise matrix division
        cv::Mat temp_ratio;
        cv::divide(grad_mag, depth, temp_ratio);

        // Initialize ratio map with zeros and copy only valid pixels
        cv::Mat ratio = cv::Mat::zeros(depth.size(), CV_32F);
        temp_ratio.copyTo(ratio, valid_mask);

        // 5. Thresholding to find edge flying pixels
        // ratio > threshold returns a CV_8UC1 mask (255 for true, 0 for false)
        cv::Mat edge_mask_raw = ratio > edge_threshold_arg; 

        // 6. Dilate the mask to ensure edges are fully covered
        // A 3x3 matrix of 1s replicates np.ones((3, 3), np.uint8)
        cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
        cv::Mat edge_mask;
        cv::dilate(edge_mask_raw, edge_mask, kernel);

        // 7. Apply mask to the output depth map
        cv::Mat depth_out = depth.clone();
        depth_out.setTo(0.0f, edge_mask);

        return depth_out;
    }


    void filter_zmin_zfar(cv::Mat& depth_mat, double zmin, double zfar) {
	    cv::Mat invalid_mask = (depth_mat > zfar);
        depth_mat.setTo(0.0, invalid_mask);

        invalid_mask = (depth_mat < zmin);
        depth_mat.setTo(0.0, invalid_mask); 
    }
};


#endif // DEPTH_PROCESSOR_H
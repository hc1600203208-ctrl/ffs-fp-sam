
#ifndef STEREO_ESTIMATOR_H
#define STEREO_ESTIMATOR_H


class StereoEstimator {
public:

    virtual bool inference(const cv::Mat& leftImg, const cv::Mat& rightImg, cv::Mat& outputDisp) = 0;
    

protected:

    // Constructor is protected to prevent direct instantiation
    //StereoEstimator(const std::string& name) : name_(name) {}
};


class InputPadder {
public:
    /**
     * @brief Constructor to calculate padding values.
     * 
     * @param height Original image height
     * @param width Original image width
     * @param mode "sintel" (pads top/bottom) or others (pads bottom only)
     * @param divis_by The factor dimensions must be divisible by (default 8)
     */
    InputPadder(int height, int width, const std::string& mode = "sintel", int divis_by = 32) {
        m_origH = height;
        m_origW = width;

        // Calculate total padding needed
        // Logic equivalent to: (((ht // 8) + 1) * 8 - ht) % 8
        int pad_ht = (((height / divis_by) + 1) * divis_by - height) % divis_by;
        int pad_wd = (((width / divis_by) + 1) * divis_by - width) % divis_by;

        // Calculate Left/Right padding (always split)
        m_padLeft = pad_wd / 2;
        m_padRight = pad_wd - m_padLeft;

        // Calculate Top/Bottom padding based on mode
        if (mode == "sintel") {
            m_padTop = pad_ht / 2;
            m_padBottom = pad_ht - m_padTop;
        } else {
            // For KITTI or others, usually pad only bottom and right
            m_padTop = 0;
            m_padBottom = pad_ht;
        }
    }

    /**
     * @brief Pad the input image using Replicate Border (matching PyTorch's 'replicate')
     * Works for CV_8UC3 (images) and CV_32F (tensors)
     */
    cv::Mat pad(const cv::Mat& input) const {
        cv::Mat output;
        // cv::copyMakeBorder is the OpenCV equivalent to F.pad
        // PyTorch F.pad order: (Left, Right, Top, Bottom)
        // OpenCV copyMakeBorder order: (Top, Bottom, Left, Right)
        cv::copyMakeBorder(input, output, 
                           m_padTop, m_padBottom, 
                           m_padLeft, m_padRight, 
                           cv::BORDER_REPLICATE);
        return output;
    }

    /**
     * @brief Crop the image back to original dimensions
     */
    cv::Mat unpad(const cv::Mat& paddedInput) const {
        // Calculate the Region of Interest (ROI) corresponding to the original image
        // Python: x[..., c[0]:c[1], c[2]:c[3]]
        // c[0]=top, c[1]=ht-bottom, c[2]=left, c[3]=wd-right
        
        int current_h = paddedInput.rows;
        int current_w = paddedInput.cols;

        // Define ROI: Rect(x, y, width, height)
        int x = m_padLeft;
        int y = m_padTop;
        int w = current_w - m_padLeft - m_padRight;
        int h = current_h - m_padTop - m_padBottom;

        // Safety check
        if (x < 0 || y < 0 || w <= 0 || h <= 0 || x+w > current_w || y+h > current_h) {
            // If dimensions don't match expectation, return as is or handle error
            return paddedInput; 
        }

        cv::Rect roi(x, y, w, h);
        
        // Return a deep copy (.clone()) to ensure memory safety
        return paddedInput(roi).clone();
    }

    // Getters for debugging or dynamic tensor sizing
    int getPadTop() const { return m_padTop; }
    int getPadBottom() const { return m_padBottom; }
    int getPadLeft() const { return m_padLeft; }
    int getPadRight() const { return m_padRight; }

private:
    int m_origH, m_origW;
    int m_padTop, m_padBottom, m_padLeft, m_padRight;
};

#endif // STEREO_ESTIMATOR_H
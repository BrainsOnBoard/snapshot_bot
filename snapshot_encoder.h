#pragma once

// Standard C++ includes
#include <array>
#include <string>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>

// TensorFlow includes
#include "tensorflow/c/c_api.h"

//----------------------------------------------------------------------------
// SnapshotEncoder
//----------------------------------------------------------------------------
class SnapshotEncoder
{
public:
    SnapshotEncoder(unsigned int inputWidth, unsigned int inputHeight, unsigned int outputSize);
    SnapshotEncoder(unsigned int inputWidth, unsigned int inputHeight, unsigned int outputSize,
                    const std::string &exportDirectory, const std::string &tag,
                    const std::string &inputOpName, const std::string &outputOpName);
    ~SnapshotEncoder();

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool openModel(const std::string &exportDirectory, const std::string &tag,
                   const std::string &inputOpName, const std::string &outputOpName);
    void encode(const cv::Mat &snapshotFloat);

    const cv::Mat &getFinalSnapshot() const{ return m_Output; }
    const cv::Mat &getFinalSnapshotFloat() const{ return m_OutputFloat; }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    std::vector<int64_t> getOpShape(const TF_Output &op);

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const unsigned int m_InputWidth;
    const unsigned int m_InputHeight;
    const unsigned int m_InputSize;
    
    const unsigned int m_OutputSize;
    TF_Status *m_Status;
    TF_Graph *m_Graph;
    TF_Session *m_Session;

    std::array<TF_Output, 1> m_InputOp;
    std::array<TF_Output, 1> m_OutputOp;

    TF_Tensor *m_InputTensor;
    TF_Tensor *m_OutputTensor;

    cv::Mat m_Output;
    cv::Mat m_OutputFloat;
};
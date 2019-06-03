#pragma once

// Standard C++ includes
#include <array>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Ardin MB includes
#include "mb_memory.h"
#include "mb_params_hog.h"

//----------------------------------------------------------------------------
// MBMemoryHOG
//----------------------------------------------------------------------------
class MBMemoryHOG : public MBMemory
{
public:
    MBMemoryHOG();

protected:
    //------------------------------------------------------------------------
    // MBMemory virtuals
    //------------------------------------------------------------------------
    virtual void beginPresent(const cv::Mat &snapshot) const override;
    virtual void endPresent() const override;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    // Sobel operator output
    mutable cv::Mat m_SobelX;
    mutable cv::Mat m_SobelY;

    // Image containing pixel orientations - one channel per orientation
    mutable cv::Mat m_PixelOrientations;

    mutable cv::Mat m_HOGFeatures;

    std::array<cv::Vec2f, MBParamsHOG::hogNumOrientations> m_HOGDirections;

    float *m_IExtPN;
};

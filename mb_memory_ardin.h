#pragma once

// Standard C++ includes
#include <array>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Ardin MB includes
#include "mb_memory.h"

//----------------------------------------------------------------------------
// MBMemoryArdin
//----------------------------------------------------------------------------
class MBMemoryArdin : public MBMemory
{
public:
    MBMemoryArdin();

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
    float *m_IExtPN;

    // CLAHE algorithm for histogram normalization
    cv::Ptr<cv::CLAHE> m_Clahe;

    mutable cv::Mat m_IntermediateSnapshotGreyscale;
    mutable cv::Mat m_FinalSnapshot;
    mutable cv::Mat m_FinalSnapshotFloat;
};

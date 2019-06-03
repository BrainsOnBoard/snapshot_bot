#define NO_HEADER_DEFINITIONS
#include "mb_memory_ardin.h"

// Antworld includes
#include "mb_params_ardin.h"

using namespace BoBRobotics;

//----------------------------------------------------------------------------
// MBMemoryArdin
//----------------------------------------------------------------------------
MBMemoryArdin::MBMemoryArdin()
    :   MBMemory(MBParamsArdin::numPN, MBParamsArdin::numKC, MBParamsArdin::numEN, MBParamsArdin::numPNSynapsesPerKC,
                 MBParamsArdin::intermediateWidth, MBParamsArdin::intermediateHeight,
                 MBParamsArdin::tauD, MBParamsArdin::kcToENWeight, MBParamsArdin::dopamineStrength,
                 MBParamsArdin::rewardTimeMs, MBParamsArdin::presentDurationMs, MBParamsArdin::timestepMs,
                 "mb_memory_ardin"),
        m_Clahe(cv::createCLAHE(40.0, cv::Size(8, 8))),
        m_IntermediateSnapshotGreyscale(MBParamsArdin::intermediateHeight, MBParamsArdin::intermediateWidth, CV_8UC1),
        m_FinalSnapshot(MBParamsArdin::inputHeight, MBParamsArdin::inputWidth, CV_8UC1),
        m_FinalSnapshotFloat(MBParamsArdin::inputHeight, MBParamsArdin::inputWidth, CV_32FC1)
{
    // Get pointers to state variables
    m_IExtPN = getSLM().getArray<float>("IextPN");
}
//----------------------------------------------------------------------------
void MBMemoryArdin::beginPresent(const cv::Mat &snapshot) const
{
    // Invert image
    cv::subtract(255, snapshot, m_IntermediateSnapshotGreyscale);

    // Apply histogram normalization
    // http://answers.opencv.org/question/15442/difference-of-clahe-between-opencv-and-matlab/
    m_Clahe->apply(m_IntermediateSnapshotGreyscale, m_IntermediateSnapshotGreyscale);

    // Finally resample down to final size
    cv::resize(m_IntermediateSnapshotGreyscale, m_FinalSnapshot,
                cv::Size(MBParamsArdin::inputWidth, MBParamsArdin::inputHeight),
                0.0, 0.0, cv::INTER_CUBIC);

    // Convert snapshot to float
    m_FinalSnapshot.convertTo(m_FinalSnapshotFloat, CV_32FC1, 1.0 / 255.0);

    // Normalise input
    cv::normalize(m_FinalSnapshotFloat, m_FinalSnapshotFloat);

    // Scale normalised input into external input current
    BOB_ASSERT(m_FinalSnapshotFloat.isContinuous());
    std::transform(m_FinalSnapshotFloat.begin<float>(), m_FinalSnapshotFloat.end<float>(), m_IExtPN,
                   [](float x){ return x * MBParamsArdin::inputCurrentScale; });

    // Copy to device
    getSLM().pushVarToDevice("PN", "Iext");
}
//----------------------------------------------------------------------------
void MBMemoryArdin::endPresent() const
{
    // Zero external input current
    std::fill_n(m_IExtPN, MBParamsArdin::numPN, 0.0f);

    // Copy external input current to device
    getSLM().pushVarToDevice("PN", "Iext");
}

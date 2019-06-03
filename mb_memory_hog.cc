#define NO_HEADER_DEFINITIONS
#include "mb_memory_hog.h"

// Standard C++ includes
#include <bitset>
#include <fstream>
#include <random>

// Standard C includes
#include <cmath>

// CLI11 includes
#include "third_party/CLI11.hpp"

// BoB robotics includes
#include "common/timer.h"
#include "genn_utils/connectors.h"

using namespace BoBRobotics;
using namespace units::literals;
using namespace units::angle;
using namespace units::math;

//----------------------------------------------------------------------------
// MBMemory
//----------------------------------------------------------------------------
MBMemoryHOG::MBMemoryHOG()
    :   MBMemory(MBParamsHOG::numPN, MBParamsHOG::numKC, MBParamsHOG::numEN, MBParamsHOG::numPNSynapsesPerKC,
                 MBParamsHOG::inputWidth, MBParamsHOG::inputHeight,
                 MBParamsHOG::tauD, MBParamsHOG::kcToENWeight, MBParamsHOG::dopamineStrength,
                 MBParamsHOG::rewardTimeMs, MBParamsHOG::presentDurationMs, MBParamsHOG::timestepMs,
                 "mb_memory_hog"),
        m_SobelX(MBParamsHOG::inputHeight, MBParamsHOG::inputWidth, CV_32FC1), m_SobelY(MBParamsHOG::inputHeight, MBParamsHOG::inputWidth, CV_32FC1),
        m_PixelOrientations(MBParamsHOG::inputHeight, MBParamsHOG::inputWidth, CV_MAKETYPE(CV_32F, MBParamsHOG::hogNumOrientations)),
        m_HOGFeatures(MBParamsHOG::hogNumRFY, MBParamsHOG::hogNumRFX, CV_MAKETYPE(CV_32F, MBParamsHOG::hogNumOrientations)), m_IExtPN(nullptr)
{
    std::cout << "HOG feature vector length:" << MBParamsHOG::hogFeatureSize << std::endl;

    // Build orientation vectors
    radian_t orient = 90_deg;
    for(auto &d : m_HOGDirections) {
        d[0] = sin(orient);
        d[1] = cos(orient);
        orient += 60_deg;
    }

    // Get pointers to state variables
    m_IExtPN = getSLM().getArray<float>("IextPN");
}
//----------------------------------------------------------------------------
void MBMemoryHOG::beginPresent(const cv::Mat &snapshot) const
{
     // Apply Sobel operator to image
    cv::Sobel(snapshot, m_SobelX, CV_32F, 1, 0, 1);
    cv::Sobel(snapshot, m_SobelY, CV_32F, 0, 1, 1);

    // At each pixel, take dot product of vector formed from x and y sobel operator and each direction vector
    typedef cv::Vec<float, MBParamsHOG::hogNumOrientations> PixelFeatures;
    static_assert(sizeof(PixelFeatures) == (MBParamsHOG::hogNumOrientations * sizeof(float)), "HOG feature size mismatch");
    std::transform(m_SobelX.begin<float>(), m_SobelX.end<float>(), m_SobelY.begin<float>(), m_PixelOrientations.begin<PixelFeatures>(),
                   [this](float x, float y)
                   {
                       PixelFeatures pix;
                       for(size_t d = 0; d < MBParamsHOG::hogNumOrientations; d++) {
                           pix[d] = std::abs((x * m_HOGDirections[d][0]) + (y * m_HOGDirections[d][1]));
                       }
                       return pix;
                   });

    // Loop through receptive fields
    auto hogOut = m_HOGFeatures.begin<PixelFeatures>();
    for(unsigned int rfY = 0; rfY <= (MBParamsHOG::inputHeight - MBParamsHOG::hogRFSize); rfY += MBParamsHOG::hogRFStride) {
        for(unsigned int rfX = 0; rfX <= (MBParamsHOG::inputWidth - MBParamsHOG::hogRFSize); rfX += MBParamsHOG::hogRFStride) {
            // Get ROI into hog directions representing pixels within receptive field
            const cv::Mat rfInput(m_PixelOrientations, cv::Rect(rfX, rfY, MBParamsHOG::hogRFSize, MBParamsHOG::hogRFSize));

            // Sum all pixels within receptive field
            const cv::Scalar sum = cv::sum(rfInput);

            // Calculate the exponential of each receptive field response
            std::array<float, MBParamsHOG::hogNumOrientations> exponentials;
            std::transform(&sum[0], &sum[MBParamsHOG::hogNumOrientations], exponentials.begin(),
                [](float s){ return std::exp(s); });

            // Sum these to get softmax scaling factor
            const float scale = std::accumulate(exponentials.cbegin(), exponentials.cend(), 0.0f);

            // Fill in features with softmax of orientations at location
            PixelFeatures features;
            std::transform(exponentials.cbegin(), exponentials.cend(), &features[0],
                           [scale](float e){ return e / scale; });
            (*hogOut++) = features;
        }
    }

    // Copy HOG features into external input current
    BOB_ASSERT(m_HOGFeatures.isContinuous());
    std::copy_n(reinterpret_cast<float*>(m_HOGFeatures.data), MBParamsHOG::hogFeatureSize, m_IExtPN);
    getSLM().pushVarToDevice("PN", "Iext");
}
//----------------------------------------------------------------------------
void MBMemoryHOG::endPresent() const
{
    std::fill_n(m_IExtPN, MBParamsHOG::numPN, 0.0f);

    // Copy external input current to device
    getSLM().pushVarToDevice("PN", "Iext");
}

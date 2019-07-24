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
                 "mb_memory_hog", MBParamsHOG::timing),
        m_SobelX(MBParamsHOG::inputHeight, MBParamsHOG::inputWidth, CV_32FC1), m_SobelY(MBParamsHOG::inputHeight, MBParamsHOG::inputWidth, CV_32FC1),
        m_PixelOrientations(MBParamsHOG::inputHeight, MBParamsHOG::inputWidth, CV_MAKETYPE(CV_32F, MBParamsHOG::numOrientations)),
        m_Features(MBParamsHOG::numRFY, MBParamsHOG::numRFX, CV_MAKETYPE(CV_32F, MBParamsHOG::numFeatures)), m_IExtPN(nullptr)
{
    std::cout << "HOG feature vector length:" << MBParamsHOG::featureSize << std::endl;

    // Build orientation vectors
    radian_t orient = 90_deg;
    for(auto &d : m_Directions) {
        d[0] = sin(orient);
        d[1] = cos(orient);
        orient += 120_deg;
    }

    // Get pointers to state variables
    m_IExtPN = getSLM().getArray<float>("IextPN");
}
//----------------------------------------------------------------------------
void MBMemoryHOG::beginPresent(const cv::Mat &snapshot) const
{
     // Apply Sobel operator to image
    cv::Sobel(snapshot, m_SobelX, CV_32F, 1, 0, 1, 1.0 / 255.0);
    cv::Sobel(snapshot, m_SobelY, CV_32F, 0, 1, 1, 1.0 / 255.0);

    // At each pixel, take dot product of vector formed from x and y sobel operator and each direction vector
    typedef cv::Vec<float, MBParamsHOG::numOrientations> OrientationFeatures;
    static_assert(sizeof(OrientationFeatures) == (MBParamsHOG::numOrientations * sizeof(float)), "Feature size mismatch");
    std::transform(m_SobelX.begin<float>(), m_SobelX.end<float>(), m_SobelY.begin<float>(), m_PixelOrientations.begin<OrientationFeatures>(),
                   [this](float x, float y)
                   {
                       OrientationFeatures pix;
                       for(size_t d = 0; d < MBParamsHOG::numOrientations; d++) {
                           pix[d] = std::abs((x * m_Directions[d][0]) + (y * m_Directions[d][1]));
                       }
                       return pix;
                   });

    // Loop through receptive fields
    typedef cv::Vec<float, MBParamsHOG::numFeatures> RFFeatures;
    auto featureOut = m_Features.begin<RFFeatures>();
    for(int rfY = 0; rfY <= (MBParamsHOG::inputHeight - MBParamsHOG::rfHeight); rfY += MBParamsHOG::rfStrideY) {
        for(int rfX = 0; rfX <= (MBParamsHOG::inputWidth - MBParamsHOG::rfWidth); rfX += MBParamsHOG::rfStrideX) {
            const cv::Rect rfROI(rfX, rfY, MBParamsHOG::rfWidth, MBParamsHOG::rfHeight);

            // Get ROI into hog directions representing pixels within receptive field
            const cv::Mat rfOrientInput(m_PixelOrientations, rfROI);
            const cv::Mat rfAreaInput(snapshot, rfROI);

            // Sum all pixels within receptive field
            const cv::Scalar orientSum = cv::sum(rfOrientInput);
            const cv::Scalar areaSum = cv::sum(rfAreaInput);

            // Calculate the exponential of each receptive field response
            std::array<float, MBParamsHOG::numOrientations> exponentials;
            std::transform(&orientSum[0], &orientSum[MBParamsHOG::numOrientations], exponentials.begin(),
                [](float s){ return std::exp(s); });

            // Sum these to get softmax scaling factor
            const float scale = std::accumulate(exponentials.cbegin(), exponentials.cend(), 0.0f);

            // Copy softmax of orientations in RF into features
            RFFeatures features;
            std::transform(exponentials.cbegin(), exponentials.cend(), &features[0],
                           [scale](float e){ return e / scale; });

            // Copy in 'area' of green in RF
            // **NOTE** sum is actually sum of sky pixels so we flip it
            features[MBParamsHOG::numOrientations] = 1.0f - ((float)areaSum[0] / (float)(255 * MBParamsHOG::rfWidth * MBParamsHOG::rfHeight));

            // Copy completed RF feature into feature vector
            (*featureOut++) = features;
        }
    }

    // Copy HOG features into external input current
    BOB_ASSERT(m_Features.isContinuous());
    std::copy_n(reinterpret_cast<float*>(m_Features.data), MBParamsHOG::featureSize, m_IExtPN);
    getSLM().pushVarToDevice("PN", "Iext");
}
//----------------------------------------------------------------------------
void MBMemoryHOG::endPresent() const
{
    std::fill_n(m_IExtPN, MBParamsHOG::numPN, 0.0f);

    // Copy external input current to device
    getSLM().pushVarToDevice("PN", "Iext");
}
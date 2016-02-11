#define NO_HEADER_DEFINITIONS
#include "memory.h"

// Standard C++ includes
#include <fstream>

// BoB robotics third party include
#include "third_party/path.h"

// Snapshot bot includes
#include "config.h"

using namespace BoBRobotics;
using namespace units::angle;
using namespace units::length;
using namespace units::literals;
using namespace units::math;

//------------------------------------------------------------------------
// MemoryBase
//------------------------------------------------------------------------
MemoryBase::MemoryBase()
:   m_BestHeading(0.0_deg), m_LowestDifference(std::numeric_limits<size_t>::max())
{
}
//------------------------------------------------------------------------
void MemoryBase::writeCSVHeader(std::ostream &os)
{
    os << "Best heading [degrees], Lowest difference";
}
//------------------------------------------------------------------------
void MemoryBase::writeCSVLine(std::ostream &os)
{
    os << getBestHeading() << ", " << getLowestDifference();
}

//------------------------------------------------------------------------
// PerfectMemory
//------------------------------------------------------------------------
PerfectMemory::PerfectMemory(const Config &config, const cv::Size &inputSize)
:   m_PM(inputSize), m_BestSnapshotIndex(std::numeric_limits<size_t>::max())
{
    // Load mask image
    if(!config.getMaskImageFilename().empty()) {
        getPM().setMaskImage(config.getMaskImageFilename());
    }
}
//------------------------------------------------------------------------
void PerfectMemory::test(const cv::Mat &snapshot)
{
    // Get heading directly from Perfect Memory
    degree_t bestHeading;
    float lowestDifference;
    std::tie(bestHeading, m_BestSnapshotIndex, lowestDifference, std::ignore) = getPM().getHeading(snapshot);

    // Set best heading and vector length
    setBestHeading(bestHeading);
    setLowestDifference(lowestDifference);
}
//------------------------------------------------------------------------
void PerfectMemory::train(const cv::Mat &snapshot)
{
    getPM().train(snapshot);
}
//------------------------------------------------------------------------
void PerfectMemory::writeCSVHeader(std::ostream &os)
{
    // Superclass
    MemoryBase::writeCSVHeader(os);

    os << ", Best snapshot index";
}
//------------------------------------------------------------------------
void PerfectMemory::writeCSVLine(std::ostream &os)
{
    // Superclass
    MemoryBase::writeCSVLine(os);

    os << ", " << getBestSnapshotIndex();
}

//------------------------------------------------------------------------
// PerfectMemoryConstrained
//------------------------------------------------------------------------
PerfectMemoryConstrained::PerfectMemoryConstrained(const Config &config, const cv::Size &inputSize)
:   PerfectMemory(config, inputSize), m_FOV(config.getMaxSnapshotRotateAngle()), m_ImageWidth(inputSize.width)
{
}
//------------------------------------------------------------------------
void PerfectMemoryConstrained::test(const cv::Mat &snapshot)
{
    // Get 'matrix' of differences from perfect memory
    const auto &allDifferences = getPM().getImageDifferences(snapshot);

    // Loop through snapshots
    // **NOTE** this currently uses a super-naive approach as more efficient solution is non-trivial because
    // columns that represent the rotations are not necessarily contiguous - there is a dis-continuity in the middle
    float lowestDifference = std::numeric_limits<float>::max();
    setBestSnapshotIndex(std::numeric_limits<size_t>::max());
    setBestHeading(0.0_deg);
    for(size_t i = 0; i < allDifferences.size(); i++) {
        const auto &snapshotDifferences = allDifferences[i];

        // Loop through acceptable range of columns
        for(int c = 0; c < m_ImageWidth; c++) {
            // If this snapshot is a better match than current best
            if(snapshotDifferences[c] < lowestDifference) {
                // Convert column into pixel rotation
                int pixelRotation = c;
                if(pixelRotation > (m_ImageWidth / 2)) {
                    pixelRotation -= m_ImageWidth;
                }

                // Convert this into angle
                const degree_t heading = turn_t((double)pixelRotation / (double)m_ImageWidth);

                // If the distance between this angle from grid and route angle is within FOV, update best
                if(fabs(heading) < m_FOV) {
                    setBestSnapshotIndex(i);
                    setBestHeading(heading);
                    lowestDifference = snapshotDifferences[c];
                }
            }
        }
    }

    // Check valid snapshot actually exists
    assert(getBestSnapshotIndex() != std::numeric_limits<size_t>::max());

    // Scale difference to match code in ridf_processors.h:57
    setLowestDifference(lowestDifference / 255.0f);
}

//------------------------------------------------------------------------
// InfoMax
//------------------------------------------------------------------------
InfoMax::InfoMax(const Config &config, const cv::Size &inputSize)
:   m_InfoMax(createInfoMax(config, inputSize))
{
    BOB_ASSERT(config.getMaskImageFilename().empty());
    std::cout << "\tUsing " << Eigen::nbThreads() << " threads" << std::endl;
}
//------------------------------------------------------------------------
void InfoMax::test(const cv::Mat &snapshot)
{
    // Get heading directly from InfoMax
    degree_t bestHeading;
    float lowestDifference;
    std::tie(bestHeading, lowestDifference, std::ignore) = getInfoMax().getHeading(snapshot);

    // Set best heading and vector length
    setBestHeading(bestHeading);
    setLowestDifference(lowestDifference);
}
//-----------------------------------------------------------------------
void InfoMax::train(const cv::Mat &snapshot)
{
    getInfoMax().train(snapshot);
}
//-----------------------------------------------------------------------
void InfoMax::saveWeights(const std::string &filename) const
{
    // Write weights to disk
    std::ofstream netFile(filename, std::ios::binary);
    const int size[2] { (int) getInfoMax().getWeights().rows(), (int) getInfoMax().getWeights().cols() };
    netFile.write(reinterpret_cast<const char *>(size), sizeof(size));
    netFile.write(reinterpret_cast<const char *>(getInfoMax().getWeights().data()), getInfoMax().getWeights().size() * sizeof(float));
}
//-----------------------------------------------------------------------
InfoMax::InfoMaxType InfoMax::createInfoMax(const Config &config, const cv::Size &inputSize)
{
    const filesystem::path weightPath = filesystem::path(config.getOutputPath()) / "weights.bin";
    if(weightPath.exists()) {
        std::cout << "\tLoading weights from " << weightPath << std::endl;
        
        std::ifstream is(weightPath.str(), std::ios::binary);
        if (!is.good()) {
            throw std::runtime_error("Could not open " + weightPath.str());
        }

        // The matrix size is encoded as 2 x int32_t
        int32_t size[2];
        is.read(reinterpret_cast<char *>(&size), sizeof(size));

        // Create data array and fill it
        InfoMaxWeightMatrixType weights(size[0], size[1]);
        is.read(reinterpret_cast<char*>(weights.data()), sizeof(float) * weights.size());
        
        return InfoMaxType(inputSize, weights);
    }
    else {
        return InfoMaxType(inputSize);
    }
}

//-----------------------------------------------------------------------
// InfoMaxConstrained
//-----------------------------------------------------------------------
InfoMaxConstrained::InfoMaxConstrained(const Config &config, const cv::Size &inputSize)
:   InfoMax(config, inputSize), m_FOV(config.getMaxSnapshotRotateAngle()), m_ImageWidth(inputSize.width)
{
}
//-----------------------------------------------------------------------
void InfoMaxConstrained::test(const cv::Mat &snapshot)
{
    // Get vector of differences from InfoMax
    const auto &allDifferences = this->getInfoMax().getImageDifferences(snapshot);

    // Loop through snapshots
    // **NOTE** this currently uses a super-naive approach as more efficient solution is non-trivial because
    // columns that represent the rotations are not necessarily contiguous - there is a dis-continuity in the middle
    this->setLowestDifference(std::numeric_limits<float>::max());
    this->setBestHeading(0_deg);
    for(size_t i = 0; i < allDifferences.size(); i++) {
        // If this snapshot is a better match than current best
        if(allDifferences[i] < this->getLowestDifference()) {
            // Convert column into pixel rotation
            int pixelRotation = i;
            if(pixelRotation > (m_ImageWidth / 2)) {
                pixelRotation -= m_ImageWidth;
            }

            // Convert this into angle
            const degree_t heading = turn_t((double)pixelRotation / (double)m_ImageWidth);

            // If the distance between this angle from grid and route angle is within FOV, update best
            if(fabs(heading) < m_FOV) {
                this->setBestHeading(heading);
                this->setLowestDifference(allDifferences[i]);
            }
        }
    }
}

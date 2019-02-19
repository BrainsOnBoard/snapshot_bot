#define NO_HEADER_DEFINITIONS
#include "memory.h"

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
MemoryBase::MemoryBase(const Config &config)
:   m_BestHeading(0.0_deg), m_LowestDifference(std::numeric_limits<size_t>::max()), m_OutputPath(config.getOutputPath())
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
PerfectMemory::PerfectMemory(const Config &config)
:   MemoryBase(config), m_PM(config.getUnwrapRes()), m_BestSnapshotIndex(std::numeric_limits<size_t>::max())
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
    
    // Save snapshot
    cv::imwrite(getSnapshotPath(getPM().getNumSnapshots() - 1).str(), snapshot);
}
//------------------------------------------------------------------------
void PerfectMemory::load()
{
    for(size_t i = 0;;i++) {
        const auto filename = getSnapshotPath(i);
        if(filename.exists()) {
            // Load image
            cv::Mat image = cv::imread(filename.str(), cv::IMREAD_GRAYSCALE);
            getPM().train(image);
        }
        else {
            break;
        }
    }
    std::cout << "Loaded " << getPM().getNumSnapshots() << " snapshots" << std::endl;
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
PerfectMemoryConstrained::PerfectMemoryConstrained(const Config &config)
:   PerfectMemory(config), m_FOV(config.getMaxSnapshotRotateAngle()), m_ImageWidth(config.getUnwrapRes().width)
{
}
//------------------------------------------------------------------------v
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
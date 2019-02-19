#pragma once

// Standard C++ includes
#include <fstream>

// BoB robotics includes
#include "navigation/infomax.h"
#include "navigation/perfect_memory.h"

// BoB robotics third-party includes
#include "third_party/units.h"

// Forward declarations
class Config;

//------------------------------------------------------------------------
// MemoryBase
//------------------------------------------------------------------------
class MemoryBase
{
public:
    MemoryBase(const Config &config);

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void test(const cv::Mat &snapshot) = 0;
    virtual void train(const cv::Mat &snapshot) = 0;
    virtual void load() = 0;

    virtual void writeCSVHeader(std::ostream &os);
    virtual void writeCSVLine(std::ostream &os);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    units::angle::degree_t getBestHeading() const{ return m_BestHeading; }
    float getLowestDifference() const{ return m_LowestDifference; }
    
protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    void setBestHeading(units::angle::degree_t bestHeading){ m_BestHeading = bestHeading; }
    void setLowestDifference(float lowestDifference){ m_LowestDifference = lowestDifference; }
    const filesystem::path &getOutputPath() const{ return m_OutputPath; }
    
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    units::angle::degree_t m_BestHeading;
    float m_LowestDifference;
    const filesystem::path &m_OutputPath;
};

//------------------------------------------------------------------------
// PerfectMemory
//------------------------------------------------------------------------
class PerfectMemory : public MemoryBase
{
public:
    PerfectMemory(const Config &config);

    //------------------------------------------------------------------------
    // MemoryBase virtuals
    //------------------------------------------------------------------------
    virtual void test(const cv::Mat &snapshot) override;
    virtual void train(const cv::Mat &snapshot) override;
    virtual void load() override;
    
    virtual void writeCSVHeader(std::ostream &os);
    virtual void writeCSVLine(std::ostream &os);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getBestSnapshotIndex() const{ return m_BestSnapshotIndex; }
    const cv::Mat &getBestSnapshot() const{ return getPM().getSnapshot(getBestSnapshotIndex()); }
    
protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    BoBRobotics::Navigation::PerfectMemoryRotater<> &getPM(){ return m_PM; }
    const BoBRobotics::Navigation::PerfectMemoryRotater<> &getPM() const{ return m_PM; }

    void setBestSnapshotIndex(size_t bestSnapshotIndex){ m_BestSnapshotIndex = bestSnapshotIndex; }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    filesystem::path getSnapshotPath(size_t index) const
    {
        return getOutputPath() / ("snapshot_" + std::to_string(index) + ".png");
    }
    
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    BoBRobotics::Navigation::PerfectMemoryRotater<> m_PM;
    size_t m_BestSnapshotIndex;
};

//------------------------------------------------------------------------
// PerfectMemoryConstrained
//------------------------------------------------------------------------
class PerfectMemoryConstrained : public PerfectMemory
{
public:
    PerfectMemoryConstrained(const Config &config);

    virtual void test(const cv::Mat &snapshot) override;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const units::angle::degree_t m_FOV;
    const int m_ImageWidth;
};

//------------------------------------------------------------------------
// InfoMax
//------------------------------------------------------------------------
/*template<typename FloatType>
class InfoMax : public MemoryBase
{
    using InfoMaxType = BoBRobotics::Navigation::InfoMaxRotater<Navigation::InSilicoRotater, FloatType>;
    using InfoMaxWeightMatrixType = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>;

public:
    InfoMax(const cv::Size &imSize)
        : m_InfoMax(createInfoMax(imSize))
    {
        // Load mask image
        if(!config.getMaskImageFilename().empty()) {
            getPM().setMaskImage(config.getMaskImageFilename());
        }
    }

    virtual void test(const cv::Mat &snapshot) override
    {
        // Get heading directly from InfoMax
        degree_t bestHeading;
        float lowestDifference;
        std::tie(bestHeading, lowestDifference, std::ignore) = m_InfoMax.getHeading(snapshot);

        // Set best heading and vector length
        setBestHeading(bestHeading);
        setLowestDifference(lowestDifference);
    }
    
    virtual void train(const cv::Mat &snapshot) override
    {
        getPM().train(snapshot);
    }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    const InfoMaxType &getInfoMax() const{ return m_InfoMax; }

private:
    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    // **TODO** move into BoB robotics
    static void writeWeights(const InfoMaxWeightMatrixType &weights, const filesystem::path &weightPath)
    {
        // Write weights to disk
        std::ofstream netFile(weightPath.str(), std::ios::binary);
        const int size[2] { (int) weights.rows(), (int) weights.cols() };
        netFile.write(reinterpret_cast<const char *>(size), sizeof(size));
        netFile.write(reinterpret_cast<const char *>(weights.data()), weights.size() * sizeof(FloatType));
    }

    // **TODO** move into BoB robotics
    static InfoMaxWeightMatrixType readWeights(const filesystem::path &weightPath)
    {
        // Open file
        std::ifstream is(weightPath.str(), std::ios::binary);
        if (!is.good()) {
            throw std::runtime_error("Could not open " + weightPath.str());
        }

        // The matrix size is encoded as 2 x int32_t
        int32_t size[2];
        is.read(reinterpret_cast<char *>(&size), sizeof(size));

        // Create data array and fill it
        InfoMaxWeightMatrixType data(size[0], size[1]);
        is.read(reinterpret_cast<char *>(data.data()), sizeof(FloatType) * data.size());

        return std::move(data);
    }

    static InfoMaxType createInfoMax(const cv::Size &imSize, const Navigation::ImageDatabase &route)
    {
        // Create path to weights from directory containing route
        const filesystem::path weightPath = filesystem::path(route.getPath()) / "infomax.bin";
        if(weightPath.exists()) {
            std::cout << "Loading weights from " << weightPath << std::endl;
            InfoMaxType infomax(imSize, readWeights(weightPath));
            return std::move(infomax);
        }
        else {
            InfoMaxType infomax(imSize);
            infomax.trainRoute(route, true);
            writeWeights(infomax.getWeights(), weightPath.str());
            std::cout << "Trained on " << route.size() << " snapshots" << std::endl;
            return std::move(infomax);
        }
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    InfoMaxType m_InfoMax;
};

//------------------------------------------------------------------------
// InfoMaxConstrained
//------------------------------------------------------------------------
template<typename FloatType>
class InfoMaxConstrained : public InfoMax<FloatType>
{
public:
    InfoMaxConstrained(const cv::Size &imSize, degree_t fov)
        : InfoMax<FloatType>(imSize), m_FOV(fov), m_ImageWidth(imSize.width)
    {
    }

    virtual void test(const cv::Mat &snapshot) override
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

        // **TODO** calculate vector length
        this->setVectorLength(1.0f);
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const degree_t m_FOV;
    const int m_ImageWidth;
};*/
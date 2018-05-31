#pragma once

// Standard C++ includes
#include <iostream>
#include <vector>

// Standard C includes
#include <cassert>
#include <cstdlib>

// OpenCV includes
#include <opencv2/opencv.hpp>

// GeNN robotics includes
#include "third_party/path.h"

// Snapshot bot includes
#include "config.h"

//------------------------------------------------------------------------
// PerfectMemoryBase
//------------------------------------------------------------------------
template<unsigned int scanStep>
class PerfectMemoryBase
{
public:
    PerfectMemoryBase(const Config &config) : SnapshotRes(config.getUnwrapRes()), m_OutputPath(config.getOutputPath())
    {
    }

    //------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------
    const cv::Size SnapshotRes;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual size_t getNumSnapshots() const = 0;

     //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void load()
    {
        for(size_t i = 0;;i++) {
            const auto filename = getSnapshotPath(i);
            if(filename.exists()) {
                // Load image
                cv::Mat image = cv::imread(filename.str(), cv::IMREAD_GRAYSCALE);
                assert(image.cols == SnapshotRes.width);
                assert(image.rows == SnapshotRes.height);
                assert(image.type() == CV_8UC1);

                // Add snapshot
                addSnapshot(image);
            }
            else {
                break;
	    }
        }
        std::cout << "Loaded " << getNumSnapshots() << " snapshots" << std::endl;
    }

    size_t train(const cv::Mat &image)
    {
        assert(image.cols == SnapshotRes.width);
        assert(image.rows == SnapshotRes.height);
        assert(image.type() == CV_8UC1);

        // Add snapshot
        const size_t index = addSnapshot(image);
        
        // Save snapshot
        cv::imwrite(getSnapshotPath(index).str(), image);
        
        // Return index to snapshot
        return index;
    }

    std::tuple<float, size_t, float> findSnapshot(cv::Mat &image) const
    {
        assert(image.cols == SnapshotRes.width);
        assert(image.rows == SnapshotRes.height);
        assert(image.type() == CV_8UC1);

        // Scan across image columns
        float minDifferenceSquared = std::numeric_limits<float>::max();
        int bestCol = 0;
        size_t bestSnapshot = std::numeric_limits<size_t>::max();
        const size_t numSnapshots = getNumSnapshots();
        for(int i = 0; i < image.cols; i += scanStep) {
            // Loop through snapshots
            for(size_t s = 0; s < numSnapshots; s++) {
                // Calculate difference
                const float differenceSquared = calcSnapshotDifferenceSquared(image, s);

                // If this is an improvement - update
                if(differenceSquared < minDifferenceSquared) {
                    minDifferenceSquared = differenceSquared;
                    bestCol = i;
                    bestSnapshot = s;
                }
            }

            // Roll image left by scanstep
            rollImage(image);
        }

        // If best column is more than 180 degrees away, flip
        if(bestCol > (SnapshotRes.width / 2)) {
            bestCol -= SnapshotRes.width;
        }

        // Convert column into angle
        constexpr float pi = 3.141592653589793238462643383279502884f;
        const float bestAngle = ((float)bestCol / (float)SnapshotRes.width) * (2.0 * pi);

        // Return result
        return std::make_tuple(bestAngle, bestSnapshot, minDifferenceSquared);
    }

protected:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    // Add a snapshot to memory and return its index
    virtual size_t addSnapshot(const cv::Mat &image) = 0;

    // Calculate difference between memory and snapshot with index
    virtual float calcSnapshotDifferenceSquared(const cv::Mat &image, size_t snapshot) const = 0;

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    filesystem::path getSnapshotPath(size_t index) const
    {
        return m_OutputPath / ("snapshot_" + std::to_string(index) + ".png");
    }

    //------------------------------------------------------------------------
    // Private static methods
    //------------------------------------------------------------------------
    // 'Rolls' an image scanStep to the left
    static void rollImage(cv::Mat &image)
    {
        // Buffer to hold scanstep of pixels
        std::array<uint8_t, scanStep> rollBuffer;

        // Loop through rows
        for(int y = 0; y < image.rows; y++) {
            // Get pointer to start of row
            uint8_t *rowPtr = image.ptr(y);

            // Copy scanStep pixels at left hand size of row into buffer
            std::copy_n(rowPtr, scanStep, rollBuffer.begin());

            // Copy rest of row back over pixels we've copied to buffer
            std::copy_n(rowPtr + scanStep, image.cols - scanStep, rowPtr);

            // Copy buffer back into row
            std::copy(rollBuffer.begin(), rollBuffer.end(), rowPtr + (image.cols - scanStep));
        }
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const filesystem::path &m_OutputPath;
};


//------------------------------------------------------------------------
// PerfectMemoryHOG
//------------------------------------------------------------------------
template<unsigned int scanStep>
class PerfectMemoryHOG : public PerfectMemoryBase<scanStep>
{
public:
    PerfectMemoryHOG(const Config &config)
    :   PerfectMemoryBase<scanStep>(config),
        HOGDescriptorSize(config.getHOGDescriptorSize()),
        m_ScratchDescriptors(HOGDescriptorSize)
    {
        std::cout << "Creating perfect memory for HOG features" << std::endl;
        
        // Configure HOG features
        m_HOG.winSize = config.getUnwrapRes(); 
        m_HOG.blockSize = cv::Size(config.getNumHOGPixelsPerCell(), config.getNumHOGPixelsPerCell());
        m_HOG.blockStride = cv::Size(config.getNumHOGPixelsPerCell(), config.getNumHOGPixelsPerCell());
        m_HOG.cellSize = cv::Size(config.getNumHOGPixelsPerCell(), config.getNumHOGPixelsPerCell());
        m_HOG.nbins = config.getNumHOGOrientations();
    }

    //------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------
    const unsigned int HOGDescriptorSize;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual size_t getNumSnapshots() const override { return m_Snapshots.size(); }

protected:
    // Add a snapshot to memory and return its index
    virtual size_t addSnapshot(const cv::Mat &image) override
    {
        m_Snapshots.emplace_back(HOGDescriptorSize);
        m_HOG.compute(image, m_Snapshots.back());
        assert(m_Snapshots.back().size() == HOGDescriptorSize);

        // Return index of new snapshot
        return (m_Snapshots.size() - 1);
    }

    // Calculate difference between memory and snapshot with index
    virtual float calcSnapshotDifferenceSquared(const cv::Mat &image, size_t snapshot) const override
    {
        // Calculate HOG descriptors of image
        m_HOG.compute(image, m_ScratchDescriptors);
        assert(m_ScratchDescriptors.size() == HOGDescriptorSize);

        // Calculate square difference between image HOG descriptors and snapshot
        std::transform(m_Snapshots[snapshot].begin(), m_Snapshots[snapshot].end(),
                       m_ScratchDescriptors.begin(), m_ScratchDescriptors.begin(),
                       [](float a, float b)
                       {
                           return (a - b) * (a - b);
                       });

        // Calculate RMS
        return sqrt(std::accumulate(m_ScratchDescriptors.begin(), m_ScratchDescriptors.end(), 0.0f));
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    mutable std::vector<float> m_ScratchDescriptors;
    std::vector<std::vector<float>> m_Snapshots;
    cv::HOGDescriptor m_HOG;
};

//------------------------------------------------------------------------
// PerfectMemoryHOG
//------------------------------------------------------------------------
template<unsigned int scanStep>
class PerfectMemoryRaw : public PerfectMemoryBase<scanStep>
{
public:
    PerfectMemoryRaw(const Config &config)
    :   PerfectMemoryBase<scanStep>(config),
        m_ScratchImage(config.getUnwrapRes(), CV_8UC1), m_ScratchImageFloat(config.getUnwrapRes(), CV_32FC1),
        m_ScratchXSumFloat(1, config.getUnwrapRes().width, CV_32FC1), m_ScratchSumFloat(1, 1, CV_32FC1)
    {
        std::cout << "Creating perfect memory for raw images" << std::endl;
    }

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual size_t getNumSnapshots() const override { return m_Snapshots.size(); }

protected:
    // Add a snapshot to memory and return its index
    virtual size_t addSnapshot(const cv::Mat &image) override
    {
        m_Snapshots.emplace_back();
        image.copyTo(m_Snapshots.back());

        // Return index of new snapshot
        return (m_Snapshots.size() - 1);
    }

    // Calculate difference between memory and snapshot with index
    virtual float calcSnapshotDifferenceSquared(const cv::Mat &image, size_t snapshot) const override
    {
        // Calculate absolute difference between image and stored image
        cv::absdiff(m_Snapshots[snapshot], image, m_ScratchImage);

        // Convert to float
        m_ScratchImage.convertTo(m_ScratchImageFloat, CV_32FC1, 1.0 / 255.0);

        // Square
        cv::multiply(m_ScratchImageFloat, m_ScratchImageFloat, m_ScratchImageFloat);

        // Reduce difference down twice to get scalar
        cv::reduce(m_ScratchImageFloat, m_ScratchXSumFloat, 0, CV_REDUCE_SUM);
        cv::reduce(m_ScratchXSumFloat, m_ScratchSumFloat, 1, CV_REDUCE_SUM);

        // Extract difference
        return m_ScratchSumFloat.at<float>(0, 0);
    }



private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<cv::Mat> m_Snapshots;
    mutable cv::Mat m_ScratchImage;
    mutable cv::Mat m_ScratchImageFloat;
    mutable cv::Mat m_ScratchXSumFloat;
    mutable cv::Mat m_ScratchSumFloat;
};
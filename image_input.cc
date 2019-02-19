#define NO_HEADER_DEFINITIONS
#include "image_input.h"

// Snapshot bot includes
#include "config.h"

using namespace BoBRobotics;

//----------------------------------------------------------------------------
// ImageInput
//----------------------------------------------------------------------------
ImageInput::ImageInput(const Config &config)
:   m_InputSize(config.getUnwrapRes())
{
}


//----------------------------------------------------------------------------
// ImageInputRaw
//----------------------------------------------------------------------------
ImageInputRaw::ImageInputRaw(const Config &config)
:   ImageInput(config), m_GreyscaleUnwrapped(config.getUnwrapRes(), CV_8UC1)
{
}
//----------------------------------------------------------------------------
const cv::Mat &ImageInputRaw::processSnapshot(const cv::Mat &snapshot)
{
    // Convert snapshot to greyscale and return
    cv::cvtColor(snapshot, m_GreyscaleUnwrapped, cv::COLOR_BGR2GRAY);
    return m_GreyscaleUnwrapped;
}

//----------------------------------------------------------------------------
// ImageInputBinary
//----------------------------------------------------------------------------
ImageInputBinary::ImageInputBinary(const Config &config)
:   ImageInput(config), m_SegmentIndices(config.getUnwrapRes(), CV_32SC1), m_SegmentedImage(config.getUnwrapRes(), CV_8UC1)
{
    // Read marker image
    // **NOTE** will read 8-bit per channel grayscale
    m_MarkerImage = cv::imread(config.getWatershedMarkerImageFilename(), cv::IMREAD_GRAYSCALE);
    BOB_ASSERT(m_MarkerImage.size() == config.getUnwrapRes());
    
    // Convert marker image to 32-bit per channel without performing any scaling
    m_MarkerImage.convertTo(m_MarkerImage, CV_32SC1);
    
    // Find minimum and maximum elements
    m_MinIndex = *std::min_element(m_MarkerImage.begin<int32_t>(), m_MarkerImage.end<int32_t>());
    m_MaxIndex = *std::max_element(m_MarkerImage.begin<int32_t>(), m_MarkerImage.end<int32_t>());
    
    // Check that there are indeed 3 markers as expected
    BOB_ASSERT(m_MinIndex == 0);
    BOB_ASSERT(m_MaxIndex == 2);
}
//----------------------------------------------------------------------------
const cv::Mat &ImageInputBinary::processSnapshot(const cv::Mat &snapshot)
{
    // Read indices of segments
    const cv::Mat &segmentedIndices = readSegmentIndices(snapshot);
    BOB_ASSERT(false);
    return segmentedIndices;
}
//----------------------------------------------------------------------------
const cv::Mat &ImageInputBinary::readSegmentIndices(const cv::Mat &snapshot)
{
    // Make a copy of marker image to perform segmentation on
    m_MarkerImage.copyTo(m_SegmentIndices);
    
    // Segment!
    cv::watershed(snapshot, m_SegmentIndices);
    return m_SegmentIndices;
}

//----------------------------------------------------------------------------
// ImageInputHorizon
//----------------------------------------------------------------------------
ImageInputHorizon::ImageInputHorizon(const Config &config)
:   ImageInputBinary(config), m_HorizonVector(1, config.getUnwrapRes().width, CV_8UC1),
    m_ColumnHorizonPixelsSum(config.getUnwrapRes().width), m_ColumnHorizonPixelsCount(config.getUnwrapRes().width)
{
    // Check image will be representable as 8-bit value
    BOB_ASSERT(config.getUnwrapRes().height < 0xFF);
}
//----------------------------------------------------------------------------
const cv::Mat &ImageInputHorizon::processSnapshot(const cv::Mat &snapshot)
{
    // Read indices of segments
    const cv::Mat &segmentedIndices = readSegmentIndices(snapshot);

    // Zero counts and sum of horizon pixels per column
    m_ColumnHorizonPixelsSum.assign(segmentedIndices.cols, 0);
    m_ColumnHorizonPixelsCount.assign(segmentedIndices.cols, 0);
    BOB_ASSERT(m_ColumnHorizonPixelsSum.size() == (size_t)segmentedIndices.cols);
    BOB_ASSERT(m_ColumnHorizonPixelsCount.size() == (size_t)segmentedIndices.cols);
    
    // Loop through image columns
    const int32_t *data = reinterpret_cast<const int32_t*>(segmentedIndices.data);
    for(int y = 0; y < segmentedIndices.rows; y++) {
        for(int x = 0; x < segmentedIndices.cols; x++) {
            // Get next index
            const int32_t index = *data++;
            
            // If this is a horizon pixel
            if(index == -1) {
                // Increment number of pixels per column
                m_ColumnHorizonPixelsCount[x]++;
                
                // Add to total in horizon
                m_ColumnHorizonPixelsCount[x] += y;
            }
        }
    }
    
    // Populate horizon vector with average horizon height
    std::transform(m_ColumnHorizonPixelsSum.cbegin(), m_ColumnHorizonPixelsSum.cend(), m_ColumnHorizonPixelsCount.cbegin(), m_HorizonVector.begin<uint8_t>(),
                   [](int sum, int count)
                   {
                       return (uint8_t)(sum / count);
                   });
    
    return m_HorizonVector;
}
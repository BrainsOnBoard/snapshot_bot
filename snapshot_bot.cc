// Standard C++ includes
#include <limits>
#include <tuple>
#include <vector>

// Standard C includes
#include <cassert>

// POSIX includes
#include <sys/stat.h>

// GeNN robotics includes
#include "fsm.h"
#include "joystick.h"
#include "motor_i2c.h"
#include "opencv_unwrap_360.h"
#include "see3cam_cu40.h"
#include "timer.h"

constexpr float pi = 3.141592653589793238462643383279502884f;

namespace Settings
{
    
    // What resolution to operate camera at
    const See3CAM_CU40::Resolution camRes = See3CAM_CU40::Resolution::_1280x720;

    // What resolution to unwrap panoramas to
    const cv::Size unwrapRes(180, 50);

    // HOG configuration
    const unsigned int numHOGOrientations = 8;
    const unsigned int numHOGPixelsPerCell = 10;

    // How large should the deadzone be on the analogue joystick
    const float joystickDeadzone = 0.25f;
}

enum class State
{
    Invalid,
    Training,
    Testing,
};

//------------------------------------------------------------------------
// PerfectMemoryBase
//------------------------------------------------------------------------
template<unsigned int scanStep>
class PerfectMemoryBase
{
public:
    PerfectMemoryBase(const cv::Size &snapshotRes) : SnapshotRes(snapshotRes)
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
        struct stat buffer;
        for(size_t i = 0;;i++) {
            char filename[128];
            sprintf(filename, "snapshot_%zu.png", i);
            if(stat(filename, &buffer) == 0) {
                // Load image
                cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
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

        // Add snapshot and return its index
        return addSnapshot(image);
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
    // Private static methods
    //------------------------------------------------------------------------
    // 'Rolls' an image scanStep to the left
    static void rollImage(cv::Mat &image)
    {
        // Buffer to hold scanstep of pixels
        std::array<uint8_t, scanStep> rollBuffer;

        // Loop through rows
        for(unsigned int y = 0; y < image.rows; y++) {
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
};


//------------------------------------------------------------------------
// PerfectMemoryHOG
//------------------------------------------------------------------------
template<unsigned int scanStep>
class PerfectMemoryHOG : public PerfectMemoryBase<scanStep>
{
public:
    PerfectMemoryHOG(const cv::Size &snapshotRes, unsigned int numHOGOrientations = 8, unsigned int numHOGPixelsPerCell = 10)
    :   PerfectMemoryBase<scanStep>(snapshotRes), HOGDescriptorSize((snapshotRes.width * snapshotRes.height * numHOGOrientations) / (numHOGPixelsPerCell * numHOGPixelsPerCell)),
        m_ScratchDescriptors(HOGDescriptorSize)
    {
        // Configure HOG features
        m_HOG.winSize = snapshotRes; 
        m_HOG.blockSize = cv::Size(numHOGPixelsPerCell, numHOGPixelsPerCell);
        m_HOG.blockStride = cv::Size(numHOGPixelsPerCell, numHOGPixelsPerCell);
        m_HOG.cellSize = cv::Size(numHOGPixelsPerCell, numHOGPixelsPerCell);
        m_HOG.nbins = numHOGOrientations;
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
    PerfectMemoryRaw(const cv::Size &snapshotRes)
    :   PerfectMemoryBase<scanStep>(snapshotRes), m_ScratchImage(snapshotRes, CV_8UC1), m_ScratchImageFloat(snapshotRes, CV_32FC1),
        m_ScratchXSumFloat(1, snapshotRes.width, CV_32FC1), m_ScratchSumFloat(1, 1, CV_32FC1)
    {
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


//------------------------------------------------------------------------
// RobotFSM
//------------------------------------------------------------------------
class RobotFSM : FSM<State>::StateHandler
{
public:
    RobotFSM(unsigned int camDevice, See3CAM_CU40::Resolution camRes, const cv::Size &unwrapRes, bool load) 
    :   m_StateMachine(this, State::Invalid), m_Camera("/dev/video" + std::to_string(camDevice), camRes),
        m_Output(m_Camera.getSuperPixelSize(), CV_8UC1), m_Unwrapped(unwrapRes, CV_8UC1),
        m_Unwrapper(See3CAM_CU40::createUnwrapper(m_Camera.getSuperPixelSize(), unwrapRes)),
        m_Memory(unwrapRes)
    {
        m_Camera.setExposure(300);
        m_Camera.setBrightness(30);
        // If we should load in existing snapshots
        if(load) {
            // Load memory
            m_Memory.load();
            
            // Start directly in testing state
            m_StateMachine.transition(State::Testing);
        }
        else {
            // Delete old snapshots
            system("rm -f snapshot_*.png");

            // Start in training state
            m_StateMachine.transition(State::Training);
        }
    }
    
    ~RobotFSM() 
    {
        // Stop motors
        m_Motor.tank(0.0f, 0.0f);
    }
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool update()
    {
        return m_StateMachine.update();
    }
    
    const cv::Mat &getOutput() const{ return m_Output; }
    const cv::Mat &getUnwrapped() const{ return m_Unwrapped; }
    
private:
    //------------------------------------------------------------------------
    // FSM::StateHandler virtuals
    //------------------------------------------------------------------------
    virtual bool handleEvent(State state, Event event) override
    {
        // If this event is an update
        if(event == Event::Update) {
            // Read joystick
            m_Joystick.read();
            
            // Exit if button 2 is pressed
            if(m_Joystick.isButtonPressed(2)) {
                return false;
            }

            // Capture greyscale frame
            if(!m_Camera.captureSuperPixelGreyscale(m_Output)) {
                return false;
            }
            
            // Unwrap frame
            m_Unwrapper.unwrap(m_Output, m_Unwrapped);
        }

        if(state == State::Training) {
            if(event == Event::Enter) {
                std::cout << "Starting training" << std::endl;
            }
            else if(event == Event::Update) {
                // Drive motors using joystick
                m_Joystick.drive(m_Motor, Settings::joystickDeadzone);
                
                if(m_Joystick.isButtonPressed(0)) {
                    const size_t snapshotID = m_Memory.train(m_Unwrapped);
                    std::cout << "\tTrained snapshot id:" << snapshotID << std::endl;
                }
                else if(m_Joystick.isButtonPressed(1)) {
                    m_StateMachine.transition(State::Testing);
                }
            }
        }
        else if(state == State::Testing) {
            if(event == Event::Enter) {
                std::cout << "Testing: finding snapshot" << std::endl;
                m_MoveTime  = 0;
            }
            else if(event == Event::Update) {
                // If it's time to move
                if(m_MoveTime == 0) {
                    // Reset move time
                    m_MoveTime = 10;

	            // Find matching snapshot
        	    float turnToAngle;
                    size_t turnToSnapshot;
              	    float minDifferenceSquared;
                    std::tie(turnToAngle, turnToSnapshot, minDifferenceSquared) = m_Memory.findSnapshot(m_Unwrapped);

                    // If a snapshot is found and it isn't the one we were previously at
                    if(turnToSnapshot != std::numeric_limits<size_t>::max()) {
                        std::cout << "\tBest match found with snapshot id " << turnToSnapshot << " (angle:" << turnToAngle << ", min difference:" << minDifferenceSquared << ")" << std::endl;

                        // If we're well oriented with snapshot, drive forward
                        const float turnMagnitude = fabs(turnToAngle);
                        if(turnMagnitude < 0.1f) {
                            m_Motor.tank(1.0f, 1.0f);
                        }
                        // Otherwise, turn towards snapshot
                        else {
                            const float motorSpeed = (turnMagnitude < 0.2f) ? 0.5f : 1.0f;
                            const float motorTurn = (turnToAngle <  0.0f) ? -motorSpeed : motorSpeed;
                            m_Motor.tank(motorTurn, -motorTurn);
                        }
                    }
                    else {
                        std::cerr << "No snapshots learned" << std::endl;
                        return false;
                    }
                }
                // Otherwise just decrement move time
                else {
                    m_MoveTime--;
                }
            }
        }
        else {
            std::cerr << "Invalid state" << std::endl;
            return false;
        }
        return true;
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    // State machine
    FSM<State> m_StateMachine;

    // Camera interface
    See3CAM_CU40 m_Camera;

    // Joystick interface
    Joystick m_Joystick;

    // OpenCV images used to store raw camera frame and unwrapped panorama
    cv::Mat m_Output;
    cv::Mat m_Unwrapped;

    // OpenCV-based panorama unwrapper
    OpenCVUnwrap360 m_Unwrapper;

    // Perfect memory
    PerfectMemoryHOG<1> m_Memory;

    // Motor driver
    MotorI2C m_Motor;

    // 'Timer' used to move between snapshot tests
    unsigned int m_MoveTime;
};

int main(int argc, char *argv[])
{
    const bool load = (argc > 1);
    RobotFSM robot(0, Settings::camRes, Settings::unwrapRes, load);
    
    {
        Timer<> timer("Total time:");

        unsigned int frame = 0;
        for(frame = 0; robot.update(); frame++) {
        }
        
        const double msPerFrame = timer.get() / (double)frame;
        std::cout << "FPS:" << 1000.0 / msPerFrame << std::endl;
    }

    return 0;
}

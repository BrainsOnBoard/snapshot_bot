// Standard C++ includes
#include <limits>
#include <tuple>
#include <vector>

// Standard C includes
#include <cassert>

// POSIX includes
#include <sys/stat.h>

// GeNN robotics includes
#include "common/fsm.h"
#include "common/joystick.h"
#include "common/timer.h"
#include "imgproc/opencv_unwrap_360.h"
#include "robots/motor_i2c.h"
#include "vicon/capture_control.h"
#include "vicon/udp.h"
#include "video/see3cam_cu40.h"

// Snapshot bot includes
#include "config.h"

using namespace GeNNRobotics;

constexpr float pi = 3.141592653589793238462643383279502884f;

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

        // Add snapshot
        const size_t index = addSnapshot(image);
        
        // Save snapshot
        char filename[128];
        sprintf(filename, "snapshot_%zu.png", index);
        cv::imwrite(filename, image);
        
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
    RobotFSM(const Config &config, bool load)
    :   m_Config(config), m_StateMachine(this, State::Invalid),
        m_Camera("/dev/video" + std::to_string(config.getCamDevice()), config.getSee3CamRes()),
        m_Output(m_Camera.getSuperPixelSize(), CV_8UC1), m_Unwrapped(config.getUnwrapRes(), CV_8UC1),
        m_Unwrapper(m_Camera.createDefaultUnwrapper(config.getUnwrapRes())),
        m_Memory(config.getUnwrapRes())
    {
        // Run auto exposure algorithm
        const cv::Mat bubblescopeMask = Video::See3CAM_CU40::createBubblescopeMask(m_Camera.getSuperPixelSize());
        m_Camera.autoExposure(bubblescopeMask);

        // If we should use Vicon tracking
        if(m_Config.shouldUseViconTracking()) {
            // Connect to port specified in config
            if(!m_ViconTracking.connect(m_Config.getViconTrackingPort())) {
                throw std::runtime_error("Cannot connect to Vicon tracking system");
            }
            
            // Wait for tracking data stream to begin
            while(m_ViconTracking.getNumObjects() == 0) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                std::cout << "Waiting for Vicon tracking data object" << std::endl;
            }
        }
        
        // If we should use Vicon capture control
        if(m_Config.shouldUseViconCaptureControl()) {
            // Connect to capture host system specified in config
            if(!m_ViconCaptureControl.connect(m_Config.getViconCaptureControlHost(), m_Config.getViconCaptureControlPort(),
                m_Config.getViconCaptureControlPath())) 
            {
                throw std::runtime_error("Cannot connect to Vicon capture control");
            }
        }
        
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
                m_Joystick.drive(m_Motor, m_Config.getJoystickDeadzone());
                
                // If 1st button is pressed, save snapshot
                if(m_Joystick.isButtonPressed(0)) {
                    const size_t snapshotID = m_Memory.train(m_Unwrapped);
                    std::cout << "\tTrained snapshot id:" << snapshotID << std::endl;
                }
                // Otherwise, if 2nd button is pressed, go to testing
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
                    m_MoveTime = m_Config.getMoveTimesteps();

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
                        
                        auto turnSpeed = m_Config.getTurnSpeed(turnMagnitude);
                        if(turnSpeed > 0.0f) {
                            const float motorTurn = (turnToAngle <  0.0f) ? -turnSpeed : turnSpeed;
                            m_Motor.tank(motorTurn, -motorTurn);
                        }
                        else {
                            m_Motor.tank(1.0f, 1.0f);
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
    // Configuration
    const Config &m_Config;

    // State machine
    FSM<State> m_StateMachine;

    // Camera interface
    Video::See3CAM_CU40 m_Camera;

    // Joystick interface
    Joystick m_Joystick;

    // OpenCV images used to store raw camera frame and unwrapped panorama
    cv::Mat m_Output;
    cv::Mat m_Unwrapped;

    // OpenCV-based panorama unwrapper
    ImgProc::OpenCVUnwrap360 m_Unwrapper;

    // Perfect memory
    //PerfectMemoryHOG<1> m_Memory;
    PerfectMemoryRaw<1> m_Memory;

    // Motor driver
    Robots::MotorI2C m_Motor;

    // 'Timer' used to move between snapshot tests
    int m_MoveTime;

    // Vicon tracking interface
    Vicon::UDPClient<Vicon::ObjectData> m_ViconTracking;

    // Vicon capture control interface
    Vicon::CaptureControl m_ViconCaptureControl;
};

int main(int argc, char *argv[])
{
    const char *configFilename = (argc > 1) ? argv[1] : "config.yaml";
    
    // Read config values from file
    Config config;
    {
        cv::FileStorage configFile(configFilename, cv::FileStorage::READ);
        if(configFile.isOpened()) {
            configFile["config"] >> config;
        }
    }

    // Re-write config file
    {
        cv::FileStorage configFile(configFilename, cv::FileStorage::WRITE);
        configFile << "config" << config;
    }
    
   // }
    // Create Vicon UDP interface
    /*Vicon::UDPClient<Vicon::ObjectData> vicon(51001);

    // Create Vicon capture control interface
    Vicon::CaptureControl viconCaptureControl("192.168.1.100", 3003,
                                              "c:\\users\\ad374\\Desktop");

    // Wait for tracking
    while(vicon.getNumObjects() == 0) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "Waiting for object" << std::endl;
    }

    // Start capture
    if(!viconCaptureControl.startRecording("camera_recorder")) {
        return EXIT_FAILURE;
    }*/

    const bool load = (argc > 1);
    RobotFSM robot(config, load);
    
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

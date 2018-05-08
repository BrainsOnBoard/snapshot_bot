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

    const size_t hogDescriptorSize = (unwrapRes.width * unwrapRes.height * 8) / (10 * 10);
    
    // How large should the deadzone be on the analogue joystick
    const float joystickDeadzone = 0.25f;

    const float  threshold = 30.0f;
}

enum class State
{
    Invalid,
    Training,
    Testing,
};

//------------------------------------------------------------------------
// PerfectMemory
//------------------------------------------------------------------------
template<unsigned int scanStep>
class PerfectMemory
{
public:
    PerfectMemory(const cv::Size &snapshotRes) 
    :   m_ScratchImage(snapshotRes, CV_8UC1), m_ScratchDescriptors(Settings::hogDescriptorSize)
    {
        // Configure HOG features
        m_HOG.winSize = snapshotRes; 
        m_HOG.blockSize = cv::Size(10, 10);
        m_HOG.blockStride = cv::Size(10, 10);
        m_HOG.cellSize = cv::Size(10, 10);
        m_HOG.nbins = 8;
    }
    
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
                assert(image.cols == m_ScratchImage.cols);
                assert(image.rows == m_ScratchImage.rows);
                assert(image.type() == CV_8UC1);
                
                // Calculate HOG features
                m_Snapshots.emplace_back(Settings::hogDescriptorSize);
                m_HOG.compute(image, m_Snapshots.back());
                assert(m_Snapshots.back().size() == Settings::hogDescriptorSize);
                
            }
            else {
                break;
	    }
        }
        std::cout << "Loaded " << m_Snapshots.size() << " snapshots" << std::endl;
    }

    size_t train(const cv::Mat &image) 
    {
        assert(image.cols == m_ScratchImage.cols);
        assert(image.rows == m_ScratchImage.rows);
        assert(image.type() == CV_8UC1);
            
        // Add a new snapshot and store HOG coefficients in it
        m_Snapshots.emplace_back(Settings::hogDescriptorSize);
        m_HOG.compute(image, m_Snapshots.back());
        assert(m_Snapshots.back().size() == Settings::hogDescriptorSize);
        
        char filename[128];
        sprintf(filename, "snapshot_%zu.png", m_Snapshots.size() - 1);
        cv::imwrite(filename, image);

        // Return index of new snapshot
        return (m_Snapshots.size() - 1);
    }
    
    std::tuple<float, size_t, float> findSnapshot(cv::Mat &image)
    {
        assert(image.cols == m_ScratchImage.cols);
        assert(image.rows == m_ScratchImage.rows);
        assert(image.type() == CV_8UC1);
        
        // Scan across image columns
        float minDifferenceSquared = std::numeric_limits<float>::max();
        int bestCol = 0;
        size_t bestSnapshot = std::numeric_limits<size_t>::max();
        for(int i = 0; i < image.cols; i += scanStep) {
            // Loop through snapshots
            for(size_t s = 0; s < m_Snapshots.size(); s++) {
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
        if(bestCol > (m_ScratchImage.cols / 2)) {
            bestCol -= m_ScratchImage.cols;
        }

        // Convert column into angle
        const float bestAngle = ((float)bestCol / (float)m_ScratchImage.cols) * (2.0 * pi);
        
        // Return result
        return std::make_tuple(bestAngle, bestSnapshot, minDifferenceSquared);
    }
    
    float calcSnapshotDifferenceSquared(const cv::Mat &image, size_t snapshot)
    {
        // Calculate HOG descriptors of image
        m_HOG.compute(image, m_ScratchDescriptors);
        assert(m_ScratchDescriptors.size() == Settings::hogDescriptorSize);
        
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

    unsigned int getNumSnapshots() const{ return m_Snapshots.size(); }

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
    
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<float> m_ScratchDescriptors;
    std::vector<std::vector<float>> m_Snapshots;
    cv::Mat m_ScratchImage;
    cv::HOGDescriptor m_HOG;
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
    PerfectMemory<1> m_Memory;

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

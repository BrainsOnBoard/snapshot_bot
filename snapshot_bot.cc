// Standard C++ includes
#include <limits>
#include <tuple>
#include <vector>

// Standard C includes
#include <cassert>

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
    const cv::Size unwrapRes(180, 20);

    // How large should the deadzone be on the analogue joystick
    const float joystickDeadzone = 0.25f;

    const float  threshold = 30.0f;
}

enum class State
{
    Invalid,
    Warmup,
    Training,
    ReturningToStart,
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
    :   m_ScratchImage(snapshotRes, CV_8UC1), m_ScratchImageFloat(snapshotRes, CV_32FC1),
        m_ScratchXSumFloat(1, snapshotRes.width, CV_32FC1), m_ScratchSumFloat(1, 1, CV_32FC1)
    {
    }
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t train(const cv::Mat &image) 
    {
        assert(image.cols == m_ScratchImage.cols);
        assert(image.rows == m_ScratchImage.rows);
        assert(image.type() == CV_8UC1);
            
        // Add a new snapshot and make copy of image into it
        m_Snapshots.emplace_back();
        image.copyTo(m_Snapshots.back());

        char filename[128];
        sprintf(filename, "snapshot_%zu.png", m_Snapshots.size() - 1);
        cv::imwrite(filename, m_Snapshots.back());

        // Return index of new snapshot
        return (m_Snapshots.size() - 1);
    }
    
    std::tuple<float, size_t, float> findSnapshot(cv::Mat &image, size_t lastSnapshot) const
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
                
                if(s == lastSnapshot && differenceSquared > Settings::threshold) {
                    continue;
                }
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
    
    float calcSnapshotDifferenceSquared(const cv::Mat &image, size_t snapshot) const
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

    void clear()
    {
        m_Snapshots.clear();
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
    std::vector<cv::Mat> m_Snapshots;
    cv::Mat m_ScratchImage;
    cv::Mat m_ScratchImageFloat;
    cv::Mat m_ScratchXSumFloat;
    cv::Mat m_ScratchSumFloat;
};

//------------------------------------------------------------------------
// RobotFSM
//------------------------------------------------------------------------
class RobotFSM : FSM<State>::StateHandler
{
public:
    RobotFSM(unsigned int camDevice, See3CAM_CU40::Resolution camRes, const cv::Size &unwrapRes) 
    :   m_StateMachine(this, State::Invalid), m_Camera("/dev/video" + std::to_string(camDevice), camRes),
        m_Output(m_Camera.getSuperPixelSize(), CV_8UC1), m_Unwrapped(unwrapRes, CV_8UC1),
        m_Unwrapper(See3CAM_CU40::createUnwrapper(m_Camera.getSuperPixelSize(), unwrapRes)),
        m_Memory(unwrapRes)
    {
        m_Camera.setExposure(200);
        
        // Start in training state
        m_StateMachine.transition(State::Warmup);
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

        if(state == State::Warmup) {
	    if(event == Event::Enter) {
                std::cout << "Starting warmup" << std::endl;
                m_Memory.train(m_Unwrapped);
            }
            else if(event == Event::Update) {
                const float differenceSinceLast = m_Memory.calcSnapshotDifferenceSquared(m_Unwrapped, m_Memory.getNumSnapshots() - 1);
                if(differenceSinceLast < 10.0f) {
                     m_Memory.clear();
                     m_StateMachine.transition(State::Training);
                }
                else {
                    m_Memory.train(m_Unwrapped);
                }
            }

        }
        else if(state == State::Training) {
            if(event == Event::Enter) {
                std::cout << "Starting training" << std::endl;
                const size_t snapshotID = m_Memory.train(m_Unwrapped);
                std::cout << "\tTrained snapshot id:" << snapshotID << std::endl;
            }
            else if(event == Event::Update) {
                // Drive motors using joystick
                m_Joystick.drive(m_Motor, Settings::joystickDeadzone);
                
                const float threshold = 20.0f - (fabs(m_Joystick.getAxisState(0)) * 10.0f);
                const float differenceSinceLast = m_Memory.calcSnapshotDifferenceSquared(m_Unwrapped, m_Memory.getNumSnapshots() - 1);
                if(differenceSinceLast > threshold) {
                    const size_t snapshotID = m_Memory.train(m_Unwrapped);
                    std::cout << "\tTrained snapshot id:" << snapshotID << " (difference since last " << differenceSinceLast << ")" << std::endl;
                }
                
                if(m_Joystick.isButtonPressed(1)) {
                    m_StateMachine.transition(State::ReturningToStart);
                }
            }
        }
        else if(state == State::ReturningToStart) {
            if(event == Event::Enter) {
                std::cout << "Return robot to start and press button 1 (B)" << std::endl;
            }
            else if(event == Event::Update) {
                m_Joystick.drive(m_Motor, Settings::joystickDeadzone);

                if(m_Joystick.isButtonPressed(1)) {
                    m_StateMachine.transition(State::Testing);
                }
            }
        }
        else if(state == State::Testing) {
            if(event == Event::Enter) {
                std::cout << "Testing: finding snapshot" << std::endl;
                m_LastSnapshot = std::numeric_limits<size_t>::max();
            }
            else if(event == Event::Update) {
                // Find matching snapshot
                float turnToAngle;
                size_t turnToSnapshot;
                float minDifferenceSquared;
                std::tie(turnToAngle, turnToSnapshot, minDifferenceSquared) = m_Memory.findSnapshot(m_Unwrapped, m_LastSnapshot);
                //m_LastSnapshot = turnToSnapshot;

                // If a snapshot is found and it isn't the one we were previously at
                if(turnToSnapshot != std::numeric_limits<size_t>::max()) {
                    std::cout << "\tBest match found with snapshot id " << turnToSnapshot << " (angle:" << turnToAngle << ", min difference:" << minDifferenceSquared << ")" << std::endl;
                    
                    // If we're well oriented with snapshot, drive forwards
                    //if(fabs(turnToAngle) < 0.1f) {
                    //    m_Motor.tank(1.0f, 1.0f);
                    //}
                    // Otherwise, turn towards snapshot
                    //else {
                    //    const float motorTurn = std::min(1.0f, std::max(-1.0f, turnToAngle * 2.0f));
                    //    m_Motor.tank(motorTurn, -motorTurn);
                    //}
                    const float theta = turnToAngle;
                    const float twoTheta = theta * 2.0f;
                    const float r = 0.6f;
                    const float halfPi = pi * 0.5f;
                    // Drive motor
            	    if(theta >= 0.0f && theta < halfPi) {
                        m_Motor.tank(r, r * cos(twoTheta));
                    }
                    else if(theta >= halfPi && theta < pi) {
                        m_Motor.tank(-r * cos(twoTheta), -r);
                    }
                    else if(theta < 0.0f && theta >= -halfPi) {
                        m_Motor.tank(r * cos(twoTheta), r);
                    }
                    else if(theta < -halfPi && theta >= -pi) {
                        m_Motor.tank(-r, -r * cos(twoTheta));
                    }
                }
                else {                    std::cerr << "No snapshots learned" << std::endl;
                    return false;
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

    size_t m_LastSnapshot;
    // OpenCV images used to store raw camera frame and unwrapped panorama
    cv::Mat m_Output;
    cv::Mat m_Unwrapped;

    // OpenCV-based panorama unwrapper
    OpenCVUnwrap360 m_Unwrapper;

    // Perfect memory
    PerfectMemory<1> m_Memory;

    // Motor driver
    MotorI2C m_Motor;

};

int main()
{
    //cv::namedWindow("Raw", CV_WINDOW_NORMAL);
    //cv::namedWindow("Unwrapped", CV_WINDOW_NORMAL);

    RobotFSM robot(0, Settings::camRes, Settings::unwrapRes);
    
    {
        Timer<> timer("Total time:");

        unsigned int frame = 0;
        for(frame = 0; robot.update(); frame++) {
            //cv::imshow("Raw", robot.getOutput());
            //cv::imshow("Unwrapped", robot.getUnwrapped());
            
            //cv::waitKey(1);
        }
        
        const double msPerFrame = timer.get() / (double)frame;
        std::cout << "FPS:" << 1000.0 / msPerFrame << std::endl;
    }

    return 0;
}

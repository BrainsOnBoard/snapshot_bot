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

// Snapshot bot includes
#include "snapshot_encoder.h"

constexpr float pi = 3.141592653589793238462643383279502884f;

namespace Settings
{
    // What resolution to operate camera at
    const See3CAM_CU40::Resolution camRes = See3CAM_CU40::Resolution::_1280x720;

    // What resolution to unwrap panoramas to
    const cv::Size unwrapRes(90, 25);

    const unsigned int encodedSnapshotSize = 64;
    
    // How large should the deadzone be on the analogue joystick
    const float joystickDeadzone = 0.25f;

    const float threshold = 30.0f;
}

enum class State
{
    Invalid,
    BuildingOfflineTrainingData,
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
    PerfectMemory(const cv::Size &snapshotRes, unsigned int encodedSnapshotSize) 
    :   m_ScratchImage(1, encodedSnapshotSize, CV_8UC1), m_ScratchImageFloat(1, encodedSnapshotSize, CV_32FC1),
        m_ScratchSumFloat(1, 1, CV_32FC1),
        m_Encoder(snapshotRes.width, snapshotRes.height, encodedSnapshotSize), m_SnapshotSize(snapshotRes)
    {
    }
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool openEncoderModel(const std::string &exportDirectory, const std::string &tag,
                          const std::string &inputOpName, const std::string &outputOpName)
    {
        return m_Encoder.openModel(exportDirectory, tag, inputOpName, outputOpName);
    }
    
    size_t train(const cv::Mat &image) 
    {
        assert(image.cols == m_SnapshotSize.width);
        assert(image.rows == m_SnapshotSize.height);
        assert(image.type() == CV_8UC1);
        assert(isEncoderModelOpen());
        
        // Convert snapshot to float and encode
        image.convertTo(m_ScratchImageFloat, CV_32FC1, 1.0 / 255.0);
        m_Encoder.encode(m_ScratchImageFloat);
        
        // Add a new snapshot and copy encoder output into it
        m_Snapshots.emplace_back();
        m_Encoder.getFinalSnapshot().copyTo(m_Snapshots.back());

        char filename[128];
        sprintf(filename, "snapshot_%zu.png", m_Snapshots.size() - 1);
        cv::imwrite(filename, m_Snapshots.back());

        // Return index of new snapshot
        return (m_Snapshots.size() - 1);
    }
    
    std::tuple<float, size_t, float> findSnapshot(cv::Mat &image)
    {
        assert(image.cols == m_SnapshotSize.width);
        assert(image.rows == m_SnapshotSize.height);
        assert(image.type() == CV_8UC1);
        assert(isEncoderModelOpen());
        
        // Scan across image columns
        float minDifferenceSquared = std::numeric_limits<float>::max();
        int bestCol = 0;
        size_t bestSnapshot = std::numeric_limits<size_t>::max();
        for(int i = 0; i < image.cols; i += scanStep) {
            // Convert rolled image to float
            image.convertTo(m_ScratchImageFloat, CV_32FC1, 1.0 / 255.0);
            
            // Encode floating point snapshot
            m_Encoder.encode(m_ScratchImageFloat);

            // Loop through snapshots
            for(size_t s = 0; s < m_Snapshots.size(); s++) {
                // Calculate difference between encoded input and snapshot
                const float differenceSquared = calcSnapshotDifferenceSquared(m_Encoder.getFinalSnapshot(), s);
                
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
        if(bestCol > (m_SnapshotSize.width / 2)) {
            bestCol -= m_SnapshotSize.width;
        }

        // Convert column into angle
        const float bestAngle = ((float)bestCol / (float)m_SnapshotSize.width) * (2.0 * pi);
        
        // Return result
        return std::make_tuple(bestAngle, bestSnapshot, minDifferenceSquared);
    }
    
    float calcSnapshotDifferenceSquared(const cv::Mat &image, size_t snapshot)
    {
        // Calculate absolute difference between image and stored image
        cv::absdiff(m_Snapshots[snapshot], image, m_ScratchImage);
        
        // Convert to float
        m_ScratchImage.convertTo(m_ScratchImageFloat, CV_32FC1, 1.0 / 255.0);
        
        // Square 
        cv::multiply(m_ScratchImageFloat, m_ScratchImageFloat, m_ScratchImageFloat);

        // Reduce difference down twice to get scalar
        cv::reduce(m_ScratchImageFloat, m_ScratchSumFloat, 1, CV_REDUCE_SUM);
        
        // Extract difference
        return m_ScratchSumFloat.at<float>(0, 0);
    }

    unsigned int getNumSnapshots() const{ return m_Snapshots.size(); }
    bool isEncoderModelOpen() const{ return m_Encoder.isModelOpen(); }
    
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
    cv::Mat m_ScratchSumFloat;
    
    SnapshotEncoder m_Encoder;
    
    const cv::Size m_SnapshotSize;
};

//------------------------------------------------------------------------
// RobotFSM
//------------------------------------------------------------------------
class RobotFSM : FSM<State>::StateHandler
{
public:
    RobotFSM(unsigned int camDevice, See3CAM_CU40::Resolution camRes, const cv::Size &unwrapRes, unsigned int encodedSnapshotSize, bool buildTrainingData) 
    :   m_StateMachine(this, State::Invalid), m_Camera("/dev/video" + std::to_string(camDevice), camRes),
        m_Output(m_Camera.getSuperPixelSize(), CV_8UC1), m_Unwrapped(unwrapRes, CV_8UC1),
        m_Unwrapper(See3CAM_CU40::createUnwrapper(m_Camera.getSuperPixelSize(), unwrapRes)),
        m_Memory(unwrapRes, encodedSnapshotSize)
    {
        m_Camera.setBrightness(20);
        
        // Start in training state
        if(buildTrainingData) {
            m_StateMachine.transition(State::BuildingOfflineTrainingData);
        }
        else {
            // Open encoder model 
            if(!m_Memory.openEncoderModel("./export_office_good", "tag", "input", "encoder_3")) {
                throw std::runtime_error("Cannot load encoder model");
            }
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

        if(state == State::BuildingOfflineTrainingData) {
            if(event == Event::Enter) {
                std::cout << "Starting to build offline training data" << std::endl;
                m_TrainingSnapshot = 0;
            }
            else if(event == Event::Update) {
                char filename[128];
                sprintf(filename, "training_%u.png", m_TrainingSnapshot++);
                
                cv::imwrite(filename, m_Unwrapped);

                m_Joystick.drive(m_Motor, Settings::joystickDeadzone);
            }
        }
        else if(state == State::Training) {
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
            }
            else if(event == Event::Update) {
                // Find matching snapshot
                float turnToAngle;
                size_t turnToSnapshot;
                float minDifferenceSquared;
                std::tie(turnToAngle, turnToSnapshot, minDifferenceSquared) = m_Memory.findSnapshot(m_Unwrapped);

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
                else {                    
                    std::cerr << "No snapshots learned" << std::endl;
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

    // OpenCV images used to store raw camera frame and unwrapped panorama
    cv::Mat m_Output;
    cv::Mat m_Unwrapped;

    // OpenCV-based panorama unwrapper
    OpenCVUnwrap360 m_Unwrapper;

    // Perfect memory
    PerfectMemory<1> m_Memory;

    // Motor driver
    MotorI2C m_Motor;
    
    unsigned int m_TrainingSnapshot;

};

int main(int argc, char *argv[])
{
    const bool buildTrainingData = (argc > 1 && strcmp(argv[1], "build") == 0);
    
    //cv::namedWindow("Raw", CV_WINDOW_NORMAL);
    //cv::namedWindow("Unwrapped", CV_WINDOW_NORMAL);

    RobotFSM robot(0, Settings::camRes, Settings::unwrapRes, Settings::encodedSnapshotSize, buildTrainingData);
    
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

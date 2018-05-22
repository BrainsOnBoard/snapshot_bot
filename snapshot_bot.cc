// Standard C++ includes
#include <fstream>
#include <limits>
#include <memory>

// Standard C includes
#include <cassert>

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
#include "perfect_memory.h"

using namespace GeNNRobotics;

enum class State
{
    Invalid,
    Training,
    Testing,
};

//------------------------------------------------------------------------
// RobotFSM
//------------------------------------------------------------------------
class RobotFSM : FSM<State>::StateHandler
{
public:
    RobotFSM(const Config &config)
    :   m_Config(config), m_StateMachine(this, State::Invalid),
        m_Camera("/dev/video" + std::to_string(config.getCamDevice()), config.getSee3CamRes()),
        m_Output(m_Camera.getSuperPixelSize(), CV_8UC1), m_Unwrapped(config.getUnwrapRes(), CV_8UC1),
        m_Unwrapper(m_Camera.createDefaultUnwrapper(config.getUnwrapRes()))
    {
        // Create appropriate type of memory
        if(config.shouldUseHOG()) {
            m_Memory.reset(new PerfectMemoryHOG<1>(config));
        }
        else {
            m_Memory.reset(new PerfectMemoryRaw<1>(config));
        }
        
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
            
            // Start capture
            if(!m_ViconCaptureControl.startRecording(m_Config.getViconCaptureControlName())) {
                throw std::runtime_error("Cannot start capture");
            }
        }
        
        // If we should train
        if(m_Config.shouldTrain()) {
            // If Vicon tracking is available, open log file and write header
            if(m_Config.shouldUseViconTracking()) {
                m_LogFile.open("snapshots.csv");
                m_LogFile << "Snapshot, Frame, X, Y, Z, Rx, Ry, Rz" << std::endl;
            }
             // Delete old snapshots
            system("rm -f snapshot_*.png");

            // Start in training state
            m_StateMachine.transition(State::Training);
        }
        else {
            // If Vicon tracking is available, open log file and write header
            if(m_Config.shouldUseViconTracking()) {
                m_LogFile.open("testing.csv");
                m_LogFile << "Best snapshot, angle difference, image difference, frame number, X, Y, Z, Rx, Ry, Rz" << std::endl;
            }
            // Otherwise, open log file and write simpler header
            else {
                m_LogFile.open("testing.csv");
                m_LogFile << "Best snapshot, angle difference, image difference" << std::endl;
            }
            
            // Load memory
            m_Memory->load();
            
            // Start directly in testing state
            m_StateMachine.transition(State::Testing);
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
                 // If we should use Vicon capture control
                if(m_Config.shouldUseViconCaptureControl()) {
                    // Stop capture
                    if(!m_ViconCaptureControl.stopRecording(m_Config.getViconCaptureControlName())) {
                        throw std::runtime_error("Cannot stop capture");
                    }
                }
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
                    const size_t snapshotID = m_Memory->train(m_Unwrapped);
                    std::cout << "\tTrained snapshot id:" << snapshotID << std::endl;
                    
                    // If Vicon tracking is available
                    if(m_Config.shouldUseViconTracking()) {
                        // Get tracking data
                        auto objectData = m_ViconTracking.getObjectData(0);
                        const auto &translation = objectData.getTranslation();
                        const auto &rotation = objectData.getRotation();

                        // Write to CSV
                        m_LogFile << snapshotID << ", " << objectData.getFrameNumber() << ", " << translation[0] << ", " << translation[1] << ", " << translation[2] << ", " << rotation[0] << ", " << rotation[1] << ", " << rotation[2] << std::endl;
                    }
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
                    std::tie(turnToAngle, turnToSnapshot, minDifferenceSquared) = m_Memory->findSnapshot(m_Unwrapped);

                    // If a snapshot is found and it isn't the one we were previously at
                    if(turnToSnapshot != std::numeric_limits<size_t>::max()) {
                        std::cout << "\tBest match found with snapshot id " << turnToSnapshot << " (angle:" << turnToAngle << ", min difference:" << minDifferenceSquared << ")" << std::endl;

                        // Write basic data to log file
                        m_LogFile << turnToSnapshot << ", " << turnToAngle << ", " << minDifferenceSquared;
                        
                        // If vicon tracking is available
                        if(m_Config.shouldUseViconTracking()) {
                            // Get tracking data
                            auto objectData = m_ViconTracking.getObjectData(0);
                            const auto &translation = objectData.getTranslation();
                            const auto &rotation = objectData.getRotation();
                            
                            // Write extra logging data
                            m_LogFile << ", " << objectData.getFrameNumber() << ", " << translation[0] << ", " << translation[1] << ", " << translation[2] << ", " << rotation[0] << ", " << rotation[1] << ", " << rotation[2];
                        }
                        m_LogFile << std::endl;
                        
                        // Determine how fast we should turn based on the absolute angle
                        auto turnSpeed = m_Config.getTurnSpeed(fabs(turnToAngle));
                        
                        // If we should turn, do so
                        if(turnSpeed > 0.0f) {
                            const float motorTurn = (turnToAngle <  0.0f) ? -turnSpeed : turnSpeed;
                            m_Motor.tank(motorTurn, -motorTurn);
                        }
                        // Otherwise drive forwards
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
    std::unique_ptr<PerfectMemoryBase<1>> m_Memory;

    // Motor driver
    Robots::MotorI2C m_Motor;

    // 'Timer' used to move between snapshot tests
    int m_MoveTime;

    // Vicon tracking interface
    Vicon::UDPClient<Vicon::ObjectData> m_ViconTracking;

    // Vicon capture control interface
    Vicon::CaptureControl m_ViconCaptureControl;
    
    // CSV file containing logging
    std::ofstream m_LogFile;
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
    
    RobotFSM robot(config);
    
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

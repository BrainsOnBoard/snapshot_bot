
// GeNN robotics includes
#include "fsm.h"
#include "joystick.h"
#include "motor_i2c.h"
#include "opencv_unwrap_360.h"
#include "see3cam_cu40.h"
#include "timer.h"

namespace Settings
{
    // What resolution to operate camera at
    const See3CAM_CU40::Resolution camRes = See3CAM_CU40::Resolution::_672x380;

    // What resolution to unwrap panoramas to
    const cv::Size unwrapRes(180, 20);

    // How large should the deadzone be on the analogue joystick
    const float joystickDeadzone = 0.25f;
}

enum class State
{
    Invalid,
    Training,
    ReturningToStart,
    Testing,
};

class RobotFSM : FSM<State>::StateHandler
{
public:
    RobotFSM(unsigned int camDevice, See3CAM_CU40::Resolution camRes, const cv::Size &unwrapRes) 
    :   m_StateMachine(this, State::Invalid), m_Camera("/dev/video" + std::to_string(camDevice), camRes),
        m_Output(m_Camera.getSuperPixelSize(), CV_8UC1), m_Unwrapped(unwrapRes, CV_8UC1),
        m_Unwrapper(See3CAM_CU40::createUnwrapper(m_Camera.getSuperPixelSize(), unwrapRes))
    {
        // Start in training state
        m_StateMachine.transition(State::Training);
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
                if(m_Joystick.isButtonPressed(0)) {
                    std::cout << "\tTrain snapshot!" << std::endl;
                }
                else if(m_Joystick.isButtonPressed(1)) {
                    m_StateMachine.transition(State::ReturningToStart);
                }
            }
            else if(event == Event::Exit) {
                std::cout << "Ending training" << std::endl;
            }
        }
        else if(state == State::ReturningToStart) {
            if(event == Event::Enter) {
                std::cout << "Starting RTS" << std::endl;
            }
            else if(event == Event::Update) {
                if(m_Joystick.isButtonPressed(1)) {
                    m_StateMachine.transition(State::Testing);
                }
            }
            else if(event == Event::Exit) {
                std::cout << "Ending RTS" << std::endl;
            }
        }
        else if(state == State::Testing) {
            if(event == Event::Enter) {
                std::cout << "Starting testing" << std::endl;
            }
            else if(event == Event::Update) {
            while(false);
            }
            else if(event == Event::Exit) {
                std::cout << "Ending testing" << std::endl;
            }
        }
        else {
            throw std::runtime_error("Invalid state");
        }
        return true;
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    FSM<State> m_StateMachine;
    See3CAM_CU40 m_Camera;
    Joystick m_Joystick;
    cv::Mat m_Output;
    cv::Mat m_Unwrapped;
    OpenCVUnwrap360 m_Unwrapper;
};

int main()
{
    cv::namedWindow("Raw", CV_WINDOW_NORMAL);
    cv::namedWindow("Unwrapped", CV_WINDOW_NORMAL);

    RobotFSM robot(0, Settings::camRes, Settings::unwrapRes);
    
    {
        Timer<> timer("Total time:");

        unsigned int frame = 0;
        for(frame = 0; robot.update(); frame++) {
            cv::imshow("Raw", robot.getOutput());
            cv::imshow("Unwrapped", robot.getUnwrapped());
            
            cv::waitKey(1);
        }
        
        const double msPerFrame = timer.get() / (double)frame;
        std::cout << "FPS:" << 1000.0 / msPerFrame << std::endl;
    }

    return 0;
}
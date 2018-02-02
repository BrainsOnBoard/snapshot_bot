
// GeNN robotics includes
#include "../common/joystick.h"
#include "../common/opencv_unwrap_360.h"
#include "../common/see3cam_cu40.h"

namespace Settings
{
    // What resolution to operate camera at
    constexpr See3CAM_CU40::Resolution camResolution = See3CAM_CU40::Resolution::_672x380;

    // What resolution to unwrap panoramas to
    constexpr cv::Size unwrapResolution(180, 20);

    // How large should the deadzone be on the analogue joystick
    constexpr float joystickDeadzone = 0.25f;
}

int main()
{

    // Create joystick interface
    Joystick joystick;

    // Create motor interface
    MotorI2C motor;

    do {
        // Read joystick
        joystick.read();

        // Use joystick to drive motor
        joystick.drive(motor, joystickDeadzone);

    } while(!joystick.isButtonDown(1));

    return 0;
}
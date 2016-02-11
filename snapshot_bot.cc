// Standard C++ includes
#include <fstream>
#include <limits>
#include <memory>

// Standard C includes
#include <cassert>

// BoB robotics includes
#include "common/fsm.h"
#include "common/timer.h"
#include "hid/joystick.h"
#include "imgproc/opencv_unwrap_360.h"
#include "navigation/perfect_memory.h"
#include "net/server.h"
#include "robots/norbot.h"
#include "third_party/path.h"
#include "vicon/capture_control.h"
#include "video/netsink.h"
#include "vicon/udp.h"
#include "video/panoramic.h"

// Snapshot bot includes
#include "config.h"

using namespace BoBRobotics;
using namespace units::angle;
using namespace units::length;
using namespace units::literals;
using namespace units::math;

//------------------------------------------------------------------------
// Anonymous namespace
//------------------------------------------------------------------------
namespace
{
enum class State
{
    Invalid,
    Training,
    Testing,
};

//------------------------------------------------------------------------
// MemoryBase
//------------------------------------------------------------------------
class MemoryBase
{
public:
    MemoryBase(const Config &config)
    :   m_BestHeading(0_deg), m_LowestDifference(std::numeric_limits<size_t>::max()), m_OutputPath(config.getOutputPath())
    {
    }

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void test(const cv::Mat &snapshot) = 0;
    virtual void train(const cv::Mat &snapshot) = 0;
    virtual void load() = 0;

    virtual void writeCSVHeader(std::ostream &os)
    {
        os << "Best heading [degrees], Lowest difference";
    }

    virtual void writeCSVLine(std::ostream &os)
    {
        os << getBestHeading() << ", " << getLowestDifference();
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    degree_t getBestHeading() const{ return m_BestHeading; }
    float getLowestDifference() const{ return m_LowestDifference; }
    
protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    void setBestHeading(degree_t bestHeading){ m_BestHeading = bestHeading; }
    void setLowestDifference(float lowestDifference){ m_LowestDifference = lowestDifference; }
    const filesystem::path &getOutputPath() const{ return m_OutputPath; }
    
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    degree_t m_BestHeading;
    float m_LowestDifference;
    const filesystem::path &m_OutputPath;
};

//------------------------------------------------------------------------
// PerfectMemory
//------------------------------------------------------------------------
class PerfectMemory : public MemoryBase
{
public:
    PerfectMemory(const Config &config)
    :   MemoryBase(config), m_PM(config.getUnwrapRes()), m_BestSnapshotIndex(std::numeric_limits<size_t>::max())
    {
        // Load mask image
        if(!config.getMaskImageFilename().empty()) {
            getPM().setMaskImage(config.getMaskImageFilename());
        }
    }

    //------------------------------------------------------------------------
    // MemoryBase virtuals
    //------------------------------------------------------------------------
    virtual void test(const cv::Mat &snapshot) override
    {
        // Get heading directly from Perfect Memory
        degree_t bestHeading;
        float lowestDifference;
        std::tie(bestHeading, m_BestSnapshotIndex, lowestDifference, std::ignore) = getPM().getHeading(snapshot);

        // Set best heading and vector length
        setBestHeading(bestHeading);
        setLowestDifference(lowestDifference);
    }

    virtual void train(const cv::Mat &snapshot) override
    {
        getPM().train(snapshot);
        
        // Save snapshot
        cv::imwrite(getSnapshotPath(getPM().getNumSnapshots() - 1).str(), snapshot);
    }
    
    virtual void load() override
    {
        for(size_t i = 0;;i++) {
            const auto filename = getSnapshotPath(i);
            if(filename.exists()) {
                // Load image
                cv::Mat image = cv::imread(filename.str(), cv::IMREAD_GRAYSCALE);
                getPM().train(image);
            }
            else {
                break;
	    }
        }
        std::cout << "Loaded " << getPM().getNumSnapshots() << " snapshots" << std::endl;
    }
    
    virtual void writeCSVHeader(std::ostream &os)
    {
        // Superclass
        MemoryBase::writeCSVHeader(os);

        os << ", Best snapshot index";
    }

    virtual void writeCSVLine(std::ostream &os)
    {
        // Superclass
        MemoryBase::writeCSVLine(os);

        os << ", " << getBestSnapshotIndex();
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getBestSnapshotIndex() const{ return m_BestSnapshotIndex; }
    const cv::Mat &getBestSnapshot() const{ return getPM().getSnapshot(getBestSnapshotIndex()); }
    
protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    Navigation::PerfectMemoryRotater<> &getPM(){ return m_PM; }
    const Navigation::PerfectMemoryRotater<> &getPM() const{ return m_PM; }

    void setBestSnapshotIndex(size_t bestSnapshotIndex){ m_BestSnapshotIndex = bestSnapshotIndex; }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    filesystem::path getSnapshotPath(size_t index) const
    {
        return getOutputPath() / ("snapshot_" + std::to_string(index) + ".png");
    }
    
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    Navigation::PerfectMemoryRotater<> m_PM;
    size_t m_BestSnapshotIndex;
};

//------------------------------------------------------------------------
// PerfectMemoryConstrained
//------------------------------------------------------------------------
class PerfectMemoryConstrained : public PerfectMemory
{
public:
    PerfectMemoryConstrained(const Config &config)
    :   PerfectMemory(config), m_FOV(config.getMaxSnapshotRotateAngle()), m_ImageWidth(config.getUnwrapRes().width)
    {
    }


    virtual void test(const cv::Mat &snapshot) override
    {
        // Get 'matrix' of differences from perfect memory
        const auto &allDifferences = getPM().getImageDifferences(snapshot);

        // Loop through snapshots
        // **NOTE** this currently uses a super-naive approach as more efficient solution is non-trivial because
        // columns that represent the rotations are not necessarily contiguous - there is a dis-continuity in the middle
        float lowestDifference = std::numeric_limits<float>::max();
        setBestSnapshotIndex(std::numeric_limits<size_t>::max());
        setBestHeading(0_deg);
        for(size_t i = 0; i < allDifferences.size(); i++) {
            const auto &snapshotDifferences = allDifferences[i];

            // Loop through acceptable range of columns
            for(int c = 0; c < m_ImageWidth; c++) {
                // If this snapshot is a better match than current best
                if(snapshotDifferences[c] < lowestDifference) {
                    // Convert column into pixel rotation
                    int pixelRotation = c;
                    if(pixelRotation > (m_ImageWidth / 2)) {
                        pixelRotation -= m_ImageWidth;
                    }

                    // Convert this into angle
                    const degree_t heading = turn_t((double)pixelRotation / (double)m_ImageWidth);

                    // If the distance between this angle from grid and route angle is within FOV, update best
                    if(fabs(heading) < m_FOV) {
                        setBestSnapshotIndex(i);
                        setBestHeading(heading);
                        lowestDifference = snapshotDifferences[c];
                    }
                }
            }
        }

        // Check valid snapshot actually exists
        assert(getBestSnapshotIndex() != std::numeric_limits<size_t>::max());

        // Scale difference to match code in ridf_processors.h:57
        setLowestDifference(lowestDifference / 255.0f);
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const degree_t m_FOV;
    const int m_ImageWidth;
};

//------------------------------------------------------------------------
// InfoMax
//------------------------------------------------------------------------
/*template<typename FloatType>
class InfoMax : public MemoryBase
{
    using InfoMaxType = Navigation::InfoMaxRotater<Navigation::InSilicoRotater, FloatType>;
    using InfoMaxWeightMatrixType = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>;

public:
    InfoMax(const cv::Size &imSize)
        : m_InfoMax(createInfoMax(imSize))
    {
        // Load mask image
        if(!config.getMaskImageFilename().empty()) {
            getPM().setMaskImage(config.getMaskImageFilename());
        }
    }

    virtual void test(const cv::Mat &snapshot) override
    {
        // Get heading directly from InfoMax
        degree_t bestHeading;
        float lowestDifference;
        std::tie(bestHeading, lowestDifference, std::ignore) = m_InfoMax.getHeading(snapshot);

        // Set best heading and vector length
        setBestHeading(bestHeading);
        setLowestDifference(lowestDifference);
    }
    
    virtual void train(const cv::Mat &snapshot) override
    {
        getPM().train(snapshot);
    }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    const InfoMaxType &getInfoMax() const{ return m_InfoMax; }

private:
    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    // **TODO** move into BoB robotics
    static void writeWeights(const InfoMaxWeightMatrixType &weights, const filesystem::path &weightPath)
    {
        // Write weights to disk
        std::ofstream netFile(weightPath.str(), std::ios::binary);
        const int size[2] { (int) weights.rows(), (int) weights.cols() };
        netFile.write(reinterpret_cast<const char *>(size), sizeof(size));
        netFile.write(reinterpret_cast<const char *>(weights.data()), weights.size() * sizeof(FloatType));
    }

    // **TODO** move into BoB robotics
    static InfoMaxWeightMatrixType readWeights(const filesystem::path &weightPath)
    {
        // Open file
        std::ifstream is(weightPath.str(), std::ios::binary);
        if (!is.good()) {
            throw std::runtime_error("Could not open " + weightPath.str());
        }

        // The matrix size is encoded as 2 x int32_t
        int32_t size[2];
        is.read(reinterpret_cast<char *>(&size), sizeof(size));

        // Create data array and fill it
        InfoMaxWeightMatrixType data(size[0], size[1]);
        is.read(reinterpret_cast<char *>(data.data()), sizeof(FloatType) * data.size());

        return std::move(data);
    }

    static InfoMaxType createInfoMax(const cv::Size &imSize, const Navigation::ImageDatabase &route)
    {
        // Create path to weights from directory containing route
        const filesystem::path weightPath = filesystem::path(route.getPath()) / "infomax.bin";
        if(weightPath.exists()) {
            std::cout << "Loading weights from " << weightPath << std::endl;
            InfoMaxType infomax(imSize, readWeights(weightPath));
            return std::move(infomax);
        }
        else {
            InfoMaxType infomax(imSize);
            infomax.trainRoute(route, true);
            writeWeights(infomax.getWeights(), weightPath.str());
            std::cout << "Trained on " << route.size() << " snapshots" << std::endl;
            return std::move(infomax);
        }
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    InfoMaxType m_InfoMax;
};

//------------------------------------------------------------------------
// InfoMaxConstrained
//------------------------------------------------------------------------
template<typename FloatType>
class InfoMaxConstrained : public InfoMax<FloatType>
{
public:
    InfoMaxConstrained(const cv::Size &imSize, degree_t fov)
        : InfoMax<FloatType>(imSize), m_FOV(fov), m_ImageWidth(imSize.width)
    {
    }

    virtual void test(const cv::Mat &snapshot) override
    {
        // Get vector of differences from InfoMax
        const auto &allDifferences = this->getInfoMax().getImageDifferences(snapshot);

        // Loop through snapshots
        // **NOTE** this currently uses a super-naive approach as more efficient solution is non-trivial because
        // columns that represent the rotations are not necessarily contiguous - there is a dis-continuity in the middle
        this->setLowestDifference(std::numeric_limits<float>::max());
        this->setBestHeading(0_deg);
        for(size_t i = 0; i < allDifferences.size(); i++) {
            // If this snapshot is a better match than current best
            if(allDifferences[i] < this->getLowestDifference()) {
                // Convert column into pixel rotation
                int pixelRotation = i;
                if(pixelRotation > (m_ImageWidth / 2)) {
                    pixelRotation -= m_ImageWidth;
                }

                // Convert this into angle
                const degree_t heading = turn_t((double)pixelRotation / (double)m_ImageWidth);

                // If the distance between this angle from grid and route angle is within FOV, update best
                if(fabs(heading) < m_FOV) {
                    this->setBestHeading(heading);
                    this->setLowestDifference(allDifferences[i]);
                }
            }
        }

        // **TODO** calculate vector length
        this->setVectorLength(1.0f);
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const degree_t m_FOV;
    const int m_ImageWidth;
};
*/

//------------------------------------------------------------------------
// RobotFSM
//------------------------------------------------------------------------
class RobotFSM : FSM<State>::StateHandler
{
public:
    RobotFSM(const Config &config)
    :   m_Config(config), m_StateMachine(this, State::Invalid), m_Camera(Video::getPanoramicCamera()),
        m_Output(m_Camera->getOutputSize(), CV_8UC1), m_Unwrapped(config.getUnwrapRes(), CV_8UC1),
        m_DifferenceImage(config.getUnwrapRes(), CV_8UC1),m_Unwrapper(m_Camera->createUnwrapper(config.getUnwrapRes()))/*,
        m_Server(config.getServerListenPort()), m_NetSink(m_Server, config.getUnwrapRes(), "unwrapped")*/
    {
        // Create output directory (if necessary)
        filesystem::create_directory(m_Config.getOutputPath());
    
        // Create appropriate type of memory
        if(m_Config.getMaxSnapshotRotateAngle() < 180_deg) {
            m_Memory.reset(new PerfectMemory(m_Config));
        }
        else {
            m_Memory.reset(new PerfectMemoryConstrained(m_Config));
        }


        // If we should stream output, run server thread
        /*if(m_Config.shouldStreamOutput()) {
            m_Server.runInBackground();
        }*/
       
        // If we should use Vicon tracking
        if(m_Config.shouldUseViconTracking()) {
            // Connect to port specified in config
            m_ViconTracking.connect(m_Config.getViconTrackingPort());
            
            // Wait for tracking data stream to begin
            while(true) {
                try {
                    m_ViconTrackingObjectID = m_ViconTracking.findObjectID(m_Config.getViconTrackingObjectName());
                    break;
                }
                catch(std::out_of_range &ex) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    std::cout << "Waiting for object '" << m_Config.getViconTrackingObjectName() << "'" << std::endl;
                }
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
            // Start in training state
            m_StateMachine.transition(State::Training);
        }
        else {
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
            m_Joystick.update();
            
            // Exit if X is pressed
            if(m_Joystick.isPressed(HID::JButton::X)) {
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
            if(!m_Camera->readGreyscaleFrame(m_Output)) {
                return false;
            }
            
            // Unwrap frame
            m_Unwrapper.unwrap(m_Output, m_Unwrapped);
        }

        if(state == State::Training) {
            if(event == Event::Enter) {
                std::cout << "Starting training" << std::endl;

                // Open settings file and write unwrapper settings to it
                cv::FileStorage settingsFile((m_Config.getOutputPath() / "training_settings.yaml").str().c_str(), cv::FileStorage::WRITE);
                settingsFile << "unwrapper" << m_Unwrapper;
                
                // Close log file if it's already open
                if(m_LogFile.is_open()) {
                    m_LogFile.close();
                }

                // If Vicon tracking is available, open log file and write header
                if(m_Config.shouldUseViconTracking()) {
                    m_LogFile.open((m_Config.getOutputPath() / "snapshots.csv").str());
                    m_LogFile << "Frame, X, Y, Z, Rx, Ry, Rz" << std::endl;
                }

                // Delete old snapshots
                const std::string snapshotWildcard = (m_Config.getOutputPath() / "snapshot_*.png").str();
                system(("rm -f " + snapshotWildcard).c_str());
            }
            else if(event == Event::Update) {
                // While testing, if we should stream output, send unwrapped frame
                /*if(m_Config.shouldStreamOutput()) {
                    m_NetSink.sendFrame(m_Unwrapped);
                }*/

                // Drive motors using joystick
                m_Motor.drive(m_Joystick, m_Config.getJoystickDeadzone());
                
                // If A is pressed, save snapshot
                if(m_Joystick.isPressed(HID::JButton::A)) {
                    m_Memory->train(m_Unwrapped);
                    std::cout << "\tTrained snapshot" << std::endl;
                    
                    // If Vicon tracking is available
                    if(m_Config.shouldUseViconTracking()) {
                        // Get tracking data
                        auto objectData = m_ViconTracking.getObjectData(0);
                        const auto &position = objectData.getPosition<units::length::millimeter_t>();
                        const auto &attitude = objectData.getAttitude<units::angle::degree_t>();

                        // Write to CSV
                        m_LogFile << objectData.getFrameNumber() << ", " << position[0] << ", " << position[1] << ", " << position[2] << ", " << attitude[0] << ", " << attitude[1] << ", " << attitude[2] << std::endl;
                    }
                }
                // Otherwise, if B is pressed, go to testing
                else if(m_Joystick.isPressed(HID::JButton::B)) {
                    m_StateMachine.transition(State::Testing);
                }
            }
        }
        else if(state == State::Testing) {
            if(event == Event::Enter) {
                std::cout << "Testing: finding snapshot" << std::endl;

                // Open settings file and write unwrapper settings to it
                cv::FileStorage settingsFile((m_Config.getOutputPath() / "testing_settings.yaml").str().c_str(), cv::FileStorage::WRITE);
                settingsFile << "unwrapper" << m_Unwrapper;
                
                // Close log file if it's already open
                if(m_LogFile.is_open()) {
                    m_LogFile.close();
                }

                // If we should save diagnostics when testing
                if(m_Config.shouldSaveTestingDiagnostic()) {
                    // Open log file
                    m_LogFile.open((m_Config.getOutputPath() / "testing.csv").str());

                    // Write memory-specific CSV header
                    m_Memory->writeCSVHeader(m_LogFile);
                    
                    // If Vicon tracking is available, write additional header fields
                    if(m_Config.shouldUseViconTracking()) {
                        m_LogFile << ", Frame number, X, Y, Z, Rx, Ry, Rz";
                    }
                    m_LogFile << std::endl;
                }

                // Reset move time and test image
                m_MoveTime = 0;
                m_TestImageIndex = 0;

                // Delete old testing images
                const std::string testWildcard = (m_Config.getOutputPath() / "test_*.png").str();
                system(("rm -f " + testWildcard).c_str());
            }
            else if(event == Event::Update) {
                // If it's time to move
                if(m_MoveTime == 0) {
                    // Reset move time
                    m_MoveTime = m_Config.getMoveTimesteps();

                    // Find matching snapshot
                    m_Memory->test(m_Unwrapped);

                    // If we should save diagnostics when testing
                    if(m_Config.shouldSaveTestingDiagnostic()) {
                        // Write memory-specific CSV logging
                        m_Memory->writeCSVLine(m_LogFile);
                        
                        // If vicon tracking is available
                        if(m_Config.shouldUseViconTracking()) {
                            // Get tracking data
                            auto objectData = m_ViconTracking.getObjectData(0);
                            const auto &position = objectData.getPosition<units::length::millimeter_t>();
                            const auto &attitude = objectData.getAttitude<units::angle::degree_t>();

                            // Write extra logging data
                            m_LogFile << ", " << objectData.getFrameNumber() << ", " << position[0] << ", " << position[1] << ", " << position[2] << ", " << attitude[0] << ", " << attitude[1] << ", " << attitude[2];
                        }
                        m_LogFile << std::endl;

                        // Build path to test image and save
                        const auto testImagePath = m_Config.getOutputPath() / ("test_" + std::to_string(m_TestImageIndex++) + ".png");
                        cv::imwrite(testImagePath.str(), m_Unwrapped);
                    }

                    // If we should stream output
                    if(m_Config.shouldStreamOutput()) {
                        // Attempt to dynamic cast memory to a perfect memory
                        PerfectMemory *perfectMemory = dynamic_cast<PerfectMemory*>(m_Memory.get());
                        if(perfectMemory != nullptr) {
                            // Get matched snapshot
                            const cv::Mat &matchedSnapshot = perfectMemory->getBestSnapshot();

                            // Calculate difference image
                            cv::absdiff(matchedSnapshot, m_Unwrapped, m_DifferenceImage);

                            char status[255];
                            sprintf(status, "Angle:%f deg, Min difference:%f", degree_t(perfectMemory->getBestHeading()).value(), perfectMemory->getLowestDifference());
                            cv::putText(m_DifferenceImage, status, cv::Point(0, m_Config.getUnwrapRes().height -20),
                                        cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, 0xFF);

                            // Send annotated difference image
                            //m_NetSink.sendFrame(m_DifferenceImage);
                        }
                        else {
                            std::cout << "WARNING: Can only stream output from a perfect memory" << std::endl;
                        }
                    }

                    // Determine how fast we should turn based on the absolute angle
                    auto turnSpeed = m_Config.getTurnSpeed(m_Memory->getBestHeading());
                    
                    // If we should turn, do so
                    if(turnSpeed > 0.0f) {
                        const float motorTurn = (m_Memory->getBestHeading() <  0.0_deg) ? -turnSpeed : turnSpeed;
                        m_Motor.tank(motorTurn, -motorTurn);
                    }
                    // Otherwise drive forwards
                    else {
                        m_Motor.tank(1.0f, 1.0f);
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
    std::unique_ptr<Video::Input> m_Camera;

    // Joystick interface
    HID::Joystick m_Joystick;

    // OpenCV images used to store raw camera frame and unwrapped panorama
    cv::Mat m_Output;
    cv::Mat m_Unwrapped;
    cv::Mat m_DifferenceImage;

    // OpenCV-based panorama unwrapper
    ImgProc::OpenCVUnwrap360 m_Unwrapper;

    // Perfect memory
    std::unique_ptr<MemoryBase> m_Memory;

    // Motor driver
    Robots::Norbot m_Motor;

    // 'Timer' used to move between snapshot tests
    int m_MoveTime;

    // Index of test image to write
    size_t m_TestImageIndex;

    // Vicon tracking interface
    Vicon::UDPClient<Vicon::ObjectData> m_ViconTracking;
    unsigned int m_ViconTrackingObjectID;

    // Vicon capture control interface
    Vicon::CaptureControl m_ViconCaptureControl;
    
    // CSV file containing logging
    std::ofstream m_LogFile;
    
    // Server for streaming etc
    //Net::Server m_Server;
    
    // Sink for video to send over server
    //Video::NetSink m_NetSink;
};
}   // Anonymous namespace

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

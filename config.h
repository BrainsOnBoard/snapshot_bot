#pragma once

// Standard C++ includes
#include <map>
#include <string>

// BoB robotics includes
#include "net/connection.h"
#include "third_party/path.h"
#include "third_party/units.h"

//------------------------------------------------------------------------
// Config
//------------------------------------------------------------------------
class Config
{
public:
    Config() : m_UseHOG(false), m_Train(true), m_SaveTestingDiagnostic(false), m_StreamOutput(false),
        m_MaxSnapshotRotateDegrees(180.0), m_UnwrapRes(180, 50), m_MaskImageFilename("mask.png"), m_WatershedMarkerImageFilename("segmentation.png"), m_NumHOGOrientations(8), m_NumHOGPixelsPerCell(10),
        m_JoystickDeadzone(0.25f), m_MoveTimesteps(10), m_ServerListenPort(BoBRobotics::Net::Connection::DefaultListenPort),
        m_TurnThresholds{{units::angle::degree_t(5.0), 0.5f}, {units::angle::degree_t(10.0), 1.0f}}, m_UseViconTracking(false), m_ViconTrackingPort(0), m_ViconTrackingObjectName("norbot"),
        m_UseViconCaptureControl(false), m_ViconCaptureControlPort(0)
    {
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool shouldUseHOG() const{ return m_UseHOG; }
    bool shouldTrain() const{ return m_Train; }
    bool shouldSaveTestingDiagnostic() const{ return m_SaveTestingDiagnostic; }
    bool shouldStreamOutput() const{ return m_StreamOutput; }

    units::angle::degree_t getMaxSnapshotRotateAngle() const{ return units::angle::degree_t(m_MaxSnapshotRotateDegrees); }
    
    const filesystem::path &getOutputPath() const{ return m_OutputPath; }

    const cv::Size &getUnwrapRes() const{ return m_UnwrapRes; }

    const std::string &getMaskImageFilename() const{ return m_MaskImageFilename; }
    const std::string &getWatershedMarkerImageFilename() const{ return m_WatershedMarkerImageFilename; }
    
    int getNumHOGOrientations() const{ return m_NumHOGOrientations; }
    int getNumHOGPixelsPerCell() const{ return m_NumHOGPixelsPerCell; }
    int getHOGDescriptorSize() const{ return (getUnwrapRes().width * getUnwrapRes().height * getNumHOGOrientations()) / (getNumHOGPixelsPerCell() * getNumHOGPixelsPerCell()); }
    
    float getJoystickDeadzone() const{ return m_JoystickDeadzone; }

    int getMoveTimesteps() const{ return m_MoveTimesteps; }
    
    bool shouldUseViconTracking() const{ return m_UseViconTracking; }
    int getViconTrackingPort() const{ return m_ViconTrackingPort; }
    const std::string &getViconTrackingObjectName() const{ return m_ViconTrackingObjectName; }
    
    bool shouldUseViconCaptureControl() const{ return m_UseViconCaptureControl; }
    const std::string &getViconCaptureControlName() const{ return m_ViconCaptureControlName; }
    const std::string &getViconCaptureControlHost() const{ return m_ViconCaptureControlHost; }
    int getViconCaptureControlPort() const { return m_ViconCaptureControlPort; }
    const std::string &getViconCaptureControlPath() const{ return m_ViconCaptureControlPath; }
    
    int getServerListenPort() const{ return m_ServerListenPort; }
    
    float getTurnSpeed(units::angle::degree_t angleDifference) const
    {
        const auto absoluteAngleDifference = units::math::fabs(angleDifference);
        
        // Loop through turn speed thresholds in descending order
        for(auto i = m_TurnThresholds.crbegin(); i != m_TurnThresholds.crend(); ++i) {
            // If the angle difference passes this threshold, return corresponding speed
            if(absoluteAngleDifference >= i->first) {
                return i->second;
            }
        }
        
        // No turning required!
        return 0.0f;
    }


    void write(cv::FileStorage& fs) const
    {
        fs << "{";
        fs << "shouldUseHOG" << shouldUseHOG();
        fs << "shouldTrain" << shouldTrain();
        fs << "shouldSaveTestingDiagnostic" << shouldSaveTestingDiagnostic();
        fs << "shouldStreamOutput" << shouldStreamOutput();
        fs << "outputPath" << getOutputPath().str();
        fs << "maxSnapshotRotateDegrees" << getMaxSnapshotRotateAngle().value();
        fs << "unwrapRes" << getUnwrapRes();
        fs << "maskImageFilename" << getMaskImageFilename();
        fs << "watershedMarkerImageFilename" << getWatershedMarkerImageFilename();
        fs << "numHOGOrientations" << getNumHOGOrientations();
        fs << "numHOGPixelsPerCell" << getNumHOGPixelsPerCell();
        fs << "joystickDeadzone" << getJoystickDeadzone();
        fs << "moveTimesteps" << getMoveTimesteps();
        fs << "serverListenPort" << getServerListenPort();
        fs << "turnThresholds" << "[";
        for(const auto &t : m_TurnThresholds) {
            fs << "[" << t.first.value() << t.second << "]";
        }
        fs << "]";
        
        if(shouldUseViconTracking()) {
            fs << "viconTracking" << "{";
            fs << "port" << getViconTrackingPort();
            fs << "objectName" << getViconTrackingObjectName();
            fs << "}";
        }
        
        if(shouldUseViconCaptureControl()) {
            fs << "viconCaptureControl" << "{";
            fs << "name" << getViconCaptureControlName();
            fs << "host" << getViconCaptureControlHost();
            fs << "port" << getViconCaptureControlPort();
            fs << "path" << getViconCaptureControlPath();
            fs << "}";
        }
        fs << "}";
    }

    void read(const cv::FileNode &node)
    {
        // Read settings
        // **NOTE** we use cv::read rather than stream operators as we want to use current values as defaults
        cv::read(node["shouldUseHOG"], m_UseHOG, m_UseHOG);
        cv::read(node["shouldTrain"], m_Train, m_Train);
        cv::read(node["shouldSaveTestingDiagnostic"], m_SaveTestingDiagnostic, m_SaveTestingDiagnostic);
        cv::read(node["shouldStreamOutput"], m_StreamOutput, m_StreamOutput);
        
        // **YUCK** why does OpenCV (at least my version) not have a cv::read overload for std::string!?
        cv::String outputPath;
        cv::read(node["outputPath"], outputPath, m_OutputPath.str());
        m_OutputPath = (std::string)outputPath;

        cv::read(node["maxSnapshotRotateDegrees"], m_MaxSnapshotRotateDegrees, m_MaxSnapshotRotateDegrees);
        cv::read(node["unwrapRes"], m_UnwrapRes, m_UnwrapRes);

        cv::String maskImageFilename;
        cv::read(node["maskImageFilename"], maskImageFilename, m_MaskImageFilename);
        m_MaskImageFilename = (std::string)maskImageFilename;
        
        cv::String watershedMarkerImageFilename;
        cv::read(node["watershedMarkerImageFilename"], watershedMarkerImageFilename, m_WatershedMarkerImageFilename);
        m_WatershedMarkerImageFilename = (std::string)watershedMarkerImageFilename;
        
        cv::read(node["numHOGOrientations"], m_NumHOGOrientations, m_NumHOGOrientations);
        cv::read(node["numHOGPixelsPerCell"], m_NumHOGPixelsPerCell, m_NumHOGPixelsPerCell);
        cv::read(node["joystickDeadzone"], m_JoystickDeadzone, m_JoystickDeadzone);
        cv::read(node["moveTimesteps"], m_MoveTimesteps, m_MoveTimesteps);
        cv::read(node["serverListenPort"], m_ServerListenPort, m_ServerListenPort);
        
        if(node["turnThresholds"].isSeq()) {
            m_TurnThresholds.clear();
            for(const auto &t : node["turnThresholds"]) {
                assert(t.isSeq() && t.size() == 2);
                m_TurnThresholds.emplace(units::angle::degree_t((double)t[0]), (float)t[1]);
            }
        }
        
        const auto &viconTracking = node["viconTracking"];
        if(viconTracking.isMap()) {
            m_UseViconTracking = true;
            viconTracking["port"] >> m_ViconTrackingPort;
            
            cv::String viconTrackingObjectName;
            viconTracking["objectName"] >> viconTrackingObjectName;
            m_ViconTrackingObjectName = (std::string)viconTrackingObjectName;
        }
        
        const auto &viconCaptureControl = node["viconCaptureControl"];
        if(viconCaptureControl.isMap()) {
            m_UseViconCaptureControl = true;
            
            cv::String viconCaptureControlName;
            cv::String viconCaptureControlHost;
            cv::String viconCaptureControlPath;
            viconCaptureControl["name"] >> viconCaptureControlName;
            viconCaptureControl["host"] >> viconCaptureControlHost;
            viconCaptureControl["port"] >> m_ViconCaptureControlPort;
            viconCaptureControl["path"] >> viconCaptureControlPath;
            
            m_ViconCaptureControlName = (std::string)viconCaptureControlName;
            m_ViconCaptureControlHost = (std::string)viconCaptureControlHost;
            m_ViconCaptureControlPath = (std::string)viconCaptureControlPath;
        }
    
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    // Should we use HOG features or raw images?
    bool m_UseHOG;
    
    // Should we start in training mode or use existing data?
    bool m_Train;

    // Should we write out testing diagnostic information
    bool m_SaveTestingDiagnostic;
    
    // Should we transmit visual output
    bool m_StreamOutput;
    
    // Path to store snapshots etc
    filesystem::path m_OutputPath;

    // Maximum (absolute) angle snapshots will be rotated by
    double m_MaxSnapshotRotateDegrees;
    
    // What resolution to unwrap panoramas to?
    cv::Size m_UnwrapRes;

    // Filename of mask used to crop out unwanted bits of robot
    std::string m_MaskImageFilename;

    // Filename of image used to provide markers to watershed segmentation algorithm;
    std::string m_WatershedMarkerImageFilename;
    
    // HOG configuration
    int m_NumHOGOrientations;
    int m_NumHOGPixelsPerCell;

    // How large should the deadzone be on the analogue joystick?
    float m_JoystickDeadzone;
    
    // How many timesteps do we move for before re-calculating IDF?
    int m_MoveTimesteps;
    
    // Listen port used for streaming etc
    int m_ServerListenPort;
    
    // RDF angle difference thresholds that trigger different turning speeds
    std::map<units::angle::degree_t, float> m_TurnThresholds;
    
    // Vicon tracking settings
    bool m_UseViconTracking;
    int m_ViconTrackingPort;
    std::string m_ViconTrackingObjectName;
    
    // Vicon capture control settings
    bool m_UseViconCaptureControl;
    std::string m_ViconCaptureControlName;
    std::string m_ViconCaptureControlHost;
    int m_ViconCaptureControlPort;
    std::string m_ViconCaptureControlPath;
};

static void write(cv::FileStorage &fs, const std::string&, const Config &config)
{
    config.write(fs);
}

static void read(const cv::FileNode &node, Config &x, const Config& defaultValue = Config())
{
    if(node.empty()) {
        x = defaultValue;
    }
    else {
        x.read(node);
    }
}

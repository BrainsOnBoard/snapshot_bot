// Standard C++ includes
#include <iostream>

// BoB robotics includes
#include "common/timer.h"

// BoB robotics third-party includes
#include "third_party/csv.h"

// Snapshot bot includes
#include "config.h"
#include "image_input.h"
#include "memory.h"

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

    // Create image input
    std::unique_ptr<ImageInput> imageInput = createImageInput(config);

    // Create memory
    std::unique_ptr<MemoryBase> memory = createMemory(config, imageInput->getOutputSize());
    PerfectMemory *perfectMemory = dynamic_cast<PerfectMemory*>(memory.get());
    
    // Create reader to efficiently read the three columns we care about
    // 1) Load config
    // 2) Create memory
    // 3) Load snapshots listed in training.csv and train
    // 4) Test images listed in testing csv and test
    {
        BoBRobotics::Timer<> t("Training:");
        io::CSVReader<1> trainingCSV("../snapshot_bot_reconstruction/outdoor/training.csv");
        
        // Read header, ignoring time
        trainingCSV.read_header(io::ignore_extra_column, "Filename");
        
        std::cout << "Training" << std::endl;
        std::string filename;
        while(trainingCSV.read_row(filename)){
            std::cout << "." << std::flush;
            memory->train(imageInput->processSnapshot(cv::imread("../snapshot_bot_reconstruction/" + filename)));
        }
        std::cout << std::endl;
    }
    
    {
        BoBRobotics::Timer<> t("Testing:");
        io::CSVReader<3> testingCSV("../snapshot_bot_reconstruction/outdoor/testing" + config.getTestingSuffix() + ".csv");
        
        testingCSV.read_header(io::ignore_extra_column, "Best heading [degrees]", "Best snapshot index", "Filename");
        
        std::string bestHeadingDegreesStr;
        size_t bestSnapshotIndex;
        std::string filename;
        while(testingCSV.read_row(bestHeadingDegreesStr, bestSnapshotIndex, filename)) {
            std::cout << "." << std::flush;
            memory->test(imageInput->processSnapshot(cv::imread("../snapshot_bot_reconstruction/" + config.getOutputPath().str() + "/" + filename)));
            
            const double bestHeadingDegrees = std::stod(bestHeadingDegreesStr.substr(0, bestHeadingDegreesStr.size() - 4));
            
            if(fabs(bestHeadingDegrees - perfectMemory->getBestHeading().value()) > 0.01) {
                std::cerr << "BEST HEADING ERROR:" << bestHeadingDegrees << " vs " << perfectMemory->getBestHeading().value() << std::endl;
                break;
            }
            
            if(perfectMemory && perfectMemory->getBestSnapshotIndex() != bestSnapshotIndex) {
                std::cerr << "BEST SNAPSHOT ERROR" << std::endl;
                break;
            }
        }
       
    }
    return EXIT_SUCCESS;
}

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
    
    std::unique_ptr<ImageInput> imageInput = createImageInput(config);
    
    // Perfect memory
    std::unique_ptr<MemoryBase> memory = createMemory(config, imageInput->getOutputSize());
    
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
    return EXIT_SUCCESS;
}

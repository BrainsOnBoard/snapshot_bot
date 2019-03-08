// Standard C++ includes
#include <iostream>

// BoB robotics includes
#include "common/timer.h"

// BoB robotics third-party includes
#include "third_party/csv.h"
#include "third_party/path.h"

// Snapshot bot includes
#include "config.h"
#include "image_input.h"
#include "memory.h"

int main(int argc, char *argv[])
{
    const char *configFilename = (argc > 1) ? argv[1] : "config.yaml";
    
    filesystem::path dataPath;
    
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

    std::cout << "Training" << std::endl;
    size_t numTrainingImages = 0;
    double totalTrainTime = 0.0;
    for(size_t r = 0; r < 1000; r++) {
        for(size_t i = 0; i < 10; i++) {
            char filename[255];
            sprintf(filename, "benchmark_10/snapshot_%u.png", i);
            const cv::Mat &image = cv::imread(filename);
            
            BoBRobotics::TimerAccumulate<> t(totalTrainTime);
            std::cout << "." << std::flush;
            memory->train(imageInput->processSnapshot(image));
            numTrainingImages++;
        }
    }
    std::cout << std::endl;
    std::cout << "\tTotal time:" << totalTrainTime << "ms" << std::endl;
    std::cout << "\tTime per image:" << totalTrainTime / (double)numTrainingImages << "ms" << std::endl;
    
    std::cout << "Testing" << std::endl;
    const cv::Mat &image = cv::imread("benchmark_10/test_0.png");
    size_t numTestingImages = 0;
    double totalTestTime = 0.0;
    for(size_t i = 0; i < 10; i++) {
        BoBRobotics::TimerAccumulate<> t(totalTestTime);
        
        std::cout << "." << std::flush;
        memory->test(imageInput->processSnapshot(image));
        numTestingImages++;
    }
    std::cout << std::endl;
    std::cout << "\tTotal time:" << totalTestTime << "ms" << std::endl;
    std::cout << "\tTime per image:" << totalTestTime / (double)numTestingImages << "ms" << std::endl;
    
    return EXIT_SUCCESS;
}

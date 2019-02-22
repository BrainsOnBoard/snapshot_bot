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
    const bool validate = (argc > 2) ? (strcmp(argv[2], "1") == 0) : true;
    
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
    PerfectMemory *perfectMemory = dynamic_cast<PerfectMemory*>(memory.get());
    
    // If we're not using InfoMax or pre-trained weights don't exist
    if(!config.shouldUseInfoMax() || !(dataPath  / config.getOutputPath() / "weights.bin").exists()) {
        // Create reader to read filenames from training data
        io::CSVReader<1> trainingCSV((dataPath  / config.getOutputPath() / "training.csv").str());
        trainingCSV.read_header(io::ignore_extra_column, "Filename");
        
        std::cout << "Training" << std::endl;
        std::string filename;
        double totalTrainTime = 0.0;
        size_t numTrainingImages = 0;
        while(trainingCSV.read_row(filename)){
            BoBRobotics::TimerAccumulate<> t(totalTrainTime);
            std::cout << "." << std::flush;
            memory->train(imageInput->processSnapshot(cv::imread((dataPath / filename).str())));
            numTrainingImages++;
        }
        std::cout << std::endl;
        
        std::cout << "\tTotal time:" << totalTrainTime << "ms" << std::endl;
        std::cout << "\tTime per image:" << totalTrainTime / (double)numTrainingImages << "ms" << std::endl;
    }

    // Create reader to read headings and best matching snapshot from testing data
    io::CSVReader<4> testingCSV((dataPath  / config.getOutputPath() / ("testing" + config.getTestingSuffix() + ".csv")).str());
    testingCSV.read_header(io::ignore_extra_column, "Best heading [degrees]", "Lowest difference", "Best snapshot index", "Filename");
    
    std::cout << "Testing" << std::endl;
    std::string bestHeadingDegreesStr;
    float lowestDifference;
    size_t bestSnapshotIndex;
    std::string filename;
    double totalTestTime = 0.0;
    size_t numTestingImages = 0;
    while(testingCSV.read_row(bestHeadingDegreesStr, lowestDifference, bestSnapshotIndex, filename)) {
        BoBRobotics::TimerAccumulate<> t(totalTestTime);
        
        std::cout << "." << std::flush;
        memory->test(imageInput->processSnapshot(cv::imread((dataPath / config.getOutputPath() / filename).str())));
        numTestingImages++;

        // If we're validating and the lowest differences don't match
        if(validate && fabs(memory->getLowestDifference() - lowestDifference) > 0.01) {
            const double bestHeadingDegrees = std::stod(bestHeadingDegreesStr.substr(0, bestHeadingDegreesStr.size() - 4));
            
            // If headings don't match
            if(fabs(bestHeadingDegrees - memory->getBestHeading().value()) > 0.01) {
                std::cerr << "BEST HEADING ERROR (" << bestHeadingDegrees << " rather than " << memory->getBestHeading().value() << ")" << std::endl;
               break;
            }
            
            // If snapshot doesn't mathc
            if(perfectMemory && perfectMemory->getBestSnapshotIndex() != bestSnapshotIndex) {
                std::cerr << "BEST SNAPSHOT ERROR (" << bestSnapshotIndex << " rather than " << perfectMemory->getBestSnapshotIndex() << ")" << std::endl;
                break;
            }
        }
    }
    std::cout << std::endl;
    std::cout << "\tTotal time:" << totalTestTime << "ms" << std::endl;
    std::cout << "\tTime per image:" << totalTestTime / (double)numTestingImages << "ms" << std::endl;
    
    return EXIT_SUCCESS;
}

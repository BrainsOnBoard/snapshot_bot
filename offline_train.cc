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
    
    std::unique_ptr<ImageInput> imageInput{new ImageInputRaw(config)};
    InfoMax infomax(config, imageInput->getOutputSize());
    
    std::cout << "Training" << std::endl;
    for(size_t i = 0;;i++) {
        const filesystem::path filename = config.getOutputPath() / ("snapshot_" + std::to_string(i) + ".png");
        
        // If file exists, load image and train memory on it
        if(filename.exists()) {
            std::cout << "." << std::flush;
            infomax.train(imageInput->processSnapshot(cv::imread(filename.str())));
        }
        // Otherwise, stop searching
        else {
            break;
        }
    }
    
    infomax.saveWeights((config.getOutputPath() / "weights.bin").str());
    return EXIT_SUCCESS;
}

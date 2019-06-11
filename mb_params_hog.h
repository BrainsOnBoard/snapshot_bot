#pragma once

//------------------------------------------------------------------------
// MBParams
//------------------------------------------------------------------------
namespace MBParamsHOG
{
    // Should we measure timing
    constexpr bool timing = true;
    
    constexpr double timestepMs = 1.0;

    // 90x20 = 1800 = 900 ommatidia in each eye
    constexpr unsigned int inputWidth = 90;
    constexpr unsigned int inputHeight = 20;

    // Receptive field configuration
    constexpr int rfWidth = 6;
    constexpr int rfHeight = 20;
    constexpr int rfStrideX = 4;
    constexpr int rfStrideY = 20;
    constexpr int numRFX = 22;
    constexpr int numRFY = 1;

    // Orientation configuration
    constexpr int numOrientations = 3;

    // Calculate number of features per RF (orientations + area)
    constexpr int numFeatures = numOrientations + 1;

    // Calculate total size of feature vector
    constexpr int featureSize = numFeatures * numRFX * numRFY;

    // Network dimensions
    constexpr unsigned int numPN = (unsigned int)featureSize;
    constexpr unsigned int numKC = 20000;
    constexpr unsigned int numEN = 1;

    // Regime parameters
    constexpr double rewardTimeMs = 20.0;
    constexpr double presentDurationMs = 20.0;
    constexpr double postStimuliDurationMs = 40.0;

    // Scale applied to convert image data to input currents for PNs
    constexpr double inputCurrentScale = 1.0;

    // Weight of static synapses between PN and KC populations
    // **NOTE** manually tuend to get approximately 200/20000 KC firing sparsity
    constexpr double pnToKCWeight = 0.34;//0.243;

    // Initial/maximum weight of plastic synapses between KC and EN populations
    // **NOTE** note manually tuned to get 15-20 spikes for a novel image
    constexpr double kcToENWeight = 1.0;

    // **NOTE** manually tuned to result in a maximum depolarization of 20mv (peak membrane voltage of -40mv)
    constexpr double kcToGGNWeight = 0.015;

    // **NOTE** manually tuned to normalize number of active KCs
    constexpr double ggnToKCWeight = -4.0;

    // Time constant of dopamine
    constexpr double tauD = 20.0;

    constexpr double ggnToKCVMid = -40.0;

    constexpr double ggnToKCVslope = 2.0;

    constexpr double ggnToKCVthresh = -50.0;

    constexpr double pnVthresh = -50.0;

    // Scale of each dopamine 'spike'
    // **NOTE** manually tuned for one-shot learning - also close to BA/phi
    constexpr double dopamineStrength = 0.03;

    // How many PN neurons are connected to each KC
    constexpr unsigned int numPNSynapsesPerKC = 10;



}

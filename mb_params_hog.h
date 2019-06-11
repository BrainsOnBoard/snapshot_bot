#pragma once

//------------------------------------------------------------------------
// MBParams
//------------------------------------------------------------------------
namespace MBParamsHOG
{
    // Should we measure timing
    constexpr bool timing = true;
    
    constexpr double timestepMs = 1.0;

    constexpr unsigned int inputWidth = 86;
    constexpr unsigned int inputHeight = 20;

    // HOG feature configuration
    constexpr int hogNumOrientations = 3;
    constexpr int hogRFSize = 8;
    constexpr int hogRFStride = 6;
    constexpr int hogNumRFX = 14;
    constexpr int hogNumRFY = 3;

    // Calculate hog feature size
    constexpr int hogFeatureSize = hogNumOrientations * hogNumRFX * hogNumRFY;

    // Network dimensions
    constexpr unsigned int numPN = (unsigned int)hogFeatureSize;
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

    // Midpoint of sigmoid used for graded potential synapses
    constexpr double ggnToKCVMid = -54.1;

    // Slope of sigmoid used for graded potential synapses
    constexpr double ggnToKCVslope = 1.0;

    // GGC membrane voltage where graded potential synapse activates
    constexpr double ggnToKCVthresh = -60.0;

    // Spiking thereshold for PN
    constexpr double pnVthresh = -50.0;

    // Scale of each dopamine 'spike'
    // **NOTE** manually tuned for one-shot learning - also close to BA/phi
    constexpr double dopamineStrength = 0.0289;

    // How many PN neurons are connected to each KC
    constexpr unsigned int numPNSynapsesPerKC = 10;



}

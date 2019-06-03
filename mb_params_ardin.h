#pragma once

//------------------------------------------------------------------------
// MBParamsArdin
//------------------------------------------------------------------------
namespace MBParamsArdin
{
    constexpr double timestepMs = 1.0;

    // Intermediate image size used within snapshot pre-processing
    constexpr unsigned int intermediateWidth = 74;
    constexpr unsigned int intermediateHeight = 19;

    // How much larger than this, input snapshot is expected to be
    constexpr unsigned int displayScale = 8;

    // Network dimensions
    constexpr unsigned int inputWidth = 36;
    constexpr unsigned int inputHeight = 10;
    constexpr unsigned int numPN = inputWidth * inputHeight;
    constexpr unsigned int numKC = 20000;
    constexpr unsigned int numEN = 1;

    // Regime parameters
    constexpr double rewardTimeMs = 40.0;
    constexpr double presentDurationMs = 40.0;
    constexpr double postStimuliDurationMs = 40.0;

    // Scale applied to convert image data to input currents for PNs
    // **NOTE** manually tuned to get approximately 50% PN activation
    constexpr double inputCurrentScale = 4.0;

    // Weight of static synapses between PN and KC populations
    // **NOTE** manually tuend to get approximately 200/20000 KC firing sparsity
    constexpr double pnToKCWeight = 0.05;// 0.0705;

    // Initial/maximum weight of plastic synapses between KC and EN populations
    // **NOTE** note manually tuned to get 15-20 spikes for a novel image
    constexpr double kcToENWeight = 0.6;

    // Time constant of dopamine
    constexpr double tauD = 20.0;

    // Scale of each dopamine 'spike'
    // **NOTE** manually tuned for one-shot learning - also close to BA/phi
    constexpr double dopamineStrength = 0.03;

    // How many PN neurons are connected to each KC
    constexpr unsigned int numPNSynapsesPerKC = 10;
}

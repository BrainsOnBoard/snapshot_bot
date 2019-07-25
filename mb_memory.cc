#define NO_HEADER_DEFINITIONS
#include "mb_memory.h"

// Standard C++ includes
#include <fstream>
#include <random>

// Standard C includes
#include <cmath>

// CLI11 includes
#include "third_party/CLI11.hpp"

// BoB robotics includes
#include "common/timer.h"
#include "genn_utils/connectors.h"

using namespace BoBRobotics;
using namespace units::literals;
using namespace units::angle;
using namespace units::math;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
void record(double t, unsigned int spikeCount, unsigned int *spikes, MBMemory::Spikes &spikeOutput)
{
    // Add a new entry to the cache
    spikeOutput.emplace_back();

    // Fill in time
    spikeOutput.back().first = t;

    // Reserve vector to hold spikes
    spikeOutput.back().second.reserve(spikeCount);

    // Copy spikes into vector
    std::copy_n(spikes, spikeCount, std::back_inserter(spikeOutput.back().second));
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// MBMemory
//----------------------------------------------------------------------------
MBMemory::MBMemory(unsigned int numPN, unsigned int numKC, unsigned int numEN, unsigned int numPNSynapsesPerKC,
                   int inputWidth, int inputHeight,
                   double tauD, double kcToENWeight, double dopamineStrength,
                   double rewardTimeMs, double presentDurationMs, double timestepMs,
                   const std::string &modelName, bool timing)
:   Navigation::VisualNavigationBase(cv::Size(inputWidth, inputHeight)), m_Timing(timing), m_NumPN(numPN), m_NumKC(numKC), m_NumEN(numEN), m_NumPNSynapsesPerKC(numPNSynapsesPerKC),
    m_InputWidth(inputWidth), m_InputHeight(inputHeight), m_TauD(tauD), m_KCToENWeight(kcToENWeight), m_TimestepMs(timestepMs),
    m_KCToENDopamineStrength(dopamineStrength), m_RewardTimeMs(rewardTimeMs), m_PresentDurationMs(presentDurationMs),
    m_NumPNSpikes(0), m_NumKCSpikes(0), m_NumENSpikes(0), m_NumUsedWeights(MBParamsHOG::numKC * MBParamsHOG::numEN),
    m_NumActivePN(0), m_NumActiveKC(0), m_SLM("", modelName)
{
    std::mt19937 gen;

    {
        Timer<> timer("Allocation:");
        m_SLM.allocateMem();
    }

    {
        Timer<> timer("Initialization:");
        m_SLM.initialize();
    }

    {
        Timer<> timer("Building connectivity:");

        unsigned int *rowLengthpnToKC = m_SLM.getArray<unsigned int>("rowLengthpnToKC");
        unsigned int *indpnToKC = m_SLM.getArray<unsigned int>("indpnToKC");
        unsigned int *maxRowLengthpnToKC = m_SLM.getScalar<unsigned int>("maxRowLengthpnToKC");
        float *gkcToEN = m_SLM.getArray<float>("gkcToEN");
        GeNNUtils::buildFixedNumberPreConnector(m_NumPN, m_NumKC, m_NumPNSynapsesPerKC,
                                                rowLengthpnToKC, indpnToKC, *maxRowLengthpnToKC, gen);

        // Manually initialise weights
        // **NOTE** this is a little bit of a hack as we're only doing this so repeated calls to initialise won't overwrite
        std::fill_n(&gkcToEN[0], m_NumKC * m_NumEN, m_KCToENWeight);
    }

    // Final setup
    {
        Timer<> timer("Sparse init:");
        m_SLM.initializeSparse();
    }

    // Get pointers to EGPs
    m_InjectDopamineKCToEN = m_SLM.getScalar<bool>("injectDopaminekcToEN");
    m_TDKCToEN = m_SLM.getScalar<float>("tDkcToEN");
    m_DKCToEN = m_SLM.getScalar<float>("dkcToEN");

    // Get pointers to state variables
    m_SpkCntEN = m_SLM.getArray<unsigned int>("glbSpkCntEN");
    m_SpkEN = m_SLM.getArray<unsigned int>("glbSpkEN");
    m_SpkCntKC = m_SLM.getArray<unsigned int>("glbSpkCntKC");
    m_SpkKC = m_SLM.getArray<unsigned int>("glbSpkKC");
    m_SpkCntPN = m_SLM.getArray<unsigned int>("glbSpkCntPN");
    m_SpkPN = m_SLM.getArray<unsigned int>("glbSpkPN");
    m_GKCToEN = m_SLM.getArray<float>("gkcToEN");
    
    // Get pointers to timers
    m_NeuronUpdateTime = m_SLM.getScalar<double>("neuronUpdateTime");
    m_PresynapticUpdateTime = m_SLM.getScalar<double>("presynapticUpdateTime");
    m_PostsynapticUpdateTime = m_SLM.getScalar<double>("postsynapticUpdateTime");
}
//----------------------------------------------------------------------------
MBMemory::~MBMemory()
{
}
//----------------------------------------------------------------------------
void MBMemory::train(const cv::Mat &image)
{
    present(image, true);
}
//----------------------------------------------------------------------------
float MBMemory::test(const cv::Mat &image) const
{
    // Get number of EN spikes
    return (float)std::get<2>(present(image, false));
}
//----------------------------------------------------------------------------
void MBMemory::clearMemory()
{
    throw std::runtime_error("MBMemory does not currently support clearing");
}
//----------------------------------------------------------------------------
std::tuple<unsigned int, unsigned int, unsigned int> MBMemory::present(const cv::Mat &image, bool train) const
{
    BOB_ASSERT(image.cols == m_InputWidth);
    BOB_ASSERT(image.rows == m_InputHeight);
    BOB_ASSERT(image.type() == CV_8UC1);

     // Convert simulation regime parameters to timesteps
    const unsigned long long rewardTimestep = convertMsToTimesteps(m_RewardTimeMs);
    const unsigned int endPresentTimestep = convertMsToTimesteps(m_PresentDurationMs);
    const unsigned int postStimuliDuration = convertMsToTimesteps(MBParamsHOG::postStimuliDurationMs);

    const unsigned long long duration = endPresentTimestep + postStimuliDuration;

    // Make sure KC state and GGN insyn are reset before simulation
    m_SLM.initialize();

    // Start presenting stimuli
    beginPresent(image);

    // Reset model time
    m_SLM.setTimestep(0);
    m_SLM.setTime(0.0f);

    // Clear spike records
#ifdef RECORD_INTERMEDIATES
    m_PNSpikes.clear();
    m_KCSpikes.clear();
    m_ENSpikes.clear();

    std::vector<bool> pnSpikeBitset(m_NumPN, false);
    std::vector<bool> kcSpikeBitset(m_NumKC, false);
#endif

    // Reset time of last dopamine spike
    *m_TDKCToEN = std::numeric_limits<float>::lowest();

    // Reset timers
    if(m_Timing) {
        *m_NeuronUpdateTime = 0.0;
        *m_PresynapticUpdateTime = 0.0;
        *m_PostsynapticUpdateTime = 0.0;
    }
    
    // Loop through timesteps
#ifdef RECORD_INTERMEDIATES
    m_NumPNSpikes = 0;
    m_NumKCSpikes = 0;
#endif
    m_NumENSpikes = 0;
    while(m_SLM.getTimestep() < duration) {
        // If we should stop presenting image
        if(m_SLM.getTimestep() == endPresentTimestep) {
            endPresent();
        }

        // If we should reward in this timestep, inject dopamine
        if(train && m_SLM.getTimestep() == rewardTimestep) {
            *m_InjectDopamineKCToEN = true;
        }

        // Simulate on GPU
        m_SLM.stepTime();

        // Download spikes
#ifdef RECORD_INTERMEDIATES
        m_SLM.pullCurrentSpikesFromDevice("PN");
        m_SLM.pullCurrentSpikesFromDevice("KC");
#endif
        m_SLM.pullCurrentSpikesFromDevice("EN");

        // If a dopamine spike has been injected this timestep
        if(*m_InjectDopamineKCToEN) {
            // Decay global dopamine traces
            *m_DKCToEN *= std::exp(-(m_SLM.getTime() - *m_TDKCToEN) / m_TauD);

            // Add effect of dopamine spike
            *m_DKCToEN += m_KCToENDopamineStrength;

            // Update last reward time
            *m_TDKCToEN = m_SLM.getTime();

            // Clear dopamine injection flags
            *m_InjectDopamineKCToEN = false;
        }

        m_NumENSpikes += m_SpkCntEN[0];
#ifdef RECORD_INTERMEDIATES
        m_NumPNSpikes += m_SpkCntPN[0];
        m_NumKCSpikes += m_SpkCntKC[0];
        for(unsigned int i = 0; i < m_SpkCntPN[0]; i++) {
            pnSpikeBitset[m_SpkPN[i]] = true;
        }

        for(unsigned int i = 0; i < m_SpkCntKC[0]; i++) {
            kcSpikeBitset[m_SpkKC[i]] = true;
        }

        // Record spikes
        record(m_SLM.getTime(), m_SpkCntPN[0], m_SpkPN, m_PNSpikes);
        record(m_SLM.getTime(), m_SpkCntKC[0], m_SpkKC, m_KCSpikes);
        record(m_SLM.getTime(), m_SpkCntEN[0], m_SpkEN, m_ENSpikes);
#endif
    }

    if(m_Timing) {
        std::cout << "Neuron update:" << *m_NeuronUpdateTime * 1000.0 << "ms, ";
        std::cout << "Presynaptic update:" << *m_PresynapticUpdateTime * 1000.0  << "ms, ";
        std::cout << "Postsynaptic update:" << *m_PostsynapticUpdateTime * 1000.0  << "ms" << std::endl;
    }
#ifdef RECORD_TERMINAL_SYNAPSE_STATE
    // Download synaptic state
    m_SLM.pullStateFromDevice("EN");

    std::ofstream terminalStream("terminal_synaptic_state.csv");
    terminalStream << "Weight, Eligibility" << std::endl;
    for(unsigned int s = 0; s < m_NumKC * m_NumEN; s++) {
        terminalStream << m_GKCToEN[s] << "," << m_CKCToEN[s] * std::exp(-(t - m_TCKCToEN[s]) / 40.0) << std::endl;
    }
    std::cout << "Final dopamine level:" << *m_DKCToEN * std::exp(-(t - *m_TDKCToEN) / m_TauD) << std::endl;
#endif  // RECORD_TERMINAL_SYNAPSE_STATE

#ifdef RECORD_INTERMEDIATES
    // Cache number of unique active cells
    m_NumActivePN = std::count(pnSpikeBitset.cbegin(), pnSpikeBitset.cend(), true);
    m_NumActiveKC = std::count(kcSpikeBitset.cbegin(), kcSpikeBitset.cend(), true);

    if(train) {
        const unsigned int numWeights = m_NumKC * m_NumEN;

        // Pull weights from device
        m_SLM.pullVarFromDevice("kcToEN", "g");

        // Cache number of unused weights
        m_NumUsedWeights = numWeights - std::count(&m_GKCToEN[0], &m_GKCToEN[numWeights], 0.0f);
    }

    return std::make_tuple(m_NumPNSpikes, m_NumKCSpikes, m_NumENSpikes);
#else
    return std::make_tuple(0, 0, m_NumENSpikes);
#endif
}

// GeNN includes
#include "modelSpec.h"

// BoB robotics includes
#include "genn_models/stdp_dopamine.h"
#include "genn_utils/connectors.h"

// Model includes
#include "mb_params_hog.h"

using namespace BoBRobotics;


//---------------------------------------------------------------------------
// Standard LIF model extended to take an additional
// input current from an extra global variable
//---------------------------------------------------------------------------
class LIFExtCurrent : public NeuronModels::Base
{
public:
    DECLARE_MODEL(LIFExtCurrent, 6, 3);

    SET_SIM_CODE(
        "if ($(RefracTime) <= 0.0) {\n"
        "   const scalar alpha = (($(Isyn) + $(Iext)) * $(Rmembrane)) + $(Vrest);\n"
        "   $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else {\n"
        "  $(RefracTime) -= DT;\n"
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(V) = $(Vreset);\n"
        "$(RefracTime) = $(TauRefrac);\n");

    SET_PARAM_NAMES({
        "C",            // 0 - Membrane capacitance
        "TauM",         // 1 - Membrane time constant [ms]
        "Vrest",        // 2 - Resting membrane potential [mV]
        "Vreset",       // 3 - Reset voltage [mV]
        "Vthresh",      // 4 - Spiking threshold [mV]
        "TauRefrac"});  // 5 - Refractory time [ms]


    SET_DERIVED_PARAMS({
        {"ExpTC", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const std::vector<double> &pars, double){ return  pars[1] / pars[0]; }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}, {"Iext", "scalar"}});
};
IMPLEMENT_MODEL(LIFExtCurrent);

//---------------------------------------------------------------------------
// ExpStaticGraded
//---------------------------------------------------------------------------
class ExpStaticGraded : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(ExpStaticGraded, 3, 1, 0, 0);

    SET_EVENT_CODE("$(addToInSyn, DT * $(g) * max(0.0, 1.0 / (1.0 + exp(($(Vmid) - $(V_pre)) / $(Vslope)))));\n");

    SET_EVENT_THRESHOLD_CONDITION_CODE("$(V_pre) > $(Vthresh)");

    SET_PARAM_NAMES({
        "Vmid",         // 0 - Mid point of sigmoid
        "Vslope",       // 1 - Slope of sigmoid
        "Vthresh"});    // 2 - Presynaptic membrane threshold

    SET_VARS({{"g", "scalar"}});
};
IMPLEMENT_MODEL(ExpStaticGraded);

void modelDefinition(NNmodel &model)
{
    model.setDT(MBParamsHOG::timestepMs);
    model.setName("mb_memory_hog");
    model.setTiming(MBParamsHOG::timing);

    //---------------------------------------------------------------------------
    // Neuron model parameters
    //---------------------------------------------------------------------------
    // LIF model parameters
    LIFExtCurrent::ParamValues pnParams(
        1.0,                                // 0 - C
        20.0,                               // 1 - TauM
        -60.0,                              // 2 - Vrest
        -60.0,                              // 3 - Vreset
        MBParamsHOG::pnVthresh,             // 4 - Vthresh
        200.0);                             // 5 - TauRefrac **NOTE** essentially make neurons fire once

    NeuronModels::LIF::ParamValues kcParams(
        0.2,                                // 0 - C
        20.0,                               // 1 - TauM
        -60.0,                              // 2 - Vrest
        -60.0,                              // 3 - Vreset
        -50.0,                              // 4 - Vthresh
        0.0,                                // 5 - Ioffset
        2.0);                               // 6 - TauRefrac

    NeuronModels::LIF::ParamValues enParams(
        0.2,                                // 0 - C
        20.0,                               // 1 - TauM
        -60.0,                              // 2 - Vrest
        -60.0,                              // 3 - Vreset
        -50.0,                              // 4 - Vthresh
        0.0,                                // 5 - Ioffset
        1.0);                               // 6 - TauRefrac

    NeuronModels::LIF::ParamValues ggnParams(
        0.2,                                // 0 - C
        20.0,                               // 1 - TauM
        -60.0,                              // 2 - Vrest
        -60.0,                              // 3 - Vreset
        10000.0,                            // 4 - Vthresh **NOTE** essentially non-spiking
        0.0,                                // 5 - Ioffset
        2.0);                               // 6 - TauRefrac

    // LIF initial conditions
    NeuronModels::LIF::VarValues lifInit(
        -60.0,  // 0 - V
        0.0);   // 1 - RefracTime

    // PN initial conditions
    LIFExtCurrent::VarValues pnInit(
        -60.0,  // 0 - V
        0.0,    // 1 - RefracTime
        0.0);   // 2 - Iext

    //---------------------------------------------------------------------------
    // Postsynaptic model parameters
    //---------------------------------------------------------------------------
     PostsynapticModels::ExpCurr::ParamValues pnToKCPostsynapticParams(
        3.0);   // 0 - Synaptic time constant [from Ardin] (ms)

    PostsynapticModels::ExpCurr::ParamValues kcToENPostsynapticParams(
        8.0);   // 0 - Synaptic time constant [from Ardin] (ms)

    // **TODO** experiment with tuning these
    PostsynapticModels::ExpCurr::ParamValues kcToGGNPostsynapticParams(
        5.0);   // 0 - Synaptic time constant (ms)

    PostsynapticModels::ExpCurr::ParamValues ggnToKCPostsynapticParams(
        4.0);   // 0 - Synaptic time constant [from Nowotny](ms)

    //---------------------------------------------------------------------------
    // Weight update model parameters
    //---------------------------------------------------------------------------
    GeNNModels::STDPDopamine::ParamValues kcToENWeightUpdateParams(
        15.0,                           // 0 - Potentiation time constant (ms)
        15.0,                           // 1 - Depression time constant (ms)
        40.0,                           // 2 - Synaptic tag time constant (ms)
        MBParamsHOG::tauD,              // 3 - Dopamine time constant (ms)
        -1.0,                           // 4 - Rate of potentiation
        1.0,                            // 5 - Rate of depression
        0.0,                            // 6 - Minimum weight
        MBParamsHOG::kcToENWeight);     // 7 - Maximum weight

    GeNNModels::STDPDopamine::VarValues kcToENWeightUpdateInitVars(
        uninitialisedVar(),             // 0 - Synaptic weight
        0.0,                            // 1 - Synaptic tag
        0.0);                           // 2 - Time of last synaptic tag update

    WeightUpdateModels::StaticPulse::VarValues pnToKCInitVars(
        MBParamsHOG::pnToKCWeight);    // 0 - Synaptic weight

    WeightUpdateModels::StaticPulse::VarValues kcToGGNInitVars(
        MBParamsHOG::kcToGGNWeight);    // 0 - Synaptic weight

    ExpStaticGraded::ParamValues ggnToKCParams(
        MBParamsHOG::ggnToKCVMid,       // 0 - Mid point of sigmoid
        MBParamsHOG::ggnToKCVslope,     // 1 - Slope of sigmoid
        MBParamsHOG::ggnToKCVthresh);   // 2 - Presynaptic membrane threshold

    ExpStaticGraded::VarValues ggnToKCInitVars(
        MBParamsHOG::ggnToKCWeight);    // 0 - Synaptic weight

    // Create neuron populations
    auto *pn = model.addNeuronPopulation<LIFExtCurrent>("PN", MBParamsHOG::numPN, pnParams, pnInit);
    model.addNeuronPopulation<NeuronModels::LIF>("KC", MBParamsHOG::numKC, kcParams, lifInit);
    auto *en = model.addNeuronPopulation<NeuronModels::LIF>("EN", MBParamsHOG::numEN, enParams, lifInit);
    model.addNeuronPopulation<NeuronModels::LIF>("GGN", 1, ggnParams, lifInit);

/*#ifdef __aarch64__
    pn->setVarLocation("Iext", VarLocation::HOST_DEVICE_ZERO_COPY);
    en->setSpikeLocation(VarLocation::HOST_DEVICE_ZERO_COPY);
#endif*/
    
    auto pnToKC = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "pnToKC", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "PN", "KC",
        {}, pnToKCInitVars,
        pnToKCPostsynapticParams, {});

    model.addSynapsePopulation<GeNNModels::STDPDopamine, PostsynapticModels::ExpCurr>(
        "kcToEN", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "KC", "EN",
        kcToENWeightUpdateParams, kcToENWeightUpdateInitVars,
        kcToENPostsynapticParams, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "kcToGGN", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
        "KC", "GGN",
        {}, kcToGGNInitVars,
        kcToGGNPostsynapticParams, {});

    model.addSynapsePopulation<ExpStaticGraded, PostsynapticModels::ExpCurr>(
        "ggnToKC", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
        "GGN","KC",
        ggnToKCParams, ggnToKCInitVars,
        ggnToKCPostsynapticParams, {});

    // Calculate max connections
    const unsigned int maxConn = GeNNUtils::calcFixedNumberPreConnectorMaxConnections(MBParamsHOG::numPN, MBParamsHOG::numKC,
                                                                                      MBParamsHOG::numPNSynapsesPerKC);

    std::cout << "Max connections:" << maxConn << std::endl;
    pnToKC->setMaxConnections(maxConn);
}

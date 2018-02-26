#include "snapshot_encoder.h"

#include <iostream>

// Standard C includes
#include <cassert>

//----------------------------------------------------------------------------
// SnapshotEncoder
//----------------------------------------------------------------------------
SnapshotEncoder::SnapshotEncoder(unsigned int inputWidth, unsigned int inputHeight, unsigned int outputSize)
:   m_InputWidth(inputWidth), m_InputHeight(inputHeight), m_InputSize(inputWidth * inputHeight), m_OutputSize(outputSize),
    m_Status(nullptr), m_Graph(nullptr), m_Session(nullptr), m_InputTensor(nullptr), m_OutputTensor(nullptr), 
    m_Output(1, m_OutputSize, CV_8UC1)
{
    
}
//----------------------------------------------------------------------------
SnapshotEncoder::SnapshotEncoder(unsigned int inputWidth, unsigned int inputHeight, unsigned int outputSize,
                                 const std::string &exportDirectory, const std::string &tag,
                                 const std::string &inputOpName, const std::string &outputOpName)
:   SnapshotEncoder(inputWidth, inputHeight, outputSize)
{
    if(!openModel(exportDirectory, tag, inputOpName, outputOpName)) {
        throw std::runtime_error("Cannot open model");
    }
}
//----------------------------------------------------------------------------
SnapshotEncoder::~SnapshotEncoder()
{
    if(m_OutputTensor) {
        TF_DeleteTensor(m_OutputTensor);
    }

    if(m_InputTensor) {
        TF_DeleteTensor(m_InputTensor);
    }

    if(m_Session) {
        TF_DeleteSession(m_Session, m_Status);
    }

    if(m_Graph) {
        TF_DeleteGraph(m_Graph);
    }

    if(m_Status) {
        TF_DeleteStatus(m_Status);
    }
}
//----------------------------------------------------------------------------
bool SnapshotEncoder::openModel(const std::string &exportDirectory, const std::string &tag,
                                const std::string &inputOpName, const std::string &outputOpName)
{
    // Create a status object for all subsequent calls
    m_Status = TF_NewStatus();

    // Create graph
    m_Graph = TF_NewGraph();

    // Load session from saved model
    std::array<const char*, 1> tags = {tag.c_str()};
    TF_SessionOptions *sessionOptions = TF_NewSessionOptions();
    m_Session = TF_LoadSessionFromSavedModel(sessionOptions, nullptr,
                                             exportDirectory.c_str(),
                                             tags.data(), tags.size(),
                                             m_Graph, nullptr, m_Status);
    TF_DeleteSessionOptions(sessionOptions);

    // Check status
    if(TF_GetCode(m_Status) != TF_OK) {
        std::cerr << "Cannot load session:" <<  TF_Message(m_Status) << std::endl;
        return false;
    }

    // Get input operator
    m_InputOp[0].index = 0;
    m_InputOp[0].oper = TF_GraphOperationByName(m_Graph, inputOpName.c_str());
    if(m_InputOp[0].oper == nullptr) {
        std::cerr << "Cannot get find input operation in graph" << std::endl;
        return false;
    }

    // Check input shape matches snapshot size
    const auto inputShape = getOpShape(m_InputOp[0]);
    assert(inputShape.size() == 2);
    assert(inputShape[0] == -1);
    assert(inputShape[1] == m_InputSize);

    // Get output operator
    m_OutputOp[0].index = 0;
    m_OutputOp[0].oper = TF_GraphOperationByName(m_Graph, outputOpName.c_str());
    if(m_OutputOp[0].oper == nullptr) {
        std::cerr << "Cannot get find input operation in graph" << std::endl;
        return false;
    }

    // Check output shape matches encoded snapshot size
    const auto outputShape = getOpShape(m_OutputOp[0]);
    assert(outputShape.size() == 2);
    assert(outputShape[0] == -1);
    assert(outputShape[1] == m_OutputSize);

    // Create a 1D input tensor to hold input snapshots
    std::array<int64_t, 2> inputDims = {1, m_InputSize};
    m_InputTensor = TF_AllocateTensor(TF_FLOAT, inputDims.data(), inputDims.size(), m_InputSize * sizeof(float));
    
    return true;
}
//----------------------------------------------------------------------------
void SnapshotEncoder::encode(const cv::Mat &snapshotFloat)
{
    assert(snapshotFloat.cols == m_InputWidth);
    assert(snapshotFloat.rows == m_InputHeight);
    assert(snapshotFloat.type() == CV_32FC1);

    // If we have an output tensor from a previous run, delete it
    if(m_OutputTensor) {
        TF_DeleteTensor(m_OutputTensor);
        m_OutputTensor = nullptr;
    }

    // Copy snapshot data into input Tensor
    memcpy(TF_TensorData(m_InputTensor), snapshotFloat.data, m_InputSize * sizeof(float));

    // Create arrays of input and output tensors
    // **TODO** can output tensors be re-used?
    std::array<TF_Tensor*, 1> inputTensors = {m_InputTensor};
    std::array<TF_Tensor*, 1> outputTensors = {nullptr};

    TF_SessionRun(m_Session,                                                    // Session
                  nullptr,                                                      // Run options
                  m_InputOp.data(), inputTensors.data(), m_InputOp.size(),      // Input tensors
                  m_OutputOp.data(), outputTensors.data(), m_OutputOp.size(),   // Output tensors
                  nullptr, 0,                                                   // Target operations
                  nullptr,                                                      // Run metadata
                  m_Status);                                                    // Status
    if(TF_GetCode(m_Status) != TF_OK) {
        throw std::runtime_error(std::string("Cannot run session:") + TF_Message(m_Status));
    }

    // If we have an output tensor
    m_OutputTensor = outputTensors[0];
    if(outputTensors[0] == nullptr) {
        throw std::runtime_error("No output");
    }
    else {
        m_OutputFloat = cv::Mat(1, m_OutputSize, CV_32FC1, TF_TensorData(m_OutputTensor));
        m_OutputFloat.convertTo(m_Output, CV_8UC1, 255.0);
    }
}
//----------------------------------------------------------------------------
std::vector<int64_t> SnapshotEncoder::getOpShape(const TF_Output &op)
{
    // Get number of dimensions
    const int numDims = TF_GraphGetTensorNumDims(m_Graph, op, m_Status);
    if(TF_GetCode(m_Status) != TF_OK) {
        throw std::runtime_error(std::string("Cannot get op dimensions:") + TF_Message(m_Status));
    }

    // Get shape
    std::vector<int64_t> shape(numDims);
    TF_GraphGetTensorShape(m_Graph, op, shape.data(), shape.size(), m_Status);
    if(TF_GetCode(m_Status) != TF_OK) {
        throw std::runtime_error(std::string("Cannot get op shapse:") + TF_Message(m_Status));
    }
    return shape;
}
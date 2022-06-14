#include "process_link.hpp"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <fstream>

/*
  Faster parallel version of the example
  
  In this example, the Parallel job option is replaced with the
  ParallelNonBlocking job option.
  
  ParallelNonBlocking means that a batch of link operations is sent to
  the scheduler to be distributed to worker threads, but this thread
  doesn't wait for the batch to finish, continuing to assign more
  operations and do other things. The worker threads will work in the
  background.

  Consecutive batches will be executed in sequence; each batch will be
  completed before the next is started. Within each batch, operations
  will be done in any order and simultaneously, subject only to the
  near-node guarantee.

  The comments below highlight the changes necessary to use
  ParallelNonBlocking instead of Parallel.  Recommended to read
  ex1.cpp first, because that version of the example has comments of a
  more general nature.
 */

using namespace llrt;

struct IFNeuron{
    float v[2]={0,0};
    float x[2]={0,0};
};

struct IFDendrite{
    float w=0;
};

template <typename T>
T sigmoid(const T x){
    return (1 / (1 + std::exp(-x)));
}

const float mu = 0.99;
const float k = 0.01;

using TTypes = std::tuple<IFNeuron, IFDendrite>;
using LTypes = std::tuple<DenseLink>;
using TL=std::pair<TTypes, LTypes>;

void initializeNetwork(Network<TL> &net){
    ProcessNetLinks_Er(net, [](IFDendrite &E, ThreadsafeRNG &r){
        E.w = std::normal_distribution<float>(0,1.0)(r);
    }, Dendrites
        | ParallelNonBlocking); // don't wait for it to finish. This operation counts as one batch.
}

size_t advanceNetwork(Network<TL> &net, size_t timestep, Component<TL> &inputCmp, std::vector<float> &inputs){
    size_t _0 = timestep % 2;
    size_t _1 = 1 - _0;
    
    ProcessNetCmps_N(net, [=](IFNeuron &N){
        if (N.x[_0] == 0)
            N.v[_1] = mu * N.v[_0];
        else
            N.v[_1] = 0;
    }, ParallelNonBlocking
        // we give this operation a name so that it shows up in the performance report
        | KernelName("SelfPotential"));

    // The various Process operations have a return value: the batch
    // number.  We save this number so that later, in main(), we can
    // wait for the inputBatch operation to finish before writing new
    // inputs to the inputs array.
    size_t inputBatch = ProcessCmp_NNi(inputCmp, [=, &inputs](IFNeuron &N, const size_t Ni){
        N.v[_1] += inputs[Ni];
    }, ParallelNonBlocking | KernelName("Input"));
    
    ProcessNetLinks_NEn(net, [_1, _0](IFNeuron &N, const IFDendrite &E, const IFNeuron &n){
        N.v[_1] += E.w * n.x[_0];
    }, Dendrites | ParallelNonBlocking | KernelName("EdgeSum"));
    
    ProcessNetCmps_Nr(net, [=](IFNeuron &N, ThreadsafeRNG &r){
        float activateProb = sigmoid(k*N.v[_1]);
        if(std::uniform_real_distribution<float>(0,1.0)(r) < activateProb)
            N.x[_1] = 1;
        else
            N.x[_1] = 0;
    }, ParallelNonBlocking | KernelName("Activate"));

    // return the input batch number back to main()
    return inputBatch;
}

int main(){
    Network<TL> net(7);

    index_t inSize = 500, outSize = 500, iters = 2000;

    Component<TL> &c1 = net.template component<IFNeuron>({inSize});
    Component<TL> &c2 = c1.template connect<DenseLink, NoData, IFDendrite, IFNeuron>({outSize});
    c2.template connect<DenseLink, NoData, IFDendrite>(c1);

    net.display();

    initializeNetwork(net);

    std::vector<float> inputs(inSize);
    std::mt19937_64 rng(136);
    
    // batch 0 is a dummy batch that is already done when the network
    // is created. So, our first call to net.finishBatch(0) will
    // return immediately.
    size_t inputBatch = 0;
    for (size_t i=0; i < iters; i++){
        // wait until the input batch is finished before we start writing to the input array
        // other batches can keep running while we write to the input array.
        net.finishBatch(inputBatch);
        for(size_t i=0; i < inSize; i++){
            inputs[i] = std::normal_distribution<float>(0,1.0)(rng);
        }

        // get the batch number
        inputBatch = advanceNetwork(net, i, c1, inputs);
    }

    net.perfReport("perfReport.json");
}

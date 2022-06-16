#include "process_link.hpp"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <fstream>

/*
  A demonstration of how to use multiple neuron types in the same network.
 */

using namespace llrt;

struct IFNeuron{
    float v[2]={0,0};
    float x[2]={0,0};
};

struct IFDendrite{
    float w=0;
};

// A neuron which will hold the same activation value for the duration
// of the network, after it's initialized
struct ConstNeuron{
    float x=0;
};

template <typename T>
T sigmoid(const T x){
    return (1 / (1 + std::exp(-x)));
}

const float mu = 0.99;
const float k = 0.01;

// Make sure to add ConstNeuron to TTypes so the tensor data can be a
// variant containing an std::vector<ConstNeuron>
using TTypes = std::tuple<IFNeuron, IFDendrite, ConstNeuron>;
using LTypes = std::tuple<DenseLink>;
using TL=std::pair<TTypes, LTypes>;

void initializeNetwork(Network<TL> &net){
    // ParallelPart means that this operation can run at the same time
    // as the next one, in the same batch. This is more efficient
    // because it avoids the need for the worker threads to
    // synchronize with each other before running the next operation.
    ProcessNetLinks_Er(net, [](IFDendrite &E, ThreadsafeRNG &r){
        E.w = std::normal_distribution<float>(0,1.0)(r);
    }, Dendrites | ParallelPart | KernelName("InitEdges"));

    // Initialize the ConstNeuron initial x values to be randomly 0 or 1.
    
    // This operation will run only on the ConstNeuron components, not
    // on the IFNeuron components. In general, operations only run on
    // links where the kernel parameter types for the N, E, e, and n
    // parameters match the type of data in the component or link end.
    ProcessNetCmps_Nr(net, [](ConstNeuron &N, ThreadsafeRNG &r){
        N.x = std::uniform_int_distribution<int>(0, 1)(r);
    }, ParallelNonBlocking | KernelName("InitConstNeurons"));
}

size_t advanceNetwork(Network<TL> &net, size_t timestep, Component<TL> &inputCmp, std::vector<float> &inputs){
    size_t _0 = timestep % 2;
    size_t _1 = 1 - _0;

    // Again, this runs only on the IFNeurons, not the ConstNeurons
    ProcessNetCmps_N(net, [=](IFNeuron &N){
        if (N.x[_0] == 0)
            N.v[_1] = mu * N.v[_0];
        else
            N.v[_1] = 0;
    }, ParallelNonBlocking | KernelName("SelfPotential"));

    size_t inputBatch = ProcessCmp_NNi(inputCmp, [=, &inputs](IFNeuron &N, const size_t Ni){
        N.v[_1] += inputs[Ni];
    }, ParallelNonBlocking | KernelName("Input"));

    // This operation runs only where the near-node is an IFNeuron,
    // the near-edge is an IFDendrite, and the far-node is an
    // IFNeuron. Again, ParallelPart on this operation allows it to
    // run concurrently with the next operation. This is threadsafe
    // because the near-node guarantee means that N.v[_1] is not going
    // to be updated by two worker threads at the same time, even when
    // two different network operations are running at once.
    ProcessNetLinks_NEn(net, [_1, _0](IFNeuron &N, const IFDendrite &E, const IFNeuron &n){
        N.v[_1] += E.w * n.x[_0];
    }, Dendrites | KernelName("EdgeSum") | ParallelPart);

    // We want ConstNeurons to provide input to IFNeurons in the same
    // way that IFNeurons provide input to each other, so we need
    // another operation to provide that input.
    ProcessNetLinks_NEn(net, [_1, _0](IFNeuron &N, const IFDendrite &E, const ConstNeuron &n){
        N.v[_1] += E.w * n.x;
    }, Dendrites | KernelName("ConstEdgeSum") | ParallelNonBlocking);
    
    ProcessNetCmps_Nr(net, [=](IFNeuron &N, ThreadsafeRNG &r){
        float activateProb = sigmoid(k*N.v[_1]);
        if(std::uniform_real_distribution<float>(0,1.0)(r) < activateProb)
            N.x[_1] = 1;
        else
            N.x[_1] = 0;
    }, ParallelNonBlocking | KernelName("Activate"));
    
    return inputBatch;
}

int main(){
    Network<TL> net(7);

    index_t inSize = 500, midSize = 500, outSize = 500, iters = 2000;

    Component<TL> &c1 = net.template component<IFNeuron>({inSize});
    Component<TL> &c2 = c1.template connect<DenseLink, NoData, IFDendrite, IFNeuron>({midSize});
    c2.template connect<DenseLink, NoData, IFDendrite>(c1);

    // Create a component of ConstNeurons and connect it to c2.
    
    // We want the dendrite end to be on the c2 side, not the c3 side,
    // so we set swapAxon=true (the last parameter). Note also the
    // order of template parameters has changed: IFDendrite comes
    // before NoData, because we want end 0 to contain IFDendrites and
    // end 1 to contain no data, instead of the reverse.

    // The second-to-last parameter is swapEnds, which would allow us
    // to swap the 0 end and the 1 end, so that the 0 end would be on
    // c3 and the 1 end would be on c2. We don't want to do that, so
    // we leave that one as false.
    Component<TL> &c3 = c2.template connect<DenseLink, IFDendrite, NoData, ConstNeuron>({outSize}, false, true);

    net.display();

    initializeNetwork(net);

    std::vector<float> inputs(inSize);
    std::mt19937_64 rng(136);
    
    size_t inputBatch = 0;
    for (size_t i=0; i < iters; i++){
        net.finishBatch(inputBatch);
        for(size_t i=0; i < inSize; i++){
            inputs[i] = std::normal_distribution<float>(0,1.0)(rng);
        }

        inputBatch = advanceNetwork(net, i, c1, inputs);
    }

    // If you build this with -DPROFILER=1, and view the tracing data
    // in Chrome (see the "Profiling" section of README.md), you can
    // see how InitConstNeurons happens in the middle of the InitEdges
    // operations, as opposed to after them. This is the effect of
    // ParallelPart on the InitEdges operation. It may be difficult to
    // spot because InitConstNeurons takes much less time than
    // InitEdges. Zoom way in on the beginnings or ends of the
    // InitEdges job chunks, and the InitConstNeurons operation should
    // show up on one of them.

    // However, all the ConstEdgeSum operations actually run after the
    // EdgeSum operations. This is because to help grant the near-node
    // guarantee, the scheduler can't run two link operations with the
    // same near component at the same time.  ParallelPart does not
    // ensure that the operations will always be run concurrently; it
    // only grants the scheduler permission to do so if possible.
    net.perfReport("perfReport.json");
}

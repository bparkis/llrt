#include "process_link.hpp"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <fstream>

using namespace llrt;

/*
  Example to demonstrate some additional LinkTypes

  The Local2DLink connects two components in the pattern of a 2d
  convolution. You may wish to look at include/local2dlink.hpp.
  
  The AdjListLink can connect nodes in any arbitrary way, using
  adjacency lists. However, that adds some overhead compared to other
  links. You may wish to look at include/adjlistlink.hpp.
 */

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

// parameters of the Local2DLink. These terms are from 2d convolutions
// in machine learning. This will be the connectivity pattern of a
// strided convolution with a 3x3 filter size and a stride of 2, and
// "Same" padding.  It won't actually be a convolution, because our
// edges here will each have independent weights, instead of sharing
// weights. It is strided local2d connectivity.
constexpr index_t radius=1, stride=2, atrous=1;

using LTypes = std::tuple<DenseLink,
                          AdjListLink,
                          Local2DLink<radius, stride, atrous, Same> > ;
// We don't list SameLink here, despite using it. That's because
// SameLink is always included, because the selfLink of every
// Component is a SameLink.

using TL=std::pair<TTypes, LTypes>;

void initializeNetwork(Network<TL> &net){
    ProcessNetLinks_Er(net, [](IFDendrite &E, ThreadsafeRNG &r){
        E.w = std::normal_distribution<float>(0,1.0)(r);
    }, Dendrites | Parallel);
}

// Note that advanceNetwork hasn't changed at all from the last
// example, even though we'll be changing the network structure a
// lot. This is the design separation between neuron behavior and
// network connectivity described in README.md.
void advanceNetwork(Network<TL> &net, size_t timestep, Component<TL> &inputCmp, std::vector<float> &inputs){
    size_t _0 = timestep % 2;
    size_t _1 = 1 - _0;
    ProcessNetCmps_N(net, [=](IFNeuron &N){
        if (N.x[_0] == 0)
            N.v[_1] = mu * N.v[_0];
        else
            N.v[_1] = 0;
    },  KernelName("SelfPotential") | Parallel);

    ProcessCmp_NNi(inputCmp, [=, &inputs](IFNeuron &N, const size_t Ni){
        N.v[_1] += inputs[Ni];
    },  KernelName("Input") | Parallel);

    ProcessNetLinks_NEn(net, [_1, _0](IFNeuron &N, const IFDendrite &E, const IFNeuron &n){
        N.v[_1] += E.w * n.x[_0];
    }, Dendrites | KernelName("EdgeSum") | Parallel);
    
    ProcessNetCmps_Nr(net, [=](IFNeuron &N, ThreadsafeRNG &r){
        float activateProb = sigmoid(k*N.v[_1]);
        if(std::uniform_real_distribution<float>(0,1.0)(r) < activateProb)
            N.x[_1] = 1;
        else
            N.x[_1] = 0;
    }, KernelName("Activate") | Parallel);
}

int main(){
    Network<TL> net(7);

    index_t inSize = 500;
    index_t secondSize = 500;
    size_t iters = 2000;
    std::vector<index_t> thirdSize = {25, 20};
    Component<TL> &c1 = net.template component<IFNeuron>({inSize});
    Component<TL> &c2 = c1.template connect<DenseLink, NoData, IFDendrite, IFNeuron>({secondSize});

    // Here we use our AdjListLink. The third component will be
    // two-dimensional, but the AdjListLink doesn't care about that,
    // only dealing with the flattened index. The Local2DLink we use
    // later does care, however.
    Component<TL> &c3 = c2.template connect<AdjListLink, NoData, IFDendrite, IFNeuron>(thirdSize);
    // We need to manage the AdjListLink to add edges, so we get the
    // LinkType object that was just created
    auto &adjLink1 = net.template prevLinkType<AdjListLink>();
    // Let's connect the node numbered i with the nodes numbered (i-20)%500, (i-19)%500, etc. up to (i+20)%500.
    std::vector<std::pair<size_t, size_t> > adjEdges1;
    size_t thirdsizeFlat = thirdSize[0] * thirdSize[1];
    for(int i=0; i < secondSize; i++)
        for(int j=-20; j < 21; j++){
            size_t other = (i+j)%thirdsizeFlat;
            adjEdges1.push_back({i,other});
        }
    // now adjEdges1 is a vector of pairs of indexes. Each pair holds
    // an end 0 index and the corresponding end 1 index. Insert these
    // pairs into the link as edges:
    adjLink1.insertEdges(adjEdges1);

    // Next let's use a SameLink. The SameLink always connects two
    // components of the same dimensions, so it is not necessary to
    // specify the dimensions. They can be deduced automatically.
    // Again remember that we should not include the SameLink in
    // TTypes, because it is already included as the selfLink of each
    // Component.
    Component<TL> &c4 = c3.template connect<SameLink, NoData, IFDendrite, IFNeuron>();

    // Now for our Local2DLink. The dimensions here can also be
    // deduced.
    Component<TL> &c5 = c4.template connect<Local2DLink<radius, stride, atrous, Same>, NoData, IFDendrite, IFNeuron>();
    
    // Now we have five components, that we've connected only in one
    // direction. That's a little boring. Let's join up the last with
    // the first so that it's recurrent.
    c5.template connect<DenseLink, NoData, IFDendrite>(c1);

    // This will show us our network including the deduced dimensions
    // from the Local2DLink (13x10).
    //
    // It will look like:
    // 8IFNeuron_1(500) ──NoData─(Dense3)─10IFDendrite──> 8IFNeuron_2(500) ──NoData─(AdjList5)─10IFDendrite──> 8IFNeuron_3(25x20) ──NoData─(Same7)─10IFDendrite──> 8IFNeuron_4(25x20) ──NoData─(Local2D9)─10IFDendrite──> 8IFNeuron_5(13x10) ──NoData─(Dense10)─10IFDendrite──> 8IFNeuron_1(500)

    // To help make sense of it, first look at the 8IFNeuron_1,
    // 8IFNeuron_2, .... Abbreviate those C1, C2, ...  and ignore the
    // other data. Then the display looks like:
    // C1 -> C2 -> C3 -> C4 -> C5 -> C1
    // This is the network structure: from component 1 to component 5
    // and then back to 1. The other information says how that
    // happens.
    net.display();

    // The rest of the code is the same.
    initializeNetwork(net);

    std::vector<float> inputs(inSize);
    std::mt19937_64 rng(136);
    
    for (size_t i=0; i < iters; i++){
        for(size_t i=0; i < inSize; i++){
            inputs[i] = std::normal_distribution<float>(0,1.0)(rng);
        }
        
        advanceNetwork(net, i, c1, inputs);
    }
    clock_t done = clock();

    net.perfReport("perfReport.json");
}

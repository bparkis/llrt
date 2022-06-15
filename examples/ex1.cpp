// "process_link.hpp" is a generated file that contains the LLRT operations used in this program
#include "process_link.hpp"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <fstream>

using namespace llrt;

// Galves–Löcherbach integrate and fire
// https://en.wikipedia.org/wiki/Galves%E2%80%93L%C3%B6cherbach_model#Leaky_integrate_and_fire_variants
//
//           ⎧ 0  if Xᵢ[t] = 1     ⎫
// Vᵢ[t+1] = ⎨                     ⎬ + Eᵢ[t] + ∑ⱼ w_{j->i} Xⱼ[t]
//           ⎩ μᵢ Vᵢ[t] otherwise  ⎭
//
//  where Vᵢ is neuron potential
//        μᵢ is decay factor
//        Eᵢ is external input
//        w_{j->i} is edge weight
//        Xⱼ is 0 or 1 depending if the neuron fired this timestep
//
//
// Xᵢ[t+1] = 1 with probability sigmoid(k * Vᵢ[t+1])
//           0 otherwise

// This is the state of each neuron body
struct IFNeuron{
    float v[2]={0,0}; // potential on the current timestep and the next time step
    float x[2]={0,0}; // activation on the current timestep and the next time step
};

// This is the state of each edge end:  just the edge weight.
struct IFDendrite{
    float w=0;
};

template <typename T>
T sigmoid(const T x){
    return (1 / (1 + std::exp(-x)));
}

const float mu = 0.99; // potential decay
const float k = 0.01; // scale factor for activation probability

// TL gives the Tensor types (T) and Link types (L) that may be used
// in the network
// it is necessary to know this ahead of time so that the std::variant
// types used internally by llrt can be declared
using TTypes = std::tuple<IFNeuron, IFDendrite>;
using LTypes = std::tuple<DenseLink>;
using TL=std::pair<TTypes, LTypes>;

void initializeNetwork(Network<TL> &net){

    // This applies a kernel to all dendrites in the network,
    // initializing the weights to random values. "_Er" here denotes
    // the parameters to the kernel: E for the near edge-end (in this
    // case the dendrite side of a synapse), and r for a
    // ThreadsafeRNG.  A ThreadsafeRNG is a random number generator
    // that can make copies of itself to distribute to each worker
    // thread.
    ProcessNetLinks_Er(net, [](IFDendrite &E, ThreadsafeRNG &r){
        E.w = std::normal_distribution<float>(0,1.0)(r);
    }, Dendrites // run this only on the dendrite end of each link.
        | Parallel); // use multiple worker threads to run this operation
}


// This function gives the neuron behavior. It is independent of the
// network architecture. Combining the network architecture with this
// function gives the network behavior.
void advanceNetwork(Network<TL> &net, size_t timestep, Component<TL> &inputCmp, std::vector<float> &inputs){
    size_t _0 = timestep % 2; // parity of current timestep
    size_t _1 = 1 - _0; // parity of timestep we're about to calculate

// First term for Vᵢ[t+1]:
// Vᵢ[t+1] = 0  if Xᵢ[t] = 1, μᵢ Vᵢ[t] otherwise, + ...
// every node calculates its potential based on its prior activation
// value.
    ProcessNetCmps_N(net, // "_N" denotes the near node
        [=](IFNeuron &N){
            if (N.x[_0] == 0)
                N.v[_1] = mu * N.v[_0];
            else
                N.v[_1] = 0;
        }, Parallel);

// input term:
// Vᵢ[t+1] = ... + Eᵢ[t] + ...

// "_NNi" denotes the near node (N) and the flat index of the near node (Ni)
// We'll only run this operation for the designated input component
// the nodes in the input component add the input to their potential.
    ProcessCmp_NNi(inputCmp, [=, &inputs](IFNeuron &N, const size_t Ni){
        N.v[_1] += inputs[Ni];
    }, Parallel);

// Remaining terms for Vᵢ[t+1]:

// Vᵢ[t+1] = ... + ∑ⱼ w_{j->i} Xⱼ[t]
// every node sums up activations of its neighbors multiplied by the
// dendrite weights.
    ProcessNetLinks_NEn(net, // "_NEn" denotes the near-node (N), the near-edge end (E), and the far-node (n)
        [_1, _0](IFNeuron &N, const IFDendrite &E, const IFNeuron &n){
            N.v[_1] += E.w * n.x[_0];
        }, Dendrites | Parallel);
    // As mentioned before, the above operation runs "on the
    // dendrites."  In more detail: Each link can be processed from
    // either end, giving two possibilities.  When a link is processed
    // on an end, data on that end is "near" and data on the other end
    // is "far."  So, to run the operation "on the dendrites" means to
    // run it on link-ends for which the dendrites are the "near"
    // position and the axons are the "far" position.  Data can be
    // stored in four places: N (near-node), E (near edge-end), e (far
    // edge-end), and n (far-node).  Because we're running "on the
    // dendrites," the near edge-end E will be a dendrite, and the far
    // edge-end e (which we didn't happen to use here) will be an
    // axon. If we were running on the axons it would be the other way
    // around.

// Xᵢ[t+1] = 1 with probability sigmoid(k * Vᵢ[t+1])
//           0 otherwise
// every node randomly activates based on its potential.
    ProcessNetCmps_Nr(net, // "_Nr" is near-node (N), and ThreadsafeRNG (r)
        [=](IFNeuron &N, ThreadsafeRNG &r){
            float activateProb = sigmoid(k*N.v[_1]);
            if(std::uniform_real_distribution<float>(0,1.0)(r) < activateProb)
                N.x[_1] = 1;
            else
                N.x[_1] = 0;
        }, Parallel);
}

int main(){
    // 7 is the number of worker threads to spawn, which will be used
    // for Parallel operations.  You may find it efficient to spawn a
    // number of workers equal to your number of cores minus 1.
    // In addition to the 7 workers, there is a scheduler thread, and
    // this main() thread, for 9 threads total.
    Network<TL> net(7);

    // index_t is an integer type used for the tensor indices
    index_t inSize = 500, outSize = 500, iters = 2000;
    // make a component of 500 IFNeurons
    Component<TL> &c1 = net.template component<IFNeuron>({inSize});
    // connect it using a DenseLink to another component of 500
    // IFNeurons.  We use the special type NoData to indicate we don't
    // want to store anything on end 0 of this link (the axon end).
    // We'll store IFDendrites on end 1.
    Component<TL> &c2 = c1.template connect<DenseLink, NoData, IFDendrite, IFNeuron>({outSize});
    // connect the second component back to the first, with another
    // DenseLink. Again, we store NoData on end 0 (which in this case
    // is the end closest to c2) and IFDendrites on end 1.
    c2.template connect<DenseLink, NoData, IFDendrite>(c1);

    // outputs a short text description of the components and links we just created
    // The description includes type names that may vary by compiler.
    // e.g. gcc gives us
    // 8IFNeuron_1(500) ──NoData─(Dense3)─10IFDendrite──> 8IFNeuron_2(500) ──NoData─(Dense4)─10IFDendrite──> 8IFNeuron_1(500)
    //
    // The description consists of a listing of paths through the
    // graph, including component names, the size of each component,
    // the type of data stored on each end of the link, and the type
    // of link.
    net.display();

    // initialize the weights randomly
    initializeNetwork(net);

    std::vector<float> inputs(inSize);
    std::mt19937_64 rng(136);

    // run the network for 2000 time steps. Since the network has
    // 50,000 edges (not counting the nodes), this will be 1 billion
    // executions of the dendrite kernel doing
    //   N.v[_1] += E.w * n.x[_0];
    // plus a relatively small number of other kernel executions.

    for (size_t i=0; i < iters; i++){
        // we don't actually have any useful inputs to give the network
        // so we'll just supply a random vector

        // wait until the input batch is finished before we start writing to the input array
        // other batches can keep running while we write to the input array.

        for(size_t i=0; i < inSize; i++){
            inputs[i] = std::normal_distribution<float>(0,1.0)(rng);
        }

        advanceNetwork(net, i, c1, inputs);
    }

    net.perfReport("perfReport.json");
}

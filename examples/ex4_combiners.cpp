#include "process_link.hpp"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <fstream>

/*
  Example to demonstrate combiners

  In this example, we gather summary statistics from the network using
  the Combiner() JobOption.
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
        | ParallelNonBlocking);
}

size_t advanceNetwork(Network<TL> &net, size_t timestep, Component<TL> &inputCmp, std::vector<float> &inputs){
    size_t _0 = timestep % 2;
    size_t _1 = 1 - _0;
    
    ProcessNetCmps_N(net, [=](IFNeuron &N){
        if (N.x[_0] == 0)
            N.v[_1] = mu * N.v[_0];
        else
            N.v[_1] = 0;
    }, ParallelNonBlocking | KernelName("SelfPotential"));

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

    // give the network a random seed, and ask the scheduler to assign
    // job chunks to workers in a deterministic way.
    net.setDeterminism();
    net.seed(157);

    index_t inSize = 500, outSize = 500;
    size_t iters = 2000;

    Component<TL> &c1 = net.template component<IFNeuron>({inSize});
    Component<TL> &c2 = c1.template connect<DenseLink, NoData, IFDendrite, IFNeuron>({outSize});
    c2.template connect<DenseLink, NoData, IFDendrite>(c1);

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

        if (i % 20 == 0){
            //finish batches first, because we're about to run a
            //single-threaded operation and we don't want the
            //Scheduler running parallel operations at the same time.
            net.finishBatches();
            // summarize the network in a single-threaded way
            // perfectly okay, potentially a little slow
            float totalActivation=0;
            float totalPotential=0;
            ProcessNetCmps_N(net, [&totalActivation,&totalPotential](const IFNeuron &N){
                totalActivation += N.x[1];
                totalPotential += N.v[1];
            });
            std::cout << "Total activation: " << totalActivation << " Total potential: " << totalPotential << std::endl;
            
            // summarize the network in a parallel way
            // When we run the operation, this kernel will be copied
            // into each worker thread, and each copy will get the sum
            // from just the nodes assigned to that worker thread
            struct PotentialSumKernel_N{
                float totalPotential=0;
                float totalActivation=0;
                void operator()(const IFNeuron &N){
                    totalPotential += N.v[1];
                    totalActivation += N.x[1];
                }

            } PSK_N;

            // The "Combiner" will add up the results from all the
            // copies of PSK_N, putting them in the original PSK_N
            auto PSK_Combiner = [](PotentialSumKernel_N &psk_original, PotentialSumKernel_N &psk_copy){
                psk_original.totalPotential += psk_copy.totalPotential;
                psk_original.totalActivation += psk_copy.totalActivation;
            };

            ProcessNetCmps_N(net, PSK_N, Parallel | Combiner(PSK_Combiner) | KernelName("Combiner"));
            
            std::cout << "With combiners, total activation: " << PSK_N.totalActivation << " Total potential: " << PSK_N.totalPotential << std::endl;
        }
    }

    net.perfReport("perfReport.json");
}

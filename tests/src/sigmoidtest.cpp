#include "sigmoidtest.hpp"

using namespace llrt;

// The concept of this test is to run a network of sigmoid activation
// components for a few time steps, summing and recording activations
// from each step, with determinism turned on. The result of the
// network evolution, and the sums, should be *exactly* the same on
// subsequent runs.

struct SNode{
    float x[2]={0,0};
};

using SEdge = float; // the edge weight

using TL=std::pair<std::tuple<SNode, SEdge>, std::tuple<DenseLink, Local2DLink<1,1,1>, Local2DLink<2,3,2> > >;


void initializeNetwork(Network<TL> &net){
    ProcessNetCmps_Nr(net, [](SNode &N, ThreadsafeRNG &r){
        N.x[1] = std::normal_distribution<float>(0,1.0)(r);
    }, ParallelPart);

    ProcessNetLinks_Er(net, [](SEdge &E, ThreadsafeRNG &r){
        E = std::normal_distribution<float>(0,1.0)(r);
    }, ParallelNonBlocking);
}

void advanceNetwork(Network<TL> &net, size_t iter){
    size_t _0 = iter % 2;
    size_t _1 = 1 - _0;
    // sum weights
    ProcessNetLinks_NEn(net, [=](SNode &N, const SEdge &E, const SNode &n){
        N.x[_1] += n.x[_0] * E;
    }, ParallelNonBlocking);
    
    // activate, and reset the other timestep to 0
    ProcessNetCmps_Nr(net, [=](SNode &N, ThreadsafeRNG &r){
        // add some random noise to the activation, to better test determinism
        N.x[_1] = sigmoid(N.x[_1]) + 0.1 * std::normal_distribution<float>(0,1.0)(r);
        N.x[_0] = 0;
    }, ParallelNonBlocking);
}


float sumNetwork(Network<TL> &net, size_t iter){
    size_t _0 = iter % 2;
    size_t _1 = 1 - _0;
    float result1 = 0;
    // single-threaded sum
    net.finishBatches();
    ProcessNetCmps_N(net, [=,&result1](const SNode &N){
        result1 += N.x[_1];
    });

    // parallel sum with combiner (should be same)
    struct Sum_N{
        const size_t _1;
        float result = 0;
        void operator()(const SNode &N){
            result += N.x[_1];
        }
    }SN{_1};

    ProcessNetCmps_N(net, SN, Combiner([](Sum_N &orig, Sum_N &copy){
        orig.result += copy.result;
    }) | Parallel);

    REQUIRE(std::abs(result1 - SN.result) < 0.001);
    
    return result1;
}

float testSigmoidOnce(bool useSched, bool secondIter){
        Network<TL> net(useSched ? 7 : 0);
        net.seed(157);
        net.setDeterminism();
        Component<TL> &one = net.component<SNode>({100});
        Component<TL> &two = one.template connect<SameLink, SEdge, SEdge, SNode>({100});
        Component<TL> &three = two.template connect<DenseLink, SEdge, SEdge, SNode>({10, 7});
        Component<TL> &four = three.template connect<Local2DLink<1, 1, 1>, SEdge, SEdge, SNode>();
        Component<TL> &five = four.template connect<Local2DLink<2, 3, 2>, SEdge, SEdge, SNode>();

        initializeNetwork(net);

        for(size_t i=0; i < 100; i++){
            advanceNetwork(net, i);
            sumNetwork(net, i);
        }
        return sumNetwork(net, 99);
}

void sigmoidTest(){
    std::vector<std::vector<float> > cmpOneVals;
    std::vector<int> sums;
    GIVEN("Sigmoid determinism test"){
        float sum1 = testSigmoidOnce(true,false);
        float sum2 = testSigmoidOnce(true,true);
        REQUIRE(sum1 == sum2);
        assert(sum1 == sum2);
        float sum3 = testSigmoidOnce(false,false);
        float sum4 = testSigmoidOnce(false,true);
        REQUIRE(sum3 == sum4);
    }
}

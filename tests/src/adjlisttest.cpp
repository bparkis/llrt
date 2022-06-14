#include "adjlisttest.hpp"
#include "catch.hpp"
#include "process_link.hpp"
#include <vector>
#include <set>
#include <map>
#include <random>

using namespace llrt;

struct AdjResultNode{
    float local2dresult=0;
    float adjlistresult=0;
};
using AdjEdge = float;

using TTypes = std::tuple<AdjResultNode, float>;
using LTypes = std::tuple<Local2DLink<3,2,2>, Local2DLink<3,2,1>, AdjListLink>;
using TL = std::pair<TTypes, LTypes>;

// test whether we can make adjlink behave the same as local2d by inserting/removing edges to match
void test_equivalence(Component<TL> &c1, Component<TL> &c2, Link<TL> &local2d, Link<TL> &adjlink, AdjListLink &adjlinktype){
    
    std::vector<std::pair<size_t, size_t> > adjEdges;
    std::vector<std::pair<size_t, size_t> > adjEdgesToRemove;
    std::map<std::pair<size_t, size_t>, std::pair<float, float> > adjEdgeWts;

    std::map<std::pair<size_t, size_t>, int> currentAdjEdges;
    // collect the current edges of the adjlink
    int countEdgesAdj = 0;
    ProcessLink_Nini(adjlink, 0, [&](const size_t Ni, const size_t ni){
        currentAdjEdges[std::make_pair(Ni, ni)] = 0;
        countEdgesAdj++;
    });

    ProcessCmp_N(c2, [](AdjResultNode &N){
        N.local2dresult = 0;
        N.adjlistresult = 0;
    });
    
    int countEdges = 0;
    float smedges = 0;
    ProcessLink_NiEenir(local2d, 0, [&](const size_t Ni, float &E, float &e, const size_t ni, ThreadsafeRNG &r){
        countEdges++;
        E = std::uniform_real_distribution<float>(-1,1)(r);
        e = std::uniform_real_distribution<float>(-1,1)(r);
        smedges += E*2 + e;
        auto p = std::make_pair(Ni, ni);
        if(currentAdjEdges.contains(p))
            currentAdjEdges[p] = 1;
        else
            adjEdges.emplace_back(Ni, ni);
        adjEdgeWts.insert(std::make_pair(std::make_pair(Ni, ni), std::make_pair(E, e)));
    });

    for(auto &val: currentAdjEdges){
        auto key = std::get<0>(val);
        if(currentAdjEdges[key] == 0)
            adjEdgesToRemove.emplace_back(key);
    }
    adjlinktype.insertEdges(adjEdges);
    adjlinktype.removeEdges(adjEdgesToRemove);
    //REQUIRE(countEdges == countEdgesAdj - adjEdgesToRemove.size() + adjEdges.size());
    REQUIRE(countEdges == countEdgesAdj - adjEdgesToRemove.size() + adjEdges.size());

    int recount = 0;
    float smedges_2 = 0;
    ProcessLink_NiEeni(adjlink, 0, [&](const size_t Ni, float &E, float &e, const size_t ni){
        REQUIRE(adjEdgeWts.contains(std::make_pair(Ni,ni)));
        std::pair<float, float> es = adjEdgeWts[std::make_pair(Ni, ni)];
        E = std::get<0>(es);
        e = std::get<1>(es);
        smedges_2 += e + E*2;
        recount++;
    });
    REQUIRE(recount == countEdges);
    REQUIRE(std::abs(smedges - smedges_2) < 0.001f);

    ProcessCmp_Nr(c1, [](AdjResultNode &N, ThreadsafeRNG &r){
        N.local2dresult=std::uniform_real_distribution<float>(-1, 1)(r);
        N.adjlistresult=std::uniform_real_distribution<float>(-1, 1)(r);
    }, Parallel);

ProcessLink_NEen(adjlink, 1, [](AdjResultNode &N, float &E, float &e, AdjResultNode &n){
            N.adjlistresult += n.adjlistresult * e + n.local2dresult * E;
        }, Parallel);

    ProcessLink_NEen(local2d, 1, [](AdjResultNode &N, float &E, float &e, AdjResultNode &n){
        N.local2dresult += n.adjlistresult * e + n.local2dresult * E;
    }, Parallel);

    ProcessCmp_Nr(c2, [](AdjResultNode &N, ThreadsafeRNG &r){
        REQUIRE(std::abs(N.adjlistresult - N.local2dresult) < 0.001);
        N.local2dresult=std::uniform_real_distribution<float>(-1, 1)(r);
        N.adjlistresult=std::uniform_real_distribution<float>(-1, 1)(r);
    }, Parallel);

    // now ensure the end 1 to end 0 direction works as well
    ProcessCmp_N(c1, [](AdjResultNode &N){
        N.local2dresult = 0;
        N.adjlistresult = 0;
    }, Parallel);
    
    ProcessLink_NEen(adjlink, 0, [](AdjResultNode &N, float &E, float &e, AdjResultNode &n){
        N.adjlistresult += n.adjlistresult * e + n.local2dresult * E;
    }, Parallel);

    ProcessLink_NEen(local2d, 0, [](AdjResultNode &N, float &E, float &e, AdjResultNode &n){
        N.local2dresult += n.adjlistresult * e + n.local2dresult * E;
    }, Parallel);

    ProcessCmp_N(c1, [](AdjResultNode &N){
        REQUIRE(std::abs(N.adjlistresult - N.local2dresult) < 0.001);
    });
}

void adjListTest(){
    // We'll test that an AdjListLink can be used to duplicate the
    // behavior of a Local2DLink
    Network<TL> net(7);
    Component<TL> &c1 = net.template component<AdjResultNode>({100,100});
    Component<TL> &c2 = c1.template connect<Local2DLink<3,2,2>, float, float, AdjResultNode>();
    Link<TL> &local2d_1 = *c1.links[0][0];
    c1.template connect<Local2DLink<3,2,1>, float, float>(c2);
    Link<TL> &local2d_2 = *c1.links[0][1];
    c1.template connect<AdjListLink, float, float>(c2);
    Link<TL> &adjlink = *c1.links[0][2];
    

    auto & adjlinktype = net.prevLinkType<AdjListLink>();
    GIVEN("A network with AdjListLinks running alongside Local2DLinks"){

        test_equivalence(c1, c2, local2d_1, adjlink, adjlinktype);
        test_equivalence(c1, c2, local2d_2, adjlink, adjlinktype);
        adjlinktype.defragmentEdges();
        test_equivalence(c1, c2, local2d_1, adjlink, adjlinktype);
        adjlinktype.defragmentEdges();
        test_equivalence(c1, c2, local2d_2, adjlink, adjlinktype);
    }
}

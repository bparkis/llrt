#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "process_link.hpp"
#include "local2dtester.hpp"
#include "sigmoidtest.hpp"
#include "adjlisttest.hpp"

using namespace llrt;

template<typename LinkType>
std::vector<float> simpleLinkTest(std::vector<index_t> inDims, std::vector<float> inVals, std::vector<float> weights, std::vector<index_t> outDims, bool swapAxon){
    using TL=std::pair<std::tuple<float>, std::tuple<LinkType> >;
    Network<TL> net(8);
    if(!swapAxon){
        auto & A = net.template component<float>(inDims);
        auto & B = A.template connect<LinkType, float, float, float>(outDims);
        B.links[1][0]->ends[1].data.values = weights;
        A.data.values = inVals;
        ProcessLink_NEn(*B.links[1][0], 1, [](float &N, const float E, const float n){
            N += E * n;
        });
        auto & t = B.data;
        std::vector<float> &tvals = std::get<std::vector<float>>(t.values);
        return tvals;
    }
    else{
        auto & A = net.template component<float>(outDims);
        auto & B = A.template connect<LinkType, float, float, float>(inDims, false, true);
        A.links[0][0]->ends[0].data.values = weights;
        B.data.values = inVals;
        ProcessLink_NEn(*A.links[0][0], 0, [](float &N, const float E, const float n){
            N += E * n;
        });
        auto & t = A.data;
        std::vector<float> &tvals = std::get<std::vector<float>>(t.values);
        return tvals;
    }
}

std::vector<float> local2dLinkTest(std::vector<index_t> inDims, std::vector<float> inVals, std::vector<float> weights, std::vector<index_t> outDims, size_t radius, size_t stride, size_t atrous, bool swapAxon){
    using TL=std::pair<std::tuple<float>, std::tuple<GeneralLocal2DLink> >;
    Network<TL> net(8);
    
    int startRow = -static_cast<int>(radius*atrous);
    if(!swapAxon){
        auto & A = net.template component<float>(inDims);
        auto & B = A.template connect<GeneralLocal2DLink, float, float, float>(outDims);
        auto & l = net.template prevLinkType<GeneralLocal2DLink>();
        l.setParams(startRow, startRow, radius*2+1, radius*2+1, stride, stride, atrous, atrous);
        B.links[1][0]->ends[1].data.values = weights;
        A.data.values = inVals;
        ProcessLink_NEn(*B.links[1][0], 1, [](float &N, const float E, const float n){
            N += E * n;
        });
        auto & t = B.data;
        std::vector<float> &tvals = std::get<std::vector<float>>(t.values);
        return tvals;
    }
    else{
        auto & A = net.template component<float>(outDims);
        auto & B = A.template connect<GeneralLocal2DLink, float, float, float>(inDims, false, true);
        auto & l = net.template prevLinkType<GeneralLocal2DLink>();
        l.setParams(startRow, startRow, radius*2+1, radius*2+1, stride, stride, atrous, atrous);
        A.links[0][0]->ends[0].data.values = weights;
        B.data.values = inVals;
        ProcessLink_NEn(*A.links[0][0], 0, [](float &N, const float E, const float n){
            N += E * n;
        });
        auto & t = A.data;
        std::vector<float> &tvals = std::get<std::vector<float>>(t.values);
        return tvals;
    }
}




SCENARIO("Basic link tests", "[link][basic]"){
    REQUIRE(simpleLinkTest<DenseLink>({3}, {7,8,9}, {1, 2, 3, 4, 5, 6}, {2}, false) == std::vector<float>{50, 122});
    REQUIRE(simpleLinkTest<DenseLink>({2}, {7,8}, {1, 2, 3, 4, 5, 6}, {3}, true) == std::vector<float>{23, 53, 83});

    REQUIRE(local2dLinkTest({3, 3}, {1, 3, 5, 0, 2, 7, 6, 7, 1}, {8, 6, 0, 5, 9, 7, 1, 1, 9, 3, 8, 9, 3, 9, 3, 3, 10, 0, 2, 1, 9, 8, 10, 6, 0, 1, 3, 1, 6, 5, 6, 1, 0, 7, 6, 5, 5, 0, 1, 6, 8, 2, 5, 3, 9, 4, 8, 3, 7, 3, 10, 4, 9, 3, 10, 1, 7, 8, 4, 3, 8, 3, 6, 10, 2, 8, 6, 4, 7, 10, 10, 3, 2, 2, 9, 1, 6, 6, 4, 9, 2}
        , {3, 3}, 1, 1, 1, false) == std::vector<float>{53, 107, 66, 92, 112, 119, 82, 100, 117});

    REQUIRE(local2dLinkTest({3, 3}, {1, 3, 5, 0, 2, 7, 6, 7, 1}, {0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 8 , 3 , 9 , 9 , 3 , 3 , 10 , 0 , 0 , 6 , 6 , 5 , 1 , 7 , 0 , 6 , 0 , 0 , 1 , 8 , 9 , 10 , 0 , 6 , 1 , 0 , 0 , 0 , 6 , 1 , 8 , 5 , 2 , 3 , 0 , 0 , 1 , 8 , 7 , 4 , 8 , 3 , 3 , 0 , 0 , 8 , 7 , 3 , 3 , 4 , 10 , 9 , 0 , 0 , 2 , 6 , 8 , 4 , 10 , 7 , 10 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0}
        , {3, 3}, 1, 1, 1, true) == std::vector<float>{53, 107, 66, 92, 112, 119, 82, 100, 117});

    REQUIRE(local2dLinkTest({4, 4}, {5, 2, 6, 5, 10, 5, 6, 9, 0, 9, 0, 8, 10, 4, 6, 0}, {5, 0, 7, 1, 4, 9, 6, 5, 1, 4, 4, 8, 10, 1, 6, 5, 6, 4, 0, 7, 9, 3, 4, 6, 3, 9, 7, 2, 8, 5, 6, 1, 5, 6, 7, 4}
        ,{2, 2}, 1, 2, 1, false) == std::vector<float>{67, 169, 208, 217});
}

SCENARIO("2D link tests", "[link]"){
    testLocal2d();
}

SCENARIO("Sigmoid tests", "[sigmoid]"){
    sigmoidTest();
}

SCENARIO("AdjListLink tests", "[adjlist]"){
    adjListTest();
}

#include <iostream>
#include <algorithm>
#include <fstream>
#include <set>
#include <tuple>
#include <cassert>
#include <vector>
#include <random>
#include <chrono>
#include <iterator>
#include <sstream>
#include <string>
#include "local2dtester.hpp"
#include "process_link.hpp"
#include "catch.hpp"

using namespace llrt;

template<typename Kernel>
void allEdges(GeneralLocal2DLink &l, Kernel &k, int whichEnd){
    size_t iters=0;
    for(size_t end1row=0; end1row < l.end1rows; end1row++)
        for(size_t end1col=0; end1col < l.end1cols; end1col++)
            for(size_t filterRow=0; filterRow < l.filterRows; filterRow++)
                for(size_t filterCol=0; filterCol < l.filterCols; filterCol++){
                    int64_t end0row = end1row * l.strideRows + l.startRow + filterRow * l.atrousRows;
                    int64_t end0col = end1col * l.strideCols + l.startCol + filterCol * l.atrousCols;
                    if (!(end0row >= 0 && end0row < l.end0rows && end0col >= 0 && end0col < l.end0cols))
                        continue;

                    size_t edgeInfo = filterRow * l.filterCols + filterCol;
                    for(size_t depth1=0; depth1 < l.end1depth; depth1++)
                        for(size_t depth0=0; depth0 < l.end0depth; depth0++){
                            iters++;
                            size_t node0 = end0row * l.end0cols * l.end0depth // nodes from prior rows
                                + end0col * l.end0depth // prior columns from this row
                                + depth0; // prior nodes from this depth
                            size_t node1 = end1row * l.end1cols * l.end1depth
                                + end1col * l.end1depth
                                + depth1;
                            size_t edgeIndex = end1row * (l.end1cols * l.filterRows * l.filterCols * l.end0depth * l.end1depth)
                                + filterRow * (l.end1cols * l.filterCols * l.end1depth * l.end0depth)
                                + end1col * (l.filterCols * l.end1depth * l.end0depth)
                                + filterCol * (l.end1depth * l.end0depth)
                                + depth1 * (l.end0depth)
                                + depth0;
                            if (false){
                                std::cout << "end0row " << end0row << " end0col " << end0col << " depth0 " << depth0 << " end1row " << end1row << " end1col " << end1col << " depth1 " << depth1 << " edgeIndex " << edgeIndex << " node0 " << node0 << " node1 " << node1 << std::endl;
                            }
                            if (whichEnd == 0)
                                k(node0, edgeIndex, node1, edgeIndex, edgeInfo);
                            else
                                k(node1, edgeIndex, node0, edgeIndex, edgeInfo);

                        }
                }
}

void test(GeneralLocal2DLink &l, int whichEnd, std::vector<int> values, std::mt19937_64 &g){
    
    size_t maxP = l.maxProgress(0);

    size_t firstRequestedProgress=std::uniform_int_distribution<size_t>(1, maxP)(g);
    size_t secondRequestedProgress = 0;
    if (firstRequestedProgress < maxP){
        secondRequestedProgress=std::uniform_int_distribution<size_t>(1, maxP-firstRequestedProgress)(g);
    }
    //     size_t Ni, ni, Ei, edgeInfo;
    using info = std::tuple<size_t, size_t, size_t, size_t>;

    struct Collector{
        std::multiset<info> seen;
        void operator()(size_t Ni, size_t Ei, size_t ni, size_t ei, size_t edgeInfo){
            seen.insert(std::make_tuple(Ni, ni, Ei, edgeInfo));
        }
    } c1, c2;

    allEdges(l, c1, whichEnd);

    if(firstRequestedProgress == 0)
        firstRequestedProgress=1;
    size_t firstProgress = l.requestPartialProgress(whichEnd, firstRequestedProgress);
    if (firstProgress > maxP)
        firstProgress = maxP;
    l(whichEnd, c2, 0, firstProgress);
    size_t secondProgress=0;
    if (firstProgress < maxP){
        if (secondRequestedProgress == 0)
            secondRequestedProgress = 1;
        secondProgress = l.requestPartialProgress(whichEnd, firstProgress+secondRequestedProgress);
        if (secondProgress > maxP)
            secondProgress = maxP;
        l(whichEnd, c2, firstProgress, secondProgress);
        if (secondProgress < maxP)
            l(whichEnd, c2, secondProgress, maxP);
    }

    for(const info &f : c1.seen){
        REQUIRE(c1.seen.count(f) == c2.seen.count(f));
    }
    for(const info &f : c2.seen){
        REQUIRE(c1.seen.count(f) == c2.seen.count(f));
    }
    //assert(c1.seen == c2.seen);
}


void testLocal2d(){
    GeneralLocal2DLink l;
    struct DummyTensorWrapper : VariantVectorWrapper{
        virtual void apply(void *capture, void(*f)(void *, AnyVector&)){
        }
    }t0, t1;

    l.setLinkData(t0, t1);

    std::vector<std::vector<int> > testSetup =
        {{1, 2, 3, 4, 5}, // filter rows 0
         {1, 2, 3, 4, 5}, // filter cols 1
         {1, 2, 3}, // stride rows       2
         {1, 2, 3}, // stride cols       3
         {1, 2, 3}, // atrous rows       4
         {1, 2, 3}, // atrous cols       5
         {1, 2, 5, 6, 10}, // end 1 rows 6
         {1, 2, 5, 6, 10}, // end 1 cols 7
         {1, 2, 3}, // end 1 depth       8
         {1, 2, 5, 6, 10}, // end 0 rows 9
         {1, 2, 5, 6, 10}, // end 0 cols 10
         {1, 2, 3}, // end 0 depth       11
         {0, -1, -3, 1, 3}, // start row 12
         {0, -1, -3, 1, 3}, // start column 13
         {0, 1} };  // whichEnd          14
    //
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::mt19937_64 generator(seed);
    for(size_t i=0; i < 1000; i++){
        std::vector<int> config(testSetup.size());
        for(size_t j=0; j < testSetup.size(); j++){
            size_t rval = std::uniform_int_distribution<size_t>(0,testSetup[j].size()-1)(generator);
            config[j] = testSetup[j][rval];
        }

        
        // l.filterRows = config[0];
        // l.filterCols = config[1];
        // l.strideRows = config[2];
        // l.strideCols = config[3];
        // l.atrousRows = config[4];
        // l.atrousCols = config[5];
        // l.end1rows = config[6];
        // l.end1cols = config[7];
        // l.end1depth = config[8];
        // l.end0rows = config[9];
        // l.end0cols = config[10];
        // l.end0depth = config[11];
        // l.startRow = config[12];
        // l.startCol = config[13];

        l.setParams(config[12], config[13], config[0], config[1], config[2], config[3], config[4], config[5]);
        l.setDimensions({static_cast<index_t>(config[9]), static_cast<index_t>(config[10]), static_cast<index_t>(config[11])}, {static_cast<index_t>(config[6]), static_cast<index_t>(config[7]), static_cast<index_t>(config[8])});
        test(l, config[14], config, generator);
    }
}

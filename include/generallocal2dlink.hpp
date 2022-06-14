#ifndef GENERALLOCAL2DLINK_HPP_
#define GENERALLOCAL2DLINK_HPP_

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
#include <any>


namespace llrt{

    /**
       A locally connected 2D link.

       This link has the connectivity pattern of a convolution.  It
       corresponds to several layer types from machine learning:
       convolutions, atrous convolutions, transpose convolutions,
       transpose atrous convolutions, locally connected 2d layers,
       transpose locally connected 2d layers, atrous locally connected
       2d layers, transpose atrous locally connected 2d layers, even
       pooling layers.  It's possible to use the Local2DLink in
       combination with custom kernels to implement any of the above.

       Each component may have either 2 dimensions: (rows, columns) or
       3 dimensions: (rows, columns, depth).  If depth is not given,
       it is treated as 1.  When a 2D cell (r0,c0) at end0 is
       connected with a 2D cell (r1,c1) at end1, all end0 nodes at any
       depth at (r0,c0) are fully connected with all end1 nodes at any
       depth at (r1,c1).

       The link iteration moves a filter over end0 of the link.  The
       filter is centered at a particular 2D cell, and is a rectangle
       of the specified dimensions.  Each position of the filter
       connects the end0 cells covered by the filter with a single
       end1 2D cell.

       The filter moves in steps of size strideRows (vertically) and
       strideCols (horizontally).  If either stride is greater than 1,
       then end1 will be smaller than end0.

       If atrousRows>1 or atrousCols>1, the filter will be spread out, as in an
       atrous convolution.  The filter shape changes like this:

                    atrous==1:  xxx         atrous==2:  x x x
                                xxx
                                xxx                     x x x

                                                        x x x

       This may be useful if you wish to combine information from far
       away parts of the image, without reducing the size of the image
       or having a large and computationally intensive filter.

       The edgeInfo values for an end1 node are in left-to-right,
       top-to-bottom order. For example, if filterRows == filterCols
       == 3, then they will be numbered:

                  0 1 2
                  3 4 5
                  6 7 8

       so that edge 0 of a node at end1 is to the end1 node northwest
       of it, edge 5 of a node at end1 is to the end1 node east of it,
       etc. These numbers may be used as indices into a convolution
       kernel (not included).
       
       This link works row by row, and can divide up a task into sets
       of rows.  It is a bit slower at the sides of each row, because
       it must check bounds there, and because there may be function
       call overhead for each row.  Thus, it's best to have long rows.
       If one dimension of your input is much longer than another, for
       best performance the longer dimension should be the second one
       (the columns dimension) so that the rows have many columns.

    */

    struct GeneralLocal2DLink : public BaseLinkType{
        virtual std::string identifier(){
            return "GeneralLocal2D";
        }

        virtual bool canConnectDimensions(const std::vector<index_t> &dim0, const std::vector<index_t> &dim1){
            if(dim0.size() != 2 && dim0.size() != 3)
                return false;
            if(dim1.size() != 2 && dim1.size() != 3)
                return false;
            return true;
        }

        virtual bool deduceComponentDimensions(const std::vector<index_t> &dimF, std::vector<index_t> &result, int whichEnd){
            return false; // can't deduce dimensions
        }

        static const bool needsLink = true;

        VariantVectorWrapper *end0LinkData=nullptr, *end1LinkData=nullptr;

        bool linkDataWasSet = false;

        virtual void setLinkData(VariantVectorWrapper &valuesEnd0, VariantVectorWrapper &valuesEnd1){
            end0LinkData = &valuesEnd0;
            end1LinkData = &valuesEnd1;
            linkDataWasSet = true;
        }

        // Configured parameters of the filter
        int startRow=0; // top left corner of the top-left filter placed on end 0
        int startCol=0; // can be positive or negative

        size_t filterRows=0, filterCols=0;
        size_t strideRows=0, strideCols=0;
        size_t atrousRows=0, atrousCols=0;

        size_t end1rows=0;
        size_t end1cols=0;
        size_t end1depth=0;
        size_t end0rows=0;
        size_t end0cols=0;
        size_t end0depth=0;
        // end configured parameters of the filter

        bool dirty = false;
        void setParams(const int startRow_, const int startCol_, const size_t filterRows_, const size_t filterCols_, const size_t strideRows_, const size_t strideCols_, const size_t atrousRows_, const size_t atrousCols_){
            startRow = startRow_;
            startCol = startCol_;
            filterRows = filterRows_;
            filterCols = filterCols_;
            strideRows = strideRows_;
            strideCols = strideCols_;
            atrousRows = atrousRows_;
            atrousCols = atrousCols_;

            dirty = true;
            initialize();
        }

        virtual void setDimensions(const std::vector<index_t> &dim0, const std::vector<index_t> &dim1){
            end0rows = dim0.at(0);
            end0cols = dim0.at(1);
            if (dim0.size() == 3)
                end0depth = dim0.at(2);
            else
                end0depth = 1;

            end1rows = dim1.at(0);
            end1cols = dim1.at(1);
            if (dim1.size() == 3)
                end1depth = dim1.at(2);
            else
                end1depth = 1;
            dirty = true;
            initialize();
        }

        virtual std::vector<index_t> linkEndSize(const std::vector<index_t> &dimN, const std::vector<index_t> &dimF, int whichEnd){
            const std::vector<index_t> &dim1 = whichEnd == 1 ? dimN : dimF;
            const std::vector<index_t> &dim0 = whichEnd == 0 ? dimN : dimF;
            size_t end1depth = 1;
            if (dim1.size() == 3)
                end1depth = dim1[2];
            size_t end0depth = 1;
            if (dim0.size() == 3)
                end0depth = dim0[2];
            size_t size = dim1[0] * dim1[1] * end1depth * end0depth * filterRows * filterCols;

            std::vector<index_t> v{size};
            return v;
        }

        void resize(){
            std::vector<index_t> dimN{end1rows, end1cols, end1depth};
            std::vector<index_t> dimF{end0rows, end0cols, end0depth};
            size_t size = linkEndSize(dimN, dimF, 1).at(0);

            auto rsz = [&](VariantVectorWrapper *v){
                v->apply(&size, [](void *size, AnyVector &v2){
                    v2.resize(*static_cast<size_t *>(size));
                });
            };

            rsz(end0LinkData);
            rsz(end1LinkData);
        }

        std::string showParams(){
            std::stringstream st("", std::ios_base::app | std::ios_base::out);;
            st << "end0 " << end0rows << "x" << end0cols << "x" << end0depth << " end1 " << end1rows << "x" << end1cols << "x" << end1depth << " start " << startRow << "," << startCol << " filter " << filterRows << "x" << filterCols << " stride " << strideRows << "," << strideCols << " atrous " << atrousRows << "," << atrousCols << std::endl;
            return st.str();
        }

        /**
           Round a/b towards neg inf.  a may be positive or negative but b must be positive.
        */
        static int64_t div_round_neginf(int64_t a, int64_t b){
            return a >= 0 || a%b == 0 ? a/b : a/b-1;
        }

        /**
           Round a/b towards pos inf.  a may be positive or negative but b must be positive.
        */
        static int64_t div_round_posinf(int64_t a, int64_t b){
            if (a >= 0){
                if (a%b == 0){
                    return a/b;
                }
                else
                    return a/b+1;
            }
            else{
                if ((-a)%b == 0){
                    return a/b;
                }
                else{
                    return a/b + 1;
                }
            }
        }

        template<typename Kernel>
        void RowRowIteration(size_t filterRow, size_t end1row, Kernel &k, bool end1){


            int64_t end0row = end1row * strideRows + filterRow * atrousRows + startRow;

            if (end0row < 0 || end0row >= end0rows)
                return; // nothing to do, filter location is outside array bounds

            size_t edgeInfoStart = filterRow * filterCols, edgeInfoEnd = filterRow * filterCols + filterCols - 1;


            size_t end0BaseRowIx = end0row * end0cols * end0depth;

            size_t end1BaseRowIx = end1row * end1cols * end1depth;

            size_t edgeIx = end1row * (end1cols * filterRows * filterCols * end0depth * end1depth) // from end1 complete rows above this
                + filterRow * (end1cols * filterCols * end1depth * end0depth); // from complete filter rows above this
            int64_t curLeftSideFilter = startCol;

            for(int64_t end1col=0; end1col < end1cols; end1col++){
                size_t edgeInfo = edgeInfoStart;
                for(int64_t end0col=curLeftSideFilter; end0col < curLeftSideFilter + static_cast<int64_t>(filterCols*atrousCols); end0col += atrousCols){
                    if(end0col < 0 || end0col >= end0cols){
                        edgeInfo++;
                        edgeIx += end0depth * end1depth;
                        continue; // out of bounds
                    }
                    size_t end0BaseDepthIx = end0BaseRowIx + end0col*end0depth;
                    size_t end1BaseDepthIx = end1BaseRowIx + end1col*end1depth;
                    for(size_t i=0; i < end1depth; i++){
                        for(size_t j=0; j < end0depth; j++){
                            size_t end0ix = end0BaseDepthIx + j;
                            size_t end1ix = end1BaseDepthIx + i;

                            if(end1)
                                k(end1ix, edgeIx, end0ix, edgeIx, edgeInfo);
                            else
                                k(end0ix, edgeIx, end1ix, edgeIx, edgeInfo);
                            edgeIx++;
                        }
                    }
                    edgeInfo++;
                }
                curLeftSideFilter += strideCols;
            }
        }

        template<typename Kernel>
        void RowFindingIteration(size_t end0row_start, size_t end0row_end, Kernel &k){

            // end0row_start = end1row_start * strideRows + filterRow * atrousRows + startRow
            // end1row_start = (end0row_start - filterRow * atrousRows - startRow) / strideRows
            int64_t end1row_start_signed = div_round_neginf(static_cast<int64_t>(end0row_start) - static_cast<int64_t>(startRow) - static_cast<int64_t>(filterRows * atrousRows), static_cast<int64_t>(strideRows));
            size_t end1row_start;
            if (end1row_start_signed < 0)
                end1row_start=0;
            else
                end1row_start = end1row_start_signed;
            end1row_start = std::min(end1row_start, end1rows-1);

            //end0row_end = end1row_end * strideRows + startRow
            // end1row_end = (end0row_end - startRow)/strideRows
            int64_t end1row_end_signed = div_round_posinf(static_cast<int64_t>(end0row_end) - static_cast<int64_t>(startRow), static_cast<int64_t>(strideRows));
            size_t end1row_end;
            if (end1row_end_signed < 0)
                end1row_end = 0;
            else
                end1row_end = end1row_end_signed;
            end1row_end = std::min(end1row_end, end1rows);

            for(size_t end1row = end1row_start; end1row < end1row_end; end1row++){
                for(size_t filterRow=0; filterRow < filterRows; filterRow++){
                    int64_t end0row = end1row * strideRows + filterRow * atrousRows + startRow;
                    if (end0row >= end0row_start && end0row < end0row_end){
                        RowRowIteration(filterRow, end1row, k, 0);
                    }
                }
            }
        }

        std::vector<size_t> cumulativeEnd0RowSizes, cumulativeEnd1RowSizes;

        void initialize(){
            if (!dirty)
                return;
            if (end1rows == 0 || filterRows == 0) // incomplete params
                return;
            cumulativeEnd0RowSizes.clear();
            cumulativeEnd1RowSizes.clear();
            cumulativeEnd0RowSizes.resize(end0rows,0);
            cumulativeEnd1RowSizes.resize(end1rows,0);
            size_t rowrowsize=0;
            for(size_t end1row=0; end1row < end1rows; end1row++){
                for(size_t filterRow=0; filterRow < filterRows; filterRow++){
                    int64_t end0row = static_cast<int64_t>(end1row * strideRows) + startRow + static_cast<int64_t>(filterRow * atrousRows);
                    if (end0row >= 0 && end0row < end0rows){
                        if (rowrowsize == 0){
                            struct RowRowFinder{
                                size_t &rowrowsize;
                                void operator()(size_t, size_t, size_t, size_t, size_t){
                                    rowrowsize++;
                                }
                            } rrs{rowrowsize};
                            RowRowIteration(filterRow, end1row, rrs, 1);
                        }
                        cumulativeEnd0RowSizes[end0row] += rowrowsize;
                        cumulativeEnd1RowSizes[end1row] += rowrowsize;
                    }
                }
            }
            size_t cumulative = 0;
            for(size_t end1row=0; end1row < end1rows; end1row++){
                size_t tmp = cumulativeEnd1RowSizes[end1row];
                cumulativeEnd1RowSizes[end1row] += cumulative;
                cumulative += tmp;
            }
            cumulative = 0;
            for(size_t end0row=0; end0row < end0rows; end0row++){
                size_t tmp = cumulativeEnd0RowSizes[end0row];
                cumulativeEnd0RowSizes[end0row] += cumulative;
                cumulative += tmp;
            }
            resize();
            dirty = false;
        }

        virtual size_t maxProgress(int){
            return cumulativeEnd0RowSizes.at(cumulativeEnd0RowSizes.size()-1);
        }

        virtual size_t requestPartialProgress(int whichEnd, size_t requestedProgress){
            std::vector<size_t> &arr = whichEnd == 0 ? cumulativeEnd0RowSizes : cumulativeEnd1RowSizes;
            if (arr.empty())
                return 0;
            // This std function is misnamed. std::lower_bound returns the least upper bound for requestedProgress within arr
            auto result = std::lower_bound(arr.begin(), arr.end(), requestedProgress);
            if (result == arr.end())
                return arr.back();
            return *result;
        }


        template<typename Kernel>
        void operator()(int whichEnd,
                        Kernel &k,
                        size_t start,
                        size_t end){

            if(whichEnd == 1){
                auto it = std::lower_bound(cumulativeEnd1RowSizes.begin(), cumulativeEnd1RowSizes.end(), start+1);
                size_t end1row_start = std::distance(cumulativeEnd1RowSizes.begin(), it);
                it = std::lower_bound(cumulativeEnd1RowSizes.begin(), cumulativeEnd1RowSizes.end(), end);
                it++;
                size_t end1row_end = std::distance(cumulativeEnd1RowSizes.begin(), it);
                for(size_t end1row=end1row_start; end1row < end1row_end; end1row++){
                    for(size_t filterRow=0; filterRow < filterRows; filterRow++){
                        RowRowIteration(filterRow, end1row, k, 1);
                    }
                }
            }
            else{
                auto it = std::lower_bound(cumulativeEnd0RowSizes.begin(), cumulativeEnd0RowSizes.end(), start+1);
                size_t end0row_start = std::distance(cumulativeEnd0RowSizes.begin(), it);
                it = std::lower_bound(cumulativeEnd0RowSizes.begin(), cumulativeEnd0RowSizes.end(), end);
                it++;
                size_t end0row_end = std::distance(cumulativeEnd0RowSizes.begin(), it);

                RowFindingIteration(end0row_start, end0row_end, k);
            }
        }
    };
}

#endif

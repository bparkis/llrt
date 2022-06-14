#ifndef LOCAL2DLINK_HPP_
#define LOCAL2DLINK_HPP_

#include "generallocal2dlink.hpp"

namespace llrt{
    enum PaddingTypes{
        Same, /// Same padding: if stride==1, end1 will be the same size as end0
        Valid /// Valid padding: the filter will always be fully contained within end0
    };

    /**
       A Local2DLink is a GeneralLocal2DLink with a square filter, and
       parameters given as template parameters, for convenience.

       @tparam filterSize is the side length of the square filter
       @tparam stride is how many squares to move the filter,
       vertically or horizontally, each time it is applied
       @tparam atrous is how spread-out the filter is, vertically or
       horizontally
       @tparam padding is either Same or Valid
     */
    template<size_t filterSize, size_t stride=1, size_t atrous=1, PaddingTypes padding=Same>
    struct Local2DLink : public GeneralLocal2DLink{
        static constexpr int start = padding == Same ?
            -static_cast<int>((filterSize/2)*stride) : 0;
        
        Local2DLink() : GeneralLocal2DLink(){
            GeneralLocal2DLink::setParams(start, start, filterSize, filterSize, stride, stride, atrous, atrous);
        }

        virtual bool canConnectDimensions(const std::vector<index_t> &dim0, const std::vector<index_t> &dim1){
            if (dim0.size() != 2 && dim0.size() != 3)
                return false;
            if (dim1.size() != 2 && dim1.size() != 3)
                return false;
            if (padding == Same){
                if ((dim0[0] - 1) / stride + 1 != dim1[0])
                    return false;
                if ((dim0[1] - 1) / stride + 1 != dim1[1])
                    return false;
                return true;
            }
            assert(padding == Valid);
            if ((dim0[0] - filterSize) / stride + 1 != dim1[0])
                return false;
            if ((dim0[1] - filterSize) / stride + 1 != dim1[1])
                return false;
            return true;

        }
        
        virtual bool deduceComponentDimensions(const std::vector<index_t> &dimF, std::vector<index_t> &result, int whichEnd){
            if(dimF.size() != 2 && dimF.size() != 3)
                return false;
            // we'll assume channel depth for the output is the same
            // as for the input.
            if(whichEnd == 1){
                // dimF == dim0, result = dim1
                result = dimF;
                if (padding == Same){
                    result[0] = (dimF[0] - 1) / stride + 1;
                    result[1] = (dimF[1] - 1) / stride + 1;
                }
                else{
                    assert(padding == Valid);
                    result[0] = (dimF[0] - filterSize) / stride + 1;
                    result[1] = (dimF[1] - filterSize) / stride + 1;
                }
                return true;
            }
            else{
                // dimF == dim1, result = dim0
                result = dimF;
                // a variety of dimensions could work here
                if (padding == Same){
                    // (dim0[0] - 1) / stride + 1 = dim1[0]
                    // .: dim0[0] = (dim1[0] - 1) * stride + 1
                    result[0] = (dimF[0] - 1) * stride + 1;
                    result[1] = (dimF[1] - 1) * stride + 1;
                }
                else{
                    assert(padding == Valid);
                    // (dim0[0] - filterSize) / stride + 1 = dim1[0]
                    // .: dim0[0] = (dim1[0] - 1) * stride + filterSize
                    result[0] = (dimF[0] - 1) * stride + filterSize;
                    result[1] = (dimF[1] - 1) * stride + filterSize;
                }
                return true;
            }
        }
        
        virtual std::string identifier(){
            return "Local2D";
        }

    };
}

#endif

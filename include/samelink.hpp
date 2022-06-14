// this file is included at the end of linktypes.hpp and may therefore
// refer to definitions from there.

#ifndef SAMELINK_HPP_
#define SAMELINK_HPP_

namespace llrt{
    /**
       A "same"-link:  links (i,j,...) with (i,j,...)
       You don't need to, and should not, include SameLink as part of TL.
       It's there already because it's on the selfLink of each Component.
    */
    struct SameLink : public BaseLinkType{
        virtual std::string identifier(){
            return "Same";
        }

        virtual bool canConnectDimensions(const std::vector<index_t> &dim0, const std::vector<index_t> &dim1){
            return dim0 == dim1;
        }

        virtual bool deduceComponentDimensions(const std::vector<index_t> &dimF, std::vector<index_t> &result, int whichEnd){
            result = dimF;
            return true;
        }

        std::vector<index_t> dim0, dim1;

        virtual void setDimensions(const std::vector<index_t> &dim0_, const std::vector<index_t> &dim1_){
            dim0.clear();
            dim0.insert(dim0.end(), dim0_.begin(), dim0_.end());
            dim1.clear();
            dim1.insert(dim1.end(), dim1_.begin(), dim1_.end());
        }
        
        virtual std::vector<index_t> linkEndSize(const std::vector<index_t> &dimN, const std::vector<index_t> &dimF, int whichEnd){
            std::vector<index_t> v = dimN;
            v.push_back(1);
            return v;
        }

        virtual size_t maxProgress(int whichEnd){
            return std::accumulate(dim0.begin(), dim0.end(), 1, std::multiplies<>());
        }

        virtual size_t requestPartialProgress(int whichEnd, size_t requestedProgress){
            return requestedProgress;
        }

        template<typename Kernel>
        void operator()(int whichEnd,
                        Kernel &k,
                        size_t start,
                        size_t end
            ){
            for(size_t i = start; i < end; i++){
                k(i, i, i, i, 0);
            }
        }
    };
}

#endif

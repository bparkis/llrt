// this file is included at the end of linktypes.hpp and may therefore
// refer to definitions from there.

#ifndef DENSELINK_HPP_
#define DENSELINK_HPP_
namespace llrt{
    /**
       A Dense link, that connects every node in one component with
       every node in the other component.
    */
    struct DenseLink : public BaseLinkType{
        virtual std::string identifier(){
            return "Dense";
        }

        virtual bool canConnectDimensions(const std::vector<index_t> &dim0, const std::vector<index_t> &dim1){
            return true;
        }

        virtual bool deduceComponentDimensions(const std::vector<index_t> &dimF, std::vector<index_t> &result, int whichEnd){
            return false; // can't deduce dimensions
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
            v.push_back(std::accumulate(dimF.begin(), dimF.end(), 1, std::multiplies<>()));
            return v;
        }

        virtual size_t maxProgress(int whichEnd){
            size_t dim0tot = std::accumulate(dim0.begin(), dim0.end(), 1, std::multiplies<>());
            size_t dim1tot = std::accumulate(dim1.begin(), dim1.end(), 1, std::multiplies<>());
            return dim0tot*dim1tot;
        }

        virtual size_t requestPartialProgress(int whichEnd, index_t requestedProgress){
            auto &dimF = whichEnd == 0 ? dim1 : dim0;
            size_t dimFtot = std::accumulate(dimF.begin(), dimF.end(), 1, std::multiplies<>());
            if(requestedProgress == 0)
                return dimFtot;
            return ((requestedProgress-1)/dimFtot)*dimFtot+dimFtot; // next whole near node
        }

        template<typename Kernel>
        void operator()(int whichEnd,
                        Kernel &k,
                        size_t start,
                        size_t end
            ){
            auto &dimN = whichEnd == 0 ? dim0 : dim1;
            auto &dimF = whichEnd == 0 ? dim1 : dim0;
            size_t vNsize = std::accumulate(dimN.begin(), dimN.end(), 1, std::multiplies<>());
            size_t vFsize = std::accumulate(dimF.begin(), dimF.end(), 1, std::multiplies<>());
            size_t vNlink = start;
            for(size_t i = start/vFsize; i < end/vFsize; i++){
                size_t vFlink = i;
                for(size_t j = 0; j < vFsize; j++){
#ifdef DEBUG_EDGE_LEVEL
                        std::cout << "Dense link at " << i << ", " << j << std::endl;
#endif
                    k(i, vNlink, j, vFlink, j);
                    vNlink++;
                    vFlink += vNsize;
                }
            }
        }
    };
}
#endif

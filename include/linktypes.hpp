#ifndef LINKTYPES_HPP_
#define LINKTYPES_HPP_
#include "common.hpp"

// core LinkTypes

namespace llrt{

    struct BaseLinkType{
        /**
           @return the name of this link type
         */
        virtual std::string identifier() = 0;

        /**
           @param dim0 dimensions of component on end 0
           @param dim1 dimensions of component on end 1
         
           @return true if a Component of dimensions dim0 at end 0 can connect to a Component of dimensions dim1 at end 1
         */
        virtual bool canConnectDimensions(const std::vector<index_t> &dim0, const std::vector<index_t> &dim1) = 0;

        /**
           Deduce the dimensions of the near Component, if the far Component has dimensions dimN
         
           @param dimF the input dimensions of the far Component
           @param result the output dimensions of the near Component
           @param whichEnd 0 or 1:  which link end is the near Component
         
           @return false if the Component dimensions at the far Component can't be deduced,
           or if dimF are invalid dimensions for the far Component.
         */
        virtual bool deduceComponentDimensions(const std::vector<index_t> &dimF, std::vector<index_t> &result, int whichEnd) = 0;

        /**
           Inform the link about the dimensions of the components it connects
         
           @param dim0 dimensions of component on end 0
           @param dim1 dimensions of component on end 1
         */
        virtual void setDimensions(const std::vector<index_t> &dim0, const std::vector<index_t> &dim1) = 0;

        /**
           Obtain the dimensions of a link end.
         
           @param dimN the dimensions of the Component whose LinkEnd's dimensions will be found
         
           @param dimF the dimensions of the other Component on the Link.
           dimN and dimF must already be known valid for this Link.
         
           @param whichEnd 0 or 1:  which LinkEnd dimN corresponds to.
         
           @return the dimensions of the LinkEnd at position whichEnd
         */
        virtual std::vector<index_t> linkEndSize(const std::vector<index_t> &dimN, const std::vector<index_t> &dimF, int whichEnd) = 0;

        /**
           @return the amount of progress at which the iteration is complete
         */
        virtual size_t maxProgress(int whichEnd){
            return 100;
        }

        /**
           This function may be used if the link needs to manage some
           aspects of the data stored on the link ends, such as
           reordering or resizing that data when parameters of the
           link change. The network will call it when the link is
           created.
         */
        virtual void setLinkData(VariantVectorWrapper &valuesEnd0, VariantVectorWrapper &valuesEnd1){
            
        }

        /**
           If the job scheduler wants the link iterator to run until requestedProgress,
           return the minimum progress level the iterator can do that's at least as large
           as requestedProgress.
         */
        virtual size_t requestPartialProgress(int whichEnd, size_t requestedProgress){
            return maxProgress(whichEnd); // by default, the iterator can't split the job up
        }

        // All LinkTypes must also implement operator() with the following signature:
        //
        // template<typename Kernel>
        // void operator()(const std::vector<index_t> &dimN,
        //                 const std::vector<index_t> &dimF,
        //                 int whichEnd,
        //                 Kernel &k,
        //                 size_t start,
        //                 size_t end
        //     );
        //
        //
        // operator() must iterate over edges, and call:
        //  k(near_node_index,
        //    near_edge_index,
        //    far_node_index,
        //    far_edge_index,
        //    edgeInfo)
        //
        // for every edge in the Link with progress >= start and
        // progress < end.  progress is an arbitrary number that
        // starts from 0, and when progress == maxProgress(...), the
        // link is finished iterating.  (progress roughly corresponds
        // to the number of edges processed so far).
        //
        // edgeInfo is a size_t describing what kind of edge it is.
        // For a dense link, edgeInfo gives the index of the far node.
        // For local2d, edgeInfo gives the index into a convolution
        // kernel.
        // 
        // If a link supports partial execution, operator() must only
        // divide the task on whole near-node boundaries!!!
        // otherwise it will create a race condition when multithreading
        // This is necessary for the near-node guarantee
        //
        // See examples, e.g. denselink.hpp
        //
        // Note that operator() can't be a virtual function because
        // it's templated.  Which is important for performance because
        // it allows the compiler to generate a copy of operator() for
        // each Kernel k it's called with, which in turn allows the
        // compiler to inline the calls to k.  Unless k is a function
        // pointer, because then the Kernel type does not identify the
        // specific function, so the compiler can't inline it.  So,
        // for performance, don't use function pointers as kernels.
        // Use lambdas or custom callables ("functors").
    };
}

#include "samelink.hpp"
#include "denselink.hpp"
// the following links are included at the end of network.hpp instead of here, for dependency reasons
// #include "generallocal2dlink.hpp"
// #include "local2dlink.hpp"
// #include "adjlistlink.hpp" 

#endif

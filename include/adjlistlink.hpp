// this file is included at the end of network.hpp

#ifndef ADJLISTLINK_HPP_
#define ADJLISTLINK_HPP_

namespace llrt{
/**
   a link that connects nodes based on configurable adjacency lists

   To create and populate this link, do something like:
   Component<TL> &c2 = c1.template connect<AdjListLink<TTypes>, MyEdge, MyEdge, MyNode>({100});
   AdjListLink &adj = net.prevLinkType<AdjListLink>();
   adj.insertEdges({{10, 12}, {13, 3}, {5, 3}});  list of {end0 node index, end1 node index} pairs

   Now the link is ready to go. If you want to remove edges, you can do:
   adj.removeEdges({{10, 12}, {5, 3}});

   And if you've removed a lot of edges, you can optionally clean up a
   bit and reduce memory afterwards by calling:
   adj.defragmentEdges();

   Do not change the edges at the same time as a link operation is
   running on the link, as it is not threadsafe.
*/
    struct AdjListLink : public BaseLinkType{
        virtual std::string identifier(){
            return "AdjList";
        }

        virtual bool canConnectDimensions(const std::vector<index_t> &dim0, const std::vector<index_t> &dim1){
            return true;
        }

        virtual bool deduceComponentDimensions(const std::vector<index_t> &dimF, std::vector<index_t> &result, int whichEnd){
            return false; // can't deduce dimensions
        }

        std::vector<index_t> dim0, dim1;

        struct NeighborIndices{
            // the index of the near edge and the far edge are always the same
            // both given by edgeIx
            size_t edgeIx, farNode;
        };

        std::vector<std::vector<NeighborIndices> > end0Adjacency;
        std::vector<std::vector<NeighborIndices> > end1Adjacency;

        virtual void setDimensions(const std::vector<index_t> &dim0_, const std::vector<index_t> &dim1_){
            dim0.clear();
            dim0.insert(dim0.end(), dim0_.begin(), dim0_.end());
            dim1.clear();
            dim1.insert(dim1.end(), dim1_.begin(), dim1_.end());
            size_t v0size = std::accumulate(dim0.begin(), dim0.end(), 1, std::multiplies<>());
            size_t v1size = std::accumulate(dim1.begin(), dim1.end(), 1, std::multiplies<>());
            end0Adjacency.resize(v0size);
            end1Adjacency.resize(v1size);
        }

        static const bool needsLink = true;

        VariantVectorWrapper *end0LinkData, *end1LinkData;

        // the size of the edge data arrays
        // This will be larger than the actual number of edges,
        // if some have been deleted since the last call to recopyEdges
        size_t edgeIxBound = 0;

        // end0CumulativeEdgeCounts[i] is the total number of edges incident to nodes on end 0
        // with index <= i.
        //
        std::vector<size_t> end0CumulativeEdgeCounts;
        std::vector<size_t> end1CumulativeEdgeCounts;

        bool dirty = false;

        void resetCumulativeEdgeCounts(){
            if (!dirty)
                return;
            auto f = [&](std::vector<size_t> &counts, std::vector<std::vector<NeighborIndices> > &adjacency){
                counts.resize(adjacency.size());
                size_t count = 0;
                for(size_t i=0; i < adjacency.size(); i++){
                    count += adjacency[i].size();
                    counts[i] = count;
                }
            };
            f(end0CumulativeEdgeCounts, end0Adjacency);
            f(end1CumulativeEdgeCounts, end1Adjacency);
            dirty = false;
        }

        virtual void setLinkData(VariantVectorWrapper &valuesEnd0, VariantVectorWrapper &valuesEnd1){
            end0LinkData = &valuesEnd0;
            end1LinkData = &valuesEnd1;
        }

        // template<typename TL>
        // void setLink(Link<TL> &l){
        //     end0LinkData = &l.ends[0].data;
        //     end1LinkData = &l.ends[1].data;
        // }

        // true if an edgeIx has been destructed, false otherwise
        std::vector<bool> destructedStatus;

        /**
           insert a set of edges
           This operation may cause the edge Tensor data to be moved
           when the underlying vector is resized, invalidating
           pointers to them.  However, the indices of the edge Tensor
           data will not change.

           @param nodeIndices is a vector of pairs. The first element
           of each pair is the end 0 index, and the second element is
           the end 1 index.
        */
        void insertEdges(const std::vector<std::pair<size_t, size_t> > &nodeIndices){
            // TODO: if >50% of edges have been deleted, start inserting new edges
            // by overwriting deleted ones from the beginning, instead of adding them
            // at the end
            for(auto &pair : nodeIndices){
                size_t end0ix = std::get<0>(pair);
                size_t end1ix = std::get<1>(pair);
                end0Adjacency.at(end0ix).push_back({static_cast<size_t>(edgeIxBound), end1ix});
                end1Adjacency.at(end1ix).push_back({static_cast<size_t>(edgeIxBound), end0ix});
                edgeIxBound++;
            }
            
            static_assert(std::is_same_v<decltype(edgeIxBound), size_t>);
            end0LinkData->apply(&edgeIxBound, [](void * edgeIxBound, AnyVector &v){
                v.resize(*static_cast<size_t *>(edgeIxBound));
            });
            end1LinkData->apply(&edgeIxBound, [](void * edgeIxBound, AnyVector &v){
                v.resize(*static_cast<size_t *>(edgeIxBound));
            });
            // end0LinkData->resize({edgeIxBound});
            // end1LinkData->resize({edgeIxBound});
            destructedStatus.resize(edgeIxBound, false);
            dirty = true;
        }

        template <typename T>
        struct VecTypeUnpacker;

        template <typename T>
        struct VecTypeUnpacker<std::vector<T> >{
            using type = T;
        };


        void callEdgeDataDestructor(size_t edgeIx){

            // I mean, it's declared on the line above, but we're
            // about to cast a void*, so take no chances.
            static_assert(std::is_same_v<decltype(edgeIx), size_t>);
            end0LinkData->apply(&edgeIx, [](void * edgeIx, AnyVector &v){
                v.refreshIndex(*static_cast<size_t *>(edgeIx));
            });
            end1LinkData->apply(&edgeIx, [](void *edgeIx, AnyVector &v){
                v.refreshIndex(*static_cast<size_t *>(edgeIx));
            });
            
            // if(!end0LinkData->noData)
            //     std::visit([&](auto && vec){
            //         using T = VecTypeUnpacker<std::decay_t<decltype(vec)>>::type;
            //         vec.at(edgeIx) = T();
            //         //vec.at(edgeIx).~T();
            //     },end0LinkData->values);
            // if(!end1LinkData->noData)
            //     std::visit([&](auto && vec){
            //         using T = VecTypeUnpacker<std::decay_t<decltype(vec)>>::type;
            //         vec.at(edgeIx) = T();
            //         //vec.at(edgeIx).~T();
            //     },end1LinkData->values);
            destructedStatus[edgeIx] = true;
        }

        /**
           Remove a set of edges.  Removed edges are destructed
           in-place without moving any edges.  Indices and pointers to
           the remaining edges remain valid.

           @param nodeIndices is a vector of pairs. The first element
           of each pair is the end 0 index, and the second element is
           the end 1 index.
        */
        void removeEdges(const std::vector<std::pair<size_t, size_t> > &nodeIndices){
            for(auto &pair : nodeIndices){
                size_t end0ix = std::get<0>(pair);
                size_t end1ix = std::get<1>(pair);
                std::vector<NeighborIndices> &v0 = end0Adjacency.at(end0ix);
                std::vector<NeighborIndices> &v1 = end1Adjacency.at(end1ix);
                for(auto i=v0.begin(); i!= v0.end(); i++){
                    NeighborIndices &ixs = *i;
                    if (ixs.farNode == end1ix){
                        size_t edgeIx = ixs.edgeIx;
                        v0.erase(i);
                        callEdgeDataDestructor(edgeIx);
                        break;
                    }
                }
                for(auto i=v1.begin(); i != v1.end(); i++){
                    NeighborIndices &ixs = *i;
                    if (ixs.farNode == end0ix){
                        v1.erase(i);
                        break;
                    }
                }
            }
            dirty = true;
        }

        /**
           After removing some edges, you may optionally call this function to make the link more efficient
           by copying existing edges to the beginning of the edge Tensor, overwriting the removed ones,
           reducing the memory footprint.
           This invalidates any pointers to the remaining edges as well as changing the indices of the remaining edges.
        */
        void defragmentEdges(){
            // TODO: defragment with a better access pattern, by rearranging the edges in the order of an end0 iteration,
            // with neighbors of each end0 node sorted by end1 node index
            std::vector<size_t> partialSums(edgeIxBound);
            size_t edgeCount=0;
            for(size_t i=0; i < edgeIxBound; i++){
                if(!destructedStatus[i])
                    edgeCount++;
                partialSums[i] = edgeCount;
            }
            // now partialSums[i] is the number of not-deleted edges between indices 0 and i inclusive
            // so if i is a not-deleted edge, partialSums[i]-1 is its new index
            for(auto &a : end0Adjacency)
                for(NeighborIndices & ixs : a)
                    ixs.edgeIx = partialSums[ixs.edgeIx]-1;

            for(auto &a : end1Adjacency)
                for(NeighborIndices & ixs : a)
                    ixs.edgeIx = partialSums[ixs.edgeIx]-1;

            edgeIxBound = edgeCount;

            // auto dataMover = [&](Tensor<TType> *linkData){

            //     if(!linkData->noData)
            //         std::visit([&](auto && vec){
                        // size_t j=0;
                        // // an edge index i is not-deleted if either i==0 and partialSums[i] = 1,
                        // // or i > 0 and partialSums[i] > partialSums[i-1]
                        // for (size_t i=0; i < vec.size(); i++){
                        //     if (partialSums[i] > j){
                        //         j = partialSums[i];
                        //         vec[partialSums[i]-1] = std::move(vec[i]);
                        //     }
                        // }
                        // vec.resize(edgeCount);
            //         },linkData->values);
            // };

            struct Capturer{
                std::vector<size_t> &partialSums;
                size_t edgeCount;
            }cap{partialSums, edgeCount};

            auto dataMover = [&](VariantVectorWrapper *linkData){
                linkData->apply(&cap, [](void *cap_, AnyVector &vec){
                    size_t j=0;
                    Capturer *cap = static_cast<Capturer *>(cap_);
                    // an edge index i is not-deleted if either i==0 and partialSums[i] = 1,
                    // or i > 0 and partialSums[i] > partialSums[i-1]
                    for (size_t i=0; i < vec.size(); i++){
                        
                        if (cap->partialSums[i] > j){
                            j = cap->partialSums[i];
                            vec.move(i, cap->partialSums[i]-1);
                        }
                    }
                    vec.resize(cap->edgeCount);
                    
                });
            };

            dataMover(end0LinkData);
            dataMover(end1LinkData);
            destructedStatus.resize(edgeIxBound);
            std::fill(destructedStatus.begin(), destructedStatus.end(), false);
        }

        virtual std::vector<index_t> linkEndSize(const std::vector<index_t> &dimN, const std::vector<index_t> &dimF, int whichEnd){
            return {0}; // user can increase this by adding edges
        }

        virtual size_t maxProgress(int whichEnd){
            resetCumulativeEdgeCounts();
            if (end0CumulativeEdgeCounts.empty())
                return 0;
            return end0CumulativeEdgeCounts.back(); // total number of edges
        }

        virtual size_t requestPartialProgress(int whichEnd, index_t requestedProgress){
            resetCumulativeEdgeCounts();
            std::vector<size_t> &arr = whichEnd == 0 ? end0CumulativeEdgeCounts : end1CumulativeEdgeCounts;
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
                        size_t end
            ){
            std::vector<std::vector<NeighborIndices> > &adj = whichEnd == 0 ? end0Adjacency : end1Adjacency;
            std::vector<size_t> &arr = whichEnd == 0 ? end0CumulativeEdgeCounts : end1CumulativeEdgeCounts;
            size_t progress = start;

            size_t ix = std::lower_bound(arr.begin(), arr.end(), start+1) - arr.begin();
            while(ix < adj.size()){
                progress += adj[ix].size();
                if (progress > end)
                    break;
                size_t f=0;
                for(NeighborIndices &ixs : adj[ix]){
#ifdef DEBUG_EDGE_LEVEL
                    std::cout << "AdjList link at " << ix << ", " << ixs.farNode << std::endl;
#endif
                    k(ix, ixs.edgeIx, ixs.farNode, ixs.edgeIx, f++);
                }
                ix++;
            }
        }
    };
}
#endif

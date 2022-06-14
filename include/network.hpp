/**
   network.hpp
  
    Created on: Oct 27, 2019
        Author: bparkis
  
   DESIGN OVERVIEW
   
   A Network stores a few Components, e.g. expected to be <100.
   
   The Network structure is expected to remain unchanged after its creation.
   
   A Component stores data. The data is a Tensor, a multidimensional
   array.  The data is intended to be large, with millions of
   entries and operations on it
   
   A Component also stores two sets of Links:
   the Links where the Component is in position 0,
   the Links where the Component is in position 1.
   It also stores a selfLink that loops back to the same Component.
   
   The purpose of a Link is to specify the connection pattern between
   indices of two components.  The connection pattern can be described
   as a set of edges.  (edges are not directly named in this file; I
   introduce them for explanatory purposes only).  For example, if
   indices (1, 5, 7) of Component 1 happen to connect to indices (2,
   5) of Component 2, then (1,5,7) --- (2,5) would be an edge
   specified by the Link.  A particular piece of data on a Component,
   corresponding to a set of indices on that Component, will be
   described as a node.  (again, for explanatory purposes only)
   Components and Links describe the large-scale structure of the
   Network, and nodes and edges, which have little formal existence in
   the code, describe the small-scale structure of the Network.
   
   A Link has two LinkEnds.
   one LinkEnd is for the Component in position 0 of the Link
   the other LinkEnd is for the Component in position 1 of the Link
   
   The distinction between position 0 and position 1 is based on the
   connection pattern for the Link.  For example, a Link that connects
   in the pattern of a strided convolution would have a larger
   Component always at position 0 and a reduced-size Component always
   at position 1.  Links always support a transfer of information in
   both directions, even when there is a distinction between their
   ends.  To implement a strided convolution Link, it would also be
   required to implement the transpose strided convolution pattern for
   going the other way.
   
   A LinkEnd stores a Component reference, and data. Each index into a
   LinkEnd's data represents information attached to an edge end and
   owned/managed by the Component of that LinkEnd.
     
   Therefore a LinkEnd's data will often be several times larger than
   the corresponding Component's data.
  
   A Link has a type, which specifies the pattern of connectivity
   between the two Components of the Link.  Link type is an
   std::variant of LinkTypes, which is a template parameter.  The
   possible types in LinkTypes are all derived from BaseLinkType.
   (There's a good reason to use std::variant here: it means the
   possible types of the link are known at compile time, which allows
   the use of templates to generate inlined code for the inner loops.)
   
   All Links in the same Network have the same possible set of LinkTypes.
   
   The Link type specifies only the pattern of connectivity, not the
   operation performed, in contrast to layers in ML frameworks.

   The operation performed is called a "kernel." This is a function to be
   applied to every edge in the link from one of the ends.
     
   This means that the compiler must generate a 2 dimensional grid of
   inlined functions, one function for each possible combination of
   kernel type and Link type, which is achieved through the use of
   templates.
   
   A Tensor stores values, which is an std::variant<std::vector<TensorTypes>...> >.
     TensorTypes is a template parameter, which can be a list of arbitrary data types
   All Tensors in the same Network have the same possible set of TensorTypes.
   A Tensor also has a list of its dimensions.


        Network
          ├───────────┬───────────┬ ...
         Component1 Component2  ...
            │
            ├── Tensor data
            ├────────────────┬───────────────╮
           links[0]         links[1]        selfLink
            ├───────┬── ...  ├──────┬── ...
           Link1  Link2     Link3 Link4
            │
            ├── std::variant<LinkTypes...> type
            │                               ╰ each LinkType inherits from BaseLinkType
            ├──────────────────────────────╮
           ends[0]                        ends[1]
            │                              │
           LinkEnd1                       LinkEnd2
            ├ Tensor data                  ├ Tensor data
            ╰ Component1                   ╰ the Component linked to Component1
    
           Tensor
            │
            ├── std::variant<std::vector<TensorTypes>...> values
            ╰── std::vector<int64_t> dimensions
          
        
   
 */

#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <vector>
#include <algorithm>
#include <memory>
#include <numeric>
#include <cassert>
#include <exception>
#include <variant>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include <thread>
#include <atomic>
#include <functional>
#include <random>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include "common.hpp"
#include "linktypes.hpp"
#include "function_traits.hpp"
#include "scheduler.hpp"
#include "network_perf_logger.hpp"

namespace llrt{

    // used to indicate a variant vector will not be initialized with values
    struct NoData{};

    template <typename TL>
    struct TLTypes;

    template <typename ...Ts, typename ...Ls>
    struct TLTypes<std::pair<std::tuple<Ts...>, std::tuple<Ls...> > >{
        using TType = std::tuple<Ts...>;
        using LType = std::tuple<Ls...>;
    };

    template <typename T>
    struct UnpackTType;

    template <typename ...Ts>
    struct UnpackTType<std::tuple<Ts...> >{
        // type for data stored in Tensors
        using varType = std::variant<std::vector<Ts>...>;
    };

    template <typename L>
    struct UnpackLType;
    
    template <typename ...Ls>
    struct UnpackLType<std::tuple<Ls...> >{
        // type of links
        using varType = std::variant<SameLink, Ls...>;
    };

    
/**
   Variable-type tensor.
 */
    template <typename TType>
    struct Tensor{
        UnpackTType<TType>::varType values;
        std::vector<index_t> dimensions;
        bool initialized = false;
        bool noData = true;

        size_t num_values;

        void resize(const std::vector<index_t> &dims){
            dimensions = dims;
            num_values = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<>());
            if (!noData)
                std::visit([&](auto && vec){vec.resize(num_values);},values);
        }

        template <typename T>
        struct ValueNameHelper;

        template <typename T>
        struct ValueNameHelper<std::vector<T> >{
            std::string s = typeid(T).name();
        };

        std::string valueTypeName(){
            if (noData)
                return "NoData";
            return std::visit([](auto &&arg){
                using T = std::decay_t<decltype(arg)>;
                return ValueNameHelper<T>().s;
            }, values);
        }

        template <typename T>
        void initialize(){
            assert(!initialized);
            if constexpr(!std::is_same_v<T, NoData>){
                values = std::vector<T>(num_values, T());
                noData = false;
            }
            initialized = true;
        }

        /**
           Given a multi-dimensional index, flatten it into an index
           into the values vector For efficiency, it's not recommended
           to rely on this when implementing LinkTypes.  It serves as a
           specification of what the indices mean.

           Specifically, the indices are in column-major format.
           An increment in the last index increases the flat index by
           1, and an increment in any index increases the flat index
           by the product of the dimensions that follow that position.
         */
        size_t flattenIndex(const std::vector<index_t> && indices){
            size_t flat_ix = 0;
            int mult = 1;
            for(index_t i = indices.size()-1; i >= 0; i++){
                index_t ii = indices[i];
                if (!(0 <= ii && ii < dimensions[i]))
                    throw "Invalid dimensions";
                flat_ix += ii*mult;
                mult*=dimensions.at(i);
            }
            return flat_ix;
        }

        void clear(){
            std::visit([](auto && vec){Tensor::_clear(vec);},values);
        }

        template<typename T>
        static void _clear(std::vector<T> & vec){
            std::fill(vec.begin(), vec.end(), T());
        }

        ~Tensor(){
        }

        struct TensorWrapper : VariantVectorWrapper{
            Tensor<TType> &t;

            TensorWrapper(Tensor<TType> &t) : t(t){}

            template <typename VecT>
            struct VecTGetter;

            template <typename T>
            struct VecTGetter<std::vector<T> >{
                using ItemType = T;
            };

            virtual void apply(void *capture, void (*f)(void *, AnyVector &)){
                if (t.noData)
                    return;
                std::visit([&f, capture](auto &&arg){
                    using VecT = std::decay_t<decltype(arg)>;
                    using T = VecTGetter<VecT>::ItemType;
                    TVectorWrapper<T> wrap(arg);
                    f(capture, wrap);
                }, t.values);
            }
        };

        TensorWrapper wrapper;

        Tensor(const std::vector<index_t> &dims) : wrapper(*this){
            dimensions = dims;
            num_values = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<>());
        }
    };

    // template<typename TTypes>
    // struct TensorWrapper : VariantVectorWrapper{
    //     Tensor<TTypes> &t;
    //     TensorWrapper(Tensor<TTypes> &t) : t(t) {}

    //     template <typename VecT>
    //     struct VecTGetter;

    //     template <typename T>
    //     struct VecTGetter<std::vector<T> >{
    //         using ItemType = T;
    //     };

    //     virtual void apply(void (*f)(AnyVector &)){
    //         if (t.noData)
    //             return;
    //         std::visit([&f](auto &&vec){
    //             using VecT = std::decay_t<decltype(arg)>;
    //             using T = VecTGetter<VecT>::ItemType;
    //             TVectorWrapper<T> wrap(arg);
    //             f(wrap);
    //         }, t.values);

    //     }
    // };

    template<typename TL>
    struct Network;

    template<typename TL>
    struct Component;

    template<typename TL>
    struct Link;

    /**
       One end of a Link. Stores the data.
    */
    template<typename TL>
    struct LinkEnd{
        using TType = TLTypes<TL>::TType;
        Tensor<TType> data;
        Component<TL> & c;
        Link<TL> &l;
        size_t whichEnd=0;
        /// one LinkEnd is an axon, and the other is a dendrite
        /// (entirely optional whether a link operation cares about this)
        /// by default end 0 is the axon and end 1 is the dendrite
        bool _isAxon;
        bool isAxon() const{
            return _isAxon;
        }
        bool isDendrite() const{
            return !_isAxon;
        }
        LinkEnd(const std::vector<index_t> &dims, Component<TL> & c, Link<TL> &l, size_t whichEnd) :
            data(dims), c(c), l(l), whichEnd(whichEnd){}

    };

    /**
       A Link is two LinkEnds, plus the link type that gives the
       iteration pattern.
     */
    template<typename TL>
    struct Link{
        LinkEnd<TL> ends[2];
        using LType = TLTypes<TL>::LType;
        UnpackLType<LType>::varType type;
        
        std::string name;
        size_t id;
        template<typename LinkType>
        Link(Component<TL> &c1, LinkType h, Component<TL> &c2, bool swapAxon, size_t numericId) :
            ends{LinkEnd<TL>(h.linkEndSize(c1.data.dimensions, c2.data.dimensions, 0),c1,*this,0),
            LinkEnd<TL>(h.linkEndSize(c2.data.dimensions, c1.data.dimensions, 1),c2,*this,1)},
            type(h),
            id(numericId)
            {
                ends[0]._isAxon = true != swapAxon;
                ends[1]._isAxon = false != swapAxon;
                name = identifier() + "_" + std::to_string(id);

                std::visit([&](auto &&t){
                    
                    t.setLinkData(ends[0].data.wrapper, ends[1].data.wrapper);
                    t.setDimensions(c1.data.dimensions, c2.data.dimensions);
                }, type);

            }

        const std::string cmpIdentifier(int whichEnd){
            return ends[whichEnd].c.type.identifier();
        }

        std::string identifier(){
            return std::visit([](auto && arg){return arg.identifier();}, type);
        }

        std::string endName(int whichEnd){
            return name + "_" + std::to_string(whichEnd);
        }

        template<typename T>
        std::vector<T> & compData(int whichEnd){
            return std::get<std::vector<T> >(ends[whichEnd].c.data.values);
        }

        template<typename T>
        std::vector<T> & linkData(int whichEnd){
            return std::get<std::vector<T> >(ends[whichEnd].data.values);
        }

        size_t getMaxProgress(int whichEnd){
            return std::visit(
                [whichEnd, this](auto &&arg){
                    return arg.maxProgress(whichEnd);
                }, type);
        }

    };

    /**
       A random number generator that is threadsafe for use in a ProcessLink operation.
      
       It has two properties:
       1. It is to be used within a kernel, so each kernel copy in each different thread gets its own generator.
          This prevents race conditions when different threads access the same generator.
       2. When copied, the copy generates a different sequence from the original.  This makes it actually
          (pseudo-)random when used as above.
      
          This generator is obtained within a kernel via the "g" kernel parameter specifier.
     */
    struct ThreadsafeRNG{
        std::mt19937_64 *baseRNG;
        ThreadsafeRNG(){
            baseRNG = new std::mt19937_64(std::chrono::system_clock::now().time_since_epoch().count());
        }
        ~ThreadsafeRNG(){
            delete baseRNG;
        }
        ThreadsafeRNG(const ThreadsafeRNG &mg) {
            unsigned seed = mg.baseRNG->operator()();
            baseRNG = new std::mt19937_64(seed);
        }
        void seed(size_t n){
            baseRNG->seed(n);
        }
        unsigned operator()(){
            return baseRNG->operator()();
        }
        static constexpr unsigned min(){
            return std::numeric_limits<unsigned>::min();
        }
        static constexpr unsigned max(){
            return std::numeric_limits<unsigned>::max();
        }
    };


    /**
       A collection of Components and Links, plus the Scheduler necessary
       to run parallel operations on them
     */
    template<typename TL>
    struct Network{

        std::vector<std::unique_ptr<Component<TL> > > components;

        ThreadsafeRNG rng;

        std::optional<Scheduler> sched;

        NetworkPerfLogger npl;

        size_t linkCounter = 0;


        Network(int nWorkers=0) : sched(nWorkers == 0 ?
                                        std::optional<Scheduler>() :
                                        std::optional<Scheduler>(nWorkers)) {}

        /**
           Wait until the scheduler has finished executing all batches
           assigned to it.
         */
        void finishBatches();

        /**
           Wait unti lthe scheduler has finished executing the batch
           with the given batchNumber.
         */
        void finishBatch(size_t batchNumber);
        void displayChain(Link<TL> &l, int i, std::set<Link<TL> *> &ls);

        /**
           Display a short text summary of the network structure
         */
        void display();

        /**
           Tell the scheduler to disable adaptive scheduling, so that
           it will be deterministic.
         */
        void setDeterminism();
        
        /**
           Seed the ThreadsafeRNG
         */
        void seed(size_t rngSeed);

        /**
           Give a summary of how long the network has been alive, and
           how many kernels have been executed in that time.  If the
           network was compiled with the PROFILER preprocessor
           definition, then also dump the collected performance data
           to a file.
         */
        void perfReport(const std::string dumpFilename="perfreport.json");

        int cmpId=0;
        size_t linkId=0;

        /**
           Create a new Component<TL>.

           @tparam NodeType the new component will be filled with
           instances of this type, unless it is the special type
           NoData
           @param dims the dimensions of the new component
         */
        template <typename NodeType>
        Component<TL> & component(const std::vector<index_t> &dims);

        // last created link type
        using LType = TLTypes<TL>::LType;
        UnpackLType<LType>::varType * _thatLink;

        template <typename LinkType>
        LinkType & prevLinkType(){
            return std::get<LinkType>(*_thatLink);
        }

        bool initialized = false;

        void initialize();

    };

    template<typename TL>
    struct Component{

        /// links[0] are links for which we are in position 0 and the other Component is in position 1
        /// links[1] are links for which we are in position 1 and the other is in position 0
        std::vector<std::shared_ptr<Link<TL>> > links[2];

        std::unique_ptr<Link<TL>> selfLink;

        Network<TL> &net;

        using TType=TLTypes<TL>::TType;
        Tensor<TType> data;

        std::string name;

        int id;

        const std::vector<index_t> & getDimensions() const {
            return data.dimensions;
        }


        const size_t dataSize() const{
            return data.num_values;
        }

        /**
           shows name and dimensions
        */
        std::string displayName(){
            std::string dName = name + "(";
            bool start=true;
            for(index_t i : getDimensions()){
                if (start){
                    dName += std::to_string(i);
                    start = false;
                }
                else
                    dName += "x" + std::to_string(i);
            }
            return dName + ")";
        }

        Component(Network<TL> &net, const std::vector<index_t> & dims) : net(net), data(dims){
            id = ++net.cmpId;
            selfLink = std::make_unique<Link<TL>>(*this, SameLink(), *this, false, ++net.linkId);
            selfLink->ends[0].data.template initialize<NoData>();
        }

        /**
           Create a new component of specified type, linked to this
           one by the specified LinkType, deduce the new component's
           dimensions based on the Linktype and this component's
           dimensions, and add the new component to the Network.
          
           @tparam LinkType gives the connectivity pattern of the new link.
           @tparam EdgeType0 end 0 of the new link will be filled with
           instances of this type, unless it is the special type NoData
           @tparam EdgeType1 end 1 of the new link will be filled with
           instances of this type, unless it is the special type NoData
           @tparam NodeType the new cmp will be filled with instances
           of this type, unless it is the special type NoData
           @param swapEnds normally this Component is end 0 and the
           other is end 1.  If swapEnds is true, then this component
           is end 1 and the other is end 0.
           @param swapAxon normally end 1 is the axon and end 0 is the
           dendrite.  If swapAxon is true, then end 0 is the axon and
           end 1 is the dendrite.

          
           @return the new component
         */
        template<typename LinkType, typename EdgeType0, typename EdgeType1, typename NodeType>
        Component<TL> & connect(bool swapEnds = false, bool swapAxon = false);

        /**
           Create a new component of specified type, linked to this
           one by the specified LinkType, of the specified dimensions,
           and add the new component to the Network.
          
           @tparam LinkType gives the connectivity pattern of the new link.
           @tparam EdgeType0 end 0 of the new link will be filled with
           instances of this type, unless it is the special type NoData
           @tparam EdgeType1 end 1 of the new link will be filled with
           instances of this type, unless it is the special type NoData
           @tparam NodeType the new cmp will be filled with instances
           of this type, unless it is the special type NoData
           @param swapEnds normally this Component is end 0 and the
           other is end 1.  If swapEnds is true, then this component
           is end 1 and the other is end 0.
           @param swapAxon normally end 1 is the axon and end 0 is the
           dendrite.  If swapAxon is true, then end 0 is the axon and
           end 1 is the dendrite.

           @return the new component
         */
        template<typename LinkType, typename EdgeType0, typename EdgeType1, typename NodeType>
        Component<TL> & connect(const std::vector<index_t> &newDims, bool swapEnds = false, bool swapAxon = false);

        /**
           Overload to create a new component of specified type,
           linked to this one by the specified LinkType, of the
           specified dimensions, in which the dimensions are given as
           an std::initializer_list.
        */
        template<typename LinkType, typename EdgeType0, typename EdgeType1, typename NodeType>
        Component<TL> & connect(const std::initializer_list<index_t> &newDims, bool swapEnds = false, bool swapAxon = false);

        /**
           Create a new link to an existing component.  Throws an
           exception if the dimensions are invalid based on the op.
          
           @tparam LinkType gives the connectivity pattern of the new link.
           @tparam EdgeType0 end 0 of the new link will be filled with
           instances of this type, unless it is the special type NoData
           @tparam EdgeType1 end 1 of the new link will be filled with
           instances of this type, unless it is the special type NoData
           @param otherComponent is the other component
           @param swapEnds normally this Component is end 0 and the
           other is end 1.  If swapEnds is true, then this component
           is end 1 and the other is end 0.
           @param swapAxon normally end 1 is the axon and end 0 is the
           dendrite.  If swapAxon is true, then end 0 is the axon and
           end 1 is the dendrite.

           @return the other component
         */
        template<typename LinkType, typename EdgeType0, typename EdgeType1>
        Component<TL> & connect(Component<TL> &otherComponent, bool swapEnds = false, bool swapAxon = false);
    };

    template<typename TL, typename Kernel>
    void ProcessLink(Link<TL> & link, int whichEnd, Kernel && k, index_t start, index_t end
#ifdef DEBUG_OP_LEVEL                     
                     , std::string kernelName
#endif
);

/////////////// Operation Options ///////////////////

/**
  Designed so the user may say ProcessLink_N(*l, 0, MyOp, ParallelNonBlocking | KernelName("BestKernel") | Combiner([](MyType & x, MyTypes &y){x.sum += y.sum;}));
*/ 

    template<typename Ks>
    struct JobOptions;

    template<typename K1, typename K2, typename K3>
    struct JobOptions<std::tuple<K1, K2, K3> >{
        bool parallel = false;
        bool blocking = true;
        bool endOfBatch = true;
        std::string kernelName = "";
        bool onlyAxons = false;
        bool onlyDendrites = false;
        K1 combiner;
        K2 cmpNearFilter;
        K3 cmpFarFilter;
    };

    // NullOptionType defined in common.hpp
    using NOpT3 = std::tuple<NullOptionType, NullOptionType, NullOptionType>;

    const JobOptions<NOpT3> NullJobOptions;

    // Parallel, ParallelPart, ParallelNonBlocking

    /**
       This option makes the operation run in parallel on multiple
       worker threads. See README.md
     */
    const JobOptions<NOpT3> Parallel{true};

    /**
       This option makes the operation run in parallel on multiple
       worker threads, and return immediately rather than waiting for
       the worker to finish. See README.md
    */
    const JobOptions<NOpT3> ParallelNonBlocking{true, false};

    /**
       This option submits an operation to the scheduler as part of a
       batch with more to follow. The operation will wait in a queue
       until the next Parallel or ParallelNonBlocking operation, at
       which point the whole batch will be scheduled and begin
       execution.
     */
    const JobOptions<NOpT3> ParallelPart{true, false, false};

    /**
       Filter the edge-ends so this operation runs only on the axons.
     */
    const JobOptions<NOpT3> Axons{false, true, true, "", true};
    
    /**
       Filter the edge-ends so this operation runs only on the dendrites.
     */
    const JobOptions<NOpT3> Dendrites{false, true, true, "", false, true};

    /**
       Give a name to the kernel, for performance reporting purposes.
    */
    const JobOptions<NOpT3> KernelName(std::string const name);

    /**
       Specify a Combiner for your kernel k, to combine results of
       execution in the different worker threads.

       @param combiner a function with signature void(K &k1, K &k2)
       that takes the data in k2 and merges it into k1, where k1 is
       the kernel you supplied and k2 is a copy of it.
     */
    template<typename C>
    auto Combiner(C combiner){
        return JobOptions<std::tuple<C, NullOptionType, NullOptionType> >{false, true, true, "", false, false, combiner};
    }

    /**
       Only run the operation on near components matching the filter.

       @param nearFilter a function with signature bool(Component<TL> &c).
       Returns true if you *do* want to execute on this near component,
       false otherwise.
     */
    template<typename NF>
    auto NearCmpFilter(NF nearFilter){
        return JobOptions<std::tuple<NullOptionType, NF, NullOptionType> >{false, true, true, "", false, false, NullOption, nearFilter};
    }

    /**
       Only run the operation on the near component with this exact name.
       
       @param name the name of the near component, as can be found on the
       Network<TL>::display summary
    */
    template<typename TL>
    auto NearCmpName(std::string name){
        return NearCmpFilter([name](Component<TL> &c){c.name == name;});
    }

    /**
       Only run the operation if the far component has this exact name.
       
       @param name the name of the far component, as can be found on the
       Network<TL>::display summary
    */
    template<typename TL>
    auto FarCmpName(std::string name){
        return FarCmpfilter([name](Component<TL> &c){c.name == name;});
    }

    /**
       Only run the operation with far components matching the filter.

       @param nearFilter a function with signature bool(Component<TL> &c).
       Returns true if you *do* want to execute with this far component,
       false otherwise.
     */
    template<typename FF>
    auto FarCmpFilter(FF farFilter){
        return JobOptions<std::tuple<NullOptionType, NullOptionType, FF> >{false, true, true, "", false, false, NullOption, NullOption, farFilter};
    }
    
    template<typename K>
    K & unifyKernels(K &k, NullOptionType){
        return k;
    }

    NullOptionType unifyKernels(NullOptionType c, NullOptionType);

    template<typename K>
    K & unifyKernels(NullOptionType, K &k){
        return k;
    }

    template<typename K1, typename K2>
    struct unifiedKernelType;

    template<typename K1>
    struct unifiedKernelType<K1, NullOptionType>{
        using K = K1;
    };
    template<typename K2>
    struct unifiedKernelType<NullOptionType, K2>{
        using K = K2;
    };
    template<>
    struct unifiedKernelType<NullOptionType, NullOptionType>{
        using K = NullOptionType;
    };


    template <typename K1, typename K1_, typename K2, typename K2_, typename K3, typename K3_>
    auto operator|(JobOptions<std::tuple<K1, K2, K3> > op1, JobOptions<std::tuple <K1_, K2_, K3_> > op2)
    {
        return JobOptions<std::tuple< typename unifiedKernelType<K1, K1_>::K,
                                      typename unifiedKernelType<K2, K2_>::K,
                                      typename unifiedKernelType<K3, K3_>::K > >{
            op1.parallel || op2.parallel,
            op1.blocking && op2.blocking,
            op1.endOfBatch && op2.endOfBatch,
            op1.kernelName + op2.kernelName,
            op1.onlyAxons || op2.onlyAxons,
            op1.onlyDendrites || op2.onlyDendrites,
            unifyKernels(op1.combiner, op2.combiner),
            unifyKernels(op1.cmpNearFilter, op2.cmpNearFilter),
            unifyKernels(op1.cmpFarFilter, op2.cmpFarFilter)};
    }

////////////////////////////////////////////////////

    /**
       Get a human-readable name of the kernel for performance reports
       or debugging.
    */
    template<typename Kernel>
    std::string getKernelName(){
        std::string kernelStr;
        if(strlen(typeid(Kernel).name()) < 20)
            kernelStr = typeid(Kernel).name();
        else{
            // probably a lambda of a templated class: long and garbled. So just use a number.
            kernelStr = std::to_string(typeid(Kernel).hash_code());
        }
        return kernelStr;
    }

    /**
       Execute a link operation. Depending on settings, this may be
       sent to the scheduler for later parallel execution, or executed
       immediately in this thread. Called by functions in
       process_link.hpp

       @param link the link on which to execute
       @param whichEnd 0 or 1, the end of the link on which to execute
       @param k the kernel supplied by the user, in kernel parameter
       specifier format
       @param pk a kernel that wraps a copy of the kernel supplied by
       the user, suitable for execution by the ProcessLink function
       @param pk_ref like pk, but it wraps a reference to the kernel
       rather than a copy of it
       @param li a function with signature:
       void(PureKernel &pk, size_t start, size_t end)
       The job of li is to execute the operation between progress points
       start and end.
       @param opts the options for the operation
       
       @return the client batch number, or 0 if single-threaded.
     */
    template<typename TL, typename Kernel, typename PureKernel, typename PureKernel_Ref, typename LI, typename Ks>
    size_t QueueProcessLink(Link<TL> & link, int whichEnd, Kernel &k, PureKernel &pk, PureKernel_Ref &pk_ref, LI li, JobOptions<Ks> opts){
        Component<TL> &c = link.ends[whichEnd].c;
        NETPERFREC(c.net.npl, QueueProcessLink, 0);
        size_t maxProgress = link.getMaxProgress(whichEnd);
        std::string linkName = link.endName(whichEnd);
        std::string kernelName = opts.kernelName;
        if(kernelName == ""){
            kernelName = getKernelName<Kernel>();
        }
        if(!(link.ends[0].c.net.sched.has_value() && opts.parallel)){
            // single threaded
            // track performance
#ifdef PROFILER
            size_t opId = c.net.npl.logOpStart(linkName, kernelName, maxProgress, maxProgress);
            size_t chunkId = c.net.npl.logChunkStart(opId, maxProgress, 0);
#endif
            c.net.npl.logKernels(maxProgress);
            ProcessLink(link, whichEnd, pk_ref, 0, maxProgress
#ifdef DEBUG_OP_LEVEL
                        , opts.kernelName
#endif
                );
#ifdef PROFILER
            c.net.npl.logChunkEnd(opId, chunkId);
#endif
            return 0;
        }

        struct NextProgressPoint{
            int whichEnd;
            Link<TL> &link;
            NextProgressPoint(int whichEnd, Link<TL> &link) : whichEnd(whichEnd), link(link){}
            size_t operator()(index_t requested){
                return std::visit(
                    [this, requested](auto &&arg){
                        return arg.requestPartialProgress(
                            this->whichEnd, requested);
                    }, link.type);
            }
        }npp(whichEnd, link);
        // size_t processOp(Kernel &k, std::string linkName, std::string kernelName, size_t opTypeIndex, int cmpId, size_t maxProgress, bool indivisible, Combiner combiner, NextProgressPoint nextProgressPoint, LinkIterator LI, bool endOfBatch, bool blocking)
        
        //std::type_index opTypeIndex(typeid(li));
        size_t opTypeIndex = typeid(li).hash_code();
        
        return c.net.sched->processOp(
            k,
            pk,
            linkName,
            kernelName,
            opTypeIndex,
            link.ends[whichEnd].c.id,
            maxProgress,
            !opts.parallel,
            opts.combiner,
            npp,
            li,
            opts.endOfBatch,
            opts.blocking);
    }

    /**
       This is the core function of LLRT. It executes a kernel across
       a Link, or across part of a Link. Inconvenient for the user to
       call directly.

       @param link the Link on which to execute
       @param end 0 or 1, the end of the link on which to execute
       @param pk a function with signature:
       void(const size_t Ni, const size_t Ei, const size_t ni, const size_t ei, const size_t edgeInfo)
       where Ni is the index to the near component, Ei is the index to the
       near LinkEnd, ni is the index to the far component, ei is the index
       to the far LinkEnd. edgeInfo is explained in README.md
       @param start the progress point at which we should begin execution
       @param end the progress point before which we should finish execution
       @param kernelName can be used for debugging output, if that is enabled.
    */
    template<typename TL, typename PureKernel>
    void ProcessLink(Link<TL> & link, int whichEnd, PureKernel && pk, index_t start, index_t end
#ifdef DEBUG_OP_LEVEL
                     , std::string kernelName
#endif
        ){
#ifdef DEBUG_OP_LEVEL
        std::cout << "ProcessLink for " << kernelName << " (" << start << "-" << end << ")" << std::endl;
#endif
        std::visit([whichEnd, &pk, start, end](auto&& arg){
            arg(whichEnd, pk, start, end);
        }, link.type);
    }

} // end namespace

#include "adjlistlink.hpp"
#include "generallocal2dlink.hpp"
#include "local2dlink.hpp"
#include "network_impl.hpp"

#endif /* NETWORK_HPP_ */

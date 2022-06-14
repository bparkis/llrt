// included at the end of network.hpp
// and therefore has access to definitions from there

#ifndef NETWORK_IMPL_HPP_
#define NETWORK_IMPL_HPP_

namespace llrt{

    std::string listDimensions(std::vector<index_t> dims);

    template <typename LinkType>
    std::string badDimensions(const std::vector<index_t> &dims1, const std::vector<index_t> &dims2){
        return std::string() + "Bad dimensions: link type " + typeid(LinkType).name() + " can't connect " + listDimensions(dims1) + " to " + listDimensions(dims2);
    }

    template <typename TL>
    template <typename LinkType, typename EdgeType0, typename EdgeType1, typename NodeType>
    Component<TL> & Component<TL>::connect(bool swapEnds, bool swapAxon){
        std::vector<index_t> dims2;
        int whichEnd = swapEnds ? 0 : 1;
        if (!LinkType().deduceComponentDimensions(data.dimensions, dims2, whichEnd))
            throw std::runtime_error(std::string() + "Link type " + typeid(LinkType).name() + " couldn't deduce dimensions of near component (end " + std::to_string(whichEnd) + ") when far component (end " + std::to_string(1 - whichEnd) + ") has dimensions " + listDimensions(data.dimensions));
        return connect<LinkType, EdgeType0, EdgeType1, NodeType>(dims2, swapEnds, swapAxon);
    }

    template <typename TL>
    template <typename LinkType, typename EdgeType0, typename EdgeType1, typename NodeType>
    Component<TL> & Component<TL>::connect(const std::vector<index_t> &newDims, bool swapEnds, bool swapAxon){
        if(swapEnds){
            if(!LinkType().canConnectDimensions(newDims, data.dimensions)){
                throw std::runtime_error(badDimensions<LinkType>(newDims, data.dimensions));
            }
        }
        else{
            if(!LinkType().canConnectDimensions(data.dimensions, newDims)){
                throw std::runtime_error(badDimensions<LinkType>(data.dimensions, newDims));
            }
        }

        Component<TL> &c = net.template component<NodeType>(newDims);
        connect<LinkType, EdgeType0, EdgeType1>(c, swapEnds, swapAxon);
        return c;
    }

    template <typename TL>
    template <typename LinkType, typename EdgeType0, typename EdgeType1, typename NodeType>
    Component<TL> & Component<TL>::connect(const std::initializer_list<index_t> &newDims, bool swapEnds, bool swapAxon){
        // for type overloading
        const std::vector<index_t> dims(newDims);
        return connect<LinkType, EdgeType0, EdgeType1, NodeType>(dims, swapEnds, swapAxon);
    }

/**
 * Create a new link to an existing component.  Throws an exception if the dimensions are invalid based on the LinkType.
 *
 * @return the second component
 */
    template <typename TL>
    template <typename LinkType, typename EdgeType0, typename EdgeType1>
    Component<TL> & Component<TL>::connect(Component<TL> &otherComponent, bool swapEnds, bool swapAxon){
        if(swapEnds){
            if(!LinkType().canConnectDimensions(otherComponent.data.dimensions, data.dimensions)){
                throw std::runtime_error(badDimensions<LinkType>(otherComponent.data.dimensions, data.dimensions));
            }
        }
        else{
            if(!LinkType().canConnectDimensions(data.dimensions, otherComponent.data.dimensions)){
                throw std::runtime_error(badDimensions<LinkType>(data.dimensions, otherComponent.data.dimensions));
            }
        }

        if(swapEnds){
            std::shared_ptr<Link<TL>> l = std::make_shared<Link<TL>>(otherComponent, LinkType(), *this, swapAxon, ++net.linkId);
            net._thatLink = &(l->type);
            links[1].push_back(l);
            otherComponent.links[0].push_back(l);
            l->ends[0].data.template initialize<EdgeType0>();
            l->ends[1].data.template initialize<EdgeType1>();
        }
        else{
            std::shared_ptr<Link<TL>> l = std::make_shared<Link<TL>>(*this, LinkType(), otherComponent, swapAxon, ++net.linkId);
            net._thatLink = &(l->type);
            links[0].push_back(l);
            otherComponent.links[1].push_back(l);
            l->ends[0].data.template initialize<EdgeType0>();
            l->ends[1].data.template initialize<EdgeType1>();
        }
        return otherComponent;
    }

    template <typename TL>
    template <typename NodeType>
    Component<TL> & Network<TL>::component(const std::vector<index_t> & dims){
        components.push_back(std::move(std::make_unique<Component<TL>>(*this, dims)));
        components.back()->name = std::string() + typeid(NodeType).name() + "_" + std::to_string(components.size());
        components.back()->data.template initialize<NodeType>();
        return *components.back();
    }

    template <typename TL>
    void Network<TL>::finishBatches(){
        NETPERFREC(npl, finishBatches_, 0);
        if(sched.has_value())
            sched->finishBatches();
    }

    template <typename TL>
    void Network<TL>::finishBatch(size_t batchNumber){
        NETPERFREC(npl, finishBatch_, 0);
        if(sched.has_value())
            sched->finishBatch(batchNumber);
    }
    
    template <typename TL>
    void Network<TL>::displayChain(Link<TL> &l, int i, std::set<Link<TL> *> &ls){
        std::string linkName = std::visit([](auto && arg){
            return arg.identifier();
        }, l.type) + std::to_string(l.id);
        std::string endNames[2];
        for(size_t i : {0, 1})
            endNames[i] = l.ends[i].data.valueTypeName();

        std::string linkDisplay;
        if(i == 0){
            linkDisplay = " ──" + endNames[0] + "─(" + linkName + ")─" + endNames[1] + "──> ";
        }
        else{
            linkDisplay = " <──" + endNames[0] + "─(" + linkName + ")─" + endNames[1] + "── ";
        }
        std::cout << linkDisplay;
        Component<TL> * c2 = &l.ends[1-i].c;
        std::cout << c2->displayName();
        for(int i=0; i < 2; i++){
            for(std::shared_ptr<Link<TL>> &l2 : c2->links[i]){
                if(ls.find(&(*l2)) == ls.end()){
                    ls.insert(&(*l2));
                    displayChain(*l2, i, ls);
                }
            }
        }
    }

    template <typename TL>
    void Network<TL>::display(){
        std::set<Link<TL> *> ls;
        bool beginChain = true;
        for(const auto &c : components){
            for(int i=0; i < 2; i++){
                for(std::shared_ptr<Link<TL>> &l : c->links[i]){
                    if(ls.find(&(*l)) == ls.end()){
                        ls.insert(&(*l));
                        std::cout << c->displayName();
                        displayChain(*l, i, ls);
                        std::cout << std::endl;
                    }
                }
            }
        }
    }

    template <typename TL>
    void Network<TL>::setDeterminism(){
        if (sched.has_value()){
            sched->deterministic = true;
        }
    }

    template <typename TL>
    void Network<TL>::seed(size_t rngSeed){
        rng.seed(rngSeed);
    }


    template <typename TL>
    void Network<TL>::perfReport(const std::string filename){
        finishBatches();
        if(sched.has_value()){
            // net.npl gives information about when operations are sent to the scheduler
            // net.sched->npl gives information about the internal things done by the main scheduler thread
            // net.sched->npl_op gives information about the job chunks handled by each worker thread
            // We merge all this information to see it together.
            sched->mergeLoggers(npl);
            std::cout << sched->nWorkers << " workers (hardware_concurrency = " << std::thread::hardware_concurrency() << ")" << std::endl;
        }
        else
            std::cout << "Single threaded" << std::endl;
#ifdef PROFILER
        npl.report();
        if(filename != ""){
            std::ofstream outFile;
            outFile.open(filename);
            npl.dump(outFile); // this outputs the full tracing data
            outFile.close();
            std::cout << "Logged performance data to " << filename << ". View it using chrome://tracing in the Chrome browser." << std::endl;
        }
#else
        npl.report();
#endif
    }
}
#endif

#include "network.hpp"

namespace llrt{

    NullOptionType unifyKernels(NullOptionType c, NullOptionType){
        return c;
    }
    
    const JobOptions<NOpT3> KernelName(std::string const name){
        return JobOptions<NOpT3>{false, true, true, name};
    }

    std::string listDimensions(std::vector<index_t> dims){
        std::string s = "(";
        for (size_t i=0; i < dims.size(); i++){
            s += std::to_string(dims[i]);
            if(i != dims.size() - 1)
                s += ", ";
        }
        s += ")";
        return s;
    }

}

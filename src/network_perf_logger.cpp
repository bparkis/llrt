#include <iostream>
#include <iomanip>
#include <map>
#include "network_perf_logger.hpp"
#include <cassert>

namespace llrt{
    NetworkPerfLogger::NetworkPerfLogger() : startTime(std::chrono::steady_clock::now())
    {}

    size_t NetworkPerfLogger::logOpStart(std::string linkName, std::string kernelName, size_t maxProgress, size_t numEdges,  bool isLinkKernelOperation){
        size_t sz = operations.size();
        operations.push_back(OpRecord{linkName, kernelName, maxProgress, numEdges, isLinkKernelOperation});
        return sz;
    }

    void NetworkPerfLogger::logChunk(size_t index, size_t progress, std::chrono::steady_clock::time_point startTime, std::chrono::steady_clock::time_point endTime, size_t thread){
        OpRecord &r = operations.at(index);
        r.chunks.push_back(ChunkRecord{progress, startTime, endTime, true, thread});
    }

    size_t NetworkPerfLogger::logChunkStart(size_t index, size_t progress, size_t thread){
        OpRecord &r = operations.at(index);
        size_t sz = r.chunks.size();
        r.chunks.push_back(ChunkRecord{progress, std::chrono::steady_clock::now()});
        r.chunks.back().thread = thread;
        return sz;
    }

    void NetworkPerfLogger::logChunkEnd(size_t opIndex, size_t chunkIndex){
        OpRecord &r = operations.at(opIndex);
        r.chunks.at(chunkIndex).endTime = std::chrono::steady_clock::now();
        r.chunks.at(chunkIndex).finished = true;
    }
    
    void NetworkPerfLogger::logInstant(std::chrono::steady_clock::time_point when, std::string name, size_t thread){
        instantEvents.emplace_back(InstantEvent{name, when, thread});
    }

    void NetworkPerfLogger::mergeLoggers(NetworkPerfLogger & npl){
        operations.insert(operations.end(), npl.operations.begin(), npl.operations.end());
        instantEvents.insert(instantEvents.end(), npl.instantEvents.begin(), npl.instantEvents.end());
        totKernels += npl.totKernels;
    }

    void NetworkPerfLogger::dump(std::ostream &out){
        out << "[";
        bool first = true;
        for(OpRecord &r : operations){
            for(ChunkRecord &c: r.chunks){
                if(!first)
                    out << "," << std::endl;
                first = false;
                out << "{\"name\": \"" << r.kernelName + "@" + r.linkName + ":" + std::to_string(c.progress);
                out << "\", \"cat\": \"" << "op";
                out << "\", \"ph\": \"" << "B";
                out << "\", \"pid\": " << 0;
                out << ", \"tid\": " << std::to_string(c.thread);
                double chunkStartTime = std::chrono::duration_cast<std::chrono::duration<double, std::micro> >(c.startTime - startTime).count();
                out << ", \"ts\": " << std::to_string(chunkStartTime);
                out << "}";
                //assert(c.finished);
                if(c.finished){
                    out << "," << std::endl;
                    out << "{\"name\": \"" << r.kernelName + "@" + r.linkName + ":" + std::to_string(c.progress);
                    out << "\", \"cat\": \"" << "op";
                    out << "\", \"ph\": \"" << "E";
                    out << "\", \"pid\": " << 0;
                    out << ", \"tid\": " << std::to_string(c.thread);
                    double chunkEndTime = std::chrono::duration_cast<std::chrono::duration<double, std::micro> >(c.endTime - startTime).count();
                    out << ", \"ts\": " << std::to_string(chunkEndTime);
                    out << "}";
                }
            }
        }
        for (InstantEvent &e : instantEvents){
            if(!first)
                out << "," << std::endl;
            first = false;
            out << "{\"name\": \"" << e.name;
            out << "\", \"cat\": \"" << "broadcast";
            out << "\", \"ph\": \"" << "i";
            out << "\", \"pid\": " << 0;
            out << ", \"tid\": " << std::to_string(e.thread);
            double eventStartTime = std::chrono::duration_cast<std::chrono::duration<double, std::micro> >(e.time - startTime).count();
            out << ", \"ts\": " << std::to_string(eventStartTime);
            out << "}";
        }
        out << "]";
        
        // {"name": "MyName", "cat": "PERF", "ph": "B", "pid": 22630, "tid": 22630, "ts": 829}
    }

    void NetworkPerfLogger::report(){
        double dur = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(
            std::chrono::steady_clock::now() - startTime).count();
        std::cout << std::setprecision(4) << std::fixed;
        std::cout << "Executed " << totKernels << " kernels in " << dur << " ms" << std::endl;
        std::cout << "(" << (totKernels / dur)*1000.0 << " kernels per second)" << std::endl;
    }
}

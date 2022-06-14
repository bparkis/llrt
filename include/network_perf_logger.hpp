#ifndef NETWORK_PERF_LOGGER_HPP_
#define NETWORK_PERF_LOGGER_HPP_
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <list>

/**
   Say NETPERFREC at the beginning of an event you want to time, and
   then STOPPERF at the end of the event. Alternatively, just let the
   current scope end, which will have the same effect as STOPPERF.
 */
#ifdef PROFILER
#define NETPERFREC(logger, name, thread) NetPerfRec name((logger), #name, (thread))
#define STOPPERF(name) name.stop()
#else
#define NETPERFREC(logger, name, thread)
#define STOPPERF(name)
#endif

namespace llrt{
/**
   A performance logging framework.  Tracks all operation start and
   end times, organizing them by kernel identifier and link
   identifier, so that performance reports can be generated.
 */
    struct NetworkPerfLogger{
    private:

        struct ChunkRecord{
            size_t progress; /// the amount of maxProgress that this chunk accounts for
            std::chrono::steady_clock::time_point startTime;
            std::chrono::steady_clock::time_point endTime;
            bool finished = false;
            /// 0 for main/client thread
            /// 1 for scheduler thread
            /// >1 for worker threads
            size_t thread;
        };

        struct OpRecord{
            std::string linkName;
            std::string kernelName;
            size_t maxProgress;
            size_t numEdges;
            /// true when applying a kernel to a link,
            /// false for other kinds of performance tracking
            bool isLinkKernelOperation = true;
            std::vector<ChunkRecord> chunks;
        };
        
        std::vector<OpRecord> operations;
        std::chrono::steady_clock::time_point startTime;

        struct InstantEvent{
            std::string name;
            std::chrono::steady_clock::time_point time;
            size_t thread;
        };

        std::list<InstantEvent> instantEvents;

        size_t totKernels=0;

    public:
        NetworkPerfLogger();

        /**
           Record the number of kernels.
           This is provided as a separate function so that it can be
           called even when PROFILER is not defined.
         */
        inline void logKernels(size_t numKernels){
            totKernels += numKernels;
        }

        /**
           Log the start of an operation with the given link name and
           kernel name.
         
           @param linkName is the name of the link
           @param kernelName is the name of the kernel
           @param maxProgress is the link's maxProgress
           @param numEdges is the number of edges on the link, if
           known.  Otherwise, the caller passes numEdges = maxProgress
           for an approximation.
           @return a number that should be used to log the end of the
           operation.
         */
        size_t logOpStart(std::string linkName, std::string kernelName, size_t maxProgress, size_t numEdges, bool isLinkKernelOperation = true);

        /**
           Log a chunk for which both start and end times are already
           known.
         
           @param index is the value received from logOpStart
           @param progress is the amount of progress accounted for by
           this chunk
           @param startTime is the start time
           @param endTime is the end time
           @param thread is a number identifying the thread that ran
           this chunk
         */
        void logChunk(size_t index, size_t progress, std::chrono::steady_clock::time_point startTime, std::chrono::steady_clock::time_point endTime, size_t thread);

        /**
           Log a chunk starting now, with an unknown end time.
         
           @param index is the value received from logOpStart
           @param progress is the amount of progress accounted for by
           this chunk
           @param thread is a number identifying the thread that ran
           this chunk
         */
        size_t logChunkStart(size_t index, size_t progress, size_t thread);

        /**
           Log a chunk ending now.
         
           @param opIndex is the number received from logOpStart
           @param chunkIndex is the number received from logChunkStart
         */
        void logChunkEnd(size_t opIndex, size_t chunkIndex);
        

        /**
           Log an instant event

           @param when is when
           @param name describes the event
         */
        void logInstant(std::chrono::steady_clock::time_point when, std::string name, size_t thread);

        /**
           Merge this NetworkPerfLogger together with another.  The
           intent is that the Scheduler thread has a
           NetworkPerfLogger, and the main thread has one too, and we
           merge them together at the end, avoiding the need for
           synchronization before then.  After this operation, this
           logger is the one with the combined information.
         */
        void mergeLoggers(NetworkPerfLogger & npl);

        /**
           Dump the timing information to an ostream, in a format that
           can be interpreted by chrome://tracing within google
           chrome.
         */
        void dump(std::ostream &out);
        
        /**
           Print some information of general use to someone trying to
           optimize their program.
         */
        void report();
    };

    struct NetPerfRec{
    private:
        NetworkPerfLogger &npl;

        size_t opIx=0;
        size_t chunkIx=0;
        bool active = true;

    public:
        NetPerfRec(NetworkPerfLogger &npl, std::string name, size_t thread=0):
            npl(npl){
            opIx = npl.logOpStart("", name, 0, 0, false);
            chunkIx = npl.logChunkStart(opIx, 0, thread);
        }

        void stop(){
            npl.logChunkEnd(opIx, chunkIx);
            active = false;
        }

        ~NetPerfRec(){
            if(active)
                npl.logChunkEnd(opIx, chunkIx);
        }
    };
}
#endif

#ifndef SCHEDULER_HPP_
#define SCHEDULER_HPP_
// -*- coding: utf-8 -*-
#include <iostream>
#include <vector>
#include <functional>
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <shared_mutex>              
#include <condition_variable> // std::condition_variable
#include <tuple>
#include <functional>
#include <algorithm>
#include <ctime>
#include <list>
#include <forward_list>
#include <limits>
#include <typeindex>
#include <chrono>
#include <any>
#include <iterator>
#include <map>
#include <cassert>
#include "common.hpp"
#include "network_perf_logger.hpp"


namespace llrt{
    using dur_t = std::chrono::steady_clock::duration;
    using time_t = std::chrono::steady_clock::time_point;
    using type_index_t = size_t;

/**
   Overview of how the scheduler works

   We have the client thread, the scheduler thread, and several worker
   threads.  The client thread submits a ClientBatch containing several
   jobs, to the scheduler.  A ClientBatch is a set of jobs that can be
   done in any order or simultaneously, except that each job has a
   cmpId, and two jobs with the same cmpId cannot run at the same time.
   That is because of the "near node guarantee" (see README.md).

   The scheduler waits until it receives a complete ClientBatch, then it
   schedules the ClientBatch. The scheduler divides up the ClientBatch
   into pieces called JobChunkBatches, of estimated equal size. Most
   jobs can be divided into chunks of arbitrary size, making it easier
   to do this. One JobChunkBatch is assigned to each worker, and can
   contain several JobChunks.

   The workers finish their JobChunkBatches and synchronize with each
   other. This constitutes a Barrier. A single ClientBatch may result in
   multiple Barriers due to the need to avoid running two jobs with the
   same cmpId at the same time.  After synchronizing, the workers can
   immediately begin work on the next Barrier, if available. The workers
   also notify the scheduler that they have finished their Barrier.

   The scheduler collects statistics on how long it took to finish each
   job.  This allows it to more accurately estimate how to chop up a
   ClientBatch into JobChunkBatches of equal size.

   The scheduler also maintains a record of which Barriers and
   ClientBatches are finished, and notifies the client thread, if the
   client thread is waiting for a ClientBatch to finish. The scheduler
   is also responsible for cleaning up old information from previous
   Barriers.

   The scheduler manages its communication using a channel. The
   channel contains a mutex and a condition variable. The workers have
   their own worker bulletin for alerting them all at once.

   There is one more mutex that is used by the scheduler to tell the
   client which ClientBatches are finished.
*/
    class Scheduler{

    private:

        size_t sequence=0; ///< an ID number for each synchronization barrier. increments with each barrier. This number is equal to the highest scheduled barrier. Accessed only by the scheduler thread.

        size_t clientBatchNumber=1; ///< ID numbers for batches submitted by clients. One client batch may correspond to multiple synchronization barriers. Need to lock schedChan.mtx to access.


        /// maps from barrier sequence numbers to client batch numbers.
        /// if clientSequenceMap[num1] = num2, and the barrier with sequence num1 is done, then the client batch numbered num2 is done.
        std::map<size_t, size_t> sequenceClientMap;

        /// any client batches with this number or earlier are finished
        /// this is how we communicate that fact with the client
        size_t completedClientBatchNum=0;
        std::mutex completedClientBatchMtx;
        std::condition_variable completedClientBatchCv;



        /// A Job submitted to the scheduler by a client
        /// normally this represents a Link operation
        struct Job{
            std::forward_list<std::any> kernelCopies;
            std::any originalKernelPtr;
            std::any firstCopy;
            std::function<std::function<void(int64_t, int64_t)>(Job &)> copier;
            std::function<int64_t(int64_t)> nextProgressPoint;
            std::any combiner;
            std::function<void(Job &)> combineAll;
            std::string kernelName;
            type_index_t opTypeIndex;
            size_t opPerfLogId;
            size_t progress=0; ///< how much of the job has been assigned to workers
            size_t maxProgress=100;
            bool indivisible = false;
            /// cmpId of the near component (see the near-node guarantee)
            /// If multiple Jobs have the same cmpId, only one of them can run at a time.
            /// This would be because the Jobs might access and update the same data, causing a race condition.
            int cmpId;
            std::vector<int64_t> chunkVec;
            dur_t estimatedTime;
        };

        /// A piece of a Job given to a worker by the scheduler
        ///
        struct JobChunk{
            std::function<void(int64_t, int64_t)> task;
            const int64_t start; // the point at which this job should start
            const int64_t end;   // the point before which this job should end

            JobChunk(std::function<void(int64_t, int64_t)> task, const int64_t start, const int64_t end, Job *job) : task(task), start(start), end(end), job(job) {}

            time_t startTime;
            time_t endTime;
            Job *job;
        };

        /// Possibly several pieces of a Job given to a worker,
        /// which are part of the same synchronization barrier.
        struct JobChunkBatch{
            std::list<JobChunk> chunks;

            bool statsRecorded = false;
            bool neededByWorker = true;
        };

        /// A synchronization barrier
        /// access controlled by schedChan.mtx
        struct Barrier{
            size_t doneWorkers=0; ///< number of workers who have finished their JobChunkBatches for this barrier

            const size_t sequence;
            std::list<Job *> jobs;
            bool singleThreaded = false;
            bool singleThreadedStartedYet = false;
            bool finalized = false;

            std::vector<JobChunkBatch> workerBatches;

            Barrier *next = nullptr;

            bool finished(size_t nWorkers){
                if (singleThreaded)
                    return doneWorkers >= 1;
                return doneWorkers == nWorkers;
            }

            Barrier(size_t nWorkers, size_t sequence) : sequence(sequence), workerBatches(nWorkers){}

        };

        Barrier *lastBarrier = nullptr;
        Barrier *firstBarrier = nullptr;

        Barrier & newBarrier();

        /// the barrier that the scheduler will finalize next, once it is done
        /// updated by the scheduler after we finish a barrier, if there is already another
        /// also updated by the scheduler if we add another barrier, if we already finished one
        Barrier *schedBarrier = nullptr;

        /// All the information necessary to communicate with a worker
        /// and all the information the worker uses to decide when to wake up
        struct WorkerChannel{
            std::mutex mtx;
            std::condition_variable cv;
            bool terminate = false;
            size_t latestSequence = 0;
            size_t completedSequence = 0;
        };

        std::vector<WorkerChannel> workChans;

        /**
           Broadcast to other workers the fact that the current barrier is finished
           @param completed the sequence of the completed barrier
           @param workerWhoNotifies index of the worker who notifies
           @return true if there is another barrier already ready to work on after this one
         */
        bool broadcastCompleted(size_t completed, size_t workerWhoNotifies);

        /**
           Broadcast to the workers that a new Barrier has been scheduled.
           @param latest the sequence of the new barrier that was scheduled
        */
        void broadcastLatest(size_t latest);

        /**
           Broadcast a shutdown signal to the workers
         */
        void broadcastTerminate();

        /**
           A batch of jobs, submitted by the client, that may be
           scheduled in any order and in parallel. Except that two
           jobs with the same cmpId can't run at the same time.
        */
        struct ClientBatch{
            size_t clientBatchNumber=0;
            std::list<Job> jobs;
            bool readyToSchedule = false;
            bool scheduled = false;
        };

        /**
           all the information needed for another thread to
           communicate with the scheduler
        */
        struct SchedChannel{
            std::mutex mtx;
            std::condition_variable cv;
            std::list<ClientBatch> batches;
            bool shutdown = false;
        } schedChan;

        /**
           Clean up information when the Scheduler terminates.
         */
        void finalCleanup();

        /**
           Collect timing information from a completed job batch.
         */
        void collectStats(JobChunkBatch &batch, size_t worker);

        /**
           Clean up information relating to a barrier after it has completed.
         */
        void cleanupBarrier();

        /**
           The Scheduler calls this after the workers have finished a
           barrier, to notify the client and clean up.
         */
        void recordFinishedJobs();

        /**
           Assign a job chunk to a worker.

           @param job
           @param batch the batch into which the job will go
           @param desiredDuration the amount of time we want the job
           chunk to take. If this is zero, schedule the whole job.

           @return the actual amount of time we expect the job chunk to take
         */
        dur_t assignJob(Job *job, JobChunkBatch *batch, dur_t desiredDuration
#ifdef DEBUG_OP_LEVEL
                        , int worker
#endif
            );

        /**
           Choose a subset of jobs to be performed in the same synchronization barrier
           @param buckets the jobs left to assign from the ClientBatch. Jobs assigned to the current barrier will be removed from this list.
           @param waterToPour The jobs that will go into the current barrier. An output parameter.
           @return the total amount of time we expect all the jobs to take
         */
        dur_t selectWater(std::list<Job *> &buckets, std::list<Job *> &waterToPour);

        /**
           Assign jobs to workers. The jobs can be divided into
           pieces, so metaphorically it is like pouring water. We
           obtain the expected completion time for the barrier, by
           dividing the time for all jobs by the number of workers,
           and assign job chunks to each worker aiming to meet that
           time.

           @param buckets the jobs to assign in this barrier
           @param totWater the total expected time of all the jobs in this barrier
         */
        void pourWater(std::list<Job *> &buckets, dur_t totWater);

        /**
           Assign all jobs in this barrier to a single worker. This is
           done if the total expected time of all jobs is very short.
           @param jobs the jobs in the barrier
         */
        void singleThreadedSchedule(std::list<Job *> &jobs);

        /**
           Given the jobs in a ClientBatch, assign them all to workers, creating possibly multiple Barriers.
           @param jobs the jobs
         */
        void planAllStages(std::list<Job *> &jobs);
        /**
           Run all combiners for jobs in a barrier, after the main
           work for the barrier is complete.
           @param jobs the jobs in the barrier.
         */
        void runCombiners(std::list<Job *> &jobs);

        /**
           When the Scheduler started
         */
        time_t startTime;

        /**
           Timing information for instant events for each
           worker. Normally not used, but useful to turn on
           temporarily for diagnosing performance issues.
         */
        struct WorkerLogEntry{
            time_t point;
            enum Kind{
                GOT_SCHED_LCK,
                RAN_COMBINERS,
                BROADCAST_COMPLETE,
                GETTING_WORKCHAN_LCK,
                GOT_WORKCHAN_LCK
            }kind;
        };

        std::vector<std::list<WorkerLogEntry> > workerLogs;

        /**
           If this is false, we do not waste time or space logging
           instant events for each worker.
         */
        bool showInstantEvents = false;

        /**
           Log an instant event for a worker. Job chunk start and end times are logged separately from this.
         */
        void workerLog(int worker, WorkerLogEntry::Kind kind){
#ifdef PROFILER
            if(showInstantEvents)
                workerLogs[worker].emplace_back(WorkerLogEntry{std::chrono::steady_clock::now(), kind});
#endif
        }

        /**
           Tells us how much total time we have devoted to a
           particular type of job, and how many ops (usually edges) we
           have executed in total for that job.
         */
        struct PerfTracker{
            dur_t totTime = dur_t::zero();
            size_t totOps=1;
            double T_op=1;
        };

        std::map<type_index_t, PerfTracker> timeByKernel;

        inline double microseconds(dur_t dur){
            return std::chrono::duration_cast<std::chrono::duration<double, std::micro> >(dur).count();
        }

        /**
           Record how long a job chunk took.
           @param opTypeIndex an identifier for the type of operation, typically a combination of the kernel type and the link type.
           @param time the amount of time the operation took
           @param ops the amount of progress assigned to this job chunk, typically a number of edges
         */
        void trackOp(type_index_t opTypeIndex, dur_t time, size_t ops){
            if (deterministic)
                return;
            assert(timeByKernel.contains(opTypeIndex));
            PerfTracker &pt = timeByKernel[opTypeIndex];
            pt.totTime += time;
            pt.totOps += ops;
            if(pt.totOps > 0)
                pt.T_op = microseconds(pt.totTime) / pt.totOps;
        }

        /**
           Estimate how long a job chunk will take.
           @param opTypeIndex an identifier for the type of operation, typically a combination of the kernel type and the link type.
           @param ops the amount of progress assigned to this job chunk, typically a number of edges
           @return the amount of time the operation is expected to take
        */
        dur_t estimateTimeOp(type_index_t opTypeIndex, size_t ops){
            if (deterministic)
                return std::chrono::microseconds(static_cast<int64_t>(ops*1.0));
            assert(timeByKernel.contains(opTypeIndex));
            double estimate = timeByKernel[opTypeIndex].T_op * ops;
            return std::chrono::microseconds(static_cast<int64_t>(estimate));
        }

        /**
           Estimate how much of an operation we will be able to execute in a specified time.
           @param opTypeIndex an identifier for the type of operation, typically a combination of the kernel type and the link type.
           @param time is the amount of time we want the operation to take
           @return the amount of progress on the job we expect can be done in that time.
         */
        size_t estimateOpsFromTime(type_index_t opTypeIndex, dur_t time){
            assert(timeByKernel.contains(opTypeIndex));
            double estimate;
            if (deterministic)
                estimate = microseconds(time) / 1.0;
            else
                estimate = microseconds(time) / timeByKernel[opTypeIndex].T_op;
            size_t estSz = static_cast<size_t>(estimate);
            if (estSz == 0)
                estSz = 1;
            return estSz;
        }

        /**
           Workers who have finished their last JobChunkBatch wait
           here until notified the other workers are finished too and
           there is another barrier available to work on.
           @param workerIndex the number of the worker
           @param barrier the barrier that was just finished
           @return the next available barrier to work on
         */
        Scheduler::Barrier * waitForNextBarrier(int workerIndex, Barrier *barrier);

        /**
           Workers execute their JobChunkBatches here
           @param workerIndex the index of the worker
         */
        void workLoop(int workerIndex);

        /**
           The scheduler thread waits for signals from the client or
           from the workers, and handles them.
         */
        void schedLoop();

        std::thread *schedThread;

    public:
        /**
           If deterministic = true, then the Scheduler is obligated to split
           batches into chunks in exactly the same way in every execution of
           the program. This means it can't do adaptive scheduling.
        */
        bool deterministic = false;

        /**
           If a batch is estimated to be finishable in less than this
           amont of time, then it will be assigned to only a single
           worker thread, to reduce communication overhead between
           worker threads.
         */
        dur_t singleThreadThreshold = std::chrono::microseconds(30);

        size_t nWorkers; ///< number of worker threads

        NetworkPerfLogger npl_op; ///< performance logger for job chunk start and end times
        NetworkPerfLogger npl; ///< performance logger for other events
        
        /**
           Launch the scheduler
           @param nWorkers the number of worker threads
        */
        Scheduler(int nWorkers = std::thread::hardware_concurrency()) : startTime(std::chrono::steady_clock::now()), workChans(nWorkers), nWorkers(nWorkers){
            workerLogs.resize(nWorkers);
            // acquire a lock so that the scheduler thread doesn't start doing stuff until the Scheduler is fully constructed
            std::unique_lock<std::mutex> schedLck(schedChan.mtx);
            schedThread = new std::thread(&Scheduler::schedLoop, std::ref(*this));
#ifdef DEBUG_NETWORK_LEVEL
            std::cout << "Creating scheduler with " << nWorkers << " workers (hardware_concurrency = " << std::thread::hardware_concurrency() << ")" << std::endl;
#endif
        }

        /**
           terminate the Scheduler thread, which will shut down the workers too
        */
        ~Scheduler(){
            std::unique_lock<std::mutex> lck(schedChan.mtx);
            schedChan.shutdown = true;
            lck.unlock();
            schedChan.cv.notify_all();
            schedThread->join();
            delete schedThread;
        }

        /**
           Assign an operation to be run by the scheduler.
         
           The scheduler will divide the operation into variable-sized
           chunks, which it will send to worker threads in a way that
           balances the load between workers.  Each chunk gets a copy
           of the PureKernel pk, which a worker thread executes
           repeatedly over that chunk's segment of the data.
         
           When the batch is done, data from the copies of pk may
           be combined back into k, so that the caller can read off
           the result.
         
           @param k is the original kernel, to which data may be
           combined
         
           @param pk is a wrapper around k that will be applied
           repeatedly to the data for the operation. pk.k is required
           to be a copy of k.

           @param linkName is a name for the link this operation is
           running on, which may be used when reporting performance
           statistics.
         
           @param kernelName is a name for k.  kernelName may be used
           when reporting performance statistics.
         
           @param opTypeIndex is a number that uniquely identifies the
           operation which will be used by the kernel, for performance
           tracking
         
           @param cmpId is a small integer. If two jobs have the same
           cmpId, they will not be run at the same time.
         
           @param maxProgress is the maximum number of times k must be
           applied for the operation to be complete.
         
           @param indivisible is "false" if the operation can be
           divided into small chunks for better load balancing, "true"
           if the whole operation must run in a single thread (because
           it's not threadsafe).
         
           @param combiner is a function that takes as params k and a
           copy of pk, and merges the results stored in pk.k into the
           original k.  This is useful if k is stateful, so its copies
           collect state that must be merged back. If the type of
           combiner is NullOptionType, this will not be done.
         
           @param nextProgressPoint is a function with signature:
           size_t nextProgressPoint(size_t p)
           that takes as argument a desired level of progress for the
           operation, p, and returns a level of progress q at least as
           large as p, so that the next job chunk will end at q.
         
           @param LI is a function with signature:
           void(PureKernel &pk, size_t start, size_t end)
           that applies pk to every position in the chunk of data
           that begins at start and ends at the position before end.
         
           @param endOfBatch is true if the submitted batch is
           complete, so that the scheduler can begin scheduling and
           executing it.  It is false if more parts of the same batch
           will arrive later before the scheduler should schedule the
           whole batch.
         
           @param blocking is true if this call should not return until the
           entire job is done. Must have endOfBatch = true.
         
           @return the client batch number, which can be passed to
           finishBatch to wait for the batch to finish.
         */
    template<typename Kernel, typename PureKernel, typename Combiner, typename NextProgressPoint, typename LinkIterator>
    size_t processOp(Kernel &k, PureKernel &pk, std::string linkName, std::string kernelName, type_index_t opTypeIndex, int cmpId, size_t maxProgress, bool indivisible, Combiner combiner, NextProgressPoint nextProgressPoint, LinkIterator LI, bool endOfBatch, bool blocking);
        
        /**
           The client may call this to wait for all batches to finish.
         */
        void finishBatches();

        /**
           The client may call this to wait for the batch with the
           given number to finish.
         */
        void finishBatch(size_t batchNumber);

        /**
           The client may call this after submitting several jobs, to
           indicate the batch is ready to be scheduled. (As an
           alternative to this function, the client may use the
           endOfBatch option on the last job in the batch)

           @return true if we successfully ended a batch, false if
           there was sno batch to end.
         */
        bool endOfBatch();

        /**
           The client may call this to copy the scheduler's performance
           logs into npl_client.

           @param npl_client the destination to which performance logs
           will be copied.
         */
        void mergeLoggers(NetworkPerfLogger &npl_client);
    };


    template<typename Kernel, typename PureKernel, typename Combiner, typename NextProgressPoint, typename LinkIterator>
    size_t Scheduler::processOp(Kernel &k, PureKernel &pk, std::string linkName, std::string kernelName, type_index_t opTypeIndex, int cmpId, size_t maxProgress, bool indivisible, Combiner combiner, NextProgressPoint nextProgressPoint, LinkIterator LI, bool endOfBatch, bool blocking){
        std::unique_lock<std::mutex> schedLck(schedChan.mtx);
        constexpr bool hasCombiner = !std::is_same<Combiner, NullOptionType>::value;
        using _PureKernel = std::remove_reference<PureKernel>::type;
        using _Kernel = std::remove_reference<Kernel>::type;
        if (blocking)
            endOfBatch = true; // otherwise we'd block forever
        size_t opIx = 0;
#ifdef PROFILER
        opIx = npl_op.logOpStart(linkName, kernelName, maxProgress, maxProgress);
#endif
        npl_op.logKernels(maxProgress);

        ClientBatch *batch;
        if (schedChan.batches.size() == 0 || schedChan.batches.back().readyToSchedule == true){
            // new batch
            schedChan.batches.emplace_back();
            std::unique_lock<std::mutex> completedClientBatchLck(completedClientBatchMtx);
            schedChan.batches.back().clientBatchNumber = ++clientBatchNumber;
        }
        batch = &schedChan.batches.back();
        size_t batchNum = batch->clientBatchNumber;

        batch->jobs.emplace_back(Job{
                std::forward_list<std::any>(), // kernelCopies
                    hasCombiner ? &k : static_cast<_Kernel *>(nullptr), // originalKernelPtr
                    pk, // firstCopy
                    // copier
                    std::function<std::function<void(size_t, size_t)>(Job &)>(
                        [LI](Job &job){
                            job.kernelCopies.emplace_front(job.firstCopy);
                            _PureKernel *pk_ = std::any_cast<_PureKernel>(& job.kernelCopies.front());
                            return std::function<void(size_t, size_t)>(
                                [LI, pk_](size_t start, size_t end){
                                    LI(*pk_, start, end);
                                });
                        }),
                    // nextProgressPoint
                    std::function<size_t(size_t)>(nextProgressPoint),
                    combiner, // combiner
                    std::function<void(Job &)>( // combineAll
                        [](Job &job){
                            if constexpr (hasCombiner){
                                _Kernel *kptr = std::any_cast<_Kernel *>(job.originalKernelPtr);
                                std::for_each(job.kernelCopies.begin(), job.kernelCopies.end(),
                                              [kptr,&job](std::any &a){
                                                  _PureKernel *pk_ = std::any_cast<_PureKernel>(&a);
                                                  std::any_cast<Combiner>(job.combiner)(*kptr, pk_->k);
                                              });
                            }
                        }),
                    kernelName,
                    opTypeIndex,
                    opIx,
                    0, // progress
                    maxProgress,
                    indivisible,
                    cmpId,
                    });

        if (endOfBatch){
            batch->readyToSchedule = true;
        }
        schedLck.unlock();

        if (endOfBatch){
            schedChan.cv.notify_all();
        }

        if(blocking){
            finishBatches();
        }
        return batchNum;
    }
}
#endif

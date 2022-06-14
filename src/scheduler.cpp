// -*- coding: utf-8 -*-
#include "scheduler.hpp"
#include <cassert>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <sched.h>
#ifdef __linux
#include <pthread.h>
#endif
#include <set>
#include <numeric>

namespace llrt{

    void Scheduler::schedLoop(){
        std::unique_lock<std::mutex> schedLck(schedChan.mtx);
        std::vector<std::thread> workers;
        firstBarrier = new Barrier(nWorkers, sequence);
        lastBarrier = firstBarrier;
        firstBarrier->doneWorkers = nWorkers;
        schedBarrier = firstBarrier;

        for(size_t i=0; i < nWorkers; i++){
            workers.push_back(std::move(std::thread(&Scheduler::workLoop, std::ref(*this), i)));
            if(i < std::thread::hardware_concurrency()){
                // pin worker i to CPU i
#ifdef __linux
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(i, &cpuset);
                pthread_setaffinity_np(workers[i].native_handle(),
                                       sizeof(cpu_set_t), &cpuset);
#endif
            }
        }

        // scheduling plan:
        // receive batch parts
        // receive notification batch is complete
        // schedule batch

        while(true){

            if(!schedLck.owns_lock())
                schedLck.lock();
            // wait until we have an incoming job from a client
            // or if we receive a shutdown signal
            // or if it's time to finalize the batches the workers have finished
            ClientBatch *batch = nullptr;
            while(true){
                if(schedChan.shutdown)
                    break;

                for(ClientBatch &batch2: schedChan.batches){
                    if (batch2.readyToSchedule && !batch2.scheduled){
                        batch = &batch2;
                        break;
                    }
                }
                if (batch != nullptr)
                    break;

                if(!schedBarrier->finalized && schedBarrier->finished(nWorkers)){
                    // schedBarrier->doneWorkers will stay at nWorkers,
                    // so we don't need a lock to read it again later
                    break;
                }

                NETPERFREC(npl, schedLoopWait, 1);
                schedChan.cv.wait(schedLck);
                STOPPERF(schedLoopWait);
            }

            if (!schedBarrier->finalized && schedBarrier->finished(nWorkers)){
                schedLck.unlock();
                recordFinishedJobs();
                schedBarrier->finalized = true;
                if (schedBarrier->next != nullptr){
                    schedBarrier = schedBarrier->next;
                }
            }
            if (schedLck.owns_lock())
                schedLck.unlock();

            if(schedChan.shutdown)
                break;

            if (batch == nullptr)
                continue;

            NETPERFREC(npl, schedLoopScheduling, 1);
            std::list<Job> &jobs = batch->jobs;
            // copy jobs into another list so we don't need to access the channel anymore
            std::list<Job *> copyJobs;
            for(Job & j : jobs){
                copyJobs.push_back(&j);
            }

            planAllStages(copyJobs);

            for(Job *job: copyJobs){
                assert(job->progress == job->maxProgress);
            }

            if (schedBarrier->finalized && schedBarrier->next != nullptr){
                schedBarrier = schedBarrier->next;
            }
            // sequence is now the number of the last planned barrier
            sequenceClientMap[sequence] = batch->clientBatchNumber;

            batch->scheduled = true;

        }
        if(schedLck.owns_lock())
            schedLck.unlock();
        // kill workers
        broadcastTerminate();
        for(std::thread &t : workers){
            t.join();
        }
        finalCleanup();
    }

    void Scheduler::collectStats(JobChunkBatch &batch, size_t worker){
        for(JobChunk &chunk : batch.chunks){
            trackOp(chunk.job->opTypeIndex, chunk.endTime - chunk.startTime, chunk.end-chunk.start);
#ifdef PROFILER
            npl_op.logChunk(chunk.job->opPerfLogId, chunk.end - chunk.start, chunk.startTime, chunk.endTime, worker + 2);
#endif
        }
    }

    // final cleanup before shutting down
    void Scheduler::finalCleanup(){
        for(Barrier *b = firstBarrier; b != nullptr;){
            b = b->next;
            delete firstBarrier;
            firstBarrier = b;
        }
    }

    void Scheduler::cleanupBarrier(){
        // things to clean:
        // old Barriers
        //   (unless any worker is still using it)
        // old ClientBatches
        //   (including the just-completed one)
        for(Barrier *b = firstBarrier; b != nullptr;){
            if (b == schedBarrier)
                break;
            bool neededByWorker = false;
            for(int worker=0; worker < nWorkers; worker++){
                // we don't have a lock, but it's okay...
                // batch.neededByWorker can only ever change from true to false once
                // so if we get a corrupted read, the worst that can happen is
                // that we just read it as still true when it has become false
                // and in that case the only consequence is we don't clean up the barrier yet, but we will later
                if (b->workerBatches[worker].neededByWorker){
                    neededByWorker = true;
                    break;
                }
            }
            if (neededByWorker)
                break;

            assert(b != lastBarrier);
            b = b->next;
            delete firstBarrier;
            firstBarrier = b;
        }

        std::unique_lock<std::mutex> schedLck(schedChan.mtx);
        for(auto it=schedChan.batches.begin(); it != schedChan.batches.end();){
            if(it->clientBatchNumber <= completedClientBatchNum){
                ClientBatch &batch = *it;
                it = schedChan.batches.erase(it);
            }
            else
                ++it;
        }
    }

    void Scheduler::recordFinishedJobs(){
        NETPERFREC(npl, recordFinishedJobs, 1);

        // at this point, schedBarrier is completely finished
        // including combiners. We only need to record stats, record a
        // done client batch if applicable, and clean up.

        // collect stats
        NETPERFREC(npl, batchStats, 1);
#ifdef PROFILER
        // we need a lock in case we are recording timing info at the
        // same time the client is submitting a new job, which also
        // records some profiler info
        std::unique_lock<std::mutex> schedLck(schedChan.mtx);
#endif
        for(size_t worker=0; worker < nWorkers; worker++){
            JobChunkBatch &batch = schedBarrier->workerBatches[worker];
            collectStats(batch, worker);
            batch.statsRecorded = true;
        }
#ifdef PROFILER
        schedLck.unlock();
#endif
        STOPPERF(batchStats);

        // record a done client batch if applicable
        if (sequenceClientMap.contains(schedBarrier->sequence)){
            std::unique_lock<std::mutex> completedClientBatchLck(completedClientBatchMtx);
            completedClientBatchNum = sequenceClientMap[schedBarrier->sequence];
            sequenceClientMap.erase(schedBarrier->sequence);
            completedClientBatchLck.unlock();
            completedClientBatchCv.notify_all();
        }
        cleanupBarrier();
    }


    dur_t Scheduler::assignJob(Job *job, JobChunkBatch *batch, dur_t desiredDuration
#ifdef DEBUG_OP_LEVEL
                               , int worker
#endif
        ){
        size_t assignedProgress;
        if(job->indivisible || desiredDuration == dur_t::zero()){
            assignedProgress = job->maxProgress - job->progress;
        }
        else{
            assignedProgress = estimateOpsFromTime(job->opTypeIndex, desiredDuration);
            if (assignedProgress == 0)
                assignedProgress = 1;
            assignedProgress = job->nextProgressPoint(job->progress + assignedProgress) - job->progress;
            if (assignedProgress + job->progress > job->maxProgress)
                assignedProgress = job->maxProgress - job->progress;
        }
        batch->chunks.emplace_back(job->copier(*job), job->progress, job->progress + assignedProgress, job);

#ifdef DEBUG_OP_LEVEL
        std::cout << "Assigned " << job->kernelName << " " << job->progress << "-" << job->progress + assignedProgress << " to worker thread " << worker << std::endl;
#endif
        job->progress += assignedProgress;
        return estimateTimeOp(job->opTypeIndex, assignedProgress);
    }

    dur_t Scheduler::selectWater(std::list<Job *> &buckets, std::list<Job *> &waterToPour){
        std::set<int> cmpIds;
        dur_t totWater = dur_t::zero();
        for(auto it =buckets.begin(); it != buckets.end();){
            Job *j = *it;
            if (!cmpIds.contains(j->cmpId)){
                cmpIds.insert(j->cmpId);
                waterToPour.push_back(j);
                totWater += estimateTimeOp(j->opTypeIndex, j->maxProgress);
                it = buckets.erase(it);
            }
            else
                it++;
        }
        return totWater;

    }

    Scheduler::Barrier & Scheduler::newBarrier(){
        assert(lastBarrier != nullptr);
        Barrier *barrier = new Barrier(nWorkers, ++sequence);
        lastBarrier->next = barrier;
        lastBarrier = barrier;
        return *barrier;
    }

    void Scheduler::pourWater(std::list<Job *> &buckets, dur_t totWater){
        NETPERFREC(npl, bucketsOnly, 1);
        Barrier &barrier = newBarrier();

        dur_t waterLevel = totWater / nWorkers;
        auto waterIt = buckets.begin();

        auto showTime = [](dur_t t){
            return std::chrono::duration_cast<std::chrono::duration<double, std::micro> >(t).count();
        };

        for(size_t i=0; i < nWorkers; i++){


            JobChunkBatch *batch = &barrier.workerBatches[i];

            dur_t waterColumn = dur_t::zero();
            for(;waterIt != buckets.end();){
                Job *bucket = *waterIt;
                dur_t est = estimateTimeOp(bucket->opTypeIndex, bucket->maxProgress - bucket->progress);
                dur_t newHeight = waterColumn + est;
                if(newHeight < waterLevel || i == nWorkers-1){
                    // pour the whole bucket
                    // last worker gets all remaining jobs
                    waterColumn = newHeight;
                    assignJob(bucket, batch, dur_t::zero()
#ifdef DEBUG_OP_LEVEL
                              ,i
#endif
                        );
                    barrier.jobs.push_back(bucket);
                    waterIt = buckets.erase(waterIt);
                }
                else{
                    // pour as much of the bucket as we can
                    dur_t timeAvailable = waterLevel - waterColumn;
                    dur_t timeAssigned = assignJob(bucket, batch, timeAvailable
#ifdef DEBUG_OP_LEVEL
                                                   ,i
#endif
                        );
                    waterColumn += timeAssigned;
                    if (bucket->progress == bucket->maxProgress){
                        // we ended up assigning the whole job anyway
                        barrier.jobs.push_back(bucket);
                        waterIt = buckets.erase(waterIt);
                    }
                    break;
                }
            }
        }
        broadcastLatest(lastBarrier->sequence);
    }

    void Scheduler::singleThreadedSchedule(std::list<Job *> &jobs){
        NETPERFREC(npl, schedDoesWork, 1);
        Barrier &barrier = newBarrier();
        barrier.jobs.insert(barrier.jobs.end(), jobs.begin(), jobs.end());
        barrier.singleThreaded = true;
        broadcastLatest(lastBarrier->sequence);
    }

    void Scheduler::planAllStages(std::list<Job *> &jobs){
        NETPERFREC(npl, planAllStages, 1);
        for(Job *job : jobs){
            if(timeByKernel.find(job->opTypeIndex) == timeByKernel.end())
                timeByKernel[job->opTypeIndex] = PerfTracker();
        }

        std::list<Job *> waterToPour;
        while(!jobs.empty()){
            waterToPour.clear();
            dur_t totWater = selectWater(jobs, waterToPour);
            if (totWater < singleThreadThreshold){
                // if we estimate the time lost due to parallelism is greater than the gain,
                //  then the scheduler thread will just do the whole thing.)
                singleThreadedSchedule(waterToPour);
            }
            else
                pourWater(waterToPour, totWater);
        }
    }

    void Scheduler::runCombiners(std::list<Job *> &jobs){
        for(Job *j: jobs){
            j->combineAll(*j);
        }
    }

    bool Scheduler::broadcastCompleted(size_t completed, size_t workerWhoNotifies){
        bool readyBarrier = false;
        for(int worker=0; worker < nWorkers; worker++){
            WorkerChannel &chan = workChans[worker];
            std::unique_lock workLck(chan.mtx);
            chan.completedSequence = std::max<size_t>(completed, chan.completedSequence);
            if (worker == workerWhoNotifies){
                readyBarrier = chan.latestSequence > completed;
            }
            workLck.unlock();
            chan.cv.notify_all();
        }
        return readyBarrier;
    }

    void Scheduler::broadcastLatest(size_t latest){
        for(int worker=0; worker < nWorkers; worker++){
            WorkerChannel &chan = workChans[worker];
            std::unique_lock workLck(chan.mtx);
            chan.latestSequence = std::max<size_t>(latest, chan.latestSequence);
            workLck.unlock();
            chan.cv.notify_all();
        }
    }

    void Scheduler::broadcastTerminate(){
        for(int worker=0; worker < nWorkers; worker++){
            WorkerChannel &chan = workChans[worker];
            std::unique_lock workLck(chan.mtx);
            chan.terminate = true;
            workLck.unlock();
            chan.cv.notify_all();
        }
    }


    Scheduler::Barrier * Scheduler::waitForNextBarrier(int workerIndex, Barrier *barrier){
        while (true){
            WorkerChannel &chan = workChans[workerIndex];

            workerLog(workerIndex, WorkerLogEntry::Kind::GETTING_WORKCHAN_LCK);
            std::unique_lock<std::mutex> workLck(chan.mtx);
//            std::shared_lock<std::shared_mutex> bulletinLck(workerBulletin.mtx);
            workerLog(workerIndex, WorkerLogEntry::Kind::GOT_WORKCHAN_LCK);
            if (chan.terminate)
                return nullptr;
            if (chan.completedSequence >= barrier->sequence && chan.latestSequence > barrier->sequence){
                barrier->workerBatches[workerIndex].neededByWorker = false;
                return barrier->next;
            }

            chan.cv.wait(workLck);
        }
    }

    void Scheduler::workLoop(int workerIndex){
        Barrier *barrier = firstBarrier;
        bool readyBarrier = false;
        WorkerChannel &workChan = workChans[workerIndex];
        while (true){

            if (readyBarrier){
                // a faster shortcut in case this worker just notified others of a completed batch
                barrier = barrier->next;
                readyBarrier = false;
            }
            else
                barrier = waitForNextBarrier(workerIndex, barrier);
            if (workChan.terminate || barrier == nullptr)
                break;
            if (!barrier->singleThreaded){
                JobChunkBatch &batch = barrier->workerBatches[workerIndex];
                for (JobChunk &chunk : batch.chunks){
                    chunk.startTime = std::chrono::steady_clock::now();
                    chunk.task(chunk.start, chunk.end);
                    chunk.endTime = std::chrono::steady_clock::now();
                }
                std::unique_lock<std::mutex> schedLck(schedChan.mtx);
                workerLog(workerIndex, WorkerLogEntry::Kind::GOT_SCHED_LCK);
                barrier->doneWorkers++;
                if(barrier->doneWorkers == nWorkers){
                    runCombiners(barrier->jobs);
                    workerLog(workerIndex, WorkerLogEntry::Kind::RAN_COMBINERS);
                    schedLck.unlock();
                    readyBarrier = broadcastCompleted(barrier->sequence, workerIndex);
                    schedChan.cv.notify_all();
                    workerLog(workerIndex, WorkerLogEntry::Kind::BROADCAST_COMPLETE);
                }
            }
            bool singleThreadThis = false;
            if (barrier->singleThreaded && !barrier->singleThreadedStartedYet){
                std::unique_lock<std::mutex> schedLck(schedChan.mtx);
                if (!barrier->singleThreadedStartedYet){
                    barrier->singleThreadedStartedYet = true;
                    singleThreadThis = true;
                }
            }
            if (singleThreadThis){
                JobChunkBatch &batch = barrier->workerBatches[workerIndex];
                for(Job *j : barrier->jobs){
                    batch.chunks.emplace_back(j->copier(*j), 0, j->maxProgress, j);
                    JobChunk &chunk = batch.chunks.back();
                    chunk.startTime = std::chrono::steady_clock::now();
                    chunk.task(0, j->maxProgress);
                    j->combineAll(*j);
                    chunk.endTime = std::chrono::steady_clock::now();
                }
                readyBarrier = broadcastCompleted(barrier->sequence, workerIndex);
                std::unique_lock<std::mutex> schedLck(schedChan.mtx);
                barrier->doneWorkers = 1;
                schedLck.unlock();
                schedChan.cv.notify_all();
            }
        }
    }

    void Scheduler::finishBatch(size_t batchNumber){
        std::unique_lock<std::mutex> completedClientBatchLck(completedClientBatchMtx);
        while(completedClientBatchNum < batchNumber)
            completedClientBatchCv.wait(completedClientBatchLck);
    }

    void Scheduler::finishBatches(){
        // 1. find the latest clientBatchNumber
        // 2. call finishBatch on that number
        std::unique_lock<std::mutex> schedLck(schedChan.mtx);
        size_t num = clientBatchNumber;
        schedLck.unlock();
        finishBatch(num);
    }

    void Scheduler::mergeLoggers(NetworkPerfLogger &npl_client){
        finishBatches();
        std::unique_lock<std::mutex> schedLck(schedChan.mtx);
        for(int worker=0; worker < nWorkers; worker++)
            for(WorkerLogEntry &entry : workerLogs[worker]){
                std::string name;
                switch(entry.kind)
                {
                case WorkerLogEntry::Kind::GOT_SCHED_LCK:
                    name = "GOT_SCHED_LCK";
                    break;
                case WorkerLogEntry::Kind::RAN_COMBINERS:
                    name = "RAN_COMBINERS";
                    break;
                case WorkerLogEntry::Kind::BROADCAST_COMPLETE:
                    name = "BROADCAST_COMPLETE";
                    break;
                case WorkerLogEntry::Kind::GOT_WORKCHAN_LCK:
                    name = "GOT_WORKCHAN_LCK";
                    break;
                case WorkerLogEntry::Kind::GETTING_WORKCHAN_LCK:
                    name = "GETTING_WORKCHAN_LCK";
                    break;
                }
                npl.logInstant(entry.point, name, worker+2);
            }

        npl_client.mergeLoggers(npl);
        npl_client.mergeLoggers(npl_op);
    }
}

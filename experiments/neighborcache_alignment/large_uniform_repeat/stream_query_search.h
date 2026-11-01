#pragma once
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>

// The index and all cache instances survive every input batch. Only query,
// result and per-query statistics buffers are recycled. Batches are multiples
// of the worker count so static,1 preserves each query's original worker ID.
template<class T,class LabelT>
int run_stream_query_search(diskann::PQFlashIndex<T,LabelT> &index,
    const std::string &query_path,const std::string &result_prefix,
    size_t count,size_t dim,size_t aligned_dim,uint32_t threads,
    uint32_t k,uint32_t L,uint32_t beam,bool reorder,uint64_t requested_batch)
{
    if(!threads || !beam || !k || count>UINT32_MAX)
        throw std::invalid_argument("Streaming search requires positive threads/K/beam and a uint32 query count");
    const uint64_t batch=std::max<uint64_t>(threads,(requested_batch/threads)*threads);
    T *queries=nullptr;
    diskann::alloc_aligned(reinterpret_cast<void**>(&queries),batch*aligned_dim*sizeof(T),8*sizeof(T));
    struct Free {T *p;~Free(){diskann::aligned_free(p);}} cleanup{queries};
    std::ifstream input(query_path,std::ios::binary);input.seekg(8);
    std::ofstream ids_file(result_prefix+"_"+std::to_string(L)+"_idx_uint32.bin",std::ios::binary);
    std::ofstream dist_file(result_prefix+"_"+std::to_string(L)+"_dists_float.bin",std::ios::binary);
    std::ofstream windows(result_prefix+"_windows.csv");
    if(!input || !ids_file || !dist_file || !windows)throw std::runtime_error("Cannot open streaming input/output");
    const uint32_t n32=count;
    for(auto *out:{&ids_file,&dist_file}){out->write(reinterpret_cast<const char*>(&n32),4);out->write(reinterpret_cast<const char*>(&k),4);}
    windows<<"queries_completed,queries_in_window,search_seconds,window_qps,cumulative_qps,accesses,N_hits,P_hits,disk_reads,base_pages,D_reads,D_pages,latency_us_sum,io_us_sum\n";
    uint64_t accesses=0,n_hits=0,p_hits=0,disk_reads=0,base_pages=0,d_reads=0;
    double query_seconds=0,latency_us=0,io_us=0;
    std::vector<uint64_t> ids(batch*k);
    std::vector<uint32_t> ids32(batch*k);
    std::vector<float> distances(batch*k);
    std::vector<T> raw(aligned_dim==dim?0:batch*dim);
    std::vector<long> worker_tids(threads,0);
    const auto wall_start=std::chrono::steady_clock::now();
    auto report=[&](bool complete,uint64_t completed) {
        // A small checkpoint is replaced atomically. It never claims an
        // interrupted run to be a complete measurement.
        std::ofstream f(result_prefix+"_stream.json.tmp");
        f<<std::setprecision(15)<<"{\n\"complete\":"<<(complete?"true":"false")
         <<",\n\"queries\":"<<count<<",\n\"queries_completed\":"<<completed
         <<",\n\"search_seconds\":"<<query_seconds
         <<",\n\"qps\":"<<(query_seconds?completed/query_seconds:0)
         <<",\n\"end_to_end_qps\":"<<completed/std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count()
         <<",\n\"counts\":{\"n_beam_lookups\":"<<accesses<<",\"n_mem_hits\":"<<n_hits
         <<",\"n_neighborcache_prefetch_hits\":"<<p_hits<<",\"n_disk_reads\":"<<disk_reads
         <<",\"n_base_pages\":"<<base_pages<<",\"n_neighborcache_dyn_disk_reads\":"<<d_reads<<"}"
         <<",\n\"N_hit_percent\":"<<(accesses?100.0*n_hits/accesses:0)
         <<",\n\"P_hit_percent\":"<<(accesses?100.0*p_hits/accesses:0)
         <<",\n\"memory_hit_percent\":"<<(accesses?100.0*(n_hits+p_hits)/accesses:0)
         <<",\n\"mean_latency_us\":"<<(completed?latency_us/completed:0)
         <<",\n\"mean_io_us\":"<<(completed?io_us/completed:0)
         <<",\n\"batch_queries\":"<<batch<<",\n\"cache_reset_between_batches\":false\n}\n";
        f.close();
        if(std::rename((result_prefix+"_stream.json.tmp").c_str(),(result_prefix+"_stream.json").c_str())!=0)
            throw std::runtime_error("Cannot publish streaming checkpoint");
    };
    omp_set_num_threads(threads);omp_set_dynamic(0);
    for(uint64_t begin=0;begin<count;begin+=batch) {
        const size_t n=std::min<uint64_t>(batch,count-begin);
        std::vector<diskann::QueryStats> stats(n);
        if(aligned_dim==dim)input.read(reinterpret_cast<char*>(queries),n*dim*sizeof(T));
        else {
            input.read(reinterpret_cast<char*>(raw.data()),n*dim*sizeof(T));
            std::fill(queries,queries+n*aligned_dim,T{});
            for(size_t i=0;i<n;++i)std::copy(raw.data()+i*dim,raw.data()+(i+1)*dim,queries+i*aligned_dim);
        }
        if(!input)throw std::runtime_error("Short query batch read");
        const auto start=std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static,1)
        for(int64_t i=0;i<static_cast<int64_t>(n);++i) {
            neighborcache_pin_this_query_thread();
            if(static_cast<uint64_t>(i)<threads) {
                const unsigned worker=omp_get_thread_num();
                const long tid=syscall(SYS_gettid);
                if(worker_tids[worker]!=0 && worker_tids[worker]!=tid)std::abort();
                worker_tids[worker]=tid;
            }
            index.cached_beam_search(queries+i*aligned_dim,k,L,ids.data()+i*k,distances.data()+i*k,beam,reorder,&stats[i]);
        }
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        query_seconds+=seconds;
        uint64_t wa=0,wn=0,wp=0,wd=0,wb=0,wr=0;double wl=0,wi=0;
        for(size_t i=0;i<n;++i){const auto&s=stats[i];wa+=s.n_beam_lookups;wn+=s.n_cache_hits;wp+=s.n_neighborcache_prefetch_hits;
            wd+=s.n_disk_reads;wb+=s.n_unique_sectors;wr+=s.n_neighborcache_dyn_disk_reads;wl+=s.total_us;wi+=s.io_us;}
        for(size_t i=0;i<n*k;++i){if(ids[i]>UINT32_MAX)throw std::runtime_error("Result ID overflow");ids32[i]=ids[i];}
        ids_file.write(reinterpret_cast<const char*>(ids32.data()),n*k*4);
        dist_file.write(reinterpret_cast<const char*>(distances.data()),n*k*4);
        if(!ids_file || !dist_file)throw std::runtime_error("Streaming result write failed");
        accesses+=wa;n_hits+=wn;p_hits+=wp;disk_reads+=wd;base_pages+=wb;d_reads+=wr;latency_us+=wl;io_us+=wi;
        const uint64_t done=begin+n;
        windows<<std::setprecision(12)<<done<<','<<n<<','<<seconds<<','<<n/seconds<<','<<done/query_seconds
            <<','<<wa<<','<<wn<<','<<wp<<','<<wd<<','<<wb<<','<<wr<<','<<index.neighborcache_dynamic_page_count()<<','<<wl<<','<<wi<<'\n';windows.flush();
        report(false,done);
        if(done%100000==0 || done==count)diskann::cout<<"STREAM_PROGRESS queries="<<done<<" qps="<<done/query_seconds
            <<" N_hits="<<n_hits<<" P_hits="<<p_hits<<" accesses="<<accesses<<std::endl;
    }
    ids_file.close();dist_file.close();windows.close();
    index.print_neighborcache_shared_cache_stats();
    index.print_neighborcache_dynamic_3cache_stats();
    index.print_neighborcache_memory_cache_stats();
    index.enable_neighborcache_dynamic_3cache(false);
    report(true,count);
    diskann::cout<<"STREAM_COMPLETE queries="<<count<<" qps="<<count/query_seconds<<std::endl;
    return 0;
}

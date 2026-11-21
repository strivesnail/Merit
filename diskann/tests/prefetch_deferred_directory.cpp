#include "pq_flash_index.h"
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
namespace diskann {
class NeighborCacheAdapter {
 using Index=PQFlashIndex<uint8_t>;
 static constexpr uint32_t none=UINT32_MAX;
 template<class Predicate> static void wait(Predicate p) {
  const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(10);
  while(!p()){assert(std::chrono::steady_clock::now()<end);std::this_thread::yield();}
 }
 static std::array<char,4096> image(const std::vector<uint32_t>&ids) {
  std::array<char,4096> p{};
  for(size_t i=0;i<ids.size();i++){memset(p.data()+128*i,int(ids[i]%251),100);uint32_t degree=0;memcpy(p.data()+128*i+100,&degree,4);}return p;
 }
 static Index::NeighborCacheDiskLoc location(Index&idx,uint32_t page,uint16_t slot){Index::NeighborCacheDiskLoc x;x.sector=idx.NEIGHBORCACHE_DYN_SECTOR_BASE+page;x.slot=slot;x.nsectors=1;return x;}
 static bool contains(Index&idx,uint32_t id,uint32_t page,uint16_t slot){return idx.neighborcache_dc_loc_contains_unlocked(id,location(idx,page,slot));}
 static void config(Index&idx,unsigned threads,const char*path){
  idx._num_points=512;idx._max_nthreads=threads;idx._data_dim=100;idx._aligned_dim=104;idx._disk_bytes_per_point=100;idx._max_degree=6;idx._max_node_len=128;idx._nnodes_per_sector=4;
  idx._query_sector_cache_enabled=true;idx._neighborcache_shared_cache_dyn=true;assert(idx._neighborcache_shared_cache.init(2*threads,threads));
  idx.enable_neighborcache_dynamic_3cache(true,path);assert(idx.neighborcache_defer_d_locations());
 }
 static void drain(Index&idx){wait([&]{for(uint64_t p=0;p<idx._neighborcache_dyn_physical_cap;p++)if(idx._neighborcache_prefetch_writeback[p].load())return false;return true;});}
 static void rollback_race(Index&idx){
  const uint32_t page=500,seed=499;uint32_t a=200,b=201;
  while(idx.neighborcache_dyn_loc_shard_index(a)==idx.neighborcache_dyn_loc_shard_index(b)||idx.neighborcache_dyn_loc_shard_index(seed)==idx.neighborcache_dyn_loc_shard_index(b))++b;
  std::vector<uint32_t>ids{seed,a,b};Index::NeighborCacheDynWriteJob job;job.prefetch_page=page;job.prefetch_version=10;job.directory_member_count=3;std::copy(ids.begin(),ids.end(),job.directory_members.begin());
  {
   std::unique_lock<std::shared_mutex>l(idx._neighborcache_dyn_mu);const size_t begin=size_t(page)*idx._neighborcache_dyn_members_per_page;std::copy(ids.begin(),ids.end(),idx._neighborcache_dyn_page_members.ptr(begin));idx._neighborcache_dyn_page_member_count[page]=3;idx._neighborcache_dyn_page_readers[page]=1;idx._neighborcache_dyn_page_retired[page]=0;idx._neighborcache_prefetch_dirty[page]=10;idx.neighborcache_dyn_loc_add_unlocked(seed,location(idx,page,0),true);idx._neighborcache_dyn_committed_pages++;
  }
  job.prefetch_version=9;assert(!idx.neighborcache_publish_written_locations(job));assert(!contains(idx,a,page,1));assert(idx._neighborcache_prefetch_dirty[page]==10);job.prefetch_version=10;
  job.directory_members[2]=b+1;assert(!idx.neighborcache_publish_written_locations(job));assert(!contains(idx,a,page,1));job.directory_members[2]=b;
  auto &shard=idx.neighborcache_dyn_loc_shard(b);std::unique_lock<std::shared_mutex>gate(shard.mu);std::atomic<bool>retiring{false};bool published=true;
  std::thread writer([&]{published=idx.neighborcache_publish_written_locations(job);});
  wait([&]{return contains(idx,a,page,1);});
  std::thread retire([&]{std::unique_lock<std::shared_mutex>l(idx._neighborcache_dyn_mu);retiring=true;idx.neighborcache_dyn_invalidate_seed_page(seed);});
  wait([&]{return retiring.load();});gate.unlock();writer.join();retire.join();
  assert(!published&&!contains(idx,a,page,1)&&!contains(idx,b,page,2));assert(idx._neighborcache_dyn_page_member_count[page]==0);assert(idx._neighborcache_prefetch_dirty[page]==10);assert(idx._neighborcache_dyn_page_retired[page]==1);
  idx.neighborcache_dyn_release_page_pin(idx.NEIGHBORCACHE_DYN_SECTOR_BASE+page);
  {std::unique_lock<std::shared_mutex>l(idx._neighborcache_dyn_mu);idx.neighborcache_dyn_drain_reclaim_unlocked();assert(std::count(idx._neighborcache_dyn_free.begin(),idx._neighborcache_dyn_free.end(),page)==1);}
 }
 static void failure(Index&idx){
  idx.stop_neighborcache_dyn_writer();int fd=idx._neighborcache_dyn_write_fd;idx._neighborcache_dyn_write_fd=-1;
  const uint32_t p=501;std::vector<uint32_t>ids{498,210,350};auto payload=image(ids);
  {std::unique_lock<std::shared_mutex>l(idx._neighborcache_dyn_mu);size_t begin=size_t(p)*idx._neighborcache_dyn_members_per_page;std::copy(ids.begin(),ids.end(),idx._neighborcache_dyn_page_members.ptr(begin));idx._neighborcache_dyn_page_member_count[p]=3;idx._neighborcache_prefetch_dirty[p]=11;idx._neighborcache_dyn_page_readers[p]=1;idx._neighborcache_prefetch_writeback[p]=1;idx.neighborcache_dyn_loc_add_unlocked(ids[0],location(idx,p,0),true);}
  idx.start_neighborcache_dyn_writer();auto job=idx.neighborcache_dyn_writer_submit(Index::NeighborCacheDynWriteOp::Write,off_t(p)*4096,payload.data(),4096,false,p,11,ids.data(),ids.size());assert(job);drain(idx);assert(idx._neighborcache_prefetch_dirty[p]==11);assert(!contains(idx,210,p,1)&&!contains(idx,350,p,2));idx.stop_neighborcache_dyn_writer();idx._neighborcache_dyn_write_fd=fd;
 }
public:
 static void replicas(){
  setenv("NEIGHBORCACHE_DCACHE_WRITER_THREADS","2",1);
  std::shared_ptr<AlignedFileReader>r;Index idx(r,Metric::L2);char path[]="/tmp/nc_replicas_XXXXXX";int fd=mkstemp(path);assert(fd>=0);close(fd);unlink(path);config(idx,2,path);
  std::atomic<unsigned> phase{0};std::atomic<uint32_t> page{none};
  std::thread a([&]{NeighborCacheSharedPageCache::ReadSection section(idx._neighborcache_shared_cache);std::vector<uint32_t>ids{1,201,351};auto bytes=image(ids);assert(idx.neighborcache_seed_prefetch_publish_page(1,ids,bytes.data()));{std::shared_lock<std::shared_mutex> l(idx._neighborcache_dyn_mu);uint32_t p;assert(idx.neighborcache_dyn_seed_page_unlocked(1,p));page=p;}phase=1;wait([&]{return phase.load()>=3;});std::array<char,128>copy{};assert(idx._neighborcache_shared_cache.copy_private_record(201,128,copy.data(),false));assert(uint8_t(copy[0])==201);std::vector<uint8_t>v;std::vector<uint32_t>n;assert(idx.neighborcache_dyn_copy_member_payload(201,v,n));assert(v[0]==201);std::array<char,4096>z{};idx._neighborcache_shared_cache.put(1,z.data());idx._neighborcache_shared_cache.put(2,z.data());phase=4;});
  std::thread b([&]{wait([&]{return phase.load()>=1;});NeighborCacheSharedPageCache::ReadSection section(idx._neighborcache_shared_cache);std::vector<uint32_t>old{1,201,351};auto bytes=image(old);assert(idx.neighborcache_seed_prefetch_publish_page(1,old,bytes.data()));std::array<char,4096>z{};idx._neighborcache_shared_cache.put(1,z.data());idx._neighborcache_shared_cache.put(2,z.data());drain(idx);assert(contains(idx,201,page,1));std::vector<uint32_t>fresh{1,202,352};auto updated=image(fresh);assert(idx.neighborcache_seed_prefetch_admit_existing(1,page,fresh,updated.data()));assert(!contains(idx,201,page,1)&&!contains(idx,202,page,1));assert(idx.neighborcache_dyn_has_payload(202));std::vector<uint8_t>v;std::vector<uint32_t>n;assert(idx.neighborcache_dyn_copy_member_payload(202,v,n));assert(v[0]==202);phase=3;wait([&]{return phase.load()>=4;});idx._neighborcache_shared_cache.put(3,z.data());idx._neighborcache_shared_cache.put(4,z.data());drain(idx);assert(!contains(idx,201,page,1)&&contains(idx,202,page,1)&&contains(idx,352,page,2));alignas(4096)std::array<char,4096>disk{};assert(pread(idx._neighborcache_dyn_write_fd,disk.data(),4096,off_t(page.load())*4096)==4096);assert(uint8_t(disk[128])==202&&uint8_t(disk[256])==352%251);assert(idx._neighborcache_prefetch_dirty[page]==0);});
  a.join();b.join();idx.enable_neighborcache_dynamic_3cache(false);unlink((std::string(path)+"_neighborcache_dc.dyn.data").c_str());
 }
 static void run(unsigned threads,unsigned writers){
  setenv("NEIGHBORCACHE_DCACHE_WRITER_THREADS",std::to_string(writers).c_str(),1);
  std::shared_ptr<AlignedFileReader>reader;Index idx(reader,Metric::L2);char path[]="/tmp/nc_dir_XXXXXX";int fd=mkstemp(path);assert(fd>=0);close(fd);unlink(path);config(idx,threads,path);
  std::vector<std::thread>workers;std::vector<std::vector<std::pair<uint32_t,uint32_t>>>pages(threads);
  for(unsigned t=0;t<threads;t++)workers.emplace_back([&,t]{
   NeighborCacheSharedPageCache::ReadSection section(idx._neighborcache_shared_cache);
   for(unsigned i=0;i<12;i++){uint32_t seed=1+i*threads+t;std::vector<uint32_t>ids{seed,200+seed%100,350+seed%100};auto p=image(ids);assert(idx.neighborcache_seed_prefetch_publish_page(seed,ids,p.data()));uint32_t page=none;{std::shared_lock<std::shared_mutex>l(idx._neighborcache_dyn_mu);assert(idx.neighborcache_dyn_seed_page_unlocked(seed,page));}pages[t].push_back({seed,page});assert(!contains(idx,ids[1],page,1));std::array<char,128>record{};assert(idx._neighborcache_shared_cache.copy_private_record(ids[1],128,record.data()));assert(uint8_t(record[0])==ids[1]%251);}
   std::array<char,4096>zeros{};assert(idx._neighborcache_shared_cache.put(1,zeros.data()));assert(idx._neighborcache_shared_cache.put(2,zeros.data()));
  });
  for(auto&t:workers)t.join();drain(idx);
  // Nonblocking query submission may lose the queue try-lock even with space.
  // A dropped cache write must stay unreadable. Retry only in this test through
  // the blocking producer API to validate the eventual metadata/readback path.
  for(auto&pp:pages)for(auto[seed,page]:pp)if(idx._neighborcache_prefetch_dirty[page].load()){
   std::vector<uint32_t>ids{seed,200+seed%100,350+seed%100};assert(!idx.neighborcache_dyn_disk_readable(idx.NEIGHBORCACHE_DYN_SECTOR_BASE+page));assert(!contains(idx,ids[1],page,1));auto payload=image(ids);auto version=idx._neighborcache_prefetch_dirty[page].load();idx._neighborcache_prefetch_writeback[page].fetch_add(1);idx._neighborcache_dyn_page_readers[page].fetch_add(1);auto retry=idx.neighborcache_dyn_writer_submit(Index::NeighborCacheDynWriteOp::Write,off_t(page)*4096,payload.data(),4096,false,page,version,ids.data(),ids.size());assert(retry);
  }drain(idx);
  for(auto&pp:pages)for(auto[seed,page]:pp){std::vector<uint32_t>ids{seed,200+seed%100,350+seed%100};assert(idx._neighborcache_prefetch_dirty[page]==0);for(unsigned i=0;i<3;i++){assert(contains(idx,ids[i],page,i));alignas(4096) std::array<char,4096>r{};assert(pread(idx._neighborcache_dyn_write_fd,r.data(),4096,off_t(page)*4096)==4096);assert(uint8_t(r[i*128])==ids[i]%251);}assert(idx.neighborcache_dc_loc_count_unlocked(seed)==1);}
  if (threads > 1)
      rollback_race(idx); // The staged race holds one location shard while observing another.
  failure(idx);idx.enable_neighborcache_dynamic_3cache(false);unlink((std::string(path)+"_neighborcache_dc.dyn.data").c_str());
 }
};}
int main(){setenv("NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE","1",1);setenv("NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY","0",1);setenv("NEIGHBORCACHE_DCACHE_UNBOUNDED","1",1);setenv("NEIGHBORCACHE_DCACHE_CAP","512",1);setenv("NEIGHBORCACHE_MCACHE_CAP","1024",1);setenv("NEIGHBORCACHE_DCACHE_SECOND_PAGE","0",1);setenv("NEIGHBORCACHE_DCACHE_MAX_PAGES","1",1);setenv("NEIGHBORCACHE_DEFER_D_LOCATIONS","1",1);setenv("NEIGHBORCACHE_DCACHE_WRITER_QUEUE",std::getenv("NC_TEST_WRITER_QUEUE") ? std::getenv("NC_TEST_WRITER_QUEUE") : "1024",1);diskann::NeighborCacheAdapter::run(1,1);diskann::NeighborCacheAdapter::run(4,2);diskann::NeighborCacheAdapter::run(10,3);diskann::NeighborCacheAdapter::replicas();}

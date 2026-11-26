// Exercise native seed publication, newest-location order, stash fallback,
// and owned record bytes during concurrent N-cache slot reuse.
#include "pq_flash_index.h"
#include <cassert>
#include <cstring>
#include <atomic>
#include <thread>
#include <unistd.h>
namespace diskann {
class NeighborCacheAdapter {
public:
 static void directory() {
  std::shared_ptr<AlignedFileReader> reader;PQFlashIndex<uint8_t> idx(reader,Metric::L2);
  idx._num_points=256;idx._neighborcache_dyn_enabled=true;idx.neighborcache_canon_reset();idx._neighborcache_dyn_page_member_count.assign(8,0);idx._neighborcache_dyn_page_secondary.assign(8,0);
  for (unsigned page=0;page<8;++page) idx._neighborcache_dyn_page_member_count[page]=1;
  const auto base=idx.NEIGHBORCACHE_DYN_SECTOR_BASE;
  for(unsigned i=0;i<5;i++){typename PQFlashIndex<uint8_t>::NeighborCacheDiskLoc loc;loc.sector=base+i;loc.slot=0;loc.nsectors=1;idx.neighborcache_dyn_loc_add_unlocked(7,loc);idx.neighborcache_dyn_loc_add_unlocked(7,loc);}
  assert(idx.neighborcache_dc_loc_count_unlocked(7)==5);
  auto verify=[&](const std::vector<unsigned>&want){std::vector<unsigned> got;idx.neighborcache_dc_for_each_loc_unlocked(7,[&](const auto &loc,size_t i){assert(i==got.size());got.push_back(loc.sector-base);});assert(got==want);for(size_t i=0;i<want.size();i++){typename PQFlashIndex<uint8_t>::NeighborCacheDiskLoc loc;assert(idx.neighborcache_dc_loc_at_unlocked(7,i,loc));assert(loc.sector==base+want[i]);}uint32_t seed=99;assert(idx.neighborcache_dyn_seed_page_unlocked(7,seed));assert(seed==want.front());typename PQFlashIndex<uint8_t>::NeighborCacheDiskLoc canon;assert(idx.neighborcache_canon_lookup(7,canon)==1);assert(canon.sector==base+want.front());};
  verify({4,3,2,1,0});typename PQFlashIndex<uint8_t>::NeighborCacheDiskLoc fresh;fresh.sector=base+5;fresh.slot=0;fresh.nsectors=1;idx.neighborcache_dyn_loc_add_unlocked(7,fresh,true);verify({5,4,3,2,1,0});idx.neighborcache_dyn_loc_erase_sector_unlocked(7,base+5);verify({4,3,2,1,0});idx.neighborcache_dyn_loc_erase_sector_unlocked(7,base+4);verify({3,2,1,0});idx.neighborcache_dyn_loc_erase_sector_unlocked(7,base+2);verify({3,1,0});
  auto &locations = idx.neighborcache_dyn_loc_shard(7).locations;
  locations.reserve(64);
  verify({3, 1, 0});
  idx.neighborcache_dyn_loc_erase_sector_unlocked(7, base + 1);
  idx.neighborcache_dyn_loc_erase_sector_unlocked(7, base + 3);
  verify({0});
  assert(locations.at(7).heap_capacity() == 0);
  idx.neighborcache_dyn_loc_add_unlocked(7, fresh, true);
  verify({5, 0});
  idx.neighborcache_dyn_loc_erase_sector_unlocked(7, base);
  verify({5});
  // Erasing node 7 moves the last value in the segmented array.
  fresh.sector = base + 6;
  idx.neighborcache_dyn_loc_add_unlocked(12, fresh, true);
  fresh.sector = base + 7;
  idx.neighborcache_dyn_loc_add_unlocked(12, fresh, true);
  idx.neighborcache_dyn_loc_erase_sector_unlocked(7, base + 5);
  assert(idx.neighborcache_dc_loc_count_unlocked(7) == 0);
  assert(idx.neighborcache_dc_loc_count_unlocked(12) == 2);
  typename PQFlashIndex<uint8_t>::NeighborCacheDiskLoc moved;
  assert(idx.neighborcache_dc_loc_at_unlocked(12, 0, moved) && moved.sector == base + 7);
  assert(idx.neighborcache_dc_loc_at_unlocked(12, 1, moved) && moved.sector == base + 6);
  idx.neighborcache_dyn_loc_add_unlocked(7, fresh);
  idx.neighborcache_dyn_loc_clear_unlocked();
  assert(idx.neighborcache_dc_loc_count_unlocked(7) == 0);
  assert(idx.neighborcache_dc_loc_count_unlocked(12) == 0);
  idx._neighborcache_dyn_enabled = false;
 }
 template<class T> static void pages(bool concurrent) {
  std::shared_ptr<AlignedFileReader> reader;PQFlashIndex<T> idx(reader,Metric::L2);
  idx._num_points=256;idx._max_nthreads=1;idx._data_dim=100;idx._aligned_dim=104;idx._disk_bytes_per_point=100*sizeof(T);idx._max_degree=6;idx._max_node_len=idx._disk_bytes_per_point+28;idx._nnodes_per_sector=4;
  idx._neighborcache_mem_pool=std::make_unique<NeighborCacheMemoryPool<T>>();idx._neighborcache_mem_pool->init(128,104,6,256);idx._neighborcache_mem_pool->commit_initial_load({},{});idx._query_sector_cache_enabled=true;idx._neighborcache_shared_cache_dyn=true;assert(idx._neighborcache_shared_cache.init(128,1));
  char prefix[]="/tmp/nc_maint_XXXXXX";int fd=mkstemp(prefix);assert(fd>=0);close(fd);unlink(prefix);idx.enable_neighborcache_dynamic_3cache(true,prefix);auto section=std::make_unique<NeighborCacheSharedPageCache::ReadSection>(idx._neighborcache_shared_cache);
  auto make=[&](uint32_t id,bool seed){std::vector<char> r(idx._max_node_len,0);std::vector<T> v(100,T(id%19));memcpy(r.data(),v.data(),idx._disk_bytes_per_point);uint32_t hood[]={6,3,3,4,250,5,UINT32_MAX};if(!seed)hood[0]=0;memcpy(r.data()+idx._disk_bytes_per_point,hood,sizeof(hood));return r;};
  auto three=make(3,false),four=make(4,false);idx._neighborcache_mem_pool->try_admit(3,three.data(),idx._disk_bytes_per_point,idx._max_node_len,0);idx._neighborcache_mem_pool->try_admit(4,four.data(),idx._disk_bytes_per_point,idx._max_node_len,0);
  {std::lock_guard<std::mutex> lock(idx._neighborcache_stash_mu);auto &e=idx._neighborcache_payload_stash[5];e.coords.assign(104,T(5));e.nbrs={};}
  std::atomic<bool> stop{false};std::thread writer;if(concurrent)writer=std::thread([&]{while(!stop.load()){idx._neighborcache_mem_pool->erase(3);idx._neighborcache_mem_pool->try_admit(3,three.data(),idx._disk_bytes_per_point,idx._max_node_len,0);}});
  for(uint32_t seed=10;seed<90;seed++){
   auto record=make(seed,true);idx._neighborcache_mem_pool->try_admit(seed,record.data(),idx._disk_bytes_per_point,idx._max_node_len,0);assert(idx._neighborcache_mem_pool->contains(seed));std::array<char,4096> check{};assert(idx.neighborcache_copy_resident_record(seed,check.data(),idx._max_node_len));idx.neighborcache_seed_page_open(seed);
   std::array<char,defaults::SECTOR_LEN> p{};std::vector<uint32_t> ids;uint32_t slot=99;assert(idx._neighborcache_shared_cache.private_record_page_snapshot(seed,idx._max_node_len,slot,ids,p.data()));assert(ids.front()==seed);if(!concurrent)assert((ids==std::vector<uint32_t>{seed,3,4,5}));
   for(size_t k=0;k<ids.size();k++){T first;memcpy(&first,p.data()+k*idx._max_node_len,sizeof(T));assert(first==T(ids[k]%19));}assert(std::find(ids.begin(),ids.end(),4)!=ids.end());assert(std::find(ids.begin(),ids.end(),5)!=ids.end());
  }
  stop=true;if(writer.joinable())writer.join();
  if (!concurrent)
  {
      const auto unrelated = make(6, false);
      idx._neighborcache_mem_pool->try_admit(6, unrelated.data(), idx._disk_bytes_per_point, idx._max_node_len, 0);
      assert(idx._neighborcache_mem_pool->contains(6));
      for (const uint32_t seed : {90u, 91u})
      {
          const auto record = make(seed, true);
          idx._neighborcache_mem_pool->try_admit(seed, record.data(), idx._disk_bytes_per_point, idx._max_node_len, 0);
          NeighborCacheMetadataCache::Snapshot ranking;
          if (seed == 90)
              ranking.significant = {{6, 10}, {4, 3}};
          assert(idx.neighborcache_seed_page_open(seed, &ranking) == PQFlashIndex<T>::NeighborCacheSeedOpen::Published);
          std::array<char, defaults::SECTOR_LEN> payload{};
          std::vector<uint32_t> ids;
          uint32_t slot = UINT32_MAX;
          assert(idx._neighborcache_shared_cache.private_record_page_snapshot(
              seed, idx._max_node_len, slot, ids, payload.data()));
          const std::vector<uint32_t> expected = seed == 90 ? std::vector<uint32_t>{seed, 4, 3}
                                                           : std::vector<uint32_t>{seed, 3, 4};
          assert(ids == expected);
      }
  }
  {std::lock_guard<std::mutex> lock(idx._neighborcache_stash_mu);idx._neighborcache_payload_stash.clear();}
  section.reset();idx.enable_neighborcache_dynamic_3cache(false);unlink((std::string(prefix)+"_neighborcache_dc.dyn.data").c_str());
 }
};}
int main(){setenv("NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE","1",1);setenv("NEIGHBORCACHE_DCACHE_CAP","128",1);setenv("NEIGHBORCACHE_MCACHE_CAP","512",1);setenv("NEIGHBORCACHE_DCACHE_UNBOUNDED","1",1);setenv("NEIGHBORCACHE_DCACHE_SECOND_PAGE","0",1);diskann::NeighborCacheAdapter::directory();diskann::NeighborCacheAdapter::pages<uint8_t>(false);diskann::NeighborCacheAdapter::pages<int8_t>(false);diskann::NeighborCacheAdapter::pages<float>(false);diskann::NeighborCacheAdapter::pages<uint8_t>(true);}

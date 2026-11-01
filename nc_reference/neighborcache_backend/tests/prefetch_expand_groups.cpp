#include "nc_client.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <set>
int main() {
 namespace fs=std::filesystem;
 auto root=fs::temp_directory_path()/("nc_expand_groups_"+std::to_string(getpid()));fs::create_directory(root);
 auto prefix=(root/"prefill").string(),runtime=(root/"runtime").string();
 auto record=[](uint32_t id){std::array<char,388> r{};for(size_t j=0;j<128;++j)r[j]=char((id*17+j)%251);uint32_t degree=1,next=id+1;memcpy(r.data()+128,&degree,4);memcpy(r.data()+132,&next,4);return r;};
 {
  std::ofstream slots(prefix+".slots",std::ios::binary),pages(prefix+".pages",std::ios::binary);
  uint64_t header[]{2048,388,128,64,10,3};slots.write("NCHOT2P\0",8);slots.write((char*)header,sizeof(header));
  uint32_t ids[3][10]{{1,2,3,4,5,6,7,8,9,10},{1,11,12,13,14,15,16,17,18,19},{30,31,30,30,30,30,30,30,30,30}};
  for(auto&row:ids){std::array<char,4096> page{};for(int j=0;j<10;j++)memcpy(page.data()+j*388,record(row[j]).data(),388);slots.write((char*)row,sizeof(row));pages.write(page.data(),page.size());}
 }
 setenv("NEIGHBORCACHE_HOT_PREFILL_PREFIX",prefix.c_str(),1);setenv("NEIGHBORCACHE_DYNAMIC_PREFILL_STATIC","1",1);
 setenv("NEIGHBORCACHE_FREEZE_MAINTENANCE","1",1);setenv("NEIGHBORCACHE_DCACHE_SECOND_PAGE","1",1);
 setenv("NEIGHBORCACHE_DCACHE_MAX_PAGES","2",1);setenv("NEIGHBORCACHE_DCACHE_CAP","3",1);
 setenv("NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES","3",1);setenv("NEIGHBORCACHE_EXPAND_PREFETCH_PAGE","1",1);
 {
  neighborcache::Client c;NCConfig cfg{NC_BACKEND_ABI,4,2048,388,128,64,10,128,128,1,runtime.c_str()};c.init(cfg);
  c.set_verifier([&](uint32_t id,const char*raw){assert(memcmp(raw,record(id).data(),388)==0);});
  neighborcache::Client::QueryGuard query(c);char raw[388]{};NCRead req{};
  assert(c.resolve(11,raw,req)==3&&req.bytes==8192);
  assert(c.complete(req,pread(req.fd,req.buffer,req.bytes,req.offset),raw));
  std::vector<std::pair<uint32_t,std::vector<char>>> extras;c.append_prefetch_members(11,extras);assert(extras.empty());
  for(uint32_t target:{2u,19u}) {
   assert(c.resolve(target,raw,req)==2);extras.clear();c.append_prefetch_members(target,extras);assert(extras.size()==18);
   std::set<uint32_t> got;for(auto &x:extras){assert(x.first!=target);assert(memcmp(x.second.data(),record(x.first).data(),388)==0);assert(got.insert(x.first).second);}
   for(uint32_t id=1;id<=19;++id)assert(got.count(id)==unsigned(id!=target));
   c.append_prefetch_members(target,extras);assert(extras.size()==18);
  }
  assert(c.stats().d_reads==1);
  assert(c.resolve(31,raw,req)==3&&req.bytes==4096);assert(c.complete(req,pread(req.fd,req.buffer,req.bytes,req.offset),raw));
  assert(c.resolve(31,raw,req)==2);extras.clear();c.append_prefetch_members(31,extras);assert(extras.size()==1&&extras[0].first==30);
  auto r=record(100);c.expand(100,r.data());assert(c.resolve(100,raw,req)==1);extras.clear();c.append_prefetch_members(100,extras);assert(extras.empty());
  assert(c.resolve(200,raw,req)==0);c.append_prefetch_members(200,extras);assert(extras.empty());assert(c.stats().m_entries==0);
 }
 fs::remove_all(root);
}

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
constexpr size_t PAGE=4096,REC=388,N=100000000,FOOT=4052,CHUNK=8192;
uint32_t u32(const char*p){uint32_t x;memcpy(&x,p,4);return x;}
int main(int argc,char**argv){
 if(argc!=4)return 2;
 std::ifstream sf(argv[2],std::ios::binary|std::ios::ate);size_t sz=sf.tellg();if(!sf||!sz||sz%4)throw std::runtime_error("seed IDs");
 std::vector<uint32_t> seeds(sz/4);sf.seekg(0);sf.read((char*)seeds.data(),sz);if(!sf)throw std::runtime_error("seed list read");
 std::vector<uint8_t> visited(N,0);for(auto s:seeds){if(s>=N||visited[s])throw std::runtime_error("invalid seed");visited[s]=1;}
 int input=open(argv[1],O_RDONLY);struct stat st;if(input<0||fstat(input,&st))throw std::runtime_error("input");
 const char *base=(const char*)mmap(nullptr,st.st_size,PROT_READ,MAP_SHARED,input,0);if(base==MAP_FAILED)throw std::runtime_error("mmap");
 auto raw=[&](uint32_t id){return base+(1+size_t(id)/10)*PAGE+(id%10)*REC;};
 char *buf=(char*)aligned_alloc(PAGE,CHUNK*PAGE);if(!buf)throw std::runtime_error("buffer");
 int out=open(argv[3],O_WRONLY|O_CREAT|O_EXCL|O_DIRECT,0644);if(out<0||posix_fallocate(out,0,seeds.size()*PAGE))throw std::runtime_error("output");
 uint64_t members=0;auto start=std::chrono::steady_clock::now();
 for(size_t begin=0;begin<seeds.size();begin+=CHUNK){size_t count=std::min(CHUNK,seeds.size()-begin);
 #pragma omp parallel for num_threads(8) schedule(static) reduction(+:members)
  for(size_t j=0;j<count;j++){
   uint32_t seed=seeds[begin+j],ids[10]={seed},n=1;const char *rec=raw(seed);uint32_t degree=u32(rec+128);if(degree>64)std::abort();
   for(uint32_t k=0;k<degree&&n<10;k++){uint32_t id=u32(rec+132+4*k);if(id>=N)std::abort();if(!visited[id])continue;bool found=false;for(uint32_t i=0;i<n;i++)found|=ids[i]==id;if(!found)ids[n++]=id;}
   char *dst=buf+j*PAGE;memset(dst,0,PAGE);for(uint32_t i=0;i<n;i++)memcpy(dst+i*REC,raw(ids[i]),REC);memcpy(dst+FOOT,&n,4);memcpy(dst+FOOT+4,ids,n*4);members+=n;
  }
  size_t todo=count*PAGE;char *p=buf;while(todo){ssize_t v=write(out,p,todo);if(v<=0)throw std::runtime_error("write");p+=v;todo-=v;}
  if(begin%(CHUNK*16)==0)std::cout<<"BUILT "<<begin+count<<" / "<<seeds.size()<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<std::endl;
 }
 fsync(out);close(out);munmap((void*)base,st.st_size);close(input);free(buf);std::cout<<"COMPLETE seeds="<<seeds.size()<<" bytes="<<seeds.size()*PAGE<<" nonpadding_members="<<members<<std::endl;
}

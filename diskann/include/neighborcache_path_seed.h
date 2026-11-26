#include "residual_probe.h"
#pragma once
#include "aligned_file_reader.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <libaio.h>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <unistd.h>

namespace diskann {
class PathSeedCache {
    static constexpr unsigned block=4096, shards=64;
    struct alignas(block) Bytes { char data[block]{}; };
    struct Image { std::shared_ptr<Bytes> bytes; unsigned count=0; int worker=-1; };
    struct Page {
        uint64_t pos;
        std::mutex io;
        Image dirty;
        unsigned saved=0;
        std::atomic<unsigned> readable{0};
        bool queued=false;
        std::shared_future<Image> flight;
        const uint32_t owner;
        uint8_t owners=1;
        // Slots are immutable once published. Concurrent readers only scan a released prefix.
        std::array<uint32_t,10> ids;
        std::atomic<unsigned> count{0};
        Page(uint64_t n,uint32_t seed):pos(n),owner(seed){}
        unsigned slot(uint32_t node) const {
            unsigned n=count.load(std::memory_order_acquire);
            auto at=std::find(ids.begin(),ids.begin()+n,node);
            return at==ids.begin()+n ? UINT32_MAX : unsigned(at-ids.begin());
        }
    };
    struct Seed { std::mutex mu; std::vector<std::shared_ptr<Page>> pages; };
    struct Directory { std::shared_mutex mu; std::unordered_map<uint32_t,std::unique_ptr<Seed>> seeds; };
    struct Cache {
        std::mutex mu;
        size_t cap=0;
        std::list<Page*> lru;
        struct Entry { Image image; std::list<Page*>::iterator at; };
        std::unordered_map<Page*,Entry> entries;
        Image get(Page* p,unsigned slot) {
            std::lock_guard lock(mu);auto it=entries.find(p);
            if(it==entries.end() || it->second.image.count<=slot)return {};
            lru.splice(lru.begin(),lru,it->second.at); return it->second.image;
        }
        void put(Page* p,const Image& img) {
            std::lock_guard lock(mu);auto it=entries.find(p);
            if(it!=entries.end()){
                if(it->second.image.count<=img.count)it->second.image=img;
                lru.splice(lru.begin(),lru,it->second.at);return;
            }
            if(entries.size()==cap){entries.erase(lru.back());lru.pop_back();}
            lru.push_front(p);entries.emplace(p,Entry{img,lru.begin()});
        }
    };
    Directory dir[shards];
    std::unique_ptr<Cache[]> buffers;
    unsigned width, slots;
    bool shared;
    std::array<int, 2> files{-1, -1};
    std::array<std::atomic<uint64_t>, 2> writes_per_disk{};
    int page_file(uint64_t page) const {
        return files[files[1] >= 0 ? (page & 1) : 0];
    }
    uint64_t page_offset(uint64_t page) const {
        return (files[1] >= 0 ? page >> 1 : page) * block;
    }
    void close_files() {
        for (int file : files) if (file >= 0) ::close(file);
    }
    std::atomic<uint64_t> allocated{0},writes{0},merge_reads{0},cross{0},flights{0};
    std::mutex queue_mu; std::condition_variable ready,space;
    std::deque<Page*> queue;
    size_t queue_capacity=1024;
    size_t peak_queued=0;
    // Both producer submissions and writer retries hold queue_mu here.
    void push_locked(Page* page) {
        queue.push_back(page);
        peak_queued=std::max(peak_queued,queue.size());
        ready.notify_one();
    }
    std::atomic<int64_t> last_completion_ns{0};
    bool stopping=false;
    std::vector<std::thread> writers;
    void finish_writes() {
        { std::lock_guard lock(queue_mu); stopping=true; }
        ready.notify_all();
        for(auto& writer : writers) writer.join();
    }
    static unsigned shard(uint32_t n){n^=n>>16;n*=0x7feb352du;n^=n>>15;return n&(shards-1);}
    Cache& cache(uint32_t owner,int worker){return buffers[shared?shard(owner):worker];}
    Seed* find(uint32_t id,bool create=false){
        if(id==UINT32_MAX)return nullptr;
        auto& d=dir[shard(id)];
        {std::shared_lock lock(d.mu);auto it=d.seeds.find(id);if(it!=d.seeds.end())return it->second.get();}
        if(!create)return nullptr;
        std::unique_lock lock(d.mu);auto& s=d.seeds[id];if(!s)s=std::make_unique<Seed>();return s.get();
    }
    struct Op {Page* p;Image image;};
    void transfer(io_context_t ctx,std::vector<Op>& ops,bool writing){
        for(size_t off=0;off<ops.size();off+=64){
            const size_t n=std::min<size_t>(64,ops.size()-off);
            struct iocb cb[64]; struct iocb* ptr[64]; struct io_event events[64];
            for(size_t i=0;i<n;++i){
                auto& o=ops[off+i];auto offset=page_offset(o.p->pos); const int fd=page_file(o.p->pos);
                if(writing)io_prep_pwrite(&cb[i],fd,o.image.bytes->data,block,offset);
                else io_prep_pread(&cb[i],fd,o.image.bytes->data,block,offset);
                ptr[i]=&cb[i];
            }
            size_t done=0;
            while(done<n){
                long sent=io_submit(ctx,n-done,ptr+done);
                if(sent==-EINTR || sent==-EAGAIN)continue;
                if(sent<=0)throw std::runtime_error("path cache io_submit failed");
                long completed=0;
                while(completed<sent){
                    long nr=io_getevents(ctx,1,sent-completed,events,nullptr);
                    if(nr==-EINTR)continue;
                    if(nr<=0)throw std::runtime_error("path cache io_getevents failed");
                    for(long i=0;i<nr;i++)if(events[i].res!=block)throw std::runtime_error("short path page I/O");
                    completed+=nr;
                }
                done+=sent;
            }
        }
    }
    void enqueue(Page* p){
        residual::Scope sample_enqueue(residual::Enqueue);
        std::unique_lock lock(queue_mu);
        if (queue_capacity && queue.size()>=queue_capacity) {
            residual::Scope sample_full(residual::FullQueue);
            space.wait(lock,[&]{return queue.size()<queue_capacity;});
        }
        push_locked(p);
    }
    void write_loop(void(*pin)()){
        int64_t last_batch_ns=0;
        uint64_t local_writes[2]{};
        pin();io_context_t ctx=0;if(io_setup(64,&ctx)!=0)throw std::runtime_error("path writer io_setup failed");
        for(;;){
            std::vector<Page*> batch;
            {std::unique_lock lock(queue_mu);ready.wait(lock,[&]{return stopping||!queue.empty();});
             if(stopping&&queue.empty())break;
             while(!queue.empty()&&batch.size()<64){batch.push_back(queue.front());queue.pop_front();}space.notify_all();}
            std::vector<Op> ops;
            for(auto p:batch){std::lock_guard lock(p->io);ops.push_back({p,p->dirty});}
            residual::writing.fetch_add(1,std::memory_order_relaxed);
            transfer(ctx,ops,true);
            residual::writing.fetch_sub(1,std::memory_order_relaxed);
            residual::writer_batches.fetch_add(1,std::memory_order_relaxed);
            residual::writer_pages.fetch_add(ops.size(),std::memory_order_relaxed);
            writes+=ops.size();
            for(auto& op:ops){
                ++local_writes[files[1] >= 0 ? (op.p->pos & 1) : 0];
                auto p=op.p;std::lock_guard lock(p->io);
                p->saved=op.image.count;p->readable.store(p->saved,std::memory_order_release);
                if(p->saved==p->dirty.count){p->dirty={};p->queued=false;}
                else {std::lock_guard qlock(queue_mu);push_locked(p);}
            }
            // Set after the batch is written and the saved slot counts have been published.
            last_batch_ns=residual::now_ns();
        }
        auto prior=last_completion_ns.load(std::memory_order_relaxed);
        while(prior<last_batch_ns && !last_completion_ns.compare_exchange_weak(prior,last_batch_ns,std::memory_order_relaxed)) {}
        for (unsigned disk = 0; disk < 2; ++disk) writes_per_disk[disk] += local_writes[disk];
        io_destroy(ctx);
    }
public:
    size_t waiting_pages() {
        std::lock_guard lock(queue_mu); return queue.size();
    }
    struct Record {uint32_t node,parent,depth;std::vector<char> data;};
    struct Query {
        std::vector<Record> records;
        std::unordered_map<uint32_t,size_t> index;
        void add(uint32_t node,uint32_t parent,const char* coords,unsigned dim,const uint32_t* nbr,unsigned n,unsigned width){
            auto it=index.find(parent);
            if(parent!=UINT32_MAX&&it==index.end())throw std::runtime_error("unobserved path parent");
            unsigned depth=it==index.end()?0:records[it->second].depth+1;
            if(dim+4+n*4>width)throw std::runtime_error("oversized node record");
            index.emplace(node,records.size());records.push_back({node,parent,depth,{}});
            if(depth<3)return;
            auto& data=records.back().data;data.resize(width);
            memcpy(data.data(),coords,dim);memcpy(data.data()+dim,&n,4);
            if(n)memcpy(data.data()+dim+4,nbr,n*4);
        }
    };
    struct Location {Page* p=nullptr;uint32_t owner=0,slot=0;Image image;};
    struct Request {uint32_t node;Location location;};
    PathSeedCache(const std::string& file,unsigned rec,unsigned capacity,unsigned workers,bool sharing,void(*pin)())
        :width(rec),slots(std::min<unsigned>(10,block/rec)),shared(sharing){
        unsigned count=shared?shards:workers;
        if(slots<2 || capacity<count)throw std::runtime_error("invalid path page capacity");
        buffers=std::make_unique<Cache[]>(count);
        for(unsigned i=0;i<count;i++)buffers[i].cap=capacity/count+(i<capacity%count);
        files[0]=open(file.c_str(),O_CREAT|O_EXCL|O_RDWR|O_DIRECT|O_CLOEXEC,0600);
        if(files[0]<0)throw std::runtime_error("cannot create path cache "+file);
        if (const char* second = std::getenv("MERIT_D_ODD")) {
            files[1] = ::open(second, O_CREAT|O_EXCL|O_RDWR|O_DIRECT|O_CLOEXEC, 0600);
            if (files[1] < 0) { close_files(); throw std::runtime_error("cannot create second D stripe"); }
        }
        const char* setting=std::getenv("NEIGHBORCACHE_DCACHE_WRITER_THREADS");
        unsigned count_writers=setting ? std::stoul(setting) : 1;
        if(!count_writers) throw std::invalid_argument("zero path writers");
        if(const char* limit=std::getenv("MERIT_PATH_QUEUE_PAGES")) {
            queue_capacity=std::stoul(limit);
            // Zero selects unbounded producer admission for this experiment.
        }
        std::cout<<"producer_queue_limit="<<queue_capacity<<std::endl;
        writers.reserve(count_writers);
        try {
            for(unsigned i=0;i<count_writers;++i)
                writers.emplace_back([this,pin]{write_loop(pin);});
        } catch(...) {
            finish_writes(); close_files(); throw;
        }
        std::cout<<"path_writer_count="<<count_writers<<std::endl;
    }
    ~PathSeedCache(){
        finish_writes();
        auto drain_ns=std::max<int64_t>(0,last_completion_ns.load()-residual::query_end_ns.load());
        std::cout<<"peak_queued_pages="<<peak_queued<<std::endl;
        std::cout<<"after_query_drain_ns="<<drain_ns<<std::endl;
        // The measured query loop has already finished; audit the drained state.
        uint64_t checked=0;
        std::array<uint64_t,4> by_owners{}, filled_slots{};
        for(const auto& shard_dir : dir)
            for(const auto& entry : shard_dir.seeds)
                for(const auto& page : entry.second->pages) {
                    if(page->owner!=entry.first) continue;
                    ++by_owners.at(page->owners);
                    filled_slots.at(page->owners)+=page->saved;
                    if(page->queued || page->dirty.bytes || page->saved!=page->count.load() ||
                       page->readable.load()!=page->saved)
                        std::terminate();
                    ++checked;
                }
        if(!queue.empty() || checked!=allocated.load()) std::terminate();
        std::cout<<"drained_pages_checked="<<checked<<std::endl;
        for(unsigned owners=1;owners<=3;++owners)
            std::cout<<"seed_owners="<<owners<<" pages="<<by_owners[owners]
                     <<" slots_used="<<filled_slots[owners]<<std::endl;
        close_files();
        std::cout << "D_page_writes_by_disk=" << writes_per_disk[0] << "," << writes_per_disk[1] << std::endl;
        std::cout<<"path cache pages="<<allocated<<" writes="<<writes<<" merge_reads="<<merge_reads
                 <<" cross_worker_hits="<<cross<<" coalesced_worker_reads="<<flights<<std::endl;
    }
    Location lookup(uint32_t node,uint32_t parent,int worker){
        Location disk;
        for(auto owner:{node,parent}){
            auto s=find(owner);if(!s)continue;
            std::lock_guard lock(s->mu);
            for(auto& page:s->pages){
                auto p=page.get();
                unsigned slot=p->slot(node);
                if(slot==UINT32_MAX)continue;
                auto img=cache(p->owner,worker).get(p,slot);
                if(img.bytes){if(img.worker!=worker)++cross;return {p,owner,slot,img};}
                if(!disk.p && p->readable.load(std::memory_order_acquire)>slot)disk={p,owner,slot,{}};
                break;
            }
        }
        return disk;
    }
    const char* record(const Location& l)const{return l.image.bytes->data+width*l.slot;}
    unsigned fetch(std::vector<Request>& req,int worker,io_context_t ctx, AlignedFileReader& reader, std::vector<FdAlignedRead>& combined){
        struct Load {std::shared_ptr<std::promise<Image>> promise;std::shared_future<Image> future;Image image;uint32_t owner;};
        std::unordered_map<Page*,Load> loads;std::vector<Op> ops;
        for(auto& r:req){
            auto& loc=r.location;auto p=loc.p;if(loads.count(p))continue;
            std::lock_guard lock(p->io);
            Load job{};job.owner=p->owner;
            // An old prefix can remain resident while newer slots become readable.
            job.image=cache(p->owner,worker).get(p,p->saved-1);
            if(!job.image.bytes){
                if(shared&&p->flight.valid()){job.future=p->flight;++flights;}
                else {
                    job.promise=std::make_shared<std::promise<Image>>();job.future=job.promise->get_future().share();
                    if(shared)p->flight=job.future;
                    job.image={std::make_shared<Bytes>(),p->saved,worker};
                    ops.push_back({p,job.image});
                }
            }
            loads.emplace(p,std::move(job));
        }
        for (const auto& op : ops)
            combined.emplace_back(page_file(op.p->pos), page_offset(op.p->pos), block, op.image.bytes->data);
        if (!combined.empty()) reader.read_multi(combined,ctx);
        for(auto& it:loads){
            auto& j=it.second;if(!j.promise)continue;
            {std::lock_guard lock(it.first->io);cache(j.owner,worker).put(it.first,j.image);if(shared)it.first->flight={};}
            j.promise->set_value(j.image);
        }
        for(auto& r:req){
            auto& j=loads.at(r.location.p);
            if(j.image.bytes) r.location.image=j.image;
            else { residual::Scope joining(residual::InflightJoin); r.location.image=j.future.get(); }
            if(r.location.slot>=r.location.image.count){
                // An append completed after another worker submitted its read of the old prefix.
                std::vector<Op> retry{{r.location.p,{std::make_shared<Bytes>(),r.location.p->readable.load(),worker}}};
                transfer(ctx,retry,false);ops.push_back(retry.front());r.location.image=retry.front().image;
                cache(r.location.p->owner,worker).put(r.location.p,r.location.image);
            }
        }
        return ops.size();
    }
private:
    using Group=std::vector<const Record*>;
    unsigned pack_new(Query& q, const std::vector<uint32_t>& order, size_t i,
                      const std::unordered_map<uint32_t,Group>& groups, int worker) {
        Group members;
        std::array<Seed*,3> seeds{};
        unsigned count=0;
        size_t pair_size=0;
        for(;count<3 && i+count<order.size();++count) {
            uint32_t id=order[i+count];
            auto add=[&](const Record* record) {
                if(std::none_of(members.begin(),members.end(),
                                [&](const Record* e){return e->node==record->node;}))
                    members.push_back(record);
            };
            add(&q.records[q.index.at(id)]);
            for(auto r:groups.at(id)) add(r);
            if(members.size()>slots) break;
            seeds[count]=find(id,true);
            if(count==1) pair_size=members.size();
        }
        if(count<2) return 0;
        // Lock seed records in a common order even when two queries encounter
        // the same seeds in different expansion orders.
        auto sorted=seeds;
        std::sort(sorted.begin(),sorted.begin()+count,std::less<Seed*>{});
        if(std::adjacent_find(sorted.begin(),sorted.begin()+count)!=sorted.begin()+count)
            return 0;
        std::array<std::unique_lock<std::mutex>,3> locks;
        for(unsigned j=0;j<count;++j) locks[j]=std::unique_lock<std::mutex>(sorted[j]->mu);
        if(!seeds[0]->pages.empty() || !seeds[1]->pages.empty()) return 0;
        if(count==3 && !seeds[2]->pages.empty()) count=2;
        if(count==2) members.resize(pair_size);
        auto p=std::make_shared<Page>(allocated.fetch_add(1),order[i]);
        p->owners=count;
        Image img{std::make_shared<Bytes>(),0,worker};
        for(auto r:members) {
            p->ids[img.count]=r->node;
            memcpy(img.bytes->data+width*img.count++,r->data.data(),width);
        }
        p->dirty=img;
        p->queued=true;
        p->count.store(img.count,std::memory_order_release);
        for(unsigned j=0;j<count;++j) seeds[j]->pages.push_back(p);
        cache(order[i],worker).put(p.get(),img);
        enqueue(p.get());
        return count;
    }
    void append(Query& q, uint32_t seed, const Group& group, Seed& s, int worker) {
        residual::Scope meta(residual::SeedMetadata);
        Group missing;
        for(auto r:group)
            if(std::none_of(s.pages.begin(),s.pages.end(),
                            [&](const auto& p){return p->slot(r->node)!=UINT32_MAX;}))
                missing.push_back(r);
        meta.stop();
        size_t i=0;
        while(i<missing.size()) {
            if(s.pages.empty() || s.pages.back()->count.load(std::memory_order_acquire)==slots) {
                // A concurrent append by the other seed may already cover this node.
                if(!s.pages.empty() && s.pages.front()->owners>1 &&
                   s.pages.front()->slot(missing[i]->node)!=UINT32_MAX) {++i;continue;}
                s.pages.push_back(std::make_shared<Page>(allocated.fetch_add(1),seed));
            }
            auto p=s.pages.back().get();
            bool schedule=false;
            Image img;
            {
                std::lock_guard lock(p->io);
                unsigned old=p->count.load(std::memory_order_relaxed);
                if(old==slots) continue;
                unsigned n=old;
                if(!old) p->ids[n++]=seed;
                auto begin=i;
                while(i<missing.size() && n<slots) {
                    auto node=missing[i++]->node;
                    if(std::find(p->ids.begin(),p->ids.begin()+n,node)==p->ids.begin()+n)
                        p->ids[n++]=node;
                }
                if(n==old) continue;
                img={std::make_shared<Bytes>(),n,worker};
                if(!old)
                    memcpy(img.bytes->data,q.records[q.index.at(seed)].data.data(),width);
                else {
                    auto prefix=p->dirty.bytes?p->dirty:cache(p->owner,worker).get(p,old-1);
                    if(prefix.bytes) memcpy(img.bytes->data,prefix.bytes->data,old*width);
                    else {
                        residual::Scope sample(residual::OldRead);
                        ssize_t bytes;
                        do {bytes=pread(page_file(p->pos),img.bytes->data,block,page_offset(p->pos));}
                        while(bytes<0 && errno==EINTR);
                        if(bytes!=block) throw std::runtime_error("short path merge read");
                        ++merge_reads;
                    }
                }
                unsigned slot=old?old:1;
                for(size_t j=begin;j<i && slot<n;++j) {
                    if(missing[j]->node!=p->ids[slot]) continue;
                    memcpy(img.bytes->data+width*slot++,missing[j]->data.data(),width);
                }
                if(slot!=n) std::terminate();
                p->dirty=img;
                p->count.store(n,std::memory_order_release);
                if(!p->queued){p->queued=true;schedule=true;}
            }
            cache(p->owner,worker).put(p,img);
            if(schedule)enqueue(p);
        }
    }
public:
    void commit(Query& q,int worker) {
        std::unordered_map<uint32_t,Group> groups;
        for(auto& r:q.records) if(r.depth>=4) groups[r.parent].push_back(&r);
        std::vector<uint32_t> order;
        for(auto& r:q.records) if(groups.count(r.node)) order.push_back(r.node);
        for(size_t i=0;i<order.size();++i) {
            uint32_t a=order[i];
            auto s=find(a,true);
            std::unique_lock lock(s->mu);
            if(s->pages.empty() && i+1<order.size()) {
                lock.unlock();
                unsigned packed=pack_new(q,order,i,groups,worker);
                if(packed) {
                    i+=packed-1;
                    continue;
                }
                lock.lock();
            }
            append(q,a,groups.at(a),*s,worker);
        }
    }
};
}

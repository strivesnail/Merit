#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <unordered_set>

namespace pipeann {

  template<typename T>
  class ConcurrentQueue {
    typedef std::chrono::microseconds chrono_us_t;
    typedef std::unique_lock<std::mutex> mutex_locker;

    std::queue<T> q;
    std::mutex mut;
    std::mutex push_mut;
    std::mutex pop_mut;
    std::condition_variable push_cv;
    std::condition_variable pop_cv;

   public:
    T null_T;  // default value for pop() when queue is empty

    ConcurrentQueue() {
    }

    ConcurrentQueue(T nullT) {
      this->null_T = nullT;
    }

    ~ConcurrentQueue() {
      this->push_cv.notify_all();
      this->pop_cv.notify_all();
    }

    // queue stats
    uint64_t size() {
      mutex_locker lk(this->mut);
      uint64_t ret = q.size();
      lk.unlock();
      return ret;
    }

    bool empty() {
      return (this->size() == 0);
    }

    // PUSH BACK
    void push(const T &new_val) {
      mutex_locker lk(this->mut);
      this->q.push(new_val);
      lk.unlock();
    }

    template<class Iterator>
    void insert(Iterator iter_begin, Iterator iter_end) {
      mutex_locker lk(this->mut);
      for (Iterator it = iter_begin; it != iter_end; it++) {
        this->q.push(*it);
      }
      lk.unlock();
    }

    // POP FRONT
    T pop() {
      mutex_locker lk(this->mut);
      if (this->q.empty()) {
        lk.unlock();
        return this->null_T;
      } else {
        T ret = this->q.front();
        this->q.pop();
        // LOG(INFO) << "thread_id: " << std::this_thread::get_id() << ",
        // ctx: "
        // << ret.ctx << "\n";
        lk.unlock();
        return ret;
      }
    }

    // register for notifications
    void wait_for_push_notify(chrono_us_t wait_time = chrono_us_t{10}) {
      mutex_locker lk(this->push_mut);
      this->push_cv.wait_for(lk, wait_time);
      lk.unlock();
    }

    void wait_for_pop_notify(chrono_us_t wait_time = chrono_us_t{10}) {
      mutex_locker lk(this->pop_mut);
      this->pop_cv.wait_for(lk, wait_time);
      lk.unlock();
    }

    // just notify functions
    void push_notify_one() {
      this->push_cv.notify_one();
    }
    void push_notify_all() {
      this->push_cv.notify_all();
    }
    void pop_notify_one() {
      this->pop_cv.notify_one();
    }
    void pop_notify_all() {
      this->pop_cv.notify_all();
    }
  };

  // Lock-free SPSC queue (single-producer, single-consumer)
  template<typename T>
  struct SPSCQueue {
    alignas(128) std::atomic<int> sq_head{0};
    alignas(128) std::atomic<int> sq_tail{0};

    int cap;
    T *sq_data;

    SPSCQueue(int cap) : cap(cap) { sq_data = new T[cap];  }
    ~SPSCQueue() { delete[] sq_data; }

    bool push(const T &x) {
      int t = sq_tail.load(std::memory_order_relaxed);
      int next = (t + 1) % cap;
      if (next == sq_head.load(std::memory_order_acquire)) return false;
      sq_data[t] = x;
      sq_tail.store(next, std::memory_order_release);
      return true;
    }

    bool pop(T &x) {
      int h = sq_head.load(std::memory_order_relaxed);
      int t = sq_tail.load(std::memory_order_acquire);
      if (h == t) return false;
      x = sq_data[h];
      sq_head.store((h + 1) % cap, std::memory_order_release);
      return true;
    }

    template<typename Func>
    void pop_all_fn(Func &&func, int producer_id) {
      int h = sq_head.load(std::memory_order_relaxed);
      int t = sq_tail.load(std::memory_order_acquire);
      if (h == t) return;
      while (h != t) {
        func(sq_data[h], producer_id);
        h = (h + 1) % cap;
      }
      sq_head.store(h, std::memory_order_release);
    }
  };

  // MPSC queue: one SPSC per producer, consumer round-robins
  template<typename T>
  struct MPSCQueue {
    int n_producers;
    int cur = 0;
    std::vector<SPSCQueue<T> *> qp;
    
    MPSCQueue(int n_producers, int cap_per_producer) : n_producers(n_producers) {
      for (int i = 0; i < n_producers; i++) {
        qp.push_back(new SPSCQueue<T>(cap_per_producer + 1));
      }
    }
  
    ~MPSCQueue() { for (auto *q : qp) delete q; }

    bool push(const T &x, int producer_id) {
      return qp[producer_id]->push(x);
    }

    bool pop(T &x) {
      for (int i = 0; i < n_producers; i++) {
        int t = cur;
        cur = (cur + 1) % n_producers;
        if (qp[t]->pop(x)) {
          return true;
        }
      }
      return false;
    }

    template<typename Func>
    void pop_all_fn(Func &&func) {
      for (int i = 0; i < n_producers; i++) {
        qp[i]->pop_all_fn(func, i);
      }
    }
  };
}  // namespace pipeann

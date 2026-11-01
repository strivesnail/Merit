// Fault injection against the real reader: interrupted and partial completions.
#include "linux_aligned_file_reader.h"
#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
extern "C" int __real_io_getevents(io_context_t, long, long, io_event *, timespec *);
static unsigned calls = 0, interrupts = 0, partials = 0;
extern "C" int __wrap_io_getevents(io_context_t ctx, long min, long max, io_event *ev, timespec *ts) {
    if (++calls % 2) { ++interrupts; return -EINTR; }
    if (max > 1) ++partials;
    return __real_io_getevents(ctx, 1, 1, ev, ts);
}
int main() {
    char path[] = "/tmp/nc_aio_test_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) throw std::runtime_error("mkstemp failed");
    std::vector<char> original(8192);
    for (size_t i = 0; i < original.size(); ++i) original[i] = i % 251;
    if (write(fd, original.data(), original.size()) != 8192) throw std::runtime_error("write failed");
    ::close(fd);
    LinuxAlignedFileReader r;
    r.open(path); r.register_thread(); auto &ctx = r.get_ctx();
    void *buf = nullptr;
    if (posix_memalign(&buf, 4096, 8192)) throw std::bad_alloc();
    std::vector<AlignedRead> reads{{0, 4096, buf}, {4096, 4096, (char *)buf + 4096}};
    r.read(reads, ctx);
    if (memcmp(buf, original.data(), 8192)) throw std::runtime_error("single-fd data mismatch");
    memset(buf, 0, 8192);
    std::vector<FdAlignedRead> multi{{r.get_file_desc(), 0, 4096, buf},
                                    {r.get_file_desc(), 4096, 4096, (char *)buf + 4096}};
    r.read_multi(multi, ctx);
    if (memcmp(buf, original.data(), 8192)) throw std::runtime_error("multi-fd data mismatch");
    if (interrupts < 4 || partials < 2) throw std::runtime_error("faults not exercised");
    free(buf); r.deregister_thread(); r.close(); unlink(path);
    std::cout << "PASS EINTR=" << interrupts << " partial=" << partials << '\n';
}

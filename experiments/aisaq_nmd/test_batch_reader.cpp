// Linux regression: mixed direct/buffered reads, short/error completion, context reuse.
#include "linux_aligned_file_reader.h"
#include <array>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <unistd.h>
static void check(bool ok) { if (!ok) throw std::runtime_error("batch reader regression failed"); }
int main() {
    char path[] = "/tmp/aisaq-batch-reader-XXXXXX";
    int fd = mkstemp(path);
    check(fd >= 0);
    std::array<char,4096> original;
    for (size_t i=0;i<original.size();++i) original[i]=char(i%127);
    check(write(fd,original.data(),original.size())==(ssize_t)original.size());
    check(fsync(fd)==0);
    LinuxAlignedFileReader reader;
    reader.open(path);reader.register_thread();
    void *direct=nullptr;check(posix_memalign(&direct,512,4096)==0);
    std::array<char,100> buffered{},shortbuf{};
    AlignedRead d;d.fd_override=fd;d.offset=3;d.len=100;d.buf=buffered.data();
    AlignedRead shortread=d;shortread.offset=4090;shortread.buf=shortbuf.data();
    AlignedRead bad=d;bad.fd_override=1000000000;
    std::vector<AlignedRead> batch={AlignedRead(0,4096,direct),d,shortread,bad};
    reader.read(batch,reader.get_ctx());
    check(batch[0].result==4096 && !memcmp(direct,original.data(),4096));
    check(batch[1].result==100 && !memcmp(buffered.data(),original.data()+3,100));
    check(batch[2].result==6 && batch[3].result<0);
    // Error did not leave outstanding requests or poison the next batch.
    batch={d,AlignedRead(0,4096,direct)};reader.read(batch,reader.get_ctx());
    check(batch[0].result==100 && batch[1].result==4096);
    batch={AlignedRead(4096,4096,direct)};
    bool threw=false;try {reader.read(batch,reader.get_ctx());} catch(const std::runtime_error &) {threw=true;}
    check(threw);
    reader.deregister_thread();reader.close();close(fd);unlink(path);free(direct);
    puts("mixed reads, short/error completion, base failure, context reuse: PASS");
}

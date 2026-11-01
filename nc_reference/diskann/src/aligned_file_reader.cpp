#include "aligned_file_reader.h"

#include <map>

void AlignedFileReader::read_multi(std::vector<FdAlignedRead> &read_reqs, IOContext &ctx)
{
    if (read_reqs.empty())
        return;
    std::map<int, std::vector<AlignedRead>> by_fd;
    for (auto &fr : read_reqs)
    {
        by_fd[fr.fd].push_back(fr.read);
    }
    for (auto &kv : by_fd)
    {
        (void)kv.first;
        read(kv.second, ctx);
    }
}

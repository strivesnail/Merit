#include "neighborcache_thread_page_cache.h"
#include <cassert>
#include <cstring>
#include <unordered_map>

using Bytes = std::array<char, diskann::defaults::SECTOR_LEN>;
struct Writes {
    std::unordered_map<uint32_t, Bytes> images;
    unsigned seen = 0;
    static bool write(void *context, uint32_t key, const char *image, bool discard)
    {
        auto &self = *static_cast<Writes *>(context);
        if (!discard) { assert(std::memcmp(image, self.images.at(key).data(), image_size()) == 0); ++self.seen; }
        return true;
    }
    static constexpr size_t image_size() { return diskann::defaults::SECTOR_LEN; }
};
void roundtrips(bool trim)
{
    setenv("NEIGHBORCACHE_PREFETCH_TRIM_ZERO_TAIL", trim ? "1" : "0", 1);
    diskann::NeighborCacheThreadPageCache cache;
    assert(cache.init(2, 1, false, false));
    cache.configure_dynamic(1000, 8);
    Writes writes;
    cache.set_evict_hook(Writes::write, &writes);
    cache.enter();
    std::array<std::atomic<uint64_t>, 8> dirty{};
    for (unsigned pattern = 0; pattern < 40; ++pattern)
    {
        Bytes image{};
        if (pattern % 7 == 0) image.fill('F');
        else if (pattern % 7 == 1) image[0] = 'A';
        else if (pattern % 7 == 2) image[511] = 'B';
        else if (pattern % 7 == 3) image[512] = 'C';
        else if (pattern % 7 == 4) image[4095] = 'D';
        else if (pattern % 7 == 5) { image[130] = 'E'; image[1997] = 'G'; }
        const uint32_t key = 1000 + pattern % 4;
        assert(cache.put(key, image.data(), true, &dirty[key - 1000]));
        writes.images[key] = image;
        const uint32_t ids[] = {100 + pattern, 200 + pattern};
        cache.register_records(key, ids, 2);
        Bytes output;
        output.fill(99);
        assert(cache.get_range(key, 1, output.data()) && output == image);
        uint32_t seed_key = 0;
        std::vector<uint32_t> members;
        output.fill(99);
        assert(cache.seed_snapshot(ids[0], seed_key, members, output.data()));
        assert(seed_key == key && output == image && members == std::vector<uint32_t>(ids, ids + 2));
        std::array<char, 128> record{};
        assert(cache.copy_record(ids[1], record.size(), record.data()));
        assert(std::memcmp(record.data(), image.data() + 128, record.size()) == 0);
    }
    assert(writes.seen >= 38);
    cache.exit();
    // Static immutable snapshots compare the whole reconstructed image.
    diskann::NeighborCacheThreadPageCache immutable;
    assert(immutable.init(2, 1, false, true));
    immutable.enter();
    Bytes first{}; first[1] = 17;
    assert(immutable.put(1, first.data()));
    assert(immutable.put(1, first.data(), true));
    Bytes changed = first; changed.back() = 19;
    bool rejected = false;
    try { immutable.put(1, changed.data(), true); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    Bytes restored{};
    assert(immutable.get_range(1, 1, restored.data()) && restored == first);
    immutable.exit();
}
int main() { roundtrips(false); roundtrips(true); }

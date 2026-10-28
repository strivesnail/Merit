#pragma once
#include "nc_client.h"
namespace aisaq_nmd {
class NMDCache : public neighborcache::NMDCache {
 public:
  void init(uint64_t points, uint64_t record, uint64_t nps, uint64_t vector,
            const std::string &path, const std::string &index_path, uint32_t medoid, uint64_t degree = 0) {
    neighborcache::NMDCache::init("AISAQ_NMD", 4, points, record, nps, vector,
                                degree ? degree : (record - vector - 4) / 4, path, medoid);
    if (enabled() && neighborcache::env("AISAQ_NMD_VERIFY", 0))
      set_verifier([index_path, record, nps](uint32_t id, const char *raw) {
        neighborcache::verify_file_record(index_path, (1 + id / nps) * 4096 + (id % nps) * record, record, id, raw);
      });
  }
};
}

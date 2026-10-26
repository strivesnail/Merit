#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "utils.h"

namespace pipeann {
  struct QueryStats {
    double total_us = 0;        // total time to process query in micros
    double n_ios = 0;           // total # of IOs issued
    double n_ncache = 0;        // N-cache hits (no base IO)
    double n_pref = 0;          // prefetch-page hits
    double n_dcache = 0;        // D-cache hits
    double io_us = 0;           // total time spent in IO
    double io_us1 = 0;          // total time spent in IO
    double head_us = 0;         // total time spent in in-memory index
    double cpu_us = 0;          // total time spent in CPU
    double cpu_us1 = 0;         // total time spent in CPU
    double cpu_us2 = 0;         // total time spent in CPU
    double n_cmps = 0;          // # cmps
    double n_hops = 0;          // # search hops
    double n_current_used = 0;  // # force return for latency limit
    double n_filter[N_FILTER_TYPES] = {0};
    double n_est_filter_reads[N_FILTER_TYPES] = {0};
    double n_est_filter_cmps[N_FILTER_TYPES] = {0};
    double n_filter_reads[N_FILTER_TYPES] = {0};
    double n_filter_accessed_vectors[N_FILTER_TYPES] = {0};
    double n_filter_false_positives[N_FILTER_TYPES] = {0};

    double filter_io_us[N_FILTER_TYPES] = {0};
    double filter_cpu_us[N_FILTER_TYPES] = {0};
    double filter_io_us1[N_FILTER_TYPES] = {0};
  };

  inline double get_percentile_stats(QueryStats *stats, uint64_t len, float percentile,
                                     const std::function<double(const QueryStats &)> &member_fn) {
    std::vector<double> vals(len);
    for (uint64_t i = 0; i < len; i++) {
      vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(), [](const double &left, const double &right) { return left < right; });

    auto retval = vals[(uint64_t) (percentile * ((float) len))];
    vals.clear();
    return retval;
  }

  inline double get_mean_stats(QueryStats *stats, uint64_t len,
                               const std::function<double(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      avg += member_fn(stats[i]);
    }
    return avg / ((double) len);
  }

  inline double get_mean_stats(QueryStats *stats, uint64_t cnt, uint64_t tot,
                               const std::function<double(const QueryStats &)> &member_fn) {
    if (cnt == 0) {
      return 0;
    }

    double avg = 0;
    for (uint64_t i = 0; i < tot; i++) {
      avg += member_fn(stats[i]);
    }
    return avg / ((double) cnt);
  }
}  // namespace pipeann

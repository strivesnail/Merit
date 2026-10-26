#ifndef FILTER_UTILS_H_
#define FILTER_UTILS_H_

#include "ssd_index_defs.h"
#include <vector>

namespace pipeann {
  using VectorIDList = std::vector<uint32_t>;
  inline VectorIDList or_sorted_unique(const uint32_t *a, size_t a_size, const uint32_t *b, size_t b_size) {
    VectorIDList result;
    result.reserve(a_size + b_size);

    size_t i = 0, j = 0;
    while (i < a_size && j < b_size) {
      if (a[i] < b[j]) {
        result.push_back(a[i]);
        i++;
      } else if (b[j] < a[i]) {
        result.push_back(b[j]);
        j++;
      } else {
        result.push_back(a[i]);
        i++;
        j++;
      }
    }

    if (i < a_size) {
      result.insert(result.end(), a + i, a + a_size);
    }
    if (j < b_size) {
      result.insert(result.end(), b + j, b + b_size);
    }
    return result;
  }

  inline VectorIDList or_sorted_unique(const VectorIDList &a, const VectorIDList &b) {
    return or_sorted_unique(a.data(), a.size(), b.data(), b.size());
  }

  inline VectorIDList and_sorted_unique(const uint32_t *a, size_t a_size, const uint32_t *b, size_t b_size) {
    VectorIDList result;
    result.reserve(std::min(a_size, b_size));

    size_t i = 0, j = 0;
    while (i < a_size && j < b_size) {
      if (a[i] < b[j]) {
        i++;
      } else if (b[j] < a[i]) {
        j++;
      } else {
        result.push_back(a[i]);
        i++;
        j++;
      }
    }
    return result;
  }

  inline VectorIDList and_sorted_unique(const VectorIDList &a, const VectorIDList &b) {
    return and_sorted_unique(a.data(), a.size(), b.data(), b.size());
  }
}  // namespace pipeann

#endif
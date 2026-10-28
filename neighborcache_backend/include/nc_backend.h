#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define NC_BACKEND_ABI 2
typedef struct {
    uint32_t abi, count_bytes;
    uint64_t points, record_bytes, vector_bytes, degree, nodes_per_page;
    uint64_t n_capacity, prefetch_pages, threads;
    const char *path;
    const uint32_t *medoids;
    uint64_t medoid_count;
} NCConfig;
typedef struct {
    int fd;
    uint64_t offset, bytes;
    void *buffer;
    void *ticket;
    uint32_t node_id;
} NCRead;
typedef struct {
    uint64_t n_hits, prefetch_hits, d_attempts, d_hits, d_reads, base_misses;
    uint64_t seeds, capacity, m_entries, opens, appends, replaces, writes;
    uint64_t acquired_records, recorded_parents;
} NCStats;
// A query and its tickets belong to one calling thread. Finish every issued IO
// before ending its query. Errors are returned, never thrown across the ABI.
typedef struct {
    uint32_t abi;
    const char *(*error)(void);
    void *(*create)(const NCConfig *);
    void (*destroy)(void *);
    void *(*begin)(void *);
    int (*end)(void *, void *);
    // ABI 2: carry source metadata only. No parent map/counter update here.
    // expand(raw) commits the relation after a full record has been acquired.
    int (*visit)(void *, void *, uint32_t, uint32_t);
    int (*resolve)(void *, void *, uint32_t, char *, NCRead *);
    int (*complete)(void *, void *, NCRead *, int64_t, char *);
    int (*expand)(void *, void *, uint32_t, const char *, uint32_t);
    int (*stats)(void *, NCStats *);
    int (*contains)(void *, uint32_t);
    int (*base_reads)(void *, void *, uint32_t, double);
} NCBackend;
const NCBackend *nc_backend_api(void);
#ifdef __cplusplus
}
#endif

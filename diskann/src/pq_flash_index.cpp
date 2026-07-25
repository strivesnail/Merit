// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"

#include <limits>
#include <numeric>
#include <sstream>

#include "timer.h"
#include "pq.h"
#include "pq_scratch.h"
#include "pq_flash_index.h"
#include "cosine_similarity.h"
#include "defaults.h"
#include "relayout_utils.h"

#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#endif

namespace
{
inline float issue_merit_and_base_disk_reads(std::shared_ptr<AlignedFileReader> &base_reader,
                                            std::shared_ptr<AlignedFileReader> &merit_reader, IOContext &base_ctx,
                                            std::vector<AlignedRead> &merit_reqs, std::vector<AlignedRead> &base_reqs,
                                            diskann::Timer &io_timer, diskann::QueryStats *stats)
{
    if (merit_reqs.empty() && base_reqs.empty())
        return 0.f;

    io_timer.reset();
#ifndef _WINDOWS
    if (!merit_reqs.empty() && !base_reqs.empty() && merit_reader != nullptr)
    {
        auto *base_linux = dynamic_cast<LinuxAlignedFileReader *>(base_reader.get());
        auto *merit_linux = dynamic_cast<LinuxAlignedFileReader *>(merit_reader.get());
        if (base_linux != nullptr && merit_linux != nullptr)
        {
            std::vector<FdAlignedRead> multi;
            multi.reserve(merit_reqs.size() + base_reqs.size());
            const int merit_fd = merit_linux->get_file_desc();
            const int base_fd = base_linux->get_file_desc();
            for (auto &r : merit_reqs)
                multi.emplace_back(merit_fd, r.offset, r.len, r.buf);
            for (auto &r : base_reqs)
                multi.emplace_back(base_fd, r.offset, r.len, r.buf);
            base_reader->read_multi(multi, base_ctx);
            const float us = (float)io_timer.elapsed();
            if (stats != nullptr)
                stats->io_us += us;
            return us;
        }
    }
#endif
    if (!merit_reqs.empty() && merit_reader != nullptr)
        merit_reader->read(merit_reqs, merit_reader->get_ctx());
    if (!base_reqs.empty())
        base_reader->read(base_reqs, base_ctx);
    const float us = (float)io_timer.elapsed();
    if (stats != nullptr)
        stats->io_us += us;
    return us;
}
} // namespace

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) (((uint64_t)(id)) / _nvecs_per_sector + _reorder_data_start_sector)

namespace
{
inline void record_disk_read_batch_stats(diskann::QueryStats *stats, float batch_us,
                                         const std::vector<uint64_t> &miss_sector_ids, uint64_t max_graph_sector)
{
    const size_t bs = miss_sector_ids.size();
    stats->n_disk_read_batches++;
    stats->sum_batch_io_us += batch_us;
    if (bs == 1)
    {
        stats->n_batches_size1++;
        stats->sum_batch_us_size1 += batch_us;
    }
    else if (bs == 2)
    {
        stats->n_batches_size2++;
        stats->sum_batch_us_size2 += batch_us;
    }
    else
    {
        stats->n_batches_size_other++;
        stats->sum_batch_us_size_other += batch_us;
    }
    for (uint64_t sid : miss_sector_ids)
    {
        unsigned b = 0;
        if (max_graph_sector > 0)
        {
            b = (unsigned)((sid * diskann::QueryStats::DISK_SECTOR_BUCKETS) / max_graph_sector);
            if (b >= diskann::QueryStats::DISK_SECTOR_BUCKETS)
                b = (unsigned)(diskann::QueryStats::DISK_SECTOR_BUCKETS - 1);
        }
        stats->disk_sector_bucket[b]++;
    }
    stats->n_disk_sector_bucket_samples += (unsigned)miss_sector_ids.size();
}
} // namespace

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) ((((uint64_t)(id)) % _nvecs_per_sector) * _data_dim * sizeof(float))

namespace diskann
{

template <typename T, typename LabelT>
PQFlashIndex<T, LabelT>::PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader, diskann::Metric m)
    : reader(fileReader), metric(m), _thread_data(nullptr)
{
    diskann::Metric metric_to_invoke = m;
    if (m == diskann::Metric::COSINE || m == diskann::Metric::INNER_PRODUCT)
    {
        if (std::is_floating_point<T>::value)
        {
            diskann::cout << "Since data is floating point, we assume that it has been appropriately pre-processed "
                             "(normalization for cosine, and convert-to-l2 by adding extra dimension for MIPS). So we "
                             "shall invoke an l2 distance function."
                          << std::endl;
            metric_to_invoke = diskann::Metric::L2;
        }
        else
        {
            diskann::cerr << "WARNING: Cannot normalize integral data types."
                          << " This may result in erroneous results or poor recall."
                          << " Consider using L2 distance with integral data types." << std::endl;
        }
    }

    this->_dist_cmp.reset(diskann::get_distance_function<T>(metric_to_invoke));
    this->_dist_cmp_float.reset(diskann::get_distance_function<float>(metric_to_invoke));
}

template <typename T, typename LabelT> PQFlashIndex<T, LabelT>::~PQFlashIndex()
{
#ifndef EXEC_ENV_OLS
    if (data != nullptr)
    {
        delete[] data;
    }
#endif

    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    // delete backing bufs for nhood and coord cache
    if (_nhood_cache_buf != nullptr)
    {
        delete[] _nhood_cache_buf;
        diskann::aligned_free(_coord_cache_buf);
    }

    if (_merit_disk_reader)
    {
        _merit_disk_reader->close();
        _merit_disk_reader.reset();
    }
    _merit_dc_map.clear();

    if (_load_flag)
    {
        diskann::cout << "Clearing scratch" << std::endl;
        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        manager.destroy();
        this->reader->deregister_all_threads();
        reader->close();
    }
    if (_pts_to_label_offsets != nullptr)
    {
        delete[] _pts_to_label_offsets;
    }
    if (_pts_to_label_counts != nullptr)
    {
        delete[] _pts_to_label_counts;
    }
    if (_pts_to_labels != nullptr)
    {
        delete[] _pts_to_labels;
    }
    if (_medoids != nullptr)
    {
        delete[] _medoids;
    }
}

template <typename T, typename LabelT> inline uint64_t PQFlashIndex<T, LabelT>::get_node_sector(uint64_t node_id)
{
    return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector
                                       : node_id * DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN));
}

template <typename T, typename LabelT>
inline char *PQFlashIndex<T, LabelT>::offset_to_node(char *sector_buf, uint64_t node_id)
{
    return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * _max_node_len);
}

template <typename T, typename LabelT> inline uint32_t *PQFlashIndex<T, LabelT>::offset_to_node_nhood(char *node_buf)
{
    return (unsigned *)(node_buf + _disk_bytes_per_point);
}

template <typename T, typename LabelT> inline T *PQFlashIndex<T, LabelT>::offset_to_node_coords(char *node_buf)
{
    return (T *)(node_buf);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::setup_thread_data(uint64_t nthreads, uint64_t visited_reserve)
{
    diskann::cout << "Setting up thread-specific contexts for nthreads: " << nthreads << std::endl;
// omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int)nthreads)
    for (int64_t thread = 0; thread < (int64_t)nthreads; thread++)
    {
#pragma omp critical
        {
            SSDThreadData<T> *data = new SSDThreadData<T>(this->_aligned_dim, visited_reserve);
            this->reader->register_thread();
            data->ctx = this->reader->get_ctx();
            this->_thread_data.push(data);
        }
    }
    _load_flag = true;
}

template <typename T, typename LabelT>
std::vector<bool> PQFlashIndex<T, LabelT>::read_nodes(const std::vector<uint32_t> &node_ids,
                                                      std::vector<T *> &coord_buffers,
                                                      std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers)
{
    std::vector<AlignedRead> read_reqs;
    std::vector<bool> retval(node_ids.size(), true);

    char *buf = nullptr;
    auto num_sectors = _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    alloc_aligned((void **)&buf, node_ids.size() * num_sectors * defaults::SECTOR_LEN, defaults::SECTOR_LEN);

    // create read requests
    for (size_t i = 0; i < node_ids.size(); ++i)
    {
        auto node_id = node_ids[i];

        AlignedRead read;
        read.len = num_sectors * defaults::SECTOR_LEN;
        read.buf = buf + i * num_sectors * defaults::SECTOR_LEN;
        read.offset = get_node_sector(node_id) * defaults::SECTOR_LEN;
        read_reqs.push_back(read);
    }

    // borrow thread data and issue reads
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;
    reader->read(read_reqs, ctx);

    // copy reads into buffers
    for (uint32_t i = 0; i < read_reqs.size(); i++)
    {
#if defined(_WINDOWS) && defined(USE_BING_INFRA) // this block is to handle failed reads in
                                                 // production settings
        if ((*ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS)
        {
            retval[i] = false;
            continue;
        }
#endif

        char *node_buf = offset_to_node((char *)read_reqs[i].buf, node_ids[i]);

        if (coord_buffers[i] != nullptr)
        {
            T *node_coords = offset_to_node_coords(node_buf);
            memcpy(coord_buffers[i], node_coords, _disk_bytes_per_point);
        }

        if (nbr_buffers[i].second != nullptr)
        {
            uint32_t *node_nhood = offset_to_node_nhood(node_buf);
            auto num_nbrs = *node_nhood;
            nbr_buffers[i].first = num_nbrs;
            memcpy(nbr_buffers[i].second, node_nhood + 1, num_nbrs * sizeof(uint32_t));
        }
    }

    aligned_free(buf);

    return retval;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::load_cache_list(std::vector<uint32_t> &node_list)
{
    diskann::cout << "Loading the cache list into memory.." << std::flush;
    size_t num_cached_nodes = node_list.size();

    // Allocate space for neighborhood cache
    _nhood_cache_buf = new uint32_t[num_cached_nodes * (_max_degree + 1)];
    memset(_nhood_cache_buf, 0, num_cached_nodes * (_max_degree + 1));

    // Allocate space for coordinate cache
    size_t coord_cache_buf_len = num_cached_nodes * _aligned_dim;
    diskann::alloc_aligned((void **)&_coord_cache_buf, coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(_coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);
    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_idx = block * BLOCK_SIZE;
        size_t end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);

        // Copy offset into buffers to read into
        std::vector<uint32_t> nodes_to_read;
        std::vector<T *> coord_buffers;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++)
        {
            nodes_to_read.push_back(node_list[node_idx]);
            coord_buffers.push_back(_coord_cache_buf + node_idx * _aligned_dim);
            nbr_buffers.emplace_back(0, _nhood_cache_buf + node_idx * (_max_degree + 1));
        }

        // issue the reads
        auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

        // check for success and insert into the cache.
        for (size_t i = 0; i < read_status.size(); i++)
        {
            if (read_status[i] == true)
            {
                _coord_cache.insert(std::make_pair(nodes_to_read[i], coord_buffers[i]));
                _nhood_cache.insert(std::make_pair(nodes_to_read[i], nbr_buffers[i]));
            }
        }
    }
    diskann::cout << "..done." << std::endl;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                      uint64_t l_search, uint64_t beamwidth,
                                                                      uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#else
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                      uint64_t beamwidth, uint64_t num_nodes_to_cache,
                                                                      uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#endif
    if (num_nodes_to_cache >= this->_num_points)
    {
        // for small num_points and big num_nodes_to_cache, use below way to get the node_list quickly
        node_list.resize(this->_num_points);
        for (uint32_t i = 0; i < this->_num_points; ++i)
        {
            node_list[i] = i;
        }
        return;
    }

    this->_count_visited_nodes = true;
    this->_node_visit_counter.clear();
    this->_node_visit_counter.resize(this->_num_points);
    for (uint32_t i = 0; i < _node_visit_counter.size(); i++)
    {
        this->_node_visit_counter[i].first = i;
        this->_node_visit_counter[i].second = 0;
    }

    uint64_t sample_num, sample_dim, sample_aligned_dim;
    T *samples;

#ifdef EXEC_ENV_OLS
    if (files.fileExists(sample_bin))
    {
        diskann::load_aligned_bin<T>(files, sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#else
    if (file_exists(sample_bin))
    {
        diskann::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#endif
    else
    {
        diskann::cerr << "Sample bin file not found. Not generating cache." << std::endl;
        return;
    }

    std::vector<uint64_t> tmp_result_ids_64(sample_num, 0);
    std::vector<float> tmp_result_dists(sample_num, 0);

    bool filtered_search = false;
    std::vector<LabelT> random_query_filters(sample_num);
    if (_filter_to_medoid_ids.size() != 0)
    {
        filtered_search = true;
        generate_random_labels(random_query_filters, (uint32_t)sample_num, nthreads);
    }

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < (int64_t)sample_num; i++)
    {
        auto &label_for_search = random_query_filters[i];
        // run a search on the sample query with a random label (sampled from base label distribution), and it will
        // concurrently update the node_visit_counter to track most visited nodes. The last false is to not use the
        // "use_reorder_data" option which enables a final reranking if the disk index itself contains only PQ data.
        cached_beam_search(samples + (i * sample_aligned_dim), 1, l_search, tmp_result_ids_64.data() + i,
                           tmp_result_dists.data() + i, beamwidth, filtered_search, label_for_search, false);
    }

    std::sort(this->_node_visit_counter.begin(), _node_visit_counter.end(),
              [](std::pair<uint32_t, uint32_t> &left, std::pair<uint32_t, uint32_t> &right) {
                  return left.second > right.second;
              });
    node_list.clear();
    node_list.shrink_to_fit();
    num_nodes_to_cache = std::min(num_nodes_to_cache, this->_node_visit_counter.size());
    node_list.reserve(num_nodes_to_cache);
    for (uint64_t i = 0; i < num_nodes_to_cache; i++)
    {
        node_list.push_back(this->_node_visit_counter[i].first);
    }
    this->_count_visited_nodes = false;

    diskann::aligned_free(samples);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                               const bool shuffle)
{
    std::random_device rng;
    std::mt19937 urng(rng());

    tsl::robin_set<uint32_t> node_set;

    // Do not cache more than 10% of the nodes in the index
    uint64_t tenp_nodes = (uint64_t)(std::round(this->_num_points * 0.1));
    if (num_nodes_to_cache > tenp_nodes)
    {
        diskann::cout << "Reducing nodes to cache from: " << num_nodes_to_cache << " to: " << tenp_nodes
                      << "(10 percent of total nodes:" << this->_num_points << ")" << std::endl;
        num_nodes_to_cache = tenp_nodes == 0 ? 1 : tenp_nodes;
    }
    diskann::cout << "Caching " << num_nodes_to_cache << "..." << std::endl;

    std::unique_ptr<tsl::robin_set<uint32_t>> cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<uint32_t>>();
    prev_level = std::make_unique<tsl::robin_set<uint32_t>>();

    for (uint64_t miter = 0; miter < _num_medoids && cur_level->size() < num_nodes_to_cache; miter++)
    {
        cur_level->insert(_medoids[miter]);
    }

    if ((_filter_to_medoid_ids.size() > 0) && (cur_level->size() < num_nodes_to_cache))
    {
        for (auto &x : _filter_to_medoid_ids)
        {
            for (auto &y : x.second)
            {
                cur_level->insert(y);
                if (cur_level->size() == num_nodes_to_cache)
                    break;
            }
            if (cur_level->size() == num_nodes_to_cache)
                break;
        }
    }

    uint64_t lvl = 1;
    uint64_t prev_node_set_size = 0;
    while ((node_set.size() + cur_level->size() < num_nodes_to_cache) && cur_level->size() != 0)
    {
        // swap prev_level and cur_level
        std::swap(prev_level, cur_level);
        // clear cur_level
        cur_level->clear();

        std::vector<uint32_t> nodes_to_expand;

        for (const uint32_t &id : *prev_level)
        {
            if (node_set.find(id) != node_set.end())
            {
                continue;
            }
            node_set.insert(id);
            nodes_to_expand.push_back(id);
        }

        if (shuffle)
            std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);
        else
            std::sort(nodes_to_expand.begin(), nodes_to_expand.end());

        diskann::cout << "Level: " << lvl << std::flush;
        bool finish_flag = false;

        uint64_t BLOCK_SIZE = 1024;
        uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);
        for (size_t block = 0; block < nblocks && !finish_flag; block++)
        {
            diskann::cout << "." << std::flush;
            size_t start = block * BLOCK_SIZE;
            size_t end = (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());

            std::vector<uint32_t> nodes_to_read;
            std::vector<T *> coord_buffers(end - start, nullptr);
            std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;

            for (size_t cur_pt = start; cur_pt < end; cur_pt++)
            {
                nodes_to_read.push_back(nodes_to_expand[cur_pt]);
                nbr_buffers.emplace_back(0, new uint32_t[_max_degree + 1]);
            }

            // issue read requests
            auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

            // process each nhood buf
            for (uint32_t i = 0; i < read_status.size(); i++)
            {
                if (read_status[i] == false)
                {
                    continue;
                }
                else
                {
                    uint32_t nnbrs = nbr_buffers[i].first;
                    uint32_t *nbrs = nbr_buffers[i].second;

                    // explore next level
                    for (uint32_t j = 0; j < nnbrs && !finish_flag; j++)
                    {
                        if (node_set.find(nbrs[j]) == node_set.end())
                        {
                            cur_level->insert(nbrs[j]);
                        }
                        if (cur_level->size() + node_set.size() >= num_nodes_to_cache)
                        {
                            finish_flag = true;
                        }
                    }
                }
                delete[] nbr_buffers[i].second;
            }
        }

        diskann::cout << ". #nodes: " << node_set.size() - prev_node_set_size
                      << ", #nodes thus far: " << node_set.size() << std::endl;
        prev_node_set_size = node_set.size();
        lvl++;
    }

    assert(node_set.size() + cur_level->size() == num_nodes_to_cache || cur_level->size() == 0);

    node_list.clear();
    node_list.reserve(node_set.size() + cur_level->size());
    for (auto node : node_set)
        node_list.push_back(node);
    for (auto node : *cur_level)
        node_list.push_back(node);

    diskann::cout << "Level: " << lvl << std::flush;
    diskann::cout << ". #nodes: " << node_list.size() - prev_node_set_size << ", #nodes thus far: " << node_list.size()
                  << std::endl;
    diskann::cout << "done" << std::endl;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::use_medoids_data_as_centroids()
{
    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    alloc_aligned(((void **)&_centroid_data), _num_medoids * _aligned_dim * sizeof(float), 32);
    std::memset(_centroid_data, 0, _num_medoids * _aligned_dim * sizeof(float));

    diskann::cout << "Loading centroid data from medoids vector data of " << _num_medoids << " medoid(s)" << std::endl;

    std::vector<uint32_t> nodes_to_read;
    std::vector<T *> medoid_bufs;
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_bufs;

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        nodes_to_read.push_back(_medoids[cur_m]);
        medoid_bufs.push_back(new T[_data_dim]);
        nbr_bufs.emplace_back(0, nullptr);
    }

    auto read_status = read_nodes(nodes_to_read, medoid_bufs, nbr_bufs);

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        if (read_status[cur_m] == true)
        {
            if (!_use_disk_index_pq)
            {
                for (uint32_t i = 0; i < _data_dim; i++)
                    _centroid_data[cur_m * _aligned_dim + i] = medoid_bufs[cur_m][i];
            }
            else
            {
                _disk_pq_table.inflate_vector((uint8_t *)medoid_bufs[cur_m], (_centroid_data + cur_m * _aligned_dim));
            }
        }
        else
        {
            throw ANNException("Unable to read a medoid", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        delete[] medoid_bufs[cur_m];
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                     const uint32_t nthreads)
{
    std::random_device rd;
    labels.clear();
    labels.resize(num_labels);

    uint64_t num_total_labels = _pts_to_label_offsets[_num_points - 1] + _pts_to_label_counts[_num_points - 1];
    std::mt19937 gen(rd());
    if (num_total_labels == 0)
    {
        std::stringstream stream;
        stream << "No labels found in data. Not sampling random labels ";
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    std::uniform_int_distribution<uint64_t> dis(0, num_total_labels - 1);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < num_labels; i++)
    {
        uint64_t rnd_loc = dis(gen);
        labels[i] = (LabelT)_pts_to_labels[rnd_loc];
    }
}

template <typename T, typename LabelT>
std::unordered_map<std::string, LabelT> PQFlashIndex<T, LabelT>::load_label_map(std::basic_istream<char> &map_reader)
{
    std::unordered_map<std::string, LabelT> string_to_int_mp;
    std::string line, token;
    LabelT token_as_num;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        token_as_num = (LabelT)std::stoul(token);
        string_to_int_mp[label_str] = token_as_num;
    }
    return string_to_int_mp;
}

template <typename T, typename LabelT>
LabelT PQFlashIndex<T, LabelT>::get_converted_label(const std::string &filter_label)
{
    if (_label_map.find(filter_label) != _label_map.end())
    {
        return _label_map[filter_label];
    }
    if (_use_universal_label)
    {
        return _universal_filter_label;
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::reset_stream_for_reading(std::basic_istream<char> &infile)
{
    infile.clear();
    infile.seekg(0);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::get_label_file_metadata(const std::string &fileContent, uint32_t &num_pts,
                                                      uint32_t &num_total_labels)
{
    num_pts = 0;
    num_total_labels = 0;

    size_t file_size = fileContent.length();

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = fileContent.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = fileContent.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label
            {
                next_lbl_pos = next_pos;
            }

            num_total_labels++;

            lbl_pos = next_lbl_pos + 1;
        }

        cur_pos = next_pos + 1;

        num_pts++;
    }

    diskann::cout << "Labels file metadata: num_points: " << num_pts << ", #total_labels: " << num_total_labels
                  << std::endl;
}

template <typename T, typename LabelT>
inline bool PQFlashIndex<T, LabelT>::point_has_label(uint32_t point_id, LabelT label_id)
{
    uint32_t start_vec = _pts_to_label_offsets[point_id];
    uint32_t num_lbls = _pts_to_label_counts[point_id];
    bool ret_val = false;
    for (uint32_t i = 0; i < num_lbls; i++)
    {
        if (_pts_to_labels[start_vec + i] == label_id)
        {
            ret_val = true;
            break;
        }
    }
    return ret_val;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::parse_label_file(std::basic_istream<char> &infile, size_t &num_points_labels)
{
    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();

    std::string buffer(file_size, ' ');

    infile.seekg(0, std::ios::beg);
    infile.read(&buffer[0], file_size);

    std::string line;
    uint32_t line_cnt = 0;

    uint32_t num_pts_in_label_file;
    uint32_t num_total_labels;
    get_label_file_metadata(buffer, num_pts_in_label_file, num_total_labels);

    _pts_to_label_offsets = new uint32_t[num_pts_in_label_file];
    _pts_to_label_counts = new uint32_t[num_pts_in_label_file];
    _pts_to_labels = new LabelT[num_total_labels];
    uint32_t labels_seen_so_far = 0;

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        _pts_to_label_offsets[line_cnt] = labels_seen_so_far;
        uint32_t &num_lbls_in_cur_pt = _pts_to_label_counts[line_cnt];
        num_lbls_in_cur_pt = 0;

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = buffer.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label in the whole file
            {
                next_lbl_pos = next_pos;
            }

            if (next_lbl_pos > next_pos) // the last label in one line, just read to the end
            {
                next_lbl_pos = next_pos;
            }

            label_str.assign(buffer.c_str() + lbl_pos, next_lbl_pos - lbl_pos);
            if (label_str[label_str.length() - 1] == '\t') // '\t' won't exist in label file?
            {
                label_str.erase(label_str.length() - 1);
            }

            LabelT token_as_num = (LabelT)std::stoul(label_str);
            _pts_to_labels[labels_seen_so_far++] = (LabelT)token_as_num;
            num_lbls_in_cur_pt++;

            // move to next label
            lbl_pos = next_lbl_pos + 1;
        }

        // move to next line
        cur_pos = next_pos + 1;

        if (num_lbls_in_cur_pt == 0)
        {
            diskann::cout << "No label found for point " << line_cnt << std::endl;
            exit(-1);
        }

        line_cnt++;
    }

    num_points_labels = line_cnt;
    reset_stream_for_reading(infile);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::set_universal_label(const LabelT &label)
{
    _use_universal_label = true;
    _universal_filter_label = label;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load(MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix)
{
#else
template <typename T, typename LabelT> int PQFlashIndex<T, LabelT>::load(uint32_t num_threads, const char *index_prefix)
{
#endif
    std::string pq_table_bin = std::string(index_prefix) + "_pq_pivots.bin";
    std::string pq_compressed_vectors = std::string(index_prefix) + "_pq_compressed.bin";
    std::string _disk_index_file = std::string(index_prefix) + "_disk.index";
#ifdef EXEC_ENV_OLS
    return load_from_separate_paths(files, num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
#else
    return load_from_separate_paths(num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
#endif
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                      const char *index_filepath, const char *pivots_filepath,
                                                      const char *compressed_filepath)
{
#else
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                      const char *pivots_filepath, const char *compressed_filepath)
{
#endif
    std::string pq_table_bin = pivots_filepath;
    std::string pq_compressed_vectors = compressed_filepath;
    std::string _disk_index_file = index_filepath;
    std::string medoids_file = std::string(_disk_index_file) + "_medoids.bin";
    std::string centroids_file = std::string(_disk_index_file) + "_centroids.bin";

    std::string labels_file = std ::string(_disk_index_file) + "_labels.txt";
    std::string labels_to_medoids = std ::string(_disk_index_file) + "_labels_to_medoids.txt";
    std::string dummy_map_file = std ::string(_disk_index_file) + "_dummy_map.txt";
    std::string labels_map_file = std ::string(_disk_index_file) + "_labels_map.txt";
    size_t num_pts_in_label_file = 0;

    size_t pq_file_dim, pq_file_num_centroids;
#ifdef EXEC_ENV_OLS
    get_bin_metadata(files, pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#else
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#endif

    this->_disk_index_file = _disk_index_file;
    _base_disk_index_bytes = get_file_size(_disk_index_file);

    if (pq_file_num_centroids != 256)
    {
        diskann::cout << "Error. Number of PQ centroids is not 256. Exiting." << std::endl;
        return -1;
    }

    this->_data_dim = pq_file_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->_disk_bytes_per_point = this->_data_dim * sizeof(T);
    this->_aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
#ifdef EXEC_ENV_OLS
    diskann::load_bin<uint8_t>(files, pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#else
    diskann::load_bin<uint8_t>(pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#endif

    this->_num_points = npts_u64;
    this->_n_chunks = nchunks_u64;
    this->_hotness_profiler.init(npts_u64);
#ifdef EXEC_ENV_OLS
    if (files.fileExists(labels_file))
    {
        FileContent &content_labels = files.getContent(labels_file);
        std::stringstream infile(std::string((const char *)content_labels._content, content_labels._size));
#else
    if (file_exists(labels_file))
    {
        std::ifstream infile(labels_file, std::ios::binary);
        if (infile.fail())
        {
            throw diskann::ANNException(std::string("Failed to open file ") + labels_file, -1);
        }
#endif
        parse_label_file(infile, num_pts_in_label_file);
        assert(num_pts_in_label_file == this->_num_points);

#ifndef EXEC_ENV_OLS
        infile.close();
#endif

#ifdef EXEC_ENV_OLS
        FileContent &content_labels_map = files.getContent(labels_map_file);
        std::stringstream map_reader(std::string((const char *)content_labels_map._content, content_labels_map._size));
#else
        std::ifstream map_reader(labels_map_file);
#endif
        _label_map = load_label_map(map_reader);

#ifndef EXEC_ENV_OLS
        map_reader.close();
#endif

#ifdef EXEC_ENV_OLS
        if (files.fileExists(labels_to_medoids))
        {
            FileContent &content_labels_to_meoids = files.getContent(labels_to_medoids);
            std::stringstream medoid_stream(
                std::string((const char *)content_labels_to_meoids._content, content_labels_to_meoids._size));
#else
        if (file_exists(labels_to_medoids))
        {
            std::ifstream medoid_stream(labels_to_medoids);
            assert(medoid_stream.is_open());
#endif
            std::string line, token;

            _filter_to_medoid_ids.clear();
            try
            {
                while (std::getline(medoid_stream, line))
                {
                    std::istringstream iss(line);
                    uint32_t cnt = 0;
                    std::vector<uint32_t> medoids;
                    LabelT label;
                    while (std::getline(iss, token, ','))
                    {
                        if (cnt == 0)
                            label = (LabelT)std::stoul(token);
                        else
                            medoids.push_back((uint32_t)stoul(token));
                        cnt++;
                    }
                    _filter_to_medoid_ids[label].swap(medoids);
                }
            }
            catch (std::system_error &e)
            {
                throw FileException(labels_to_medoids, e, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
        std::string univ_label_file = std ::string(_disk_index_file) + "_universal_label.txt";

#ifdef EXEC_ENV_OLS
        if (files.fileExists(univ_label_file))
        {
            FileContent &content_univ_label = files.getContent(univ_label_file);
            std::stringstream universal_label_reader(
                std::string((const char *)content_univ_label._content, content_univ_label._size));
#else
        if (file_exists(univ_label_file))
        {
            std::ifstream universal_label_reader(univ_label_file);
            assert(universal_label_reader.is_open());
#endif
            std::string univ_label;
            universal_label_reader >> univ_label;
#ifndef EXEC_ENV_OLS
            universal_label_reader.close();
#endif
            LabelT label_as_num = (LabelT)std::stoul(univ_label);
            set_universal_label(label_as_num);
        }

#ifdef EXEC_ENV_OLS
        if (files.fileExists(dummy_map_file))
        {
            FileContent &content_dummy_map = files.getContent(dummy_map_file);
            std::stringstream dummy_map_stream(
                std::string((const char *)content_dummy_map._content, content_dummy_map._size));
#else
        if (file_exists(dummy_map_file))
        {
            std::ifstream dummy_map_stream(dummy_map_file);
            assert(dummy_map_stream.is_open());
#endif
            std::string line, token;

            while (std::getline(dummy_map_stream, line))
            {
                std::istringstream iss(line);
                uint32_t cnt = 0;
                uint32_t dummy_id;
                uint32_t real_id;
                while (std::getline(iss, token, ','))
                {
                    if (cnt == 0)
                        dummy_id = (uint32_t)stoul(token);
                    else
                        real_id = (uint32_t)stoul(token);
                    cnt++;
                }
                _dummy_pts.insert(dummy_id);
                _has_dummy_pts.insert(real_id);
                _dummy_to_real_map[dummy_id] = real_id;

                if (_real_to_dummy_map.find(real_id) == _real_to_dummy_map.end())
                    _real_to_dummy_map[real_id] = std::vector<uint32_t>();

                _real_to_dummy_map[real_id].emplace_back(dummy_id);
            }
#ifndef EXEC_ENV_OLS
            dummy_map_stream.close();
#endif
            diskann::cout << "Loaded dummy map" << std::endl;
        }
    }

#ifdef EXEC_ENV_OLS
    _pq_table.load_pq_centroid_bin(files, pq_table_bin.c_str(), nchunks_u64);
#else
    _pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
#endif

    diskann::cout << "Loaded PQ centroids and in-memory compressed vectors. #points: " << _num_points
                  << " #dim: " << _data_dim << " #aligned_dim: " << _aligned_dim << " #chunks: " << _n_chunks
                  << std::endl;

    if (_n_chunks > MAX_PQ_CHUNKS)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max PQ bytes for in-memory "
                  "PQ data does not exceed "
               << MAX_PQ_CHUNKS << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::string disk_pq_pivots_path = this->_disk_index_file + "_pq_pivots.bin";
#ifdef EXEC_ENV_OLS
    if (files.fileExists(disk_pq_pivots_path))
    {
        _use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        _disk_pq_table.load_pq_centroid_bin(files, disk_pq_pivots_path.c_str(), 0);
#else
    if (file_exists(disk_pq_pivots_path))
    {
        _use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        _disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(), 0);
#endif
        _disk_pq_n_chunks = _disk_pq_table.get_num_chunks();
        _disk_bytes_per_point =
            _disk_pq_n_chunks * sizeof(uint8_t); // revising disk_bytes_per_point since DISK PQ is used.
        diskann::cout << "Disk index uses PQ data compressed down to " << _disk_pq_n_chunks << " bytes per point."
                      << std::endl;
    }

// read index metadata
#ifdef EXEC_ENV_OLS
    // This is a bit tricky. We have to read the header from the
    // disk_index_file. But  this is now exclusively a preserve of the
    // DiskPriorityIO class. So, we need to estimate how many
    // bytes are needed to store the header and read in that many using our
    // 'standard' aligned file reader approach.
    reader->open(_disk_index_file);
    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

    char *bytes = getHeaderBytes();
    ContentBuf buf(bytes, HEADER_SIZE);
    std::basic_istream<char> index_metadata(&buf);
#else
    std::ifstream index_metadata(_disk_index_file, std::ios::binary);
#endif

    uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
                     // metadata, nc should be 1)
    READ_U32(index_metadata, nr);
    READ_U32(index_metadata, nc);

    uint64_t disk_nnodes;
    uint64_t disk_ndims; // can be disk PQ dim if disk_PQ is set to true
    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, disk_ndims);

    if (disk_nnodes != _num_points)
    {
        diskann::cout << "Mismatch in #points for compressed data file and disk "
                         "index file: "
                      << disk_nnodes << " vs " << _num_points << std::endl;
        return -1;
    }

    size_t medoid_id_on_file;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, _max_node_len);
    READ_U64(index_metadata, _nnodes_per_sector);
    _max_degree = ((_max_node_len - _disk_bytes_per_point) / sizeof(uint32_t)) - 1;

    if (_max_degree > defaults::MAX_GRAPH_DEGREE)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max graph degree (R) does "
                  "not exceed "
               << defaults::MAX_GRAPH_DEGREE << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // setting up concept of frozen points in disk index for streaming-DiskANN
    READ_U64(index_metadata, this->_num_frozen_points);
    uint64_t file_frozen_id;
    READ_U64(index_metadata, file_frozen_id);
    if (this->_num_frozen_points == 1)
        this->_frozen_location = file_frozen_id;
    if (this->_num_frozen_points == 1)
    {
        diskann::cout << " Detected frozen point in index at location " << this->_frozen_location
                      << ". Will not output it at search time." << std::endl;
    }

    READ_U64(index_metadata, this->_reorder_data_exists);
    if (this->_reorder_data_exists)
    {
        if (this->_use_disk_index_pq == false)
        {
            throw ANNException("Reordering is designed for used with disk PQ "
                               "compression option",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        READ_U64(index_metadata, this->_reorder_data_start_sector);
        READ_U64(index_metadata, this->_ndims_reorder_vecs);
        READ_U64(index_metadata, this->_nvecs_per_sector);
    }

    diskann::cout << "Disk-Index File Meta-data: ";
    diskann::cout << "# nodes per sector: " << _nnodes_per_sector;
    diskann::cout << ", max node len (bytes): " << _max_node_len;
    diskann::cout << ", max node degree: " << _max_degree << std::endl;

#ifdef EXEC_ENV_OLS
    delete[] bytes;
#else
    index_metadata.close();
#endif

#ifndef EXEC_ENV_OLS
    // open AlignedFileReader handle to index_file
    std::string index_fname(_disk_index_file);
    reader->open(index_fname);
    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

#endif

#ifdef EXEC_ENV_OLS
    if (files.fileExists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(files, norm_file, medoids_file, _medoids, _num_medoids, tmp_dim);
#else
    if (file_exists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(medoids_file, _medoids, _num_medoids, tmp_dim);
#endif

        if (tmp_dim != 1)
        {
            std::stringstream stream;
            stream << "Error loading medoids file. Expected bin format of m times "
                      "1 vector of uint32_t."
                   << std::endl;
            throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
#ifdef EXEC_ENV_OLS
        if (!files.fileExists(centroids_file))
        {
#else
        if (!file_exists(centroids_file))
        {
#endif
            diskann::cout << "Centroid data file not found. Using corresponding vectors "
                             "for the medoids "
                          << std::endl;
            use_medoids_data_as_centroids();
        }
        else
        {
            size_t num_centroids, aligned_tmp_dim;
#ifdef EXEC_ENV_OLS
            diskann::load_aligned_bin<float>(files, centroids_file, _centroid_data, num_centroids, tmp_dim,
                                             aligned_tmp_dim);
#else
            diskann::load_aligned_bin<float>(centroids_file, _centroid_data, num_centroids, tmp_dim, aligned_tmp_dim);
#endif
            if (aligned_tmp_dim != _aligned_dim || num_centroids != _num_medoids)
            {
                std::stringstream stream;
                stream << "Error loading centroids data file. Expected bin format "
                          "of "
                          "m times data_dim vector of float, where m is number of "
                          "medoids "
                          "in medoids file.";
                diskann::cerr << stream.str() << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
    }
    else
    {
        _num_medoids = 1;
        _medoids = new uint32_t[1];
        _medoids[0] = (uint32_t)(medoid_id_on_file);
        use_medoids_data_as_centroids();
    }

    std::string norm_file = std::string(_disk_index_file) + "_max_base_norm.bin";

#ifdef EXEC_ENV_OLS
    if (files.fileExists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        uint64_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(files, norm_val, dumr, dumc);
#else
    if (file_exists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        uint64_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
#endif
        this->_max_base_norm = norm_val[0];
        diskann::cout << "Setting re-scaling factor of base vectors to " << this->_max_base_norm << std::endl;
        delete[] norm_val;
    }
    diskann::cout << "done.." << std::endl;
    return 0;
}

#ifdef USE_BING_INFRA
bool getNextCompletedRequest(std::shared_ptr<AlignedFileReader> &reader, IOContext &ctx, size_t size,
                             int &completedIndex)
{
    if ((*ctx.m_pRequests)[0].m_callback)
    {
        bool waitsRemaining = false;
        long completeCount = ctx.m_completeCount;
        do
        {
            for (int i = 0; i < size; i++)
            {
                auto ithStatus = (*ctx.m_pRequestsStatus)[i];
                if (ithStatus == IOContext::Status::READ_SUCCESS)
                {
                    completedIndex = i;
                    return true;
                }
                else if (ithStatus == IOContext::Status::READ_WAIT)
                {
                    waitsRemaining = true;
                }
            }

            // if we didn't find one in READ_SUCCESS, wait for one to complete.
            if (waitsRemaining)
            {
                WaitOnAddress(&ctx.m_completeCount, &completeCount, sizeof(completeCount), 100);
                // this assumes the knowledge of the reader behavior (implicit
                // contract). need better factoring?
            }
        } while (waitsRemaining);

        completedIndex = -1;
        return false;
    }
    else
    {
        reader->wait(ctx, completedIndex);
        return completedIndex != -1;
    }
}
#endif

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_reorder_data, QueryStats *stats)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, std::numeric_limits<uint32_t>::max(),
                       use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const bool use_reorder_data, QueryStats *stats)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, use_filter, filter_label,
                       std::numeric_limits<uint32_t>::max(), use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const uint32_t io_limit, const bool use_reorder_data,
                                                 QueryStats *stats)
{
    LabelT dummy_filter = 0;
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, false, dummy_filter, io_limit,
                       use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const uint32_t io_limit, const bool use_reorder_data,
                                                 QueryStats *stats)
{

    uint64_t num_sector_per_nodes = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    if (beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS)
        throw ANNException("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1, __FUNCSIG__, __FILE__,
                           __LINE__);

    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;
    auto query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->pq_scratch();

    // reset query scratch
    query_scratch->reset();
    if (_query_sector_cache_enabled)
    {
        query_scratch->sector_cache.clear();
        query_scratch->merit_sector_cache.clear();
        query_scratch->disk_read_sector_order.clear();
    }

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float query_norm = 0;
    T *aligned_query_T = query_scratch->aligned_query_T();
    float *query_float = pq_query_scratch->aligned_query_float;
    float *query_rotated = pq_query_scratch->rotated_query;

    // normalization step. for cosine, we simply normalize the query
    // for mips, we normalize the first d-1 dims, and add a 0 for last dim, since an extra coordinate was used to
    // convert MIPS to L2 search
    if (metric == diskann::Metric::INNER_PRODUCT || metric == diskann::Metric::COSINE)
    {
        uint64_t inherent_dim = (metric == diskann::Metric::COSINE) ? this->_data_dim : (uint64_t)(this->_data_dim - 1);
        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = query1[i];
            query_norm += query1[i] * query1[i];
        }
        if (metric == diskann::Metric::INNER_PRODUCT)
            aligned_query_T[this->_data_dim - 1] = 0;

        query_norm = std::sqrt(query_norm);

        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = (T)(aligned_query_T[i] / query_norm);
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }
    else
    {
        for (size_t i = 0; i < this->_data_dim; i++)
        {
            aligned_query_T[i] = query1[i];
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }

    // pointers to buffers for data
    T *data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *)data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    uint64_t &sector_scratch_idx = query_scratch->sector_idx;
    const uint64_t num_sectors_per_node =
        _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

    // query <-> PQ chunk centers distances
    _pq_table.preprocess_query(query_rotated); // center the query and rotate if
                                               // we have a rotation matrix
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                                            float *dists_out) {
        diskann::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch);
        diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out);
    };
    auto profile_on_expand = [this, query_scratch](uint32_t node_id) {
        if (!this->_hotness_profiler.enabled())
            return;
        this->_hotness_profiler.on_node_expand(node_id);
        auto parent_it = query_scratch->profile_parent.find(node_id);
        if (parent_it != query_scratch->profile_parent.end())
            this->_hotness_profiler.on_directed_edge(parent_it->second, node_id);
    };
    auto profile_on_first_visit = [this, query_scratch](uint32_t id, uint32_t parent) {
        if (!this->_hotness_profiler.enabled())
            return;
        if (query_scratch->profile_parent.find(id) == query_scratch->profile_parent.end())
            query_scratch->profile_parent.insert({id, parent});
    };
    Timer query_timer, io_timer, cpu_timer;

    tsl::robin_set<uint64_t> &visited = query_scratch->visited;
    NeighborPriorityQueue &retset = query_scratch->retset;
    retset.reserve(l_search);
    std::vector<Neighbor> &full_retset = query_scratch->full_retset;

    uint32_t best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();
    if (!use_filter)
    {
        for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
        {
            float cur_expanded_dist =
                _dist_cmp_float->compare(query_float, _centroid_data + _aligned_dim * cur_m, (uint32_t)_aligned_dim);
            if (cur_expanded_dist < best_dist)
            {
                best_medoid = _medoids[cur_m];
                best_dist = cur_expanded_dist;
            }
        }
    }
    else
    {
        if (_filter_to_medoid_ids.find(filter_label) != _filter_to_medoid_ids.end())
        {
            const auto &medoid_ids = _filter_to_medoid_ids[filter_label];
            for (uint64_t cur_m = 0; cur_m < medoid_ids.size(); cur_m++)
            {
                // for filtered index, we dont store global centroid data as for unfiltered index, so we use PQ distance
                // as approximation to decide closest medoid matching the query filter.
                compute_dists(&medoid_ids[cur_m], 1, dist_scratch);
                float cur_expanded_dist = dist_scratch[0];
                if (cur_expanded_dist < best_dist)
                {
                    best_medoid = medoid_ids[cur_m];
                    best_dist = cur_expanded_dist;
                }
            }
        }
        else
        {
            throw ANNException("Cannot find medoid for specified filter.", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset.insert(Neighbor(best_medoid, dist_scratch[0]));
    visited.insert(best_medoid);

    uint32_t cmps = 0;
    uint32_t hops = 0;
    uint32_t num_ios = 0;

    // cleared every iteration
        std::vector<uint32_t> frontier;
        frontier.reserve(2 * beam_width);
        std::vector<uint32_t> merit_frontier;
        merit_frontier.reserve(2 * beam_width);
        std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
        frontier_nhoods.reserve(2 * beam_width);
        std::vector<AlignedRead> frontier_read_reqs;
        frontier_read_reqs.reserve(2 * beam_width);
        std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>> cached_nhoods;
        cached_nhoods.reserve(2 * beam_width);

        while (retset.has_unexpanded_node() && num_ios < io_limit)
        {
            // clear iteration state
            frontier.clear();
            merit_frontier.clear();
            frontier_nhoods.clear();
            frontier_read_reqs.clear();
            cached_nhoods.clear();
            sector_scratch_idx = 0;
            // find new beam
            uint32_t num_seen = 0;
            while (retset.has_unexpanded_node() && frontier.size() + merit_frontier.size() < beam_width &&
                   num_seen < beam_width)
            {
                auto nbr = retset.closest_unexpanded();
                num_seen++;
                bool mem_hit = false;
                if (_merit_mem_pool != nullptr && _merit_mem_pool->active())
                {
                    T *coord_ptr = nullptr;
                    std::pair<uint32_t, uint32_t *> nh_pair;
                    if (_merit_mem_pool->lookup(nbr.id, coord_ptr, nh_pair))
                    {
                        cached_nhoods.push_back(std::make_pair(nbr.id, nh_pair));
                        _merit_mem_pool->bump(nbr.id);
                        mem_hit = true;
                        if (stats != nullptr)
                            stats->n_cache_hits++;
                    }
                }
                else
                {
                    auto iter = _nhood_cache.find(nbr.id);
                    if (iter != _nhood_cache.end())
                    {
                        cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
                        mem_hit = true;
                        if (stats != nullptr)
                            stats->n_cache_hits++;
                    }
                }
                if (!mem_hit)
                {
                    if (!_merit_dc_map.empty() && _merit_dc_map.find(nbr.id) != _merit_dc_map.end())
                    {
                        merit_frontier.push_back(nbr.id);
                    }
                    else
                    {
                        frontier.push_back(nbr.id);
                        if (_record_base_frontier && stats != nullptr)
                            stats->base_frontier_nodes.push_back(nbr.id);
                    }
                }
                profile_on_expand(nbr.id);
                if (this->_count_visited_nodes)
                {
                    reinterpret_cast<std::atomic<uint32_t> &>(this->_node_visit_counter[nbr.id].second).fetch_add(1);
                }
            }

            const size_t read_len = num_sectors_per_node * defaults::SECTOR_LEN;

            if (_record_hop_frontier && stats != nullptr && (!merit_frontier.empty() || !frontier.empty()))
            {
                QueryStats::HopFrontierRecord rec;
                rec.hop = stats->n_hops;
                rec.merit_nodes = merit_frontier;
                rec.base_nodes = frontier;
                stats->hop_frontier_trace.push_back(std::move(rec));
            }

            if (_merit_unified_disk && (!merit_frontier.empty() || !frontier.empty()))
            {
                if (stats != nullptr)
                    stats->n_hops++;

                std::vector<AlignedRead> combined_reqs;
                combined_reqs.reserve(merit_frontier.size() + frontier.size());
                std::vector<uint64_t> miss_sector_ids;
                miss_sector_ids.reserve(frontier.size());
                std::vector<std::pair<uint64_t, char *>> base_miss_bufs;
                base_miss_bufs.reserve(frontier.size());

                std::vector<MeritReadPending> merit_pending;
                std::vector<AlignedRead> merit_io;
                std::unordered_map<uint32_t, std::vector<size_t>> merit_disk_fanout;

                prepare_merit_sidecar_io(merit_frontier, query_scratch, sector_scratch, sector_scratch_idx,
                                         num_sectors_per_node, merit_pending, merit_io, merit_disk_fanout, stats,
                                         num_ios);

                for (const auto &mr : merit_io)
                    combined_reqs.push_back(mr);

                for (uint64_t i = 0; i < frontier.size(); i++)
                {
                    const auto id = frontier[i];
                    std::pair<uint32_t, char *> fnhood;
                    fnhood.first = id;
                    const uint64_t sec = get_node_sector((size_t)id);
                    query_scratch->read_sectors.insert(sec);

                    bool served_from_cache = false;
                    if (_query_sector_cache_enabled && read_len <= defaults::SECTOR_LEN)
                    {
                        const auto cache_it = query_scratch->sector_cache.find(sec);
                        if (cache_it != query_scratch->sector_cache.end())
                        {
                            fnhood.second = const_cast<char *>(cache_it->second.data());
                            if (stats != nullptr)
                            {
                                stats->n_sector_cache_hits++;
                                stats->n_4k++;
                                stats->n_ios++;
                            }
                            num_ios++;
                            served_from_cache = true;
                        }
                    }
                    if (!served_from_cache)
                    {
                        fnhood.second =
                            sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                        sector_scratch_idx++;
                        combined_reqs.emplace_back(sec * defaults::SECTOR_LEN, read_len, fnhood.second);
                        miss_sector_ids.push_back(sec);
                        base_miss_bufs.emplace_back(sec, fnhood.second);
                        if (stats != nullptr)
                        {
                            query_scratch->disk_read_sector_order.push_back(sec);
                            stats->n_4k++;
                            stats->n_ios++;
                            stats->n_disk_reads++;
                        }
                        num_ios++;
                    }
                    frontier_nhoods.push_back(fnhood);
                }

                if (!combined_reqs.empty())
                {
                    std::vector<size_t> order(combined_reqs.size());
                    std::iota(order.begin(), order.end(), 0);
                    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                        return combined_reqs[a].offset < combined_reqs[b].offset;
                    });
                    std::vector<AlignedRead> sorted_reqs;
                    sorted_reqs.reserve(combined_reqs.size());
                    for (size_t idx : order)
                        sorted_reqs.push_back(combined_reqs[idx]);
                    combined_reqs = std::move(sorted_reqs);

                    io_timer.reset();
#ifdef USE_BING_INFRA
                    reader->read(combined_reqs, ctx, true);
#else
                    reader->read(combined_reqs, ctx);
#endif
                    if (stats != nullptr)
                    {
                        const float batch_us = (float)io_timer.elapsed();
                        stats->io_us += batch_us;
                        if (!miss_sector_ids.empty())
                        {
                            const uint64_t max_graph_sector =
                                (_nnodes_per_sector > 0 && _num_points > 0)
                                    ? (1 + (_num_points - 1) / _nnodes_per_sector)
                                    : 1;
                            record_disk_read_batch_stats(stats, batch_us, miss_sector_ids, max_graph_sector);
                            if (miss_sector_ids.size() >= 2)
                            {
                                uint64_t mn = miss_sector_ids[0];
                                uint64_t mx = miss_sector_ids[0];
                                for (uint64_t sid : miss_sector_ids)
                                {
                                    mn = std::min(mn, sid);
                                    mx = std::max(mx, sid);
                                }
                                stats->sum_intra_batch_spread += (mx - mn);
                                stats->n_intra_batch_spread_samples++;
                            }
                        }
                    }
                    if (_query_sector_cache_enabled && read_len <= defaults::SECTOR_LEN)
                    {
                        for (const auto &bm : base_miss_bufs)
                        {
                            std::array<char, defaults::SECTOR_LEN> &slot = query_scratch->sector_cache[bm.first];
                            memcpy(slot.data(), bm.second, read_len);
                            for (auto &fn : frontier_nhoods)
                            {
                                if (fn.second == bm.second)
                                    fn.second = slot.data();
                            }
                        }
                    }
                }

                complete_merit_sidecar_io(query_scratch, merit_pending, merit_disk_fanout);
                finalize_merit_pending_nodes(merit_pending, query_scratch, sector_scratch, sector_scratch_idx,
                                               frontier_nhoods);
            }
            else
            {
            std::vector<MeritReadPending> merit_pending;
            std::vector<AlignedRead> merit_io;
            std::unordered_map<uint32_t, std::vector<size_t>> merit_disk_fanout;
            std::vector<uint64_t> miss_sector_ids;
            miss_sector_ids.reserve(frontier.size());

            if (!merit_frontier.empty() && _merit_disk_reader)
            {
                prepare_merit_sidecar_io(merit_frontier, query_scratch, sector_scratch, sector_scratch_idx,
                                         num_sectors_per_node, merit_pending, merit_io, merit_disk_fanout, stats,
                                         num_ios);
            }

            // read nhoods of frontier ids
            if (!frontier.empty())
            {
                const size_t read_len = num_sectors_per_node * defaults::SECTOR_LEN;

            if (_query_sector_cache_enabled && read_len <= defaults::SECTOR_LEN)
            {
                miss_sector_ids.reserve(frontier.size());
                for (uint64_t i = 0; i < frontier.size(); i++)
                {
                    auto id = frontier[i];
                    std::pair<uint32_t, char *> fnhood;
                    fnhood.first = id;
                    const uint64_t sec = get_node_sector((size_t)id);
                    query_scratch->read_sectors.insert(sec);

                    const auto cache_it = query_scratch->sector_cache.find(sec);
                    if (cache_it != query_scratch->sector_cache.end())
                    {
                        fnhood.second = const_cast<char *>(cache_it->second.data());
                        if (stats != nullptr)
                        {
                            stats->n_sector_cache_hits++;
                            stats->n_4k++;
                            stats->n_ios++;
                        }
                        num_ios++;
                    }
                    else
                    {
                        fnhood.second =
                            sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                        sector_scratch_idx++;
                        frontier_read_reqs.emplace_back(sec * defaults::SECTOR_LEN, read_len, fnhood.second);
                        miss_sector_ids.push_back(sec);
                        if (stats != nullptr)
                        {
                            query_scratch->disk_read_sector_order.push_back(sec);
                            stats->n_4k++;
                            stats->n_ios++;
                            stats->n_disk_reads++;
                        }
                        num_ios++;
                    }
                    frontier_nhoods.push_back(fnhood);
                }
                if (!merit_io.empty() || !frontier_read_reqs.empty())
                {
                    if (stats != nullptr)
                        stats->n_hops++;
                    const float batch_us = issue_merit_and_base_disk_reads(reader, _merit_disk_reader, ctx, merit_io,
                                                                           frontier_read_reqs, io_timer, stats);
                    if (stats != nullptr && !frontier_read_reqs.empty())
                    {
                        const uint64_t max_graph_sector =
                            (_nnodes_per_sector > 0 && _num_points > 0)
                                ? (1 + (_num_points - 1) / _nnodes_per_sector)
                                : 1;
                        record_disk_read_batch_stats(stats, batch_us, miss_sector_ids, max_graph_sector);
                        if (miss_sector_ids.size() >= 2)
                        {
                            uint64_t mn = miss_sector_ids[0];
                            uint64_t mx = miss_sector_ids[0];
                            for (uint64_t sid : miss_sector_ids)
                            {
                                mn = std::min(mn, sid);
                                mx = std::max(mx, sid);
                            }
                            stats->sum_intra_batch_spread += (mx - mn);
                            stats->n_intra_batch_spread_samples++;
                        }
                    }
                    for (size_t ri = 0; ri < frontier_read_reqs.size(); ri++)
                    {
                        std::array<char, defaults::SECTOR_LEN> &slot = query_scratch->sector_cache[miss_sector_ids[ri]];
                        memcpy(slot.data(), frontier_read_reqs[ri].buf, read_len);
                        for (auto &fn : frontier_nhoods)
                        {
                            if (fn.second == frontier_read_reqs[ri].buf)
                                fn.second = slot.data();
                        }
                    }
                    complete_merit_sidecar_io(query_scratch, merit_pending, merit_disk_fanout);
                }
            }
            else
            {
                for (uint64_t i = 0; i < frontier.size(); i++)
                {
                    auto id = frontier[i];
                    std::pair<uint32_t, char *> fnhood;
                    fnhood.first = id;
                    fnhood.second = sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                    sector_scratch_idx++;
                    frontier_nhoods.push_back(fnhood);
                    const uint64_t sec = get_node_sector((size_t)id);
                    query_scratch->read_sectors.insert(sec);
                    frontier_read_reqs.emplace_back(sec * defaults::SECTOR_LEN, read_len, fnhood.second);
                    if (stats != nullptr)
                    {
                        stats->n_4k++;
                        stats->n_ios++;
                        stats->n_disk_reads++;
                    }
                    num_ios++;
                }
                if (!merit_io.empty() || !frontier_read_reqs.empty())
                {
                    if (stats != nullptr)
                        stats->n_hops++;
                    issue_merit_and_base_disk_reads(reader, _merit_disk_reader, ctx, merit_io, frontier_read_reqs,
                                                    io_timer, stats);
                    complete_merit_sidecar_io(query_scratch, merit_pending, merit_disk_fanout);
                }
            }
            }
            if (frontier.empty() && !merit_io.empty())
            {
                if (stats != nullptr)
                    stats->n_hops++;
                issue_merit_and_base_disk_reads(reader, _merit_disk_reader, ctx, merit_io, frontier_read_reqs, io_timer,
                                                stats);
                complete_merit_sidecar_io(query_scratch, merit_pending, merit_disk_fanout);
            }

            if (!merit_pending.empty())
                finalize_merit_pending_nodes(merit_pending, query_scratch, sector_scratch, sector_scratch_idx,
                                             frontier_nhoods);

            }
        // process cached nhoods
        for (auto &cached_nhood : cached_nhoods)
        {
            T *node_fp_coords_copy = nullptr;
            if (_merit_mem_pool != nullptr && _merit_mem_pool->active())
            {
                std::pair<uint32_t, uint32_t *> nh_dummy;
                if (!_merit_mem_pool->lookup(cached_nhood.first, node_fp_coords_copy, nh_dummy))
                    continue;
            }
            else
            {
                auto global_cache_iter = _coord_cache.find(cached_nhood.first);
                if (global_cache_iter == _coord_cache.end())
                    continue;
                node_fp_coords_copy = global_cache_iter->second;
            }
            float cur_expanded_dist;
            if (!_use_disk_index_pq)
            {
                cur_expanded_dist = _dist_cmp->compare(aligned_query_T, node_fp_coords_copy, (uint32_t)_aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)node_fp_coords_copy);
                else
                    cur_expanded_dist = _disk_pq_table.l2_distance( // disk_pq does not support OPQ yet
                        query_float, (uint8_t *)node_fp_coords_copy);
            }
            full_retset.push_back(Neighbor((uint32_t)cached_nhood.first, cur_expanded_dist));

            uint64_t nnbrs = cached_nhood.second.first;
            uint32_t *node_nbrs = cached_nhood.second.second;

            // compute node_nbrs <-> query dists in PQ space
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            // process prefetched nhood
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    profile_on_first_visit(id, cached_nhood.first);
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                        continue;

                    if (use_filter && !(point_has_label(id, filter_label)) &&
                        (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                        continue;
                    cmps++;
                    float dist = dist_scratch[m];
                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }
        }
#ifdef USE_BING_INFRA
        // process each frontier nhood - compute distances to unvisited nodes
        int completedIndex = -1;
        long requestCount = static_cast<long>(frontier_read_reqs.size());
        // If we issued read requests and if a read is complete or there are
        // reads in wait state, then enter the while loop.
        while (requestCount > 0 && getNextCompletedRequest(reader, ctx, requestCount, completedIndex))
        {
            assert(completedIndex >= 0);
            auto &frontier_nhood = frontier_nhoods[completedIndex];
            (*ctx.m_pRequestsStatus)[completedIndex] = IOContext::PROCESS_COMPLETE;
#else
        for (auto &frontier_nhood : frontier_nhoods)
        {
#endif
            char *node_disk_buf = offset_to_node(frontier_nhood.second, frontier_nhood.first);
            if (_merit_mem_pool != nullptr && _merit_mem_pool->active() && _merit_mem_runtime_admit)
            {
                const uint32_t evicted = _merit_mem_pool->try_admit(frontier_nhood.first, node_disk_buf,
                                                                    _disk_bytes_per_point, _max_node_len);
                if (stats != nullptr && evicted != MeritMemoryPool<T>::INVALID_NODE)
                    stats->n_merit_mem_evictions++;
            }
            uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
            uint64_t nnbrs = (uint64_t)(*node_buf);
            T *node_fp_coords = offset_to_node_coords(node_disk_buf);
            memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
            float cur_expanded_dist;
            if (!_use_disk_index_pq)
            {
                cur_expanded_dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)data_buf);
                else
                    cur_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *)data_buf);
            }
            full_retset.push_back(Neighbor(frontier_nhood.first, cur_expanded_dist));
            uint32_t *node_nbrs = (node_buf + 1);
            // compute node_nbrs <-> query dist in PQ space
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            cpu_timer.reset();
            // process prefetch-ed nhood
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    profile_on_first_visit(id, frontier_nhood.first);
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                        continue;

                    if (use_filter && !(point_has_label(id, filter_label)) &&
                        (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                        continue;
                    cmps++;
                    float dist = dist_scratch[m];
                    if (stats != nullptr)
                    {
                        stats->n_cmps++;
                    }

                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }

            if (stats != nullptr)
            {
                stats->cpu_us += (float)cpu_timer.elapsed();
            }
        }

        hops++;
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end());

    if (use_reorder_data)
    {
        if (!(this->_reorder_data_exists))
        {
            throw ANNException("Requested use of reordering data which does "
                               "not exist in index "
                               "file",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        std::vector<AlignedRead> vec_read_reqs;

        if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
            full_retset.erase(full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER, full_retset.end());

        for (size_t i = 0; i < full_retset.size(); ++i)
        {
            // MULTISECTORFIX
            vec_read_reqs.emplace_back(VECTOR_SECTOR_NO(((size_t)full_retset[i].id)) * defaults::SECTOR_LEN,
                                       defaults::SECTOR_LEN, sector_scratch + i * defaults::SECTOR_LEN);

            if (stats != nullptr)
            {
                stats->n_4k++;
                stats->n_ios++;
            }
        }

        io_timer.reset();
#ifdef USE_BING_INFRA
        reader->read(vec_read_reqs, ctx, true); // async reader windows.
#else
        reader->read(vec_read_reqs, ctx); // synchronous IO linux
#endif
        if (stats != nullptr)
        {
            stats->io_us += io_timer.elapsed();
        }

        for (size_t i = 0; i < full_retset.size(); ++i)
        {
            auto id = full_retset[i].id;
            // MULTISECTORFIX
            auto location = (sector_scratch + i * defaults::SECTOR_LEN) + VECTOR_SECTOR_OFFSET(id);
            full_retset[i].distance = _dist_cmp->compare(aligned_query_T, (T *)location, (uint32_t)this->_data_dim);
        }

        std::sort(full_retset.begin(), full_retset.end());
    }

    // copy k_search values
    for (uint64_t i = 0; i < k_search; i++)
    {
        indices[i] = full_retset[i].id;
        auto key = (uint32_t)indices[i];
        if (_dummy_pts.find(key) != _dummy_pts.end())
        {
            indices[i] = _dummy_to_real_map[key];
        }

        if (distances != nullptr)
        {
            distances[i] = full_retset[i].distance;
            if (metric == diskann::Metric::INNER_PRODUCT)
            {
                // flip the sign to convert min to max
                distances[i] = (-distances[i]);
                // rescale to revert back to original norms (cancelling the
                // effect of base and query pre-processing)
                if (_max_base_norm != 0)
                    distances[i] *= (_max_base_norm * query_norm);
            }
        }
    }

#ifdef USE_BING_INFRA
    ctx.m_completeCount = 0;
#endif

    if (stats != nullptr)
    {
        stats->n_unique_sectors = (unsigned)query_scratch->read_sectors.size();
        stats->n_unique_merit_sectors = (unsigned)query_scratch->read_merit_sidecar_sectors.size();
        stats->total_us = (float)query_timer.elapsed();
        if (_query_sector_cache_enabled)
        {
            const auto &seq = query_scratch->disk_read_sector_order;
            for (size_t i = 1; i < seq.size(); i++)
            {
                const uint64_t a = seq[i - 1];
                const uint64_t b = seq[i];
                const uint64_t d = (a >= b) ? (a - b) : (b - a);
                stats->sum_abs_sector_jump += d;
                stats->n_sector_jump_samples++;
                if (d <= 8)
                    stats->n_sector_jump_le8++;
                if (d > stats->max_sector_jump)
                    stats->max_sector_jump = d;
            }
        }
    }
}

// range search returns results of all neighbors within distance of range.
// indices and distances need to be pre-allocated of size l_search and the
// return value is the number of matching hits.
template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::range_search(const T *query1, const double range, const uint64_t min_l_search,
                                               const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                               std::vector<float> &distances, const uint64_t min_beam_width,
                                               QueryStats *stats)
{
    uint32_t res_count = 0;

    bool stop_flag = false;

    uint32_t l_search = (uint32_t)min_l_search; // starting size of the candidate list
    while (!stop_flag)
    {
        indices.resize(l_search);
        distances.resize(l_search);
        uint64_t cur_bw = min_beam_width > (l_search / 5) ? min_beam_width : l_search / 5;
        cur_bw = (cur_bw > 100) ? 100 : cur_bw;
        for (auto &x : distances)
            x = std::numeric_limits<float>::max();
        this->cached_beam_search(query1, l_search, l_search, indices.data(), distances.data(), cur_bw, false, stats);
        for (uint32_t i = 0; i < l_search; i++)
        {
            if (distances[i] > (float)range)
            {
                res_count = i;
                break;
            }
            else if (i == l_search - 1)
                res_count = l_search;
        }
        if (res_count < (uint32_t)(l_search / 2.0))
            stop_flag = true;
        l_search = l_search * 2;
        if (l_search > max_l_search)
            stop_flag = true;
    }
    indices.resize(res_count);
    distances.resize(res_count);
    return res_count;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::get_data_dim()
{
    return _data_dim;
}

template <typename T, typename LabelT> diskann::Metric PQFlashIndex<T, LabelT>::get_metric()
{
    return this->metric;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT> char *PQFlashIndex<T, LabelT>::getHeaderBytes()
{
    IOContext &ctx = reader->get_ctx();
    AlignedRead readReq;
    readReq.buf = new char[PQFlashIndex<T, LabelT>::HEADER_SIZE];
    readReq.len = PQFlashIndex<T, LabelT>::HEADER_SIZE;
    readReq.offset = 0;

    std::vector<AlignedRead> readReqs;
    readReqs.push_back(readReq);

    reader->read(readReqs, ctx, false);

    return (char *)readReq.buf;
}
#endif

template <typename T, typename LabelT>
std::vector<std::uint8_t> PQFlashIndex<T, LabelT>::get_pq_vector(std::uint64_t vid)
{
    std::uint8_t *pqVec = &this->data[vid * this->_n_chunks];
    return std::vector<std::uint8_t>(pqVec, pqVec + this->_n_chunks);
}

template <typename T, typename LabelT> std::uint64_t PQFlashIndex<T, LabelT>::get_num_points()
{
    return _num_points;
}

template <typename T, typename LabelT> std::uint64_t PQFlashIndex<T, LabelT>::get_max_degree()
{
    return _max_degree;
}

namespace
{
uint64_t read_host_mem_total_bytes()
{
    std::ifstream in("/proc/meminfo");
    if (!in.is_open())
        return 0;
    std::string key;
    uint64_t kb = 0;
    while (in >> key)
    {
        if (key == "MemTotal:")
        {
            in >> kb;
            break;
        }
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return kb * 1024ULL;
}
} // namespace

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::bytes_per_cached_node() const
{
    // Matches load_cache_list allocation: aligned coords + (nnbrs + nbr ids).
    const uint64_t coord_bytes = _aligned_dim * sizeof(T);
    const uint64_t nhood_bytes = (_max_degree + 1) * sizeof(uint32_t);
    // Small per-entry map overhead allowance for robin_map nodes.
    const uint64_t map_overhead = 64;
    return coord_bytes + nhood_bytes + map_overhead;
}

template <typename T, typename LabelT>
uint64_t PQFlashIndex<T, LabelT>::estimate_baseline_resident_bytes(uint32_t num_threads) const
{
    uint64_t bytes = 0;
    // Full in-memory PQ codes for all points.
    bytes += _num_points * _n_chunks;
    // PQ pivots / tables (upper bound: 256 centroids × dim floats × a few tables).
    bytes += 256ULL * _data_dim * sizeof(float) * 4ULL;
    if (_use_disk_index_pq)
        bytes += 256ULL * _data_dim * sizeof(float) * 2ULL;
    // Medoids + centroid floats.
    bytes += _num_medoids * (_aligned_dim * sizeof(float) + sizeof(uint32_t));
    // Per-thread SSD scratch: sector buffers + aligned query + PQ scratch.
    const uint64_t per_thread =
        defaults::MAX_N_SECTOR_READS * defaults::SECTOR_LEN + _aligned_dim * sizeof(T) +
        defaults::MAX_GRAPH_DEGREE * sizeof(float) + _aligned_dim * sizeof(float) * 2ULL + (1ULL << 20);
    bytes += static_cast<uint64_t>(num_threads) * per_thread;
    // Label arrays (if present).
    if (_pts_to_label_offsets != nullptr)
        bytes += (_num_points + 1) * sizeof(uint32_t);
    return bytes;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::plan_merit_memory_cache(double merit_memory_gb, double host_memory_gb, double reserve_gb,
                                                     uint32_t num_threads, uint64_t &out_max_nodes,
                                                     std::string &report) const
{
    out_max_nodes = 0;
    report.clear();
    if (merit_memory_gb <= 0.0)
    {
        report = "merit_memory_gb <= 0; MERIT memory cache disabled.";
        return 0;
    }

    const uint64_t host_bytes =
        (host_memory_gb > 0.0) ? static_cast<uint64_t>(host_memory_gb * (1024.0 * 1024.0 * 1024.0))
                               : read_host_mem_total_bytes();
    if (host_bytes == 0)
    {
        report = "Failed to determine host memory size (pass --merit_host_memory_gb).";
        return -1;
    }

    const uint64_t reserve_bytes =
        (reserve_gb > 0.0) ? static_cast<uint64_t>(reserve_gb * (1024.0 * 1024.0 * 1024.0)) : 0;
    const uint64_t baseline = estimate_baseline_resident_bytes(num_threads);
    const uint64_t per_node = bytes_per_cached_node();
    const uint64_t requested = static_cast<uint64_t>(merit_memory_gb * (1024.0 * 1024.0 * 1024.0));

    uint64_t residual = 0;
    if (host_bytes > baseline + reserve_bytes)
        residual = host_bytes - baseline - reserve_bytes;

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << "MERIT memory budget:\n"
        << "  host_total          = " << (host_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  baseline_index      = " << (baseline / (1024.0 * 1024.0 * 1024.0))
        << " GB (PQ + pivots + thread scratch estimate)\n"
        << "  reserve             = " << (reserve_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  residual_for_cache  = " << (residual / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  requested_cache     = " << merit_memory_gb << " GB\n"
        << "  bytes_per_node      = " << per_node << " B"
        << " (aligned_dim=" << _aligned_dim << ", max_degree=" << _max_degree << ")";

    if (requested > residual)
    {
        oss << "\nERROR: requested MERIT memory cache (" << merit_memory_gb
            << " GB) exceeds residual budget (" << (residual / (1024.0 * 1024.0 * 1024.0))
            << " GB). Reduce --merit_memory_gb or --merit_memory_reserve_gb.";
        report = oss.str();
        return -1;
    }

    out_max_nodes = (per_node > 0) ? (requested / per_node) : 0;
    const uint64_t residual_cap = (per_node > 0) ? (residual / per_node) : 0;
    if (out_max_nodes > _num_points)
        out_max_nodes = _num_points;
    if (out_max_nodes > residual_cap)
        out_max_nodes = residual_cap;
    if (out_max_nodes == 0)
    {
        oss << "\nERROR: requested cache too small to hold even one node entry.";
        report = oss.str();
        return -1;
    }

    oss << "\n  max_nodes_in_residual = " << std::min<uint64_t>(residual_cap, _num_points)
        << "\n  selected_nodes      = " << out_max_nodes << " (Top-" << out_max_nodes
        << " by node_expand)\n"
        << "  approx_cache_bytes  = " << ((out_max_nodes * per_node) / (1024.0 * 1024.0 * 1024.0)) << " GB";
    report = oss.str();
    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::build_merit_memory_node_list(const std::string &profile_prefix, uint64_t max_nodes,
                                                         std::vector<uint32_t> &node_list, uint64_t rank_skip) const
{
    node_list.clear();
    if (max_nodes == 0)
        return 0;

    std::vector<uint64_t> node_expand;
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> edges;
    if (HotnessProfiler::load(profile_prefix, node_expand, edges) != 0)
        return -1;
    if (node_expand.size() != _num_points)
    {
        diskann::cerr << "MERIT profile node_expand size " << node_expand.size() << " != num_points " << _num_points
                      << std::endl;
        return -1;
    }

    std::vector<uint32_t> order(_num_points);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (node_expand[a] != node_expand[b])
            return node_expand[a] > node_expand[b];
        return a < b;
    });

    if (rank_skip >= _num_points)
        return 0;

    uint64_t skipped = 0;
    for (uint32_t id : order)
    {
        if (id >= node_expand.size() || node_expand[id] == 0)
            continue;
        if (skipped < rank_skip)
        {
            skipped++;
            continue;
        }
        node_list.push_back(id);
        if (node_list.size() >= max_nodes)
            break;
    }
    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::build_merit_disk_node_list(const std::string &profile_prefix, uint64_t max_nodes,
                                                        uint64_t memory_tier_exclude_count, uint32_t k_hops,
                                                        const std::string &layout_in,
                                                        std::vector<uint32_t> &node_list) const
{
    node_list.clear();
    if (max_nodes == 0)
        return 0;

    std::string layout = normalize_disk_cache_layout(layout_in);
    if (layout.empty())
        layout = (k_hops == 0) ? "flat" : "node";

    std::vector<uint64_t> node_expand;
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> edges;
    if (HotnessProfiler::load(profile_prefix, node_expand, edges) != 0)
        return -1;
    if (node_expand.size() != _num_points)
    {
        diskann::cerr << "MERIT profile node_expand size " << node_expand.size() << " != num_points " << _num_points
                      << std::endl;
        return -1;
    }

    std::string mem_index_path = _disk_index_file;
    const std::string disk_suffix = "_disk.index";
    if (mem_index_path.size() >= disk_suffix.size() &&
        mem_index_path.compare(mem_index_path.size() - disk_suffix.size(), disk_suffix.size(), disk_suffix) == 0)
    {
        mem_index_path.replace(mem_index_path.size() - disk_suffix.size(), disk_suffix.size(), "_mem.index");
    }
    else
    {
        diskann::cerr << "Cannot derive _mem.index from disk index path: " << _disk_index_file << std::endl;
        return -1;
    }

    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;

    std::unordered_set<uint32_t> exclude_ids;
    if (memory_tier_exclude_count > 0)
    {
        std::vector<uint32_t> mem_tier;
        if (build_merit_memory_node_list(profile_prefix, memory_tier_exclude_count, mem_tier, 0) != 0)
            return -1;
        exclude_ids.insert(mem_tier.begin(), mem_tier.end());
    }

    if (layout == "flat" || k_hops == 0)
    {
        std::vector<uint32_t> order(_num_points);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (node_expand[a] != node_expand[b])
                return node_expand[a] > node_expand[b];
            return a < b;
        });
        node_list.reserve(static_cast<size_t>(max_nodes));
        for (uint32_t id : order)
        {
            if (exclude_ids.find(id) != exclude_ids.end())
                continue;
            if (id >= node_expand.size() || node_expand[id] == 0)
                continue;
            node_list.push_back(id);
            if (node_list.size() >= max_nodes)
                break;
        }
        if (node_list.empty())
        {
            diskann::cerr << "MERIT disk-cache: flat hot-node list empty (max_nodes=" << max_nodes << ")."
                          << std::endl;
            return -1;
        }
        diskann::cout << "MERIT disk-cache node list: flat Top-" << node_list.size()
                      << " by node_expand (k_hops=0, exclude memory-tier=" << exclude_ids.size() << ")."
                      << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    VamanaGraph graph;
    if (load_vamana_graph(mem_index_path, graph) != 0)
    {
        diskann::cerr << "MERIT disk-cache: failed to load Vamana graph from " << mem_index_path << std::endl;
        return -1;
    }

    if (layout == "edge" || layout == "jiang")
    {
        if (compute_edge_disk_cache_list(graph, node_expand, edges, nps, k_hops, max_nodes, exclude_ids, node_list) !=
            0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: edge-importance packing, k_hops=" << k_hops
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (compute_hot_node_disk_cache_list(graph, node_expand, edges, nps, k_hops, max_nodes, exclude_ids, node_list) !=
        0)
        return -1;

    diskann::cout << "MERIT disk-cache node list: hot-node path packing (avg edge weight), k_hops=" << k_hops
                  << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                  << ")." << std::endl;
    append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
    return 0;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::merit_memory_cached_count() const
{
    if (_merit_mem_pool && _merit_mem_pool->active())
        return _merit_mem_pool->size();
    return _nhood_cache.size();
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::merit_mem_pool_contains(uint32_t node_id) const
{
    return _merit_mem_pool && _merit_mem_pool->active() && _merit_mem_pool->contains(node_id);
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::merit_dc_map_contains(uint32_t node_id) const
{
    return !_merit_dc_map.empty() && _merit_dc_map.find(node_id) != _merit_dc_map.end();
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::clear_merit_memory_cache()
{
    if (_merit_mem_pool)
        _merit_mem_pool->clear();
    _merit_mem_pool.reset();
    _coord_cache.clear();
    _nhood_cache.clear();
    if (_nhood_cache_buf != nullptr)
    {
        delete[] _nhood_cache_buf;
        _nhood_cache_buf = nullptr;
    }
    if (_coord_cache_buf != nullptr)
    {
        diskann::aligned_free(_coord_cache_buf);
        _coord_cache_buf = nullptr;
    }
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_merit_memory_pool(const std::string &profile_prefix,
                                                    std::vector<uint32_t> &node_list)
{
    clear_merit_memory_cache();
    if (node_list.empty())
        return 0;

    std::vector<uint64_t> node_expand;
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> edges;
    if (HotnessProfiler::load(profile_prefix, node_expand, edges) != 0)
        return -1;

    _merit_mem_pool = std::make_unique<MeritMemoryPool<T>>();
    _merit_mem_pool->init(node_list.size(), _aligned_dim, _max_degree);

    const size_t num_cached_nodes = node_list.size();
    const size_t BLOCK_SIZE = 8;
    const size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);
    for (size_t block = 0; block < num_blocks; block++)
    {
        const size_t start_idx = block * BLOCK_SIZE;
        const size_t end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);

        std::vector<uint32_t> nodes_to_read;
        std::vector<T *> coord_buffers;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        nodes_to_read.reserve(end_idx - start_idx);
        coord_buffers.reserve(end_idx - start_idx);
        nbr_buffers.reserve(end_idx - start_idx);

        for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++)
        {
            const uint32_t slot = static_cast<uint32_t>(node_idx);
            nodes_to_read.push_back(node_list[node_idx]);
            coord_buffers.push_back(_merit_mem_pool->coord_ptr(slot));
            nbr_buffers.emplace_back(0, _merit_mem_pool->nhood_ptr(slot));
        }

        const auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);
        for (size_t i = 0; i < read_status.size(); i++)
        {
            if (!read_status[i])
            {
                diskann::cerr << "MERIT memory pool: failed to read node " << nodes_to_read[i] << std::endl;
                return -1;
            }
            const uint32_t slot = static_cast<uint32_t>(start_idx + i);
            _merit_mem_pool->nhood_ptr(slot)[0] = nbr_buffers[i].first;
        }
    }

    _merit_mem_pool->commit_initial_load(node_list, node_expand);
    diskann::cout << "MERIT memory pool loaded " << _merit_mem_pool->size() << " nodes (dynamic tier; not "
                  << "DiskANN static _nhood_cache)." << std::endl;
    return 0;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::enable_merit_memory_runtime_admit(bool enable)
{
    _merit_mem_runtime_admit = enable;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::reload_merit_memory_cache(const std::string &profile_prefix, double merit_memory_gb,
                                                       double host_memory_gb, double reserve_gb, uint32_t num_threads,
                                                       uint64_t &evicted_nodes, std::string &report)
{
    evicted_nodes = 0;
    report.clear();
    const uint64_t before = merit_memory_cached_count();

    uint64_t max_nodes = 0;
    if (plan_merit_memory_cache(merit_memory_gb, host_memory_gb, reserve_gb, num_threads, max_nodes, report) != 0)
        return -1;

    clear_merit_memory_cache();
    std::vector<uint32_t> node_list;
    if (build_merit_memory_node_list(profile_prefix, max_nodes, node_list, 0) != 0)
        return -1;
    if (!node_list.empty())
    {
        if (load_merit_memory_pool(profile_prefix, node_list) != 0)
            return -1;
    }

    const uint64_t after = merit_memory_cached_count();
    if (before > after)
        evicted_nodes = before - after;

    std::ostringstream oss;
    oss << "MERIT memory reload: cached " << after << " nodes";
    if (evicted_nodes > 0)
        oss << ", evicted " << evicted_nodes << " from DRAM cache";
    oss << ".";
    report = oss.str();
    diskann::cout << report << std::endl;
    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::plan_merit_disk_cache(double base_ratio, uint64_t &out_max_nodes,
                                                   std::string &report) const
{
    out_max_nodes = 0;
    report.clear();
    if (base_ratio <= 0.0)
    {
        report = "merit_disk_cache_ratio <= 0; MERIT disk cache disabled.";
        return 0;
    }
    if (base_ratio > 1.0)
    {
        report = "ERROR: merit_disk_cache_ratio must be in (0, 1].";
        return -1;
    }
    if (_disk_index_file.empty() || _max_node_len == 0)
    {
        report = "ERROR: disk index not loaded; cannot size MERIT disk cache.";
        return -1;
    }

    const uint64_t base_bytes =
        (_base_disk_index_bytes > 0) ? _base_disk_index_bytes : get_file_size(_disk_index_file);
    const uint64_t budget = static_cast<uint64_t>(base_ratio * static_cast<double>(base_bytes));
    uint64_t nodes = 0;
    if (_nnodes_per_sector > 0)
    {
        const uint64_t sectors = budget / defaults::SECTOR_LEN;
        nodes = sectors * _nnodes_per_sector;
    }
    else
    {
        const uint64_t secs_per_node = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
        const uint64_t bytes_per_node = secs_per_node * defaults::SECTOR_LEN;
        nodes = (bytes_per_node > 0) ? (budget / bytes_per_node) : 0;
    }
    if (nodes > _num_points)
        nodes = _num_points;
    if (nodes == 0)
    {
        report = "ERROR: MERIT disk-cache budget too small for one node.";
        return -1;
    }
    out_max_nodes = nodes;

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(4);
    oss << "MERIT disk-cache plan:\n"
        << "  base_index_file     = " << _disk_index_file << "\n"
        << "  base_size           = " << (base_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  ratio               = " << base_ratio << "\n"
        << "  budget              = " << (budget / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  max_node_len        = " << _max_node_len << " B, nnodes_per_sector=" << _nnodes_per_sector << "\n"
        << "  selected_nodes      = " << out_max_nodes
        << " (disk tier; k_hops=0 flat by node_expand, else k-hop sidecar; exclude memory-tier ids)";
    report = oss.str();
    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::reload_merit_disk_cache(const std::string &profile_prefix, double base_ratio,
                                                     const std::string &output_prefix, uint64_t rank_skip,
                                                     uint64_t &evicted_nodes, std::string &report,
                                                     bool unified_single_file, uint32_t k_hops,
                                                     const std::string &layout)
{
    evicted_nodes = 0;
    report.clear();
    const uint64_t before = _merit_dc_map.size();

    uint64_t max_nodes = 0;
    if (plan_merit_disk_cache(base_ratio, max_nodes, report) != 0)
        return -1;

    if (build_and_load_merit_disk_cache(profile_prefix, max_nodes, output_prefix, rank_skip, unified_single_file,
                                        k_hops, layout) != 0)
        return -1;

    const uint64_t after = _merit_dc_map.size();
    if (before > after)
        evicted_nodes = before - after;

    std::ostringstream oss;
    oss << "MERIT disk-cache reload: " << after << " nodes in map";
    if (rank_skip > 0)
        oss << " (rank_skip=" << rank_skip << " for memory tier)";
    if (evicted_nodes > 0)
        oss << ", evicted " << evicted_nodes << " from disk-cache directory";
    oss << ".";
    report = oss.str();
    diskann::cout << report << std::endl;
    return 0;
}

namespace
{
inline int copy_file_binary(const std::string &src, const std::string &dst)
{
    std::ifstream in(src, std::ios::binary);
    if (!in.is_open())
        return -1;
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out.is_open())
        return -1;
    out << in.rdbuf();
    return out.good() ? 0 : -1;
}
} // namespace

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_merit_disk_cache_from_prefix(const std::string &output_prefix)
{
    if (_merit_disk_reader)
    {
        _merit_disk_reader->close();
        _merit_disk_reader.reset();
    }
    _merit_unified_disk = false;
    _merit_region_byte_offset = 0;
    _merit_dc_map.clear();
    _merit_dc_num_nodes = 0;
    _merit_dc_path.clear();

    const std::string data_path = output_prefix + "_merit_dc.data";
    const std::string nodes_path = output_prefix + "_merit_dc.nodes";
    if (!file_exists(data_path) || !file_exists(nodes_path))
    {
        diskann::cerr << "MERIT disk-cache reuse: missing " << data_path << " or " << nodes_path << std::endl;
        return -1;
    }

    uint32_t *nodes = nullptr;
    size_t npts = 0, nd = 0;
    diskann::load_bin<uint32_t>(nodes_path, nodes, npts, nd);
    std::vector<uint32_t> node_list(nodes, nodes + npts);
    delete[] nodes;

    if (node_list.empty() || _nnodes_per_sector == 0)
    {
        diskann::cerr << "MERIT disk-cache reuse: empty node list or nnodes_per_sector=0." << std::endl;
        return -1;
    }

    const uint64_t nps = _nnodes_per_sector;
    uint64_t slot_in_sector = 0;
    uint32_t cur_sector = 0;
    _merit_dc_map.reserve(node_list.size());
    for (uint32_t id : node_list)
    {
        MeritDiskLoc loc;
        loc.sector = cur_sector;
        loc.slot = static_cast<uint16_t>(slot_in_sector);
        loc.nsectors = 1;
        _merit_dc_map[id] = loc;
        slot_in_sector++;
        if (slot_in_sector == nps)
        {
            slot_in_sector = 0;
            cur_sector++;
        }
    }

    _merit_dc_path = data_path;
    _merit_dc_num_nodes = node_list.size();

#ifndef _WINDOWS
    _merit_disk_reader.reset(new LinuxAlignedFileReader());
#else
    _merit_disk_reader.reset(new WindowsAlignedFileReader());
#endif
    _merit_disk_reader->open(data_path);
#pragma omp parallel for num_threads((int)_max_nthreads)
    for (int64_t thread = 0; thread < (int64_t)_max_nthreads; thread++)
    {
#pragma omp critical
        {
            _merit_disk_reader->register_thread();
        }
    }

    diskann::cout << "MERIT disk-cache loaded (reuse): " << _merit_dc_num_nodes << " nodes -> " << data_path
                  << " (map size=" << _merit_dc_map.size() << ")" << std::endl;
    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::build_and_load_merit_disk_cache(const std::string &profile_prefix, uint64_t max_nodes,
                                                             const std::string &output_prefix, uint64_t rank_skip,
                                                             bool unified_single_file, uint32_t k_hops,
                                                             const std::string &layout)
{
    if (_merit_disk_reader)
    {
        _merit_disk_reader->close();
        _merit_disk_reader.reset();
    }
    _merit_unified_disk = false;
    _merit_region_byte_offset = 0;
    _merit_dc_map.clear();
    _merit_dc_num_nodes = 0;
    _merit_dc_path.clear();

    if (max_nodes == 0)
        return 0;

    std::vector<uint32_t> node_list;
    if (build_merit_disk_node_list(profile_prefix, max_nodes, rank_skip, k_hops, layout, node_list) != 0)
        return -1;

    const std::string data_path =
        unified_single_file ? (output_prefix + "_disk_merit_unified.index") : (output_prefix + "_merit_dc.data");
    const std::string nodes_path = output_prefix + "_merit_dc.nodes";

    // Persist node order for debugging / reload.
    diskann::save_bin<uint32_t>(nodes_path, node_list.data(), node_list.size(), 1);

    if (unified_single_file)
    {
        if (copy_file_binary(_disk_index_file, data_path) != 0)
        {
            diskann::cerr << "Failed to copy base index into unified MERIT file: " << data_path << std::endl;
            return -1;
        }
        uint64_t sz = get_file_size(data_path);
        const uint64_t aligned_sz = ROUND_UP(sz, defaults::SECTOR_LEN);
        if (aligned_sz > sz)
        {
            std::ofstream pad(data_path, std::ios::binary | std::ios::app);
            std::vector<char> zeros(static_cast<size_t>(aligned_sz - sz), 0);
            pad.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
            pad.close();
        }
        _merit_region_byte_offset = aligned_sz;
    }

    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    const uint64_t secs_per_node =
        (_nnodes_per_sector > 0) ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    const uint64_t num_sectors =
        (_nnodes_per_sector > 0) ? DIV_ROUND_UP(node_list.size(), nps) : (node_list.size() * secs_per_node);

    char *sector_buf = nullptr;
    alloc_aligned((void **)&sector_buf, defaults::SECTOR_LEN, defaults::SECTOR_LEN);

    std::ofstream out(data_path, std::ios::binary | (unified_single_file ? std::ios::app : std::ios::trunc));
    if (!out.is_open())
    {
        aligned_free(sector_buf);
        diskann::cerr << "Failed to open MERIT disk-cache for write: " << data_path << std::endl;
        return -1;
    }

    // Scratch for reading base nodes one block at a time.
    const size_t BLOCK = 64;
    std::vector<T *> coord_ptrs;
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_ptrs;
    std::vector<uint32_t> nbr_storage(BLOCK * (_max_degree + 1));
    std::vector<T> coord_storage(BLOCK * _aligned_dim);

    auto flush_sector = [&](char *buf) {
        out.write(buf, defaults::SECTOR_LEN);
        memset(buf, 0, defaults::SECTOR_LEN);
    };

    memset(sector_buf, 0, defaults::SECTOR_LEN);
    uint64_t slot_in_sector = 0;
    uint32_t cur_sector = 0;

    for (size_t begin = 0; begin < node_list.size(); begin += BLOCK)
    {
        const size_t end = std::min(node_list.size(), begin + BLOCK);
        const size_t bn = end - begin;
        std::vector<uint32_t> batch(node_list.begin() + begin, node_list.begin() + end);
        coord_ptrs.resize(bn);
        nbr_ptrs.resize(bn);
        for (size_t i = 0; i < bn; i++)
        {
            coord_ptrs[i] = coord_storage.data() + i * _aligned_dim;
            nbr_ptrs[i] = {0, nbr_storage.data() + i * (_max_degree + 1)};
        }
        auto ok = read_nodes(batch, coord_ptrs, nbr_ptrs);
        for (size_t i = 0; i < bn; i++)
        {
            if (!ok[i])
            {
                aligned_free(sector_buf);
                out.close();
                diskann::cerr << "Failed to read base node " << batch[i] << " for MERIT disk-cache." << std::endl;
                return -1;
            }

            if (_nnodes_per_sector > 0)
            {
                char *dst = sector_buf + slot_in_sector * _max_node_len;
                memcpy(dst, coord_ptrs[i], _disk_bytes_per_point);
                uint32_t *nhood = reinterpret_cast<uint32_t *>(dst + _disk_bytes_per_point);
                nhood[0] = nbr_ptrs[i].first;
                memcpy(nhood + 1, nbr_ptrs[i].second, nbr_ptrs[i].first * sizeof(uint32_t));

                MeritDiskLoc loc;
                loc.sector = cur_sector;
                loc.slot = static_cast<uint16_t>(slot_in_sector);
                loc.nsectors = 1;
                _merit_dc_map[batch[i]] = loc;

                slot_in_sector++;
                if (slot_in_sector == nps)
                {
                    flush_sector(sector_buf);
                    slot_in_sector = 0;
                    cur_sector++;
                }
            }
            else
            {
                // One (or more) full sectors per node.
                std::vector<char> node_pack(secs_per_node * defaults::SECTOR_LEN, 0);
                memcpy(node_pack.data(), coord_ptrs[i], _disk_bytes_per_point);
                uint32_t *nhood = reinterpret_cast<uint32_t *>(node_pack.data() + _disk_bytes_per_point);
                nhood[0] = nbr_ptrs[i].first;
                memcpy(nhood + 1, nbr_ptrs[i].second, nbr_ptrs[i].first * sizeof(uint32_t));

                MeritDiskLoc loc;
                loc.sector = cur_sector;
                loc.slot = 0;
                loc.nsectors = static_cast<uint16_t>(secs_per_node);
                _merit_dc_map[batch[i]] = loc;

                out.write(node_pack.data(), node_pack.size());
                cur_sector += static_cast<uint32_t>(secs_per_node);
            }
        }
    }
    if (_nnodes_per_sector > 0 && slot_in_sector > 0)
        flush_sector(sector_buf);

    out.close();
    aligned_free(sector_buf);

    // Pad file to sector multiple if needed (already sector-aligned writes).
    (void)num_sectors;

    _merit_dc_path = data_path;
    _merit_dc_num_nodes = node_list.size();

    if (unified_single_file)
    {
        _merit_unified_disk = true;
        reader->close();
        reader->open(data_path);
        _disk_index_file = data_path;
        diskann::cout << "MERIT unified disk-cache ready: " << _merit_dc_num_nodes << " nodes appended at offset "
                      << (_merit_region_byte_offset / (1024.0 * 1024.0)) << " MiB -> " << data_path
                      << " (map size=" << _merit_dc_map.size() << ", single-fd merged IO)" << std::endl;
        return 0;
    }

#ifndef _WINDOWS
    _merit_disk_reader.reset(new LinuxAlignedFileReader());
#else
    _merit_disk_reader.reset(new WindowsAlignedFileReader());
#endif
    _merit_disk_reader->open(data_path);
#pragma omp parallel for num_threads((int)_max_nthreads)
    for (int64_t thread = 0; thread < (int64_t)_max_nthreads; thread++)
    {
#pragma omp critical
        {
            _merit_disk_reader->register_thread();
        }
    }

    diskann::cout << "MERIT disk-cache ready: " << _merit_dc_num_nodes << " nodes -> " << data_path
                  << " (map size=" << _merit_dc_map.size() << ")" << std::endl;
    return 0;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::prepare_merit_sidecar_io(
    const std::vector<uint32_t> &merit_ids, SSDQueryScratch<T> *query_scratch, char *sector_scratch,
    uint64_t &sector_scratch_idx, size_t num_sectors_per_node, std::vector<MeritReadPending> &pending,
    std::vector<AlignedRead> &merit_io, std::unordered_map<uint32_t, std::vector<size_t>> &disk_fanout_groups,
    QueryStats *stats, uint32_t &num_ios)
{
    pending.clear();
    merit_io.clear();
    disk_fanout_groups.clear();
    if (merit_ids.empty())
        return;

    const size_t read_len = num_sectors_per_node * defaults::SECTOR_LEN;
    pending.reserve(merit_ids.size());

    for (uint32_t id : merit_ids)
    {
        const auto it = _merit_dc_map.find(id);
        if (it == _merit_dc_map.end())
            continue;
        const MeritDiskLoc loc = it->second;
        query_scratch->read_merit_sidecar_sectors.insert(loc.sector);
        pending.push_back({id, nullptr, loc});
        const size_t pidx = pending.size() - 1;

        if (stats != nullptr)
        {
            stats->n_merit_dc_hits++;
            stats->n_4k++;
            stats->n_ios++;
        }
        num_ios++;

        bool served = false;
        if (_query_sector_cache_enabled && read_len <= defaults::SECTOR_LEN && loc.nsectors == 1)
        {
            const auto cit = query_scratch->merit_sector_cache.find(loc.sector);
            if (cit != query_scratch->merit_sector_cache.end())
            {
                pending[pidx].sec_buf = const_cast<char *>(cit->second.data());
                if (stats != nullptr)
                    stats->n_sector_cache_hits++;
                served = true;
            }
        }
        if (!served)
            disk_fanout_groups[loc.sector].push_back(pidx);
    }

    for (auto &kv : disk_fanout_groups)
    {
        const uint32_t sidecar_sec = kv.first;
        std::vector<size_t> &indices = kv.second;
        if (indices.empty())
            continue;
        char *read_buf = sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
        sector_scratch_idx++;
        pending[indices[0]].sec_buf = read_buf;
        const MeritReadPending &first = pending[indices[0]];
        const uint64_t byte_len = static_cast<uint64_t>(first.loc.nsectors) * defaults::SECTOR_LEN;
        const uint64_t off =
            (_merit_unified_disk ? _merit_region_byte_offset : 0) +
            static_cast<uint64_t>(sidecar_sec) * defaults::SECTOR_LEN;
        merit_io.emplace_back(off, byte_len, read_buf);
        if (stats != nullptr)
            stats->n_disk_reads++;
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::complete_merit_sidecar_io(
    SSDQueryScratch<T> *query_scratch, std::vector<MeritReadPending> &pending,
    const std::unordered_map<uint32_t, std::vector<size_t>> &disk_fanout_groups)
{
    for (const auto &kv : disk_fanout_groups)
    {
        const std::vector<size_t> &indices = kv.second;
        if (indices.empty())
            continue;
        const char *primary = pending[indices[0]].sec_buf;
        char *shared_buf = const_cast<char *>(primary);
        if (_query_sector_cache_enabled)
        {
            std::array<char, defaults::SECTOR_LEN> &slot = query_scratch->merit_sector_cache[kv.first];
            memcpy(slot.data(), primary, defaults::SECTOR_LEN);
            shared_buf = slot.data();
        }
        for (size_t idx : indices)
            pending[idx].sec_buf = shared_buf;
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::finalize_merit_pending_nodes(const std::vector<MeritReadPending> &pending,
                                                           SSDQueryScratch<T> *query_scratch, char *sector_scratch,
                                                           uint64_t &sector_scratch_idx,
                                                           std::vector<std::pair<uint32_t, char *>> &frontier_nhoods)
{
    (void)query_scratch;
    for (const auto &mp : pending)
    {
        char *sec_buf = mp.sec_buf;
        char *out_buf = sec_buf;
        if (_nnodes_per_sector > 0)
        {
            const uint64_t base_slot = mp.id % _nnodes_per_sector;
            if (base_slot != mp.loc.slot)
            {
                out_buf = sector_scratch + sector_scratch_idx * defaults::SECTOR_LEN;
                sector_scratch_idx++;
                memcpy(out_buf, sec_buf, defaults::SECTOR_LEN);
                char *packed = sec_buf + static_cast<uint64_t>(mp.loc.slot) * _max_node_len;
                std::vector<char> tmp(_max_node_len);
                memcpy(tmp.data(), packed, _max_node_len);
                memset(out_buf, 0, defaults::SECTOR_LEN);
                memcpy(out_buf + base_slot * _max_node_len, tmp.data(), _max_node_len);
            }
        }
        frontier_nhoods.emplace_back(mp.id, out_buf);
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::enable_access_profile(bool enable)
{
    _hotness_profiler.set_enabled(enable);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::enable_query_sector_cache(bool enable)
{
    _query_sector_cache_enabled = enable;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::enable_base_frontier_recording(bool enable)
{
    _record_base_frontier = enable;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::enable_hop_frontier_recording(bool enable)
{
    _record_hop_frontier = enable;
}

template <typename T, typename LabelT> int PQFlashIndex<T, LabelT>::save_access_profile(const std::string &output_prefix) const
{
    return _hotness_profiler.save(output_prefix);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::print_access_profile_cdf() const
{
    _hotness_profiler.print_cdf_summary();
}

// instantiations
template class PQFlashIndex<uint8_t>;
template class PQFlashIndex<int8_t>;
template class PQFlashIndex<float>;
template class PQFlashIndex<uint8_t, uint16_t>;
template class PQFlashIndex<int8_t, uint16_t>;
template class PQFlashIndex<float, uint16_t>;

} // namespace diskann

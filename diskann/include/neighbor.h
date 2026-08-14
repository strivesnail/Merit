// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <algorithm>
#include <functional>
#include <mutex>
#include <vector>
#include "utils.h"

namespace diskann
{

struct Neighbor
{
    unsigned id;
    float distance;
    bool expanded;

    Neighbor() = default;

    Neighbor(unsigned id, float distance) : id{id}, distance{distance}, expanded(false)
    {
    }

    inline bool operator<(const Neighbor &other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const Neighbor &other) const
    {
        return (id == other.id);
    }
};

// Invariant: after every `insert` and `closest_unexpanded()`, `_cur` points to
//            the first Neighbor which is unexpanded.
class NeighborPriorityQueue
{
  public:
    NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0)
    {
    }

    explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1)
    {
    }

    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    void insert(const Neighbor &nbr)
    {
        if (_size == _capacity && _data[_size - 1] < nbr)
        {
            return;
        }

        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid])
            {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            }
            else if (_data[mid].id == nbr.id)
            {
                return;
            }
            else
            {
                lo = mid + 1;
            }
        }

        if (lo < _capacity)
        {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
        }
        _data[lo] = {nbr.id, nbr.distance};
        if (_size < _capacity)
        {
            _size++;
        }
        if (lo < _cur)
        {
            _cur = lo;
        }
    }

    Neighbor closest_unexpanded()
    {
        _data[_cur].expanded = true;
        size_t pre = _cur;
        while (_cur < _size && _data[_cur].expanded)
        {
            _cur++;
        }
        return _data[pre];
    }

    // Like closest_unexpanded(), but among unexpanded nodes within dist_slack of the
    // current minimum distance, prefer nodes whose io_seed(id) is in preferred_seeds.
    template <typename SeedSet>
    Neighbor closest_unexpanded_seed_biased(const std::function<uint32_t(uint32_t)> &io_seed,
                                              const SeedSet &preferred_seeds, float dist_slack)
    {
        size_t i = _cur;
        while (i < _size && _data[i].expanded)
            ++i;
        if (i >= _size)
            return closest_unexpanded();

        const float min_dist = _data[i].distance;
        const float dist_limit = min_dist + dist_slack;

        size_t pick = i;
        bool have_preferred_pick = false;
        for (size_t j = i; j < _size && !_data[j].expanded && _data[j].distance <= dist_limit; ++j)
        {
            if (preferred_seeds.empty())
            {
                pick = j;
                break;
            }
            if (preferred_seeds.count(io_seed(_data[j].id)) == 0)
                continue;
            if (!have_preferred_pick || _data[j] < _data[pick])
            {
                pick = j;
                have_preferred_pick = true;
            }
        }

        _data[pick].expanded = true;
        Neighbor result = _data[pick];
        if (pick == _cur)
        {
            while (_cur < _size && _data[_cur].expanded)
                ++_cur;
        }
        return result;
    }

    // Pick up to beam_width unexpanded nodes from the closest pool_size candidates,
    // greedily preferring nodes that share an io_seed with an already picked node.
    template <typename SeedSet>
    std::vector<Neighbor> select_beam_seed_biased(size_t beam_width, size_t pool_size,
                                                     const std::function<uint32_t(uint32_t)> &io_seed)
    {
        std::vector<size_t> pool;
        pool.reserve(pool_size);
        for (size_t i = _cur; i < _size && pool.size() < pool_size; ++i)
        {
            if (!_data[i].expanded)
                pool.push_back(i);
        }
        if (pool.empty())
            return {};

        std::vector<size_t> picked;
        picked.reserve(beam_width);
        SeedSet preferred_seeds;

        while (picked.size() < beam_width && !pool.empty())
        {
            size_t best = pool[0];
            bool have_preferred = false;
            if (!preferred_seeds.empty())
            {
                for (size_t idx : pool)
                {
                    if (preferred_seeds.count(io_seed(_data[idx].id)) == 0)
                        continue;
                    if (!have_preferred || _data[idx] < _data[best])
                    {
                        best = idx;
                        have_preferred = true;
                    }
                }
            }
            if (!have_preferred)
            {
                for (size_t idx : pool)
                {
                    if (_data[idx] < _data[best])
                        best = idx;
                }
            }
            preferred_seeds.insert(io_seed(_data[best].id));
            picked.push_back(best);
            pool.erase(std::remove(pool.begin(), pool.end(), best), pool.end());
        }

        std::vector<Neighbor> result;
        result.reserve(picked.size());
        for (size_t idx : picked)
        {
            _data[idx].expanded = true;
            result.push_back(_data[idx]);
        }
        while (_cur < _size && _data[_cur].expanded)
            ++_cur;
        return result;
    }

    void mark_expanded(uint32_t id)
    {
        for (size_t i = _cur; i < _size; ++i)
        {
            if (_data[i].id == id && !_data[i].expanded)
            {
                _data[i].expanded = true;
                if (i == _cur)
                {
                    while (_cur < _size && _data[_cur].expanded)
                        ++_cur;
                }
                return;
            }
        }
    }

    // Unexpanded retset nodes whose io_seed(id) is in seeds, excluding ids in exclude_ids.
    template <typename SeedSet, typename IdSet>
    std::vector<Neighbor> collect_unexpanded_with_io_seeds(const std::function<uint32_t(uint32_t)> &io_seed,
                                                            const SeedSet &seeds, const IdSet &exclude_ids) const
    {
        std::vector<Neighbor> out;
        for (size_t i = _cur; i < _size; ++i)
        {
            if (_data[i].expanded || exclude_ids.count(_data[i].id) != 0)
                continue;
            if (seeds.count(io_seed(_data[i].id)) == 0)
                continue;
            out.push_back(_data[i]);
        }
        return out;
    }

    bool has_unexpanded_node() const
    {
        return _cur < _size;
    }

    size_t size() const
    {
        return _size;
    }

    size_t capacity() const
    {
        return _capacity;
    }

    void reserve(size_t capacity)
    {
        if (capacity + 1 > _data.size())
        {
            _data.resize(capacity + 1);
        }
        _capacity = capacity;
    }

    Neighbor &operator[](size_t i)
    {
        return _data[i];
    }

    Neighbor operator[](size_t i) const
    {
        return _data[i];
    }

    void clear()
    {
        _size = 0;
        _cur = 0;
    }

  private:
    size_t _size, _capacity, _cur;
    std::vector<Neighbor> _data;
};

} // namespace diskann

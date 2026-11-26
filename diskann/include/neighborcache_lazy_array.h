#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace diskann
{

// Reset only while workers are stopped. Blocks are initialized on first use,
// published atomically and retain stable addresses until reset.
template <typename T> struct NeighborCacheArrayValue
{
    using Type = T;

    static void initialize(T *block, size_t length, const T &value)
    {
        std::fill_n(block, length, value);
    }
};

template <typename T> struct NeighborCacheArrayValue<std::atomic<T>>
{
    using Type = T;

    static void initialize(std::atomic<T> *block, size_t length, T value)
    {
        for (size_t i = 0; i < length; ++i)
            block[i].store(value, std::memory_order_relaxed);
    }
};

template <typename T, bool FixedBlocks = false> class NeighborCacheLazyArray
{
  public:
    using Value = typename NeighborCacheArrayValue<T>::Type;
    static constexpr size_t kBlockSize = 4096;

    NeighborCacheLazyArray() = default;
    NeighborCacheLazyArray(const NeighborCacheLazyArray &) = delete;
    NeighborCacheLazyArray &operator=(const NeighborCacheLazyArray &) = delete;

    ~NeighborCacheLazyArray() { clear(); }

    void clear()
    {
        for (size_t i = 0; i < _block_count; ++i)
            delete[] _blocks[i].load(std::memory_order_relaxed);
        _blocks.reset();
        _size = 0;
        _block_count = 0;
    }

    // The addressable size stays fixed. Only the block-pointer table is
    // allocated here. Group elements that must be contiguous into one block.
    void assign(size_t size, const Value &value = Value{}, size_t block_size = kBlockSize)
    {
        clear();
        assert(block_size != 0);
        if (FixedBlocks && block_size != kBlockSize)
            throw std::invalid_argument("Fixed NeighborCache blocks require the default block size");
        const size_t blocks = size / block_size + (size % block_size != 0);
        if (blocks != 0)
        {
            auto table = std::make_unique<std::atomic<T *>[]>(blocks);
            for (size_t i = 0; i < blocks; ++i)
                table[i].store(nullptr, std::memory_order_relaxed);
            _blocks = std::move(table);
        }
        _value = value;
        _size = size;
        _block_count = blocks;
        _block_size = block_size;
#if defined(__SIZEOF_INT128__)
        if (!FixedBlocks && block_size != kBlockSize)
            _block_reciprocal = UINT64_MAX / block_size;
#endif
    }

    size_t size() const { return _size; }
    explicit operator bool() const { return _size != 0; }

    T *find(size_t index) const
    {
        if (index >= _size)
            return nullptr;
        const auto [block_id, offset] = locate(index);
        T *block = _blocks[block_id].load(std::memory_order_acquire);
        return block == nullptr ? nullptr : block + offset;
    }

    T *ptr(size_t index) const
    {
        assert(index < _size);
        const auto [block_id, offset] = locate(index);
        auto &entry = _blocks[block_id];
        T *block = entry.load(std::memory_order_acquire);
        if (block == nullptr)
        {
            const size_t length = block_length(block_id);
            auto pending = std::unique_ptr<T[]>(new T[length]);
            NeighborCacheArrayValue<T>::initialize(pending.get(), length, _value);
            if (entry.compare_exchange_strong(block, pending.get(), std::memory_order_acq_rel,
                                              std::memory_order_acquire))
                block = pending.release();
        }
        return block + offset;
    }

    T &operator[](size_t index) const { return *ptr(index); }

    T &at(size_t index) const
    {
        if (index >= _size)
            throw std::out_of_range("NeighborCache metadata index");
        return (*this)[index];
    }

    size_t capacity() const
    {
        size_t allocated = 0;
        for (size_t i = 0; i < _block_count; ++i)
            if (_blocks[i].load(std::memory_order_acquire) != nullptr)
                allocated += block_length(i);
        return allocated;
    }

    template <typename F> void for_each_allocated(F &&fn) const
    {
        for (size_t i = 0; i < _block_count; ++i)
        {
            const T *block = _blocks[i].load(std::memory_order_acquire);
            if (block != nullptr)
                for (size_t j = 0; j < block_length(i); ++j)
                    fn(block[j]);
        }
    }

  private:
    struct BlockLocation
    {
        size_t block;
        size_t offset;
    };

    BlockLocation locate(size_t index) const
    {
        // Use compile-time division for the ordinary metadata blocks. The
        // D-cache member array can still choose a block that fits whole pages.
        if (FixedBlocks || _block_size == kBlockSize)
            return {index / kBlockSize, index % kBlockSize};
#if defined(__SIZEOF_INT128__)
        // The rounded-down multiplier can underestimate the quotient by one.
        const size_t quotient = static_cast<size_t>((static_cast<__uint128_t>(index) * _block_reciprocal) >> 64);
        const size_t remainder = index - quotient * _block_size;
        if (remainder >= _block_size)
            return {quotient + 1, remainder - _block_size};
        return {quotient, remainder};
#else
        return {index / _block_size, index % _block_size};
#endif
    }

    size_t block_length(size_t block) const
    {
        return std::min(_block_size, _size - block * _block_size);
    }

    std::unique_ptr<std::atomic<T *>[]> _blocks;
    Value _value{};
    size_t _size = 0;
    size_t _block_count = 0;
    size_t _block_size = kBlockSize;
#if defined(__SIZEOF_INT128__)
    uint64_t _block_reciprocal = 0;
#endif
};

} // namespace diskann

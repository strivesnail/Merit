#pragma once

#include <cstdint>
#include <limits>

namespace diskann
{

// Query and admission counters may wrap. Test the original counter value
// directly so the sampling schedule is independent of worker interleaving.
class NeighborCachePeriod
{
  public:
    explicit NeighborCachePeriod(uint64_t period) { reset(period); }

    void reset(uint64_t period)
    {
        _period = period;
        if (!period)
            return;
        const uint64_t power_of_two = period & (0 - period);
        _mask = power_of_two - 1;
        const uint64_t odd = period / power_of_two;
        _limit = std::numeric_limits<uint64_t>::max() / odd;
        // Invert the odd factor modulo 2^64. Each step doubles the number
        // of correct bits, starting with one correct bit.
        _inverse = 1;
        for (unsigned i = 0; i < 6; ++i)
            _inverse *= 2 - odd * _inverse;
    }

    uint64_t value() const { return _period; }

    bool matches(uint64_t counter) const
    {
        // Multiplication by the inverse maps exactly the unsigned multiples
        // of odd to [0, UINT64_MAX / odd]. The mask checks the even factor.
        return _period && (counter & _mask) == 0 && counter * _inverse <= _limit;
    }

  private:
    uint64_t _period, _mask = 0, _inverse = 1, _limit = 0;
};

} // namespace diskann

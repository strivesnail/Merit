#pragma once
namespace diskann
{
// Called only after a full record is acquired, before M-cache decisions/locks.
// Native search and every host adapter account for every acquired node. Keep
// the shared gate so callers cannot accidentally retain an older sampling rate.
inline bool neighborcache_sample_node_heat() noexcept
{
    return true;
}
}

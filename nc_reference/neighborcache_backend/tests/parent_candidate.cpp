#include "neighbor.h"
#include <cassert>

int main() {
    diskann::NeighborPriorityQueue q(3);
    for (uint32_t id=10; id>0; --id) {
        diskann::Neighbor n(id, float(id));
        n.nc_parent = id+100;
        q.insert(n);
    }
    for (uint32_t id=1; id<=3; ++id) {
        auto n=q.closest_unexpanded();
        assert(n.id==id && n.nc_parent==id+100);
    }
    assert(!q.has_unexpanded_node());
}

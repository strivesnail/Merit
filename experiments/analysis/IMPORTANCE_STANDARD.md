# Seed / Neighbor Importance Standard

## Goal

Reduce MERIT disk-cache space **on top of Adaptive** (~17 mean disk reads on SIFT1M
8 workloads), by ranking what to keep in two independent axes:

1. **Seed axis** — which parent nodes deserve a seed-replica group.
2. **Neighbor axis** — within each kept seed, which directed children occupy pages.

**Evaluation baseline:** Adaptive full-cover (`directed_seed_replica_pct100` at
profile-sized ratio + `MERIT_RECORD_DRIVEN_SEED_ACCESS=1` +
`MERIT_ADAPTIVE_PARENT_OR_SELF=1`). Not Full Extent (~18.3) and not raw DiskANN (~62).

## Metrics (offline, from same-trace profile)

### Seed importance

For each profile parent `P`:

| Symbol | Definition |
|--------|------------|
| `H(P)` | **Out-heat**: Σ directed-edge counts `(P → C)` from Run2 profile |
| `Pages(P)` | Hot pages at neighbor pct `τ` (page-aligned seed-replica packing) |
| `B(P)` | **Benefit density** `H(P) / Pages(P)` — traversal mass per disk page |
| `E(P)` | **Expand heat** `node_expand[P]` — how often `P` itself is expanded |

Interpretation under slot budget:

- Rank by `H(P)` → keep seeds that anchor the most traversal mass (default).
- Rank by `B(P)` → prefer compact high-yield seeds (few pages, high heat).
- High `E(P)` with low `H(P)` → seed is visited often but mostly as a hop, not as a parent hub.

### Neighbor importance (within seed `P`)

For each directed child `C` of `P`:

| Symbol | Definition |
|--------|------------|
| `W(P,C)` | Profile count of traversals `P → C` |
| `τ` | Neighbor pct threshold (80 / 90 / 100) — keep top children until Σ `W` ≥ τ·Σ `W` |

Interpretation: children with highest conditional frequency should fill `P`'s pages first.
Dropping tail neighbors (lower τ) shrinks per-seed footprint but may force extra base-index reads.

## Layout policies (runtime)

| Layout | Seed ranking | Neighbor pct | Budget behavior |
|--------|--------------|--------------|-----------------|
| `directed_seed_replica_pct100` | out-heat | 100% | flat append to full cover |
| `directed_seed_replica_pct90/80/50` | out-heat | 90/80/50% | flat append |
| `directed_seed_replica_benefit_pct100/80` | benefit `B(P)` | 100/80% | strict, no flat append |
| `directed_seed_replica_budget_pct100` | out-heat | 100% | strict ratio cap, no flat append |

## Two-axis reduction matrix

|  | Neighbor: keep 100% | Neighbor: pct90 | Neighbor: pct80 |
|--|---------------------|-----------------|-----------------|
| **Seed: all (full-cover ratio)** | Adaptive baseline | `pct90` layout | `pct80` layout |
| **Seed: budget @ ratio r** | `budget_pct100 @ r` | rebuild pct90 pages, cap seeds | `benefit_pct80 @ r` |

Report Pareto: `(dc_bytes / base_index, mean_disk_reads, recall@1)` vs Adaptive baseline.

## What makes a node "more important" (hypotheses to test)

1. **High out-heat parent** — many queries traverse through `P` as parent; dropping it hurts most.
2. **High benefit density** — `P` packs many traversals into few pages; good under tight budget.
3. **High edge-weight child** — `(P,C)` is the dominant exit from `P`; tail children are safe to drop.
4. **Expand-only hot node** — high `node_expand` but low out-heat; may be less critical as a *seed*
   (Adaptive may still reach it via parent extent or base index).

Future: **trace marginal disk-read benefit** — counterfactual Δreads if `(P)` or `(P,C)` were removed
from cache (not yet implemented in layout code).

## Evaluation

Always run with Adaptive env vars. Compare against stored Adaptive baseline per workload.
Scripts: `experiments/test-scripts/run_adaptive_reduction_sweep.sh`,
offline: `experiments/analysis/seed_neighbor_importance.py`.

#!/usr/bin/env bash
# Drop Linux page cache (and related caches) before cold IO benchmarks. Requires root.
#
# Usage:
#   sudo bash scripts/drop_system_caches.sh
#
# echo 3 clears page cache, dentries, and inodes. Disk reads may be slower until caches warm up.
# Use with care on shared machines.

if [[ "$(id -u)" -ne 0 ]]; then
	echo "Run as root: sudo bash $0" >&2
	exit 1
fi

sync
echo 3 >/proc/sys/vm/drop_caches
echo "Done: sync and vm.drop_caches=3"

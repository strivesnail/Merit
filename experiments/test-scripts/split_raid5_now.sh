#!/bin/bash
# Tear down md0 RAID5. Members only: nvme0n1 nvme1n1 nvme2n1 nvme3n1.
# Does not touch nvme4n1 (/mnt/nvme_single) or the HDD.
set -euo pipefail

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "需要 root：sudo $0"
  exit 1
fi

for d in /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1; do
  [[ -b "$d" ]] || { echo "missing $d"; exit 1; }
done
if findmnt -n /mnt/nvme_single | grep -q nvme4n1; then
  :
else
  echo "refuse: /mnt/nvme_single is not nvme4n1"
  findmnt /mnt/nvme_single || true
  exit 1
fi

echo "before:"
cat /proc/mdstat
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT /dev/md0 /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1

if findmnt /mnt/raid5 >/dev/null 2>&1; then
  umount /mnt/raid5
fi
mdadm --stop /dev/md0
mdadm --zero-superblock /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1

if [[ -f /etc/mdadm/mdadm.conf ]]; then
  sed -i '/^ARRAY \/dev\/md0/d; /^ARRAY \/dev\/md\/0/d' /etc/mdadm/mdadm.conf
fi
update-initramfs -u || true

echo "after:"
cat /proc/mdstat
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1
echo "RAID5 stopped. nvme0-3 are standalone disks. nvme4 untouched."

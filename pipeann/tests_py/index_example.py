import numpy as np
import time
from pipeann import IndexPipeANN, Metric
from utils import (
    SIFT_1M_GT_PATH,
    SIFT_1M_PATH,
    SIFT_2M_GT_PATH,
    SIFT_2M_PATH,
    SIFT_DATA_TYPE,
    SIFT_INDEX_PREFIX,
    SIFT_QUERY_PATH,
    bin_read,
)

def main():
    queries = bin_read(SIFT_QUERY_PATH, SIFT_DATA_TYPE)
    gt = bin_read(SIFT_1M_GT_PATH, "int32")
    gt_2M = bin_read(SIFT_2M_GT_PATH, "int32")
    full_data_2M = bin_read(SIFT_2M_PATH, SIFT_DATA_TYPE)
    print(full_data_2M.shape)

    data_dim = full_data_2M.shape[1]
    idx = IndexPipeANN(data_dim, SIFT_DATA_TYPE, Metric.L2)
    idx.omp_set_num_threads(32) # the number of search/insert threads.
    idx.set_index_prefix(SIFT_INDEX_PREFIX)
    """
print(f"Building index with prefix {index_prefix}...")
Way 1 to initialize index:
    idx.build(data_path, index_prefix) # build SSD index.
    idx.load(index_prefix) # manually load the index after building it.
Way 2: use index.add. Here we use the first half of the dataset to build the index.
    full_data = bin_read(SIFT_1M_PATH, SIFT_DATA_TYPE)
    for i in range(0, full_data.shape[0], 10000):
        print(f"Inserting data points {i} to {min(i+10000, full_data.shape[0])} ...")
        idx.add(full_data[i:min(i+10000, full_data.shape[0])], np.arange(i, min(i+10000, full_data.shape[0])))
    """

    print(f"Building index with prefix {SIFT_INDEX_PREFIX}...")
    for i in range(0, full_data_2M.shape[0] // 2, 10000):
        print(f"Inserting the first 1M points {i} to {min(i+10000, full_data_2M.shape[0] // 2)} ...")
        idx.add(full_data_2M[i:min(i+10000, full_data_2M.shape[0] // 2)], np.arange(i, min(i+10000, full_data_2M.shape[0] // 2)))
    # The index after adding vectors is inconsistent on disk, so we need to save it first.
    # Directly searching in it is fine.
    idx.save(SIFT_INDEX_PREFIX)

    print(f"Loading index with prefix {SIFT_INDEX_PREFIX}...")
    idx.load(SIFT_INDEX_PREFIX)
    topk = 10

    for L in [10, 20, 30, 40, 50]:
        print(f"Searching for {topk} nearest neighbors with L={L}...")
        t1 = time.clock_gettime(time.CLOCK_REALTIME)
        ids, dists = idx.search(queries, topk, L)
        t2 = time.clock_gettime(time.CLOCK_REALTIME)
        print(f"Search time: {t2 - t1:.4f} seconds for {len(queries)} queries, throughput: {len(queries) / (t2 - t1)} QPS.")
        recall = np.mean([
            len(set(ids[i]) & set(gt[i][:topk])) / topk
            for i in range(len(queries))
        ])
        print(f"Recall@{topk} with L={L}: {recall:.4f}")
    
    # insert vectors.
    print(f"Inserting 1M new vectors to the index ...")
    for i in range(1000000, 2000000, 10000):
        print(f"Inserting data points {i} to {min(i+10000, full_data_2M.shape[0])} ...")
        idx.add(full_data_2M[i:min(i+10000, full_data_2M.shape[0])], np.arange(i, min(i+10000, full_data_2M.shape[0])))
    print(f"Deleting the first 1M vectors from the index ...")
    idx.remove(np.arange(0, 1000000))

    # save and load.
    idx.save(SIFT_INDEX_PREFIX)
    idx.load(SIFT_INDEX_PREFIX)

    for L in [10, 20, 30, 40, 50]:
        print(f"Searching for {topk} nearest neighbors with L={L}...")
        t1 = time.clock_gettime(time.CLOCK_REALTIME)
        ids, dists = idx.search(queries, topk, L)
        t2 = time.clock_gettime(time.CLOCK_REALTIME)
        print(f"Search time: {t2 - t1:.4f} seconds for {len(queries)} queries, throughput: {len(queries) / (t2 - t1)} QPS.")
        recall = np.mean([
            len(set(ids[i]) & set(gt_2M[i][:topk])) / topk
            for i in range(len(queries))
        ])
        print(f"Recall@{topk} with L={L}: {recall:.4f}")

if __name__ == "__main__":
    main()

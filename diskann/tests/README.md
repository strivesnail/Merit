# Unit Test project

## M-cache node heat

Native search and the shared backend update M-cache heat on each acquired node
while online maintenance is active. The current decay unit is added under the
shard lock for both new and existing records.

## Cache vector correctness regression

`prefetch_vector_padding` exercises the real private prefetch, shared prefetch,
legacy seed LRU, and D-page extraction paths. It verifies vectors are padded to
the distance kernel's aligned dimension and that the padding is zero, including
after output-buffer reuse and an asynchronous dirty-page eviction/readback.
Coverage includes `int8_t`, `uint8_t`, and `float`, aligned dimensions and raw
dimensions 4 and 100. Do not replace these with only 96D/128D tests: those hid
the SPACEV100D bug. The regression fails against the old implementation.

Configure with `-DUNIT_TEST=ON`, build `prefetch_vector_padding`, and run:

```sh
ctest --test-dir <build> -R 'prefetch_vector_padding|prefetch_async_writeback|ncache_snapshot' --output-on-failure
```

Cache-only native search experiments must also check every returned ID and
distance bit against the same index/L/W reference. Matching mean Recall alone
does not validate search equivalence. The reusable check is
`experiments/neighborcache_alignment/cache_validation.py` (from the repo root).


This unit test project is based on the [boost unit test framework](https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/index.html). Below are the simple steps to add new unit test, you could find more usage from the [boost unit test document](https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/index.html).

## How to add unit test

- Create new [BOOST_AUTO_TEST_SUITE](https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/boost_test/utf_reference/test_org_reference/test_org_boost_auto_test_suite.html) for each class in an individual cpp file

- Add [BOOST_AUTO_TEST_CASE](https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/boost_test/utf_reference/test_org_reference/test_org_boost_auto_test_case.html) for each test case in the [BOOST_AUTO_TEST_SUITE](https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/boost_test/utf_reference/test_org_reference/test_org_boost_auto_test_suite.html)

- Update the [CMakeLists.txt](CMakeLists.txt) file to add the new cpp file to the test project

## Deferred D-directory publication

`NEIGHBORCACHE_DEFER_D_LOCATIONS=1` enables the guarded private, one-page
dynamic-cache implementation. Seed identity and each worker's Prefetch member
index remain immediately available. Neighbor D locations are published only
after a successful write. `NEIGHBORCACHE_D_DIRECTORY_MAINTENANCE_THREAD=1`
(default) transfers that publication to the existing maintenance worker through
a bounded queue; `=0` keeps it on the background writer for A/B testing.
Query eviction remains nonblocking. A queued publication retains its existing
page pin until version/membership validation and publication/rollback finish.

`prefetch_deferred_directory` exercises private workers and multiple writers,
real O_DIRECT readback, late publication racing retirement, stale versions and
membership, failed writes, nonblocking submission drops, and stale private
replicas after another worker updates the backing page. The test retries dropped
writes through the blocking producer API only to verify eventual readback; this
is not a retry or wait added to the query path.

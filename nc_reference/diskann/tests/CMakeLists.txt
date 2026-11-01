# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if (NOT MSVC)
    add_executable(heat_sample heat_sample.cpp)
    target_compile_options(heat_sample PRIVATE -UNDEBUG)
    target_link_libraries(heat_sample ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})
    add_test(NAME heat_sample COMMAND heat_sample)

    add_executable(prefetch_zero_tail prefetch_zero_tail.cpp)
    target_compile_options(prefetch_zero_tail PRIVATE -UNDEBUG)
    target_link_libraries(prefetch_zero_tail ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})
    add_test(NAME prefetch_zero_tail COMMAND prefetch_zero_tail)
    set_tests_properties(prefetch_zero_tail PROPERTIES TIMEOUT 60)

    add_executable(canonical_directory canonical_directory.cpp)
    target_compile_options(canonical_directory PRIVATE -UNDEBUG)
    target_link_libraries(canonical_directory ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})
    add_test(NAME canonical_directory COMMAND canonical_directory)
    set_tests_properties(canonical_directory PROPERTIES TIMEOUT 60)

    add_executable(prefetch_record_directory prefetch_record_directory.cpp)
    target_compile_options(prefetch_record_directory PRIVATE -UNDEBUG)
    target_link_libraries(prefetch_record_directory ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})
    add_test(NAME prefetch_record_directory COMMAND prefetch_record_directory)
    set_tests_properties(prefetch_record_directory PROPERTIES TIMEOUT 60)

    add_executable(mcache_lookup_bitmap mcache_lookup_bitmap.cpp)
    target_compile_options(mcache_lookup_bitmap PRIVATE -UNDEBUG)
    target_link_libraries(mcache_lookup_bitmap ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})
    add_test(NAME mcache_lookup_bitmap COMMAND mcache_lookup_bitmap)
    set_tests_properties(mcache_lookup_bitmap PROPERTIES TIMEOUT 60)

    add_executable(ncache_record_copy ncache_record_copy.cpp)
    target_compile_options(ncache_record_copy PRIVATE -UNDEBUG)
    target_link_libraries(ncache_record_copy ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})
    add_test(NAME ncache_record_copy COMMAND ncache_record_copy)
    set_tests_properties(ncache_record_copy PROPERTIES TIMEOUT 60)

    add_executable(prefetch_private_records prefetch_private_records.cpp)
    target_compile_options(prefetch_private_records PRIVATE -UNDEBUG)
    target_link_libraries(prefetch_private_records ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})
    add_test(NAME prefetch_private_records COMMAND prefetch_private_records)
    set_tests_properties(prefetch_private_records PROPERTIES TIMEOUT 60)

    add_executable(ncache_snapshot ncache_snapshot.cpp)
    target_compile_options(ncache_snapshot PRIVATE -UNDEBUG)
    target_link_libraries(ncache_snapshot ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})
    add_test(NAME ncache_snapshot COMMAND ncache_snapshot)
    set_tests_properties(ncache_snapshot PROPERTIES TIMEOUT 60)

    add_executable(ncache_admission_state ncache_admission_state.cpp)
    target_compile_options(ncache_admission_state PRIVATE -UNDEBUG)
    target_link_libraries(ncache_admission_state ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})
    add_test(NAME ncache_admission_state COMMAND ncache_admission_state)
    set_tests_properties(ncache_admission_state PROPERTIES TIMEOUT 60)

    add_executable(prefetch_async_writeback prefetch_async_writeback.cpp)
    target_compile_options(prefetch_async_writeback PRIVATE -UNDEBUG)
    target_link_libraries(prefetch_async_writeback ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})
    add_test(NAME prefetch_async_writeback COMMAND prefetch_async_writeback)
    set_tests_properties(prefetch_async_writeback PROPERTIES TIMEOUT 60)
    add_test(NAME prefetch_async_writeback_private COMMAND prefetch_async_writeback)
    set_tests_properties(prefetch_async_writeback_private PROPERTIES TIMEOUT 60
        ENVIRONMENT "NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE=1;NEIGHBORCACHE_FREEZE_MAINTENANCE=0")
endif()

find_package(Boost COMPONENTS unit_test_framework)

# For Windows, fall back to nuget version if find_package didn't find it.
if (MSVC AND NOT Boost_FOUND)
    set(DISKANN_BOOST_INCLUDE "${DISKANN_MSVC_PACKAGES}/boost/lib/native/include")
    # Multi-threaded static library.
    set(UNIT_TEST_FRAMEWORK_LIB_PATTERN "${DISKANN_MSVC_PACKAGES}/boost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}/lib/native/libboost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}-mt-x64-*.lib")
    file(GLOB DISKANN_BOOST_UNIT_TEST_FRAMEWORK_LIB ${UNIT_TEST_FRAMEWORK_LIB_PATTERN})

    set(UNIT_TEST_FRAMEWORK_DLIB_PATTERN "${DISKANN_MSVC_PACKAGES}/boost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}/lib/native/libboost_unit_test_framework-vc${MSVC_TOOLSET_VERSION}-mt-gd-x64-*.lib")
    file(GLOB DISKANN_BOOST_UNIT_TEST_FRAMEWORK_DLIB ${UNIT_TEST_FRAMEWORK_DLIB_PATTERN})

    if (EXISTS ${DISKANN_BOOST_INCLUDE} AND EXISTS ${DISKANN_BOOST_UNIT_TEST_FRAMEWORK_LIB} AND EXISTS ${DISKANN_BOOST_UNIT_TEST_FRAMEWORK_DLIB})
        set(Boost_FOUND ON)
        set(Boost_INCLUDE_DIR ${DISKANN_BOOST_INCLUDE})
        add_library(Boost::unit_test_framework STATIC IMPORTED)
        set_target_properties(Boost::unit_test_framework PROPERTIES IMPORTED_LOCATION_RELEASE "${DISKANN_BOOST_UNIT_TEST_FRAMEWORK_LIB}")
        set_target_properties(Boost::unit_test_framework PROPERTIES IMPORTED_LOCATION_DEBUG "${DISKANN_BOOST_UNIT_TEST_FRAMEWORK_DLIB}")
        message(STATUS "Falling back to using Boost from the nuget package")
    else()
        message(WARNING "Couldn't find Boost. Was looking for ${DISKANN_BOOST_INCLUDE} and ${UNIT_TEST_FRAMEWORK_LIB_PATTERN}")
    endif()
endif()

if (NOT Boost_FOUND)
    message(FATAL_ERROR "Couldn't find Boost dependency")
endif()


set(DISKANN_UNIT_TEST_SOURCES main.cpp index_write_parameters_builder_tests.cpp)

add_executable(${PROJECT_NAME}_unit_tests ${DISKANN_SOURCES} ${DISKANN_UNIT_TEST_SOURCES})
target_link_libraries(${PROJECT_NAME}_unit_tests ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::unit_test_framework)

add_test(NAME ${PROJECT_NAME}_unit_tests COMMAND ${PROJECT_NAME}_unit_tests)


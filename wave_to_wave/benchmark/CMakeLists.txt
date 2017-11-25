cmake_minimum_required(VERSION 2.8)

project(benchmark)

# source: https://pixorblog.wordpress.com/2016/05/22/cmake-google-micro-benchmarking/

if (CMAKE_VERSION VERSION_LESS 3.2)
set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()
include(DownloadProject.cmake)
download_project(PROJ                googlebenchmark
    GIT_REPOSITORY      https://github.com/google/benchmark.git
    GIT_TAG             master
    ${UPDATE_DISCONNECTED_IF_AVAILABLE}
)

add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})

include_directories("${googlebenchmark_SOURCE_DIR}/include")

add_custom_target(run-benchmarks)
set_target_properties(run-benchmarks PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)

file(GLOB_RECURSE ALL_BENCH_CPP *.cpp)

foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})

   get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)

   # Avoid name collision 
   set(TARGET_NAME Bench_${ONE_BENCH_EXEC})

   add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
   set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
   target_link_libraries(${TARGET_NAME} 

# If required, you can add your project library here
# ...

         benchmark
     ${CMAKE_THREAD_LIBS_INIT})

     add_dependencies(run-benchmarks ${TARGET_NAME})
     add_custom_command(TARGET run-benchmarks POST_BUILD 
         COMMAND $<TARGET_FILE:${TARGET_NAME}>)
endforeach()
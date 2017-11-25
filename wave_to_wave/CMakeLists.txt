cmake_minimum_required(VERSION 2.8)

project(wave_to_wave CXX)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
    cxxopts
    PREFIX ${CMAKE_BINARY_DIR}/cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    TIMEOUT 10
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
   )

ExternalProject_Get_Property(cxxopts source_dir)
set(CXXOPTS_INCLUDE_DIR ${source_dir}/include CACHE INTERNAL "Path to include folder for cxxopts")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package(PythonLibs)


add_library(CXXopts INTERFACE)
target_include_directories(CXXopts INTERFACE ${CXXOPTS_INCLUDE_DIR})

add_definitions(-std=c++11)
add_executable(${PROJECT_NAME} wave_to_wave.cpp)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
target_link_libraries(${PROJECT_NAME} fftw3f ${OpenMP_CXX_LIBRARIES} ${PYTHON_LIBRARIES} CXXopts )

add_subdirectory(test)
cmake_minimum_required(VERSION 3.10)

# 1. ADDED 'CXX' - This forces CMake to use the C++ compiler (clang++) 
project(cv_wrapper CXX)

set(CMAKE_CXX_STANDARD 17)

# 2. Force exceptions and RTTI
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -frtti")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/eigen)

add_library(cv_wrapper SHARED cv_wrapper.cpp)
set_target_properties(cv_wrapper PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../Android/libs/arm64-v8a"
)

add_library(opencv_shared SHARED IMPORTED)
set_target_properties(opencv_shared PROPERTIES IMPORTED_LOCATION 
    ${CMAKE_CURRENT_SOURCE_DIR}/../Android/libs/arm64-v8a/libopencv_java4.so
)

# 3. ADDED '-lc++_shared' - This absolutely forces the linker to include the standard library!
target_link_libraries(cv_wrapper opencv_shared log -lc++_shared)
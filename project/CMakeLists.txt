# CMakeLists.txt

# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.14)

set(This neural_net)

project(${This} C CXX)

include(cmake/UpdateSubmodules.cmake)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

enable_testing()

# Include FetchContent module
include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Include directories for header files
include_directories(include include/layers libraries)

# Collect source files from src directory and its subdirectories
file(GLOB_RECURSE Sources src/*.cpp)

# Create library using all source files
add_library(${This} STATIC ${Sources})

add_executable(${This}_main "src/neural_net.cpp")

target_link_libraries(${This}_main PRIVATE ${This})

add_subdirectory(test)

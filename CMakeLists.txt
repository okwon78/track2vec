# Copyright (c) 2020-present, Dreamus, Inc.
# All rights reserved.

cmake_minimum_required(VERSION 3.16)
project(track2vec)

# The version number.
set (TRACK2VEC_VERSION_MAJOR 0)
set (TRACK2VEC_VERSION_MINOR 1)

set (CMAKE_CXX_FLAGS_RELEASE  "-O3 -g0 -fvisibility-inlines-hidden -D_RELEASE")
set (CMAKE_CXX_FLAGS_DEBUG  "-Wall -O0 -g3 -D_DEBUG")
set (CMAKE_CXX_FLAGS "-pthread -std=c++11 -funroll-loops -march=native")
set (CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)

# dependencies 
include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.9.1
)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

include(GNUInstallDirs)

# source code
file(GLOB HEADER_FILES
    "src/*.h"
    "src/annoy/*.h"
)

file(GLOB SOURCE_FILES
    "src/*.cpp"
)

add_executable(track2vec-bin ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(track2vec-bin pthread nlohmann_json::nlohmann_json)
set_target_properties(track2vec-bin PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" OUTPUT_NAME track2vec)

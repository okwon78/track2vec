/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#pragma once

#include <chrono>
#include <fstream>
#include <ostream>
#include <vector>

namespace track2vec
{
namespace utils
{

void reset(std::istream &in);
double getDuration(const std::chrono::steady_clock::time_point&,
                  const std::chrono::steady_clock::time_point&);

void gotoLine(std::ifstream&, int64_t);

} // namespace utils
} // namespace track2vec

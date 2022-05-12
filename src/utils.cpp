/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#include "utils.h"

namespace track2vec
{
namespace utils
{

void reset(std::istream &ifs)
{
    if (ifs.eof())
    {
        ifs.clear();
        ifs.seekg(std::streampos(0));
    }
}

double getDuration(const std::chrono::steady_clock::time_point &start,
                  const std::chrono::steady_clock::time_point &end)
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

void gotoLine(std::ifstream &ifs, int64_t num)
{
    ifs.seekg(std::ios::beg);
    for (int i = 0; i < num - 1; ++i)
    {
        ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

} // namespace utils
} // namespace track2vec

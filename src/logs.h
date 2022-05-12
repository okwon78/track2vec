/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#pragma once

#include <vector>
#include <memory>

#include "track2vec.h"

namespace track2vec
{

class Logs : public std::enable_shared_from_this<Logs>
{
    std::vector<std::string> logs_;
    std::string logDir_;
    std::string s3Dir_;
    size_t buffer_size_;
    int64_t idx_;
    int64_t file_idx_;
    
public:
    Logs(const std::string&, const std::string&, size_t);
    Track2Vec::LogCallback getCallback(const std::string&);
    
private:
    const std::string getCurrentTime(const std::string &fmt = "%Y%m%d%H%M%S") const;
    void callback(const std::string&, double, double, double, double, int64_t);
    void upload(const std::string&);
};

} // namespace track2vec

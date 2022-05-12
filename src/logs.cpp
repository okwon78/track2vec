/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#include "logs.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace track2vec
{

using json = nlohmann::json;

using namespace std::placeholders;

Logs::Logs(const std::string &logDir, const std::string &s3Dir,
           size_t buffer_size)
: logs_(buffer_size), logDir_(logDir), s3Dir_(s3Dir),
buffer_size_(buffer_size), idx_(0), file_idx_(0) {}

void Logs::callback(const std::string &yyyymmddhh, double progress, double loss,
                    double tst, double lr, int64_t eta)
{
    
    int64_t _tst = int64_t(tst);
    double _progress = double(int64_t(progress * 10000000)) / 100000;
    double _loss = loss;
    double _lr = double(int64_t(lr * 1000)) / 1000;
    
    int seconds = eta;
    int minutes = seconds / 60;
    int hours = minutes / 60;
    
    std::ostringstream ss;
    ss << hours << ":" << int(minutes % 60) << ":" << int(seconds % 60);
    
    std::string _eta = ss.str();
    
    std::cout << std::fixed;
    std::cout << "Progress: " << _progress << " %";
    std::cout << " tokens/sec/thread: " << _tst;
    std::cout << " lr: " << _lr;
    std::cout << " loss: " << _loss;
    std::cout << " eta: " << _eta;
    std::cout << std::endl;
    
    std::string timepoint = getCurrentTime();
    
    json j;
    
    j["start"] = yyyymmddhh;
    j["yyyymmddhhmmss"] = timepoint;
    j["throughput"] = _tst;
    j["progress"] = _progress;
    j["loss"] = _loss;
    j["lr"] = _lr;
    j["estimated time of arrival"] = _eta;
    
    logs_[idx_++] = j.dump();
    
    if (idx_ >= buffer_size_)
    {
        upload(yyyymmddhh);
        idx_ = 0;
    }
}

const std::string Logs::getCurrentTime(const std::string &fmt) const
{
    char buffer[20];
    std::time_t rawtime;
    std::time(&rawtime);
    std::tm *timeinfo = std::localtime(&rawtime);
    std::strftime(buffer, 20, fmt.c_str(), timeinfo);
    return std::string(buffer);
}

void Logs::upload(const std::string &yyyymmddhh)
{
    
    if (logDir_.empty() || s3Dir_.empty())
        return;
    
    std::string idx = std::to_string(file_idx_++);
    std::string file_idx = std::string(10 - idx.length(), '0') + idx;
    std::string filename = "train_" + yyyymmddhh + "_" + file_idx + ".json";
    
    std::string local_path = logDir_ + "/" + filename;
    
    std::ofstream ofs(local_path);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(local_path + " cannot be opened for saving.");
    }
    
    for (int i = 0; i < buffer_size_; i++)
    {
        ofs << logs_[i] << std::endl;
    }
    
    ofs.close();
    
    if (false == s3Dir_.empty())
    {
        std::string s3_path = s3Dir_ + "/" + yyyymmddhh + "/" + filename;
        ;
        std::string cmd = "bash -lc 'aws s3 cp " + local_path + " " + s3_path + "'";
        int64_t rst = std::system(cmd.c_str());
        std::cerr << "cmd result: " << rst << std::endl;
    }
}

Track2Vec::LogCallback Logs::getCallback(const std::string &yyyymmddhh)
{
    return std::bind(&Logs::callback, shared_from_this(), yyyymmddhh, _1, _2, _3,
                     _4, _5);
}

} // namespace track2vec

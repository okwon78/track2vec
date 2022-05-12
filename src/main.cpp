/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <functional>

#include "args.h"
#include "track2vec.h"
#include "logs.h"

using namespace track2vec;

void printUsage()
{
    std::cerr
    << "usage: track2vec <command> <args> \n"
    << "The commands supported by track2vec are \n"
    << " train          train a skipgram model \n"
    << " nn          query for nearest neighbors \n"
    << std::endl;
}

void train(const std::vector<std::string> arguements)
{
    std::shared_ptr<Args> args = std::make_shared<Args>();
    args->parseArgs(arguements);
    
    std::shared_ptr<Logs> logs = std::make_shared<Logs>(args->localLog, args->s3Log, args->logBufferSize);
    
    std::shared_ptr<Track2Vec> track2vec = std::make_shared<Track2Vec>(args);
    track2vec->train(logs->getCallback(args->yyyymmddhh));
    track2vec->saveVectors(args->outputDir);
    track2vec->saveModel(args->outputDir);
}

int main(int argc, char **argv)
{
    
#ifdef _RELEASE
    std::cerr << "Release build" << std::endl;
#endif
    
#ifdef _DEBUG
    std::cerr << "Debug build" << std::endl;
#endif
    
    std::vector<std::string> args(argv, argv + argc);
    
    if (args.size() < 2)
    {
        printUsage();
        exit(EXIT_FAILURE);
    }
    
    std::string command(args[1]);
    
    if (command == "train")
    {
        train(args);
    }
    else
    {
        printUsage();
        exit(EXIT_FAILURE);
    }
    
    return 0;
}

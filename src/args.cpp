/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#include "args.h"

#include <iostream>
#include <unistd.h>

namespace track2vec
{

Args::Args()
{
    lr = 0.1;
    dim = 200;
    ntree = 2 * dim;
    ws = 3;           // size of the context window
    discard_t = 1e-4; // sampling threshold [0.0001]
    neg = 100;
    thread = sysconf(_SC_NPROCESSORS_ONLN);
    epoch = 10;
    seed = 0;
    printInterval = 5; // second
    threadInterval = 1000;
    logBufferSize = 1000;
    lrUpdateRate = 100000; // token count
    pretrained_lr = 0.2;
    loadPretrained = 1;
    verbose = 1;
    es = 0.1;
    yyyymmddhh = "0000000000";
    memory = 0;
}

void Args::printHelp() { std::cerr << "Print Help TBD" << std::endl; }

void Args::printValue()
{
    std::cerr << "input: " << input << std::endl;
    std::cerr << "outputDir: " << outputDir << std::endl;
    std::cerr << "metaFileName: " << metaFileName << std::endl;
    std::cerr << "s3Log: " << s3Log << std::endl;
    std::cerr << "localLog: " << localLog << std::endl;
    std::cerr << "yyyymmddhh: " << yyyymmddhh << std::endl;
    std::cerr << "lr: " << lr << std::endl;
    std::cerr << "lrUpdateRate: " << lrUpdateRate << std::endl;
    std::cerr << "pretrained_lr: " << pretrained_lr << std::endl;
    std::cerr << "loadPretrained: " << loadPretrained << std::endl;
    std::cerr << "memory: " << memory << std::endl;
    std::cerr << "discard_t: " << discard_t << std::endl;
    std::cerr << "dim: " << dim << std::endl;
    std::cerr << "ws: " << ws << std::endl;
    std::cerr << "epoch: " << epoch << std::endl;
    std::cerr << "neg: " << neg << std::endl;
    std::cerr << "printInterval: " << printInterval << std::endl;
    std::cerr << "logBufferSize: " << logBufferSize << std::endl;
    std::cerr << "thread: " << thread << std::endl;
    std::cerr << "threadInterval: " << threadInterval << std::endl;
    std::cerr << "verbose: " << verbose << std::endl;
    std::cerr << "es: " << es << std::endl;
}

void Args::parseArgs(const std::vector<std::string> &args)
{
    for (int i = 2; i < args.size(); i += 2)
    {
        if (args[i][0] != '-')
        {
            std::cerr << "Provided argument without a dash! Usage:" << std::endl;
            // printHelp();
            exit(EXIT_FAILURE);
        }
        try
        {
            std::string param = args[i];
            
            if (param == "-h")
            {
                std::cerr << "Here is the help! Usage:" << std::endl;
                printHelp();
                exit(EXIT_FAILURE);
            }
            else if (param == "-input")
            {
                input = std::string(args.at(i + 1));
            }
            else if (param == "-output")
            {
                outputDir = std::string(args.at(i + 1));
            }
            else if (param == "-meta")
            {
                metaFileName = std::string(args.at(i + 1));
            }
            else if (param == "-s3log")
            {
                s3Log = std::string(args.at(i + 1));
            }
            else if (param == "-locallog")
            {
                localLog = std::string(args.at(i + 1));
            }
            else if (param == "-yyyymmddhh")
            {
                yyyymmddhh = std::string(args.at(i + 1));
            }
            else if (param == "-lr")
            {
                double _lr = std::stof(args.at(i + 1));
                lr = _lr > lr ? lr : _lr;
            }
            else if (param == "-lrUpdateRate")
            {
                lrUpdateRate = std::stoi(args.at(i + 1));
            }
            else if (param == "-pretrained_lr")
            {
                pretrained_lr = std::stof(args.at(i + 1));
            }
            else if (param == "-discard_t")
            {
                discard_t = std::stof(args.at(i + 1));
            }
            else if (param == "-dim")
            {
                dim = std::stoi(args.at(i + 1));
            }
            else if (param == "-ws")
            {
                ws = std::stoi(args.at(i + 1));
            }
            else if (param == "-epoch")
            {
                epoch = std::stoi(args.at(i + 1));
            }
            else if (param == "-neg")
            {
                neg = std::stoi(args.at(i + 1));
            }
            else if (param == "-printInterval")
            {
                printInterval = std::stoi(args.at(i + 1));
            }
            else if (param == "-logBufferSize")
            {
                logBufferSize = std::stoi(args.at(i + 1));
            }
            else if (param == "-thread")
            {
                thread = std::stoi(args.at(i + 1));
            }
            else if (param == "-threadInterval")
            {
                threadInterval = std::stoi(args.at(i + 1));
            }
            else if (param == "-verbose")
            {
                verbose = std::stoi(args.at(i + 1));
            }
            else if (param == "-es")
            {
                es = std::stof(args.at(i + 1));
            }
            else if (param == "-memory")
            {
                memory = std::stoi(args.at(i + 1));
            }
            else if (param == "-loadPretrained")
            {
                loadPretrained = std::stoi(args.at(i + 1));
            }
            else
            {
                std::cerr << "Unknown argument: " << args[i] << std::endl;
                printHelp();
                exit(EXIT_FAILURE);
            }
        }
        catch (std::out_of_range)
        {
            std::cerr << args[i] << " is missing an argument" << std::endl;
            printHelp();
            exit(EXIT_FAILURE);
        }
    }
    
    printValue();
    
    if (input.empty() || outputDir.empty() || metaFileName.empty())
    {
        std::cerr << "One of the requried inputs is empty (input, meta or output)"
        << std::flush;
        printHelp();
        exit(EXIT_FAILURE);
    }
}

} // namespace track2vec

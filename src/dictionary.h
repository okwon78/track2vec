/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#pragma once

#include <memory>
#include <string>
#include <random>
#include <unordered_map>

#include "args.h"
#include "entry.h"

namespace track2vec
{

class Dictionary
{
private:
    static const int64_t MAX_TRACK_SIZE = 5000000;
    static const int64_t MAX_SEQ_SIZE = 1024;
    
    std::unordered_map<std::string, trackEntry> tracks_;
    std::unordered_map<std::string, artistEntry> artists_;
    std::unordered_map<std::string, genreEntry> genres_;
    
    std::vector<double> pdiscard_;
    int64_t ntokens_;
    
    std::shared_ptr<Args> args_;
    
    void readMeta(const std::string &);
    void indexing();
    
public:
    Dictionary(std::shared_ptr<Args>);
    
    void loadMeta(const std::string &, const std::string &);
    bool discard(const std::string &, double) const;
    void addTrack(const std::string &, int64_t, std::vector<std::string> &, std::vector<std::string> &);
    bool addGenre(const std::string &);
    bool addArtist(const std::string &);
    int64_t getSequence(std::istream &, std::vector<std::string> &, std::minstd_rand &) const;
    int64_t getRecord(std::istream &, std::vector<std::string> &) const;
    int64_t getTrackIdx(const std::string &) const;
    int64_t getArtistIdx(const std::string &) const;
    int64_t getGenreIdx(const std::string &) const;
    
    std::vector<int64_t> getTrackCount() const;
    const std::vector<int64_t> &getArtistMatrixIndices(const std::string &) const;
    const std::vector<int64_t> &getGenreMatrixIndices(const std::string &) const;
    
    inline const std::unordered_map<std::string, trackEntry> &getTrackEntries() const
    {
        return tracks_;
    }
    const std::unordered_map<std::string, artistEntry> &getArtistEntries() const
    {
        return artists_;
    }
    const std::unordered_map<std::string, genreEntry> &getGenreEntries() const
    {
        return genres_;
    }
    
    trackEntry &getTrackEntry(const std::string &);
    artistEntry &getArtistEntry(const std::string &);
    genreEntry &getGenreEntry(const std::string &);
    
    inline int64_t ntokens() const { return ntokens_; }
    inline int64_t ntracks() const { return tracks_.size(); }
    inline int64_t ngenres() const { return genres_.size(); }
    inline int64_t nartists() const { return artists_.size(); }
};

} // namespace track2vec

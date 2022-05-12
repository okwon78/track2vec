/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#include "dictionary.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <set>
#include <nlohmann/json.hpp>

#include "utils.h"

namespace track2vec
{

using json = nlohmann::json;

Dictionary::Dictionary(std::shared_ptr<Args> args)
: ntokens_(0), args_(args) {}

int64_t Dictionary::getTrackIdx(const std::string &track_id) const
{
    return tracks_.count(track_id) ? tracks_.at(track_id).idx : -1;
}

int64_t Dictionary::getArtistIdx(const std::string &artist_id) const
{
    return artists_.count(artist_id) ? artists_.at(artist_id).idx : -1;
}

int64_t Dictionary::getGenreIdx(const std::string &genre_id) const
{
    return genres_.count(genre_id) ? genres_.at(genre_id).idx : -1;
}

std::vector<int64_t> Dictionary::getTrackCount() const
{
    std::vector<int64_t> track_cnt(tracks_.size());
    for (const auto &pair : tracks_)
    {
        const auto &entry = pair.second;
        track_cnt[entry.idx] = entry.count;
    }
    return track_cnt;
}

void Dictionary::addTrack(const std::string &track_id,
                          int64_t count,
                          std::vector<std::string> &artist_id_list,
                          std::vector<std::string> &genre_id_list)
{
    ntokens_ += count;
    if (tracks_.count(track_id))
    {
        throw std::runtime_error("Invalid meta data file : duplicated track_id exist");
    }
    
    trackEntry entry(track_id);
    entry.count = count;
    entry.artist_ids = artist_id_list;
    entry.genre_ids = genre_id_list;
    tracks_.emplace(std::make_pair(track_id, entry));
}

bool Dictionary::addArtist(const std::string &artist_id)
{
    if (artists_.count(artist_id))
    {
        artists_.at(artist_id).count++;
        return false;
    }
    artists_.emplace(std::make_pair(artist_id, artistEntry(artist_id)));
    return true;
}

bool Dictionary::addGenre(const std::string &genre_id)
{
    if (genres_.count(genre_id))
    {
        genres_.at(genre_id).count++;
        return false;
    }
    
    genres_.emplace(std::make_pair(genre_id, genreEntry(genre_id)));
    return true;
}

void Dictionary::readMeta(const std::string &filename)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open())
    {
        throw std::invalid_argument(filename + " cannot be opened for loading!");
    }
    
    int64_t ntracks = 0;
    
    try
    {
        for (std::string line; std::getline(ifs, line);)
        {
            json j = json::parse(line);
            
            int64_t track_id_num = j["track_id"];
            std::string track_id = std::to_string(track_id_num);
            
            int64_t ntoken = j["ntoken"];
            
            std::vector<int64_t> artist_id_int_list = j["artist_id_list"];
            std::vector<std::string> artist_ids(artist_id_int_list.size());
            
            for (int64_t elem : artist_id_int_list)
            {
                artist_ids.push_back(std::to_string(elem));
            }
            
            std::vector<std::string> genre_ids = j["reco_genre_id_list"];
            
            addTrack(track_id, ntoken, artist_ids, genre_ids);
            
            ntracks++;
            
            if (args_->verbose > 2 && ntracks % 1000 == 0)
                std::cerr << ">> Read " << ntracks / 1000 << "K track meta data" << std::endl;
        }
    }
    catch (std::runtime_error)
    {
        std::cerr << " Invild json format in meta file: " << filename << std::endl;
    }
    
    if (ntracks > MAX_TRACK_SIZE)
    {
        throw std::out_of_range("The number of tracks exceeded the limitation of the number of tracks");
    }
    
    if (args_->verbose > 0)
        std::cerr << ">> The total number of tracks is " << ntracks << std::flush;
    
    ifs.close();
}

void Dictionary::loadMeta(const std::string &meta, const std::string &input)
{
    readMeta(meta);
    indexing();
    
    if (args_->verbose > 0)
    {
        std::cerr << "Read " << ntokens_ / 1000000 << "M tokens" << std::endl;
        std::cerr << "Number of tracks:  " << tracks_.size() << std::endl;
        std::cerr << "Number of artists:  " << artists_.size() << std::endl;
        std::cerr << "Number of reco genres:  " << genres_.size() << std::endl;
    }
}

void Dictionary::indexing()
{
    int64_t index = 0;
    
    for (auto &elem : tracks_)
    {
        auto &track_entry = elem.second;
        track_entry.idx = index++;
        double f = double(track_entry.count) / double(ntokens_);
        track_entry.pdiscard = std::sqrt(args_->discard_t / f) + args_->discard_t / f;
        
        for (const std::string &genre_id : track_entry.genre_ids)
        {
            addGenre(genre_id);
        }
        for (const std::string &artist_id : track_entry.artist_ids)
        {
            addArtist(artist_id);
        }
    }
    
    for (auto &elem : artists_)
    {
        elem.second.idx = index++;
    }
    
    for (auto &elem : genres_)
    {
        elem.second.idx = index++;
    }
    
    for (auto &elem : tracks_)
    {
        trackEntry &track_entry = elem.second;
        
        for (const auto &artist_id : track_entry.artist_ids)
        {
            int64_t artist_idx = getArtistIdx(artist_id);
            if (artist_idx > -1)
            {
                track_entry.artist_matrix_indices.push_back(artist_idx);
            }
        }
        
        for (const auto &genre_id : track_entry.genre_ids)
        {
            int64_t genre_idx = getGenreIdx(genre_id);
            if (genre_idx > -1)
            {
                track_entry.genre_matrix_indices.push_back(genre_idx);
            }
        }
    }
}

bool Dictionary::discard(const std::string &track_id, double rand) const
{
    return 0 == tracks_.count(track_id) ? true : rand > tracks_.at(track_id).pdiscard;
}

int64_t Dictionary::getSequence(std::istream &ifs,
                                std::vector<std::string> &tracks,
                                std::minstd_rand &rng) const
{
    std::uniform_real_distribution<> uniform(0, 1);
    std::string track;
    int64_t read_cnt = 0;
    
    tracks.clear();
    
    std::string line;
    
    try
    {
        do
        {
            if (std::getline(ifs, line).eof())
            {
                ifs.clear();
                ifs.seekg(std::streampos(0));
            }
        } while (0 == line.length());
        
        json j = json::parse(line);
        //int64_t character_id = j["c"];
        //int64_t length = j["l"];
        std::vector<int64_t> track_seq = j["t"];
        
        for (int64_t track_id : track_seq)
        {
            std::string track = std::to_string(track_id);
            
            if (0 == tracks_.count(track))
                continue;
            
            read_cnt++;
            if (false == discard(track, uniform(rng)))
            {
                tracks.push_back(track);
            }
        }
    }
    catch (std::logic_error)
    {
        std::cerr << " Invild json format: " << line << std::endl;
    }
    
    return read_cnt;
}

int64_t Dictionary::getRecord(std::istream &ifs, std::vector<std::string> &tracks) const
{
    std::string line;
    
    try
    {
        do
        {
            if (std::getline(ifs, line).eof())
            {
                return 0;
            }
        } while (0 == line.length());
        
        json j = json::parse(line);
        //int64_t character_id = j["c"];
        int64_t length = j["l"];
        std::vector<int64_t> track_seq = j["t"];
        
        assert(length == track_seq.size());
        
        for (int64_t track_id : track_seq)
        {
            std::string track = std::to_string(track_id);
            
            if (0 == tracks_.count(track))
                continue;
            
            tracks.push_back(track);
        }
    }
    catch (std::logic_error)
    {
        std::cerr << " Invild json format: " << line << std::endl;
    }
    
    return tracks.size();
}

const std::vector<int64_t> &Dictionary::getArtistMatrixIndices(const std::string &track_id) const
{
    if (0 == tracks_.count(track_id))
    {
        throw std::runtime_error("Invalid meta file: " + track_id + " meta data does not exist");
    }
    
    return tracks_.at(track_id).artist_matrix_indices;
}

const std::vector<int64_t> &Dictionary::getGenreMatrixIndices(const std::string &track_id) const
{
    if (0 == tracks_.count(track_id))
    {
        throw std::runtime_error("Invalid meta file: " + track_id + " meta data does not exist");
    }
    
    return tracks_.at(track_id).genre_matrix_indices;
}

trackEntry &Dictionary::getTrackEntry(const std::string &track_id)
{
    if (0 == tracks_.count(track_id))
    {
        throw std::runtime_error("Invalid meta file: " + track_id + " meta data does not exist");
    }
    
    return tracks_.at(track_id);
}

artistEntry &Dictionary::getArtistEntry(const std::string &artist_id)
{
    if (0 == artists_.count(artist_id))
    {
        throw std::runtime_error("Invalid meta file: " + artist_id + " meta data does not exist");
    }
    
    return artists_.at(artist_id);
}

genreEntry &Dictionary::getGenreEntry(const std::string &genre_id)
{
    if (0 == genres_.count(genre_id))
    {
        throw std::runtime_error("Invalid meta file: " + genre_id + " meta data does not exist");
    }
    
    return genres_.at(genre_id);
}

} // namespace track2vec

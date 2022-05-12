#include "entry.h"

namespace track2vec
{
    trackEntry::trackEntry(const std::string &track_id)
        : track_id(track_id),
          idx(-1),
          lr_alpha(1.0),
          count(1),
          pdiscard(1),
          artist_ids(0),
          genre_ids(0),
          artist_matrix_indices(0),
          genre_matrix_indices(0) {}

    artistEntry::artistEntry(const std::string &artist_id)
        : artist_id(artist_id), idx(-1), count(1) {}

    genreEntry::genreEntry(const std::string &genre_id)
        : genre_id(genre_id), idx(-1), count(1) {}

} //namespace track2vec
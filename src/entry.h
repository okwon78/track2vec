#include <vector>
#include <string>

namespace track2vec
{
    struct trackEntry
    {
        explicit trackEntry(const std::string &);
        std::string track_id;
        int64_t idx;
        double lr_alpha;
        int64_t count;
        double pdiscard;
        std::vector<std::string> artist_ids;
        std::vector<std::string> genre_ids;
        std::vector<int64_t> artist_matrix_indices;
        std::vector<int64_t> genre_matrix_indices;
    };

    struct artistEntry
    {
        explicit artistEntry(const std::string &);
        std::string artist_id;
        int64_t idx;
        int64_t count;
    };

    struct genreEntry
    {
        explicit genreEntry(const std::string &);
        std::string genre_id;
        int64_t idx;
        int64_t count;
    };
} //namespace track2vec

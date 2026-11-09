// Minimal TIFF reader for the ingest path: parses the first IFD (classic or
// BigTIFF, either byte order) and reads uncompressed chunky 8/16-bit pixel data
// straight into a caller-supplied (pinned) buffer. JPEG frames (0.6.34) are
// decoded on the reading thread through Windows Imaging Component into the same
// buffer as 8-bit RGB, so the rest of the engine sees an ordinary 8-bit frame;
// the embedded profile comes along. Compressed TIFFs are still out.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace stackcore {

struct TiffInfo {
    uint32_t width = 0, height = 0;
    uint16_t bits = 0;            // per sample: 8 or 16
    uint16_t samples = 0;         // 1, 3 or 4
    uint16_t compression = 0;     // 1 = none
    uint16_t photometric = 0;     // 2 = RGB, 1 = min-is-black
    uint16_t planar = 1;          // 1 = chunky
    uint16_t sample_format = 1;   // 1 = unsigned int
    bool big_endian = false;
    bool bigtiff = false;
    bool jpeg = false;            // a JPEG file: decoded to 8-bit RGB by tiff_read_pixels (width x height x 3 bytes)
    uint64_t file_size = 0;
    std::vector<uint64_t> strip_offsets, strip_bytes;   // (tiles are not supported yet)
    bool contiguous = false;      // strips follow each other with no gaps
    uint64_t data_offset = 0;     // first strip
    uint64_t data_bytes = 0;      // sum of strip bytes
    std::vector<uint8_t> icc;     // embedded ICC profile (tag 34675), if any
    std::string error;            // non-empty: unusable for the fast path, says why

    bool ok() const { return error.empty(); }
    size_t row_bytes() const { return (size_t)width * samples * (bits / 8); }
    size_t pixel_bytes() const { return (size_t)height * row_bytes(); }
};

// Reads the header and first IFD only (a few KB).
TiffInfo tiff_probe(const std::filesystem::path& path);

struct TiffRead {
    size_t data_offset_in_buf = 0;  // pixel data starts here inside dst (alignment slack)
    size_t bytes = 0;               // pixel bytes read
    bool unbuffered = false;        // FILE_FLAG_NO_BUFFERING was used
    double ms = 0;
    std::string error;
    bool ok() const { return error.empty(); }
};

// Alignment the unbuffered path needs for the buffer and for file offsets.
constexpr size_t kReadAlign = 4096;
// Capacity a destination buffer needs to hold any frame of `pixel_bytes`.
constexpr size_t tiff_buffer_capacity(size_t pixel_bytes) { return pixel_bytes + 2 * kReadAlign; }

// Reads the pixel data into dst (capacity cap >= tiff_buffer_capacity(info.pixel_bytes())).
// Contiguous data is read unbuffered when the volume allows it, from the
// preceding 4 KB boundary, so the pixels start at data_offset_in_buf; a
// buffered read (offset 0) is the fallback. dst should be 4 KB aligned
// (cudaHostAlloc memory is).
TiffRead tiff_read_pixels(const std::filesystem::path& path, const TiffInfo& info, uint8_t* dst, size_t cap);

}  // namespace stackcore

// Frame reader for the ingest path. Uncompressed chunky 8/16-bit TIFF (classic
// or BigTIFF, either byte order) is read straight into a caller-supplied
// (pinned) buffer. Everything else is decoded on the reading thread into the
// same buffer as plain chunky RGB, so the rest of the engine sees an ordinary
// uncompressed frame:
//   JPEG                      Windows Imaging Component, 8-bit
//   compressed / tiled TIFF   Windows Imaging Component (LZW, ZIP, PackBits), 8 or 16-bit
//   camera raw and DNG        LibRaw (raw.cpp), developed to 16-bit Adobe RGB
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
    bool wic = false;             // a compressed or tiled TIFF: decoded to 8 or 16-bit RGB by tiff_read_pixels
    bool raw = false;             // a camera raw file or DNG: developed to 16-bit RGB by tiff_read_pixels
    uint64_t file_size = 0;
    std::vector<uint64_t> strip_offsets, strip_bytes;
    bool contiguous = false;      // strips follow each other with no gaps
    uint64_t data_offset = 0;     // first strip
    uint64_t data_bytes = 0;      // sum of strip bytes
    std::vector<uint8_t> icc;     // embedded ICC profile (tag 34675), if any; Adobe RGB for a raw file
    // when the frame was taken: EXIF DateTimeOriginal (36867) with SubSecTimeOriginal (37521) appended as ".sss" when
    // present, else the TIFF DateTime (306); "YYYY:MM:DD HH:MM:SS[.sss]", sorts as text; empty = none (JPEG too)
    std::string datetime;
    std::string error;            // non-empty: unusable, says why

    bool ok() const { return error.empty(); }
    bool decoded() const { return jpeg || wic || raw; }   // not read straight from the file: decoded into the buffer
    size_t row_bytes() const { return (size_t)width * samples * (bits / 8); }
    size_t pixel_bytes() const { return (size_t)height * row_bytes(); }
};

// Reads the header only (a few KB; a raw file's metadata).
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
// (cudaHostAlloc memory is). Decoded formats land at offset 0.
TiffRead tiff_read_pixels(const std::filesystem::path& path, const TiffInfo& info, uint8_t* dst, size_t cap);

// A quick thumbnail of a JPEG (decoded small by the codec, tens of ms) or a raw file (developed at half size
// without demosaicing, a few tenths of a second), about max_w pixels wide, as 8-bit RGB. false for other
// files (their thumbnails are sampled from the full read) or when the decode fails.
bool fast_thumbnail(const std::filesystem::path& path, const TiffInfo& info, int max_w, std::vector<uint8_t>& rgb, int& w, int& h);
bool jpeg_thumbnail(const std::filesystem::path& path, int max_w, std::vector<uint8_t>& rgb, int& w, int& h);

// The file types a stack folder is searched for, by extension (lower or upper case): TIFF, JPEG, raw / DNG.
bool is_frame_file(const std::filesystem::path& path);
bool is_raw_file(const std::filesystem::path& path);
// The same list as glob patterns ("*.tif", "*.TIF", ...), for file dialogs and directory listings.
std::vector<std::string> frame_file_patterns();
bool raw_supported();   // false when the build has no LibRaw: raw files then probe with an error

// raw.cpp
TiffInfo raw_probe(const std::filesystem::path& path, TiffInfo t);
TiffRead raw_read(const std::filesystem::path& path, const TiffInfo& info, uint8_t* dst, size_t cap);
bool raw_thumbnail(const std::filesystem::path& path, int max_w, std::vector<uint8_t>& rgb, int& w, int& h);
// A matrix ICC profile equivalent to Adobe RGB (1998): what raw frames are developed to.
const std::vector<uint8_t>& adobe_rgb_icc();

}  // namespace stackcore

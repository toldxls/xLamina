// Camera raw files and DNG, developed by LibRaw on the reading thread into the caller's buffer as 16-bit
// Adobe RGB, so the rest of the engine sees an ordinary uncompressed 16-bit frame. The development is fixed
// and the same for every frame of a stack (the camera's white balance, no automatic brightening, highlights
// clipped, no noise reduction, no lens corrections): a neutral rendering, not a raw converter's look.
#include "stackcore/tiff.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>

#ifdef XLAMINA_HAVE_LIBRAW
#include <libraw/libraw.h>
#endif

namespace stackcore {

// ---- the profile: Adobe RGB (1998) as an ICC v2 matrix profile (D50-adapted primaries, gamma 563/256)
const std::vector<uint8_t>& adobe_rgb_icc() {
    static const std::vector<uint8_t> icc = [] {
        auto be32 = [](std::vector<uint8_t>& v, uint32_t x) { for (int s = 24; s >= 0; s -= 8) v.push_back((uint8_t)(x >> s)); };
        auto be16 = [](std::vector<uint8_t>& v, uint16_t x) { v.push_back((uint8_t)(x >> 8)); v.push_back((uint8_t)x); };
        auto sig = [](std::vector<uint8_t>& v, const char* s) { v.insert(v.end(), s, s + 4); };
        auto fixed = [&](std::vector<uint8_t>& v, double x) { be32(v, (uint32_t)(int32_t)std::lround(x * 65536.0)); };
        auto pad = [](std::vector<uint8_t>& v) { while (v.size() % 4) v.push_back(0); };
        auto xyz = [&](double x, double y, double z) {
            std::vector<uint8_t> v;
            sig(v, "XYZ "); be32(v, 0);
            fixed(v, x); fixed(v, y); fixed(v, z);
            return v;
        };
        std::vector<std::pair<const char*, std::vector<uint8_t>>> el;   // the distinct tag data elements
        {
            const char text[] = "Adobe RGB (1998) compatible";
            std::vector<uint8_t> v;
            sig(v, "desc"); be32(v, 0);
            be32(v, (uint32_t)sizeof text);
            v.insert(v.end(), text, text + sizeof text);
            v.insert(v.end(), 4 + 4 + 2 + 1 + 67, 0);   // no Unicode or ScriptCode description
            el.emplace_back("desc", v);
        }
        {
            const char text[] = "No copyright, use freely";
            std::vector<uint8_t> v;
            sig(v, "text"); be32(v, 0);
            v.insert(v.end(), text, text + sizeof text);
            el.emplace_back("cprt", v);
        }
        el.emplace_back("wtpt", xyz(0.95045, 1.0, 1.08905));
        el.emplace_back("rXYZ", xyz(0.60974, 0.31111, 0.01947));
        el.emplace_back("gXYZ", xyz(0.20528, 0.62567, 0.06087));
        el.emplace_back("bXYZ", xyz(0.14919, 0.06322, 0.74457));
        {
            std::vector<uint8_t> v;
            sig(v, "curv"); be32(v, 0); be32(v, 1);
            be16(v, 0x0233);   // u8Fixed8 2.19921875
            el.emplace_back("rTRC", v);
        }
        const char* tags[] = {"desc", "cprt", "wtpt", "rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC"};
        const int ntags = 9;
        std::vector<uint32_t> off(el.size()), len(el.size());
        uint32_t pos = 128 + 4 + 12 * ntags;
        for (size_t i = 0; i < el.size(); ++i) {
            off[i] = pos;
            len[i] = (uint32_t)el[i].second.size();
            pos += (len[i] + 3) & ~3u;
        }
        std::vector<uint8_t> p;
        be32(p, pos);            // profile size
        be32(p, 0);              // preferred CMM
        be32(p, 0x02100000);     // version 2.1
        sig(p, "mntr"); sig(p, "RGB "); sig(p, "XYZ ");
        for (uint16_t d : {2026, 1, 1, 0, 0, 0}) be16(p, d);
        sig(p, "acsp"); sig(p, "MSFT");
        be32(p, 0); be32(p, 0); be32(p, 0);   // flags, manufacturer, model
        be32(p, 0); be32(p, 0);               // attributes
        be32(p, 0);                           // rendering intent: perceptual
        be32(p, 0x0000F6D6); be32(p, 0x00010000); be32(p, 0x0000D32D);   // the PCS illuminant, D50
        be32(p, 0);                           // creator
        p.resize(128, 0);
        be32(p, ntags);
        for (const char* t : tags) {
            size_t i = 0;
            const char* data_of = (t[1] == 'T') ? "rTRC" : t;   // the three curves share one element
            while (std::strncmp(el[i].first, data_of, 4) != 0) ++i;
            sig(p, t); be32(p, off[i]); be32(p, len[i]);
        }
        for (auto& e : el) { p.insert(p.end(), e.second.begin(), e.second.end()); pad(p); }
        return p;
    }();
    return icc;
}

#ifdef XLAMINA_HAVE_LIBRAW

bool raw_supported() { return true; }

namespace {
// one development for every frame: nothing here may depend on the frame's own histogram
void set_params(LibRaw& lr, bool thumb) {
    auto& P = lr.imgdata.params;
    P.use_camera_wb = 1;
    P.use_auto_wb = 0;
    P.no_auto_bright = 1;
    P.highlight = 0;                 // clip
    P.output_color = 2;              // Adobe RGB
    P.gamm[0] = 1.0 / 2.19921875;    // its tone curve: a pure power
    P.gamm[1] = 0.0;
    P.output_bps = thumb ? 8 : 16;
    P.half_size = thumb ? 1 : 0;     // 2x2 cells averaged, no demosaic
    P.user_qual = 3;                 // AHD
    P.output_tiff = 0;
}
std::string raw_error(const char* what, int rc) { return std::string("raw: ") + what + ": " + libraw_strerror(rc); }
int open(LibRaw& lr, const std::filesystem::path& path) { return lr.open_file(path.c_str()); }
}  // namespace

TiffInfo raw_probe(const std::filesystem::path& path, TiffInfo t) {
    LibRaw lr;
    set_params(lr, false);
    int rc = open(lr, path);
    if (rc != LIBRAW_SUCCESS) { t.error = raw_error("cannot read", rc); return t; }
    lr.adjust_sizes_info_only();
    const auto& S = lr.imgdata.sizes;
    t.width = S.iwidth;
    t.height = S.iheight;
    if (!t.width || !t.height) { t.error = "raw: no image"; return t; }
    t.raw = true;
    t.bits = 16; t.samples = 3; t.compression = 1; t.photometric = 2; t.planar = 1; t.sample_format = 1;
    t.big_endian = false; t.contiguous = true;
    t.data_offset = 0; t.data_bytes = t.pixel_bytes();
    t.strip_offsets = {0}; t.strip_bytes = {t.data_bytes};
    t.icc = adobe_rgb_icc();
    time_t ts = lr.imgdata.other.timestamp;
    if (ts > 0) {
        struct tm tmv{};
        char buf[32];
        if (localtime_s(&tmv, &ts) == 0 && strftime(buf, sizeof buf, "%Y:%m:%d %H:%M:%S", &tmv)) t.datetime = buf;
    }
    return t;
}

namespace {
struct MemImage {
    libraw_processed_image_t* p = nullptr;
    ~MemImage() { if (p) LibRaw::dcraw_clear_mem(p); }
};
// open, unpack, develop; the result as LibRaw's bitmap
bool develop(const std::filesystem::path& path, bool thumb, MemImage& out, std::string& err) {
    LibRaw lr;
    set_params(lr, thumb);
    int rc = open(lr, path);
    if (rc != LIBRAW_SUCCESS) { err = raw_error("cannot read", rc); return false; }
    rc = lr.unpack();
    if (rc != LIBRAW_SUCCESS) { err = raw_error("cannot unpack", rc); return false; }
    rc = lr.dcraw_process();
    if (rc != LIBRAW_SUCCESS) { err = raw_error("cannot develop", rc); return false; }
    out.p = lr.dcraw_make_mem_image(&rc);
    if (!out.p || rc != LIBRAW_SUCCESS) { err = raw_error("no image", rc); return false; }
    if (out.p->type != LIBRAW_IMAGE_BITMAP || (out.p->colors != 3 && out.p->colors != 1)) { err = "raw: unexpected image layout"; return false; }
    return true;
}
}  // namespace

TiffRead raw_read(const std::filesystem::path& path, const TiffInfo& info, uint8_t* dst, size_t cap) {
    TiffRead r;
    auto t0 = std::chrono::steady_clock::now();
    MemImage img;
    if (!develop(path, false, img, r.error)) return r;
    if (img.p->width != info.width || img.p->height != info.height || img.p->bits != 16) { r.error = "raw: the developed size differs from the header's"; return r; }
    const size_t n = (size_t)info.width * info.height, bytes = n * 6;
    if (bytes > cap) { r.error = "buffer too small"; return r; }
    if (img.p->colors == 3) std::memcpy(dst, img.p->data, bytes);
    else {   // a monochrome sensor: grey to RGB
        const uint16_t* s = (const uint16_t*)img.p->data;
        uint16_t* d = (uint16_t*)dst;
        for (size_t i = 0; i < n; ++i) d[3 * i] = d[3 * i + 1] = d[3 * i + 2] = s[i];
    }
    r.bytes = bytes;
    r.data_offset_in_buf = 0;
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

bool raw_thumbnail(const std::filesystem::path& path, int max_w, std::vector<uint8_t>& rgb, int& w, int& h) {
    MemImage img;
    std::string err;
    if (!develop(path, true, img, err) || img.p->bits != 8) return false;
    const int sw = img.p->width, sh = img.p->height, c = img.p->colors;
    const int down = std::max(1, sw / std::max(1, max_w));
    w = std::max(1, sw / down);
    h = std::max(1, sh / down);
    rgb.resize((size_t)w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const uint8_t* s = img.p->data + ((size_t)(y * down) * sw + (size_t)(x * down)) * c;
            uint8_t* d = rgb.data() + ((size_t)y * w + x) * 3;
            d[0] = s[0]; d[1] = s[c == 3 ? 1 : 0]; d[2] = s[c == 3 ? 2 : 0];
        }
    return true;
}

#else   // built without LibRaw

bool raw_supported() { return false; }
TiffInfo raw_probe(const std::filesystem::path&, TiffInfo t) { t.error = "raw files need a build with LibRaw"; return t; }
TiffRead raw_read(const std::filesystem::path&, const TiffInfo&, uint8_t*, size_t) { TiffRead r; r.error = "raw files need a build with LibRaw"; return r; }
bool raw_thumbnail(const std::filesystem::path&, int, std::vector<uint8_t>&, int&, int&) { return false; }

#endif

}  // namespace stackcore

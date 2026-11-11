#include "stackcore/tiff.h"

#include <windows.h>
#include <objbase.h>
#include <wincodec.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>

namespace stackcore {
namespace {

struct File {
    HANDLE h = INVALID_HANDLE_VALUE;
    explicit File(const std::filesystem::path& p, DWORD flags = FILE_FLAG_SEQUENTIAL_SCAN) {
        h = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
    }
    ~File() { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
    bool ok() const { return h != INVALID_HANDLE_VALUE; }
    uint64_t size() const {
        LARGE_INTEGER s{};
        GetFileSizeEx(h, &s);
        return (uint64_t)s.QuadPart;
    }
    // Reads exactly n bytes at offset (fewer only at EOF). Returns bytes read.
    size_t read_at(uint64_t off, void* dst, size_t n) const {
        size_t done = 0;
        while (done < n) {
            DWORD want = (DWORD)std::min<size_t>(n - done, 64u << 20), got = 0;
            OVERLAPPED ov{};
            uint64_t o = off + done;
            ov.Offset = (DWORD)(o & 0xffffffffu);
            ov.OffsetHigh = (DWORD)(o >> 32);
            if (!ReadFile(h, (uint8_t*)dst + done, want, &got, &ov) || got == 0) break;
            done += got;
        }
        return done;
    }
};

struct Parser {
    const File& f;
    bool be = false, big = false;
    std::vector<uint8_t> tmp;

    uint64_t num(const uint8_t* p, int n) const {
        uint64_t v = 0;
        for (int i = 0; i < n; ++i) v |= (uint64_t)p[i] << (8 * (be ? n - 1 - i : i));
        return v;
    }
    static size_t type_size(uint16_t t) {
        switch (t) {
            case 1: case 2: case 6: case 7: return 1;
            case 3: case 8: return 2;
            case 4: case 9: case 11: return 4;
            case 5: case 10: case 12: case 16: case 17: case 18: return 8;
            default: return 0;
        }
    }
};

}  // namespace

// ---- JPEG through Windows Imaging Component: nothing to ship, decodes on the calling thread (the streamer's
// reader threads run 16 of them). The probe reads the header for the size and the embedded profile; the read
// decodes the whole frame to 8-bit RGB into the caller's buffer.
namespace {
struct ComScope {
    bool uninit = false;
    // RPC_E_CHANGED_MODE on a thread already in the GUI's apartment: usable, and not ours to uninitialise
    ComScope() { uninit = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED)); }
    ~ComScope() { if (uninit) CoUninitialize(); }
};
template <class T>
struct ComPtr {
    T* p = nullptr;
    ~ComPtr() { if (p) p->Release(); }
    T* operator->() const { return p; }
};
bool wic_open(const std::filesystem::path& path, ComPtr<IWICImagingFactory>& fac, ComPtr<IWICBitmapFrameDecode>& frame, std::string& err) {
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&fac.p)))) { err = "decoder: Windows Imaging Component unavailable"; return false; }
    ComPtr<IWICBitmapDecoder> dec;
    if (FAILED(fac->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec.p))) { err = "decoder: cannot read the file"; return false; }
    if (FAILED(dec->GetFrame(0, &frame.p))) { err = "decoder: no image"; return false; }
    return true;
}
// the header only, parsed here (a WIC decoder per probe cost 7.6 ms; the GUI probes every file of a folder as it
// opens): the frame's size from the SOF segment, the profile from the APP2 "ICC_PROFILE" segments
TiffInfo jpeg_probe(const std::filesystem::path& path, File& f, TiffInfo t) {
    uint64_t pos = 2;   // after SOI
    std::vector<std::pair<int, std::vector<uint8_t>>> icc_parts;   // (sequence number, bytes)
    uint8_t hdr[4];
    for (int seg = 0; seg < 256; ++seg) {
        if (f.read_at(pos, hdr, 4) < 4 || hdr[0] != 0xFF) break;
        uint8_t marker = hdr[1];
        if (marker == 0xFF) { ++pos; continue; }                     // fill byte
        if (marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD7)) { pos += 2; continue; }   // SOI / RSTn: no length
        if (marker == 0xD9 || marker == 0xDA) break;                  // EOI / SOS: the header is over
        uint32_t len = ((uint32_t)hdr[2] << 8) | hdr[3];
        if (len < 2) break;
        const bool sof = (marker >= 0xC0 && marker <= 0xCF) && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
        if (sof) {
            uint8_t s[6];
            if (f.read_at(pos + 4, s, 6) < 6) break;
            t.height = ((uint32_t)s[1] << 8) | s[2];
            t.width = ((uint32_t)s[3] << 8) | s[4];
            if (s[0] != 8) { t.error = "JPEG: only 8-bit"; return t; }
            if (s[5] != 3 && s[5] != 1) { t.error = "JPEG: CMYK is not supported"; return t; }
        } else if (marker == 0xE2 && len > 2 + 14) {               // APP2: "ICC_PROFILE\0" seq count data
            std::vector<uint8_t> seg_data(len - 2);
            if (f.read_at(pos + 4, seg_data.data(), seg_data.size()) < seg_data.size()) break;
            if (memcmp(seg_data.data(), "ICC_PROFILE", 12) == 0) icc_parts.emplace_back(seg_data[12], std::vector<uint8_t>(seg_data.begin() + 14, seg_data.end()));
        }
        pos += 2 + len;
    }
    if (t.width == 0 || t.height == 0) { t.error = "JPEG: no frame header"; return t; }
    std::sort(icc_parts.begin(), icc_parts.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (auto& [seq, bytes] : icc_parts) t.icc.insert(t.icc.end(), bytes.begin(), bytes.end());
    t.jpeg = true;
    t.bits = 8; t.samples = 3; t.compression = 7; t.photometric = 2; t.planar = 1; t.sample_format = 1;   // (decoded to RGB whatever the file holds)
    t.big_endian = false; t.contiguous = true;
    t.data_offset = 0; t.data_bytes = t.pixel_bytes();
    t.strip_offsets = {0}; t.strip_bytes = {t.data_bytes};
    return t;
}
// a JPEG (8-bit) or a compressed / tiled TIFF (8 or 16-bit, info.bits): the whole frame decoded to chunky RGB
TiffRead wic_read(const std::filesystem::path& path, const TiffInfo& info, uint8_t* dst, size_t cap) {
    TiffRead r;
    auto t0 = std::chrono::steady_clock::now();
    ComScope com;
    ComPtr<IWICImagingFactory> fac;
    ComPtr<IWICBitmapFrameDecode> frame;
    if (!wic_open(path, fac, frame, r.error)) return r;
    UINT fw = 0, fh = 0;
    if (FAILED(frame->GetSize(&fw, &fh)) || fw != info.width || fh != info.height) { r.error = "decoder: the image size differs from the header's"; return r; }
    ComPtr<IWICFormatConverter> conv;
    const GUID fmt = info.bits == 16 ? GUID_WICPixelFormat48bppRGB : GUID_WICPixelFormat24bppRGB;
    if (FAILED(fac->CreateFormatConverter(&conv.p)) ||
        FAILED(conv->Initialize(frame.p, fmt, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom))) {
        r.error = "decoder: cannot convert to RGB";
        return r;
    }
    const size_t stride = (size_t)info.width * 3 * (info.bits / 8);
    const size_t bytes = stride * info.height;
    if (bytes > cap) { r.error = "buffer too small"; return r; }
    // in bands: CopyPixels takes the buffer size as a UINT, and a 16-bit frame of 700 Mpx would pass 4 GB
    const UINT band = (UINT)std::max<size_t>(1, ((size_t)256 << 20) / stride);
    for (UINT y = 0; y < info.height; y += band) {
        WICRect rc{0, (INT)y, (INT)info.width, (INT)std::min<UINT>(band, info.height - y)};
        if (FAILED(conv->CopyPixels(&rc, (UINT)stride, (UINT)(stride * rc.Height), dst + (size_t)y * stride))) { r.error = "decoder: decode failed"; return r; }
    }
    r.bytes = bytes;
    r.data_offset_in_buf = 0;
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
}
}  // namespace

bool jpeg_thumbnail(const std::filesystem::path& path, int max_w, std::vector<uint8_t>& rgb, int& w, int& h) {
    ComScope com;
    ComPtr<IWICImagingFactory> fac;
    ComPtr<IWICBitmapFrameDecode> frame;
    std::string err;
    if (!wic_open(path, fac, frame, err)) return false;
    UINT fw = 0, fh = 0;
    if (FAILED(frame->GetSize(&fw, &fh)) || fw == 0 || fh == 0) return false;
    int down = std::max(1, (int)fw / std::max(1, max_w));
    w = std::max(1, (int)fw / down);
    h = std::max(1, (int)fh / down);
    // the scaler on the decoder's frame: the JPEG codec scales in the DCT domain, so this decodes a fraction of the pixels
    ComPtr<IWICBitmapScaler> sc;
    if (FAILED(fac->CreateBitmapScaler(&sc.p)) || FAILED(sc->Initialize(frame.p, (UINT)w, (UINT)h, WICBitmapInterpolationModeFant))) return false;
    ComPtr<IWICFormatConverter> conv;
    if (FAILED(fac->CreateFormatConverter(&conv.p)) ||
        FAILED(conv->Initialize(sc.p, GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom))) return false;
    rgb.resize((size_t)w * h * 3);
    return SUCCEEDED(conv->CopyPixels(nullptr, (UINT)w * 3, (UINT)rgb.size(), rgb.data()));
}

bool fast_thumbnail(const std::filesystem::path& path, const TiffInfo& info, int max_w, std::vector<uint8_t>& rgb, int& w, int& h) {
    if (info.jpeg) return jpeg_thumbnail(path, max_w, rgb, w, h);
    if (info.raw) return raw_thumbnail(path, max_w, rgb, w, h);
    return false;
}

namespace {
const char* const kRawExt[] = {".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf", ".orf", ".rw2", ".pef",
                               ".srw", ".3fr", ".fff", ".iiq", ".erf", ".mrw", ".x3f", ".rwl", ".mos", ".kdc", ".dcr"};
const char* const kPlainExt[] = {".tif", ".tiff", ".jpg", ".jpeg"};
std::string lower_ext(const std::filesystem::path& p) {
    std::string e = p.extension().string();
    for (char& c : e) c = (char)tolower((unsigned char)c);
    return e;
}
}  // namespace

bool is_raw_file(const std::filesystem::path& path) {
    std::string e = lower_ext(path);
    for (const char* x : kRawExt) if (e == x) return true;
    return false;
}

bool is_frame_file(const std::filesystem::path& path) {
    std::string e = lower_ext(path);
    for (const char* x : kPlainExt) if (e == x) return true;
    return raw_supported() && is_raw_file(path);
}

std::vector<std::string> frame_file_patterns() {
    std::vector<std::string> out;
    auto add = [&](const char* x) {
        std::string lo = std::string("*") + x, up = lo;
        for (char& c : up) c = (char)toupper((unsigned char)c);
        out.push_back(lo);
        out.push_back(up);
    };
    for (const char* x : kPlainExt) add(x);
    if (raw_supported()) for (const char* x : kRawExt) add(x);
    return out;
}

TiffInfo tiff_probe(const std::filesystem::path& path) {
    TiffInfo t;
    File f(path);
    if (!f.ok()) { t.error = "cannot open"; return t; }
    t.file_size = f.size();
    // by extension, before the header: a DNG (and most raw formats) is a TIFF container whose first image is a preview
    if (is_raw_file(path)) return raw_probe(path, t);
    uint8_t hdr[16];
    if (f.read_at(0, hdr, 16) < 8) { t.error = "too short"; return t; }
    if (hdr[0] == 0xFF && hdr[1] == 0xD8) return jpeg_probe(path, f, t);   // a JPEG (SOI marker)
    Parser P{f};
    if (hdr[0] == 'I' && hdr[1] == 'I') P.be = false;
    else if (hdr[0] == 'M' && hdr[1] == 'M') P.be = true;
    else { t.error = "not a TIFF"; return t; }
    t.big_endian = P.be;
    uint16_t magic = (uint16_t)P.num(hdr + 2, 2);
    uint64_t ifd;
    if (magic == 42) { ifd = P.num(hdr + 4, 4); }
    else if (magic == 43) { P.big = true; ifd = P.num(hdr + 8, 8); }
    else { t.error = "bad magic"; return t; }
    t.bigtiff = P.big;

    uint8_t cnt_buf[8];
    size_t cnt_sz = P.big ? 8 : 2, esz = P.big ? 20 : 12;
    if (f.read_at(ifd, cnt_buf, cnt_sz) < cnt_sz) { t.error = "bad IFD"; return t; }
    uint64_t n = P.num(cnt_buf, (int)cnt_sz);
    if (n == 0 || n > 4096) { t.error = "bad IFD count"; return t; }
    std::vector<uint8_t> ent(n * esz);
    if (f.read_at(ifd + cnt_sz, ent.data(), ent.size()) < ent.size()) { t.error = "bad IFD"; return t; }

    auto values = [&](const uint8_t* e, uint16_t type, uint64_t count, std::vector<uint64_t>& out) {
        size_t ts = Parser::type_size(type);
        if (!ts) return false;
        size_t total = ts * count, inl = P.big ? 8 : 4;
        const uint8_t* src = e + (P.big ? 12 : 8);
        std::vector<uint8_t> buf;
        if (total > inl) {
            uint64_t off = P.num(src, (int)inl);
            buf.resize(total);
            if (f.read_at(off, buf.data(), total) < total) return false;
            src = buf.data();
        }
        out.resize(count);
        for (uint64_t i = 0; i < count; ++i) out[i] = P.num(src + i * ts, (int)ts);
        return true;
    };

    std::vector<uint64_t> v;
    // an ASCII tag's text (type 2), trailing NULs and blanks trimmed
    auto ascii = [&](const uint8_t* e, uint16_t type, uint64_t count) {
        std::string s;
        std::vector<uint64_t> b;
        if (type != 2 || count == 0 || count > 256 || !values(e, type, count, b)) return s;
        for (uint64_t c : b) s.push_back((char)c);
        while (!s.empty() && (s.back() == '\0' || s.back() == ' ')) s.pop_back();
        return s;
    };
    std::string tiff_date, exif_date, exif_subsec;
    uint64_t exif_ifd = 0;
    bool tiled = false;
    for (uint64_t i = 0; i < n; ++i) {
        const uint8_t* e = ent.data() + i * esz;
        uint16_t tag = (uint16_t)P.num(e, 2), type = (uint16_t)P.num(e + 2, 2);
        uint64_t count = P.num(e + 4, P.big ? 8 : 4);
        if (tag == 306) { tiff_date = ascii(e, type, count); continue; }                                  // DateTime
        if (tag == 34665 && values(e, type, count, v) && !v.empty()) { exif_ifd = v[0]; continue; }     // the Exif IFD's offset
        if (tag == 34675 && count > 0 && count < (1u << 24)) {   // ICC profile: raw bytes
            size_t inl = P.big ? 8 : 4;
            const uint8_t* src = e + (P.big ? 12 : 8);
            t.icc.resize(count);
            if (count > inl) {
                uint64_t off = P.num(src, (int)inl);
                if (f.read_at(off, t.icc.data(), count) < count) t.icc.clear();
            } else std::memcpy(t.icc.data(), src, count);
            continue;
        }
        switch (tag) {
            case 256: case 257: case 258: case 259: case 262: case 273: case 277: case 278: case 279:
            case 284: case 322: case 323: case 324: case 325: case 339:
                if (!values(e, type, count, v)) { t.error = "bad tag " + std::to_string(tag); return t; }
                break;
            default: continue;
        }
        switch (tag) {
            case 256: t.width = (uint32_t)v[0]; break;
            case 257: t.height = (uint32_t)v[0]; break;
            case 258: t.bits = (uint16_t)v[0]; break;
            case 259: t.compression = (uint16_t)v[0]; break;
            case 262: t.photometric = (uint16_t)v[0]; break;
            case 273: t.strip_offsets = v; break;
            case 277: t.samples = (uint16_t)v[0]; break;
            case 279: t.strip_bytes = v; break;
            case 284: t.planar = (uint16_t)v[0]; break;
            case 339: t.sample_format = (uint16_t)v[0]; break;
            case 322: case 323: case 324: case 325: tiled = true; break;
            default: break;
        }
    }
    // the capture time from the Exif IFD (the GUI sorts frames by it); any failure here just leaves it empty
    if (exif_ifd && exif_ifd + cnt_sz < t.file_size) {
        uint8_t xc[8];
        if (f.read_at(exif_ifd, xc, cnt_sz) == cnt_sz) {
            uint64_t xn = P.num(xc, (int)cnt_sz);
            if (xn > 0 && xn <= 4096) {
                std::vector<uint8_t> xe(xn * esz);
                if (f.read_at(exif_ifd + cnt_sz, xe.data(), xe.size()) == xe.size())
                    for (uint64_t i = 0; i < xn; ++i) {
                        const uint8_t* e = xe.data() + i * esz;
                        uint16_t tag = (uint16_t)P.num(e, 2), type = (uint16_t)P.num(e + 2, 2);
                        uint64_t count = P.num(e + 4, P.big ? 8 : 4);
                        if (tag == 36867) exif_date = ascii(e, type, count);          // DateTimeOriginal
                        else if (tag == 37521) exif_subsec = ascii(e, type, count);   // SubSecTimeOriginal
                    }
            }
        }
    }
    t.datetime = !exif_date.empty() ? exif_date : tiff_date;
    if (!exif_date.empty() && !exif_subsec.empty()) t.datetime += "." + exif_subsec;
    if (!t.width || !t.height || !t.samples) { t.error = "missing size tags"; return t; }    if (t.samples != 1 && t.samples != 3 && t.samples != 4) { t.error = "unsupported sample count " + std::to_string(t.samples); return t; }
    if (t.bits == 0) t.bits = 8;
    if (t.bits != 8 && t.bits != 16) { t.error = "unsupported bits " + std::to_string(t.bits); return t; }
    if (t.sample_format != 1) { t.error = "unsupported sample format"; return t; }
    if (t.compression == 0) t.compression = 1;
    if (t.compression != 1 || t.planar != 1 || tiled) {
        // not readable straight from the file: Windows' TIFF decoder unpacks it into the buffer as chunky RGB
        // (LZW 5, ZIP 8 / 32946, PackBits 32773, with or without the predictor; tiles; planar samples)
        const uint16_t c = t.compression;
        if (c != 1 && c != 5 && c != 8 && c != 32946 && c != 32773) { t.error = "TIFF compression " + std::to_string(c) + " is not supported (use none, LZW or ZIP)"; return t; }
        if (t.photometric == 5) { t.error = "CMYK TIFF is not supported"; return t; }
        t.wic = true;
        t.samples = 3;
        t.planar = 1;
        t.big_endian = false;
        t.contiguous = true;
        t.data_offset = 0;
        t.data_bytes = t.pixel_bytes();
        t.strip_offsets = {0};
        t.strip_bytes = {t.data_bytes};
        return t;
    }
    if (t.strip_offsets.empty() || t.strip_offsets.size() != t.strip_bytes.size()) { t.error = "bad strips"; return t; }
    t.data_offset = t.strip_offsets[0];
    t.contiguous = true;
    for (size_t i = 0; i < t.strip_offsets.size(); ++i) {
        t.data_bytes += t.strip_bytes[i];
        if (i && t.strip_offsets[i - 1] + t.strip_bytes[i - 1] != t.strip_offsets[i]) t.contiguous = false;
    }
    if (t.data_bytes != t.pixel_bytes()) { t.error = "strip bytes do not match the image size"; return t; }
    if (t.data_offset + t.data_bytes > t.file_size) { t.error = "truncated file"; return t; }
    return t;
}

TiffRead tiff_read_pixels(const std::filesystem::path& path, const TiffInfo& info, uint8_t* dst, size_t cap) {
    TiffRead r;
    auto t0 = std::chrono::steady_clock::now();
    if (!info.ok()) { r.error = info.error; return r; }
    if (info.raw) return raw_read(path, info, dst, cap);                 // developed to 16-bit RGB, at offset 0
    if (info.jpeg || info.wic) return wic_read(path, info, dst, cap);   // decoded to RGB, at offset 0
    if (cap < tiff_buffer_capacity(info.pixel_bytes())) { r.error = "buffer too small"; return r; }

    if (info.contiguous && ((uintptr_t)dst % kReadAlign) == 0) {
        File f(path, FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_NO_BUFFERING);
        if (f.ok()) {
            uint64_t start = info.data_offset & ~(uint64_t)(kReadAlign - 1);
            size_t slack = (size_t)(info.data_offset - start);
            uint64_t end = (info.data_offset + info.data_bytes + kReadAlign - 1) & ~(uint64_t)(kReadAlign - 1);
            size_t want = (size_t)(end - start);
            size_t got = f.read_at(start, dst, want);
            // reading past EOF returns short: fine as long as the pixels are complete
            if (got >= slack + info.data_bytes) {
                r.data_offset_in_buf = slack;
                r.bytes = info.data_bytes;
                r.unbuffered = true;
                r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                return r;
            }
        }
        // fall through: the volume refused unbuffered I/O
    }
    File f(path);
    if (!f.ok()) { r.error = "cannot open"; return r; }
    size_t pos = 0;
    for (size_t i = 0; i < info.strip_offsets.size(); ++i) {
        size_t nb = (size_t)info.strip_bytes[i];
        if (f.read_at(info.strip_offsets[i], dst + pos, nb) < nb) { r.error = "short read"; return r; }
        pos += nb;
    }
    r.bytes = pos;
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

}  // namespace stackcore

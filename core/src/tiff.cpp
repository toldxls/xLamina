#include "stackcore/tiff.h"

#include <windows.h>

#include <algorithm>
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

TiffInfo tiff_probe(const std::filesystem::path& path) {
    TiffInfo t;
    File f(path);
    if (!f.ok()) { t.error = "cannot open"; return t; }
    t.file_size = f.size();
    uint8_t hdr[16];
    if (f.read_at(0, hdr, 16) < 8) { t.error = "too short"; return t; }
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
    for (uint64_t i = 0; i < n; ++i) {
        const uint8_t* e = ent.data() + i * esz;
        uint16_t tag = (uint16_t)P.num(e, 2), type = (uint16_t)P.num(e + 2, 2);
        uint64_t count = P.num(e + 4, P.big ? 8 : 4);
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
            case 322: case 323: case 324: case 325: t.error = "tiled TIFF not supported yet"; return t;
            default: break;
        }
    }
    if (!t.width || !t.height || !t.samples) { t.error = "missing size tags"; return t; }
    if (t.samples != 1 && t.samples != 3 && t.samples != 4) { t.error = "unsupported sample count " + std::to_string(t.samples); return t; }
    if (t.bits == 0) t.bits = 8;
    if (t.bits != 8 && t.bits != 16) { t.error = "unsupported bits " + std::to_string(t.bits); return t; }
    if (t.sample_format != 1) { t.error = "unsupported sample format"; return t; }
    if (t.planar != 1) { t.error = "planar TIFF not supported"; return t; }
    if (t.compression != 1) { t.error = "compressed TIFF (" + std::to_string(t.compression) + "): libtiff path not built yet"; return t; }
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

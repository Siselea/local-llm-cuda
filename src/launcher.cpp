// launcher.exe: Win32 GUI front-end for `engine.exe serve`.
//
// 2026-11-15 rewrite (UI round 2).  Design:
//   * "看得懂": every control has a visible label, a unit/meaning, and a tooltip
//     with the engine's own option help text.  The two sliders (GPU expert
//     layers, context length) show a live estimate of the VRAM / host-RAM / KV
//     cost while dragging.
//   * "全部 CLI 选项": the engine's configuration table is the single source of
//     truth.  `engine.exe serve --print-config --json` prints every option
//     (key / cli / type / default / range / help / effective value); this GUI
//     generates one control per option inside the collapsible "高级" panel, so a
//     new engine option shows up in the GUI without touching the GUI.
//   * The curated main-panel controls map 1:1 onto concrete engine options
//     (see the IDC_* comments); everything else lives in the advanced panel.
//   * Estimates mirror the byte accounting of src/planner.cpp; the engine's own
//     placement decider remains authoritative (its decision is echoed from the
//     engine log into the status area once the model is loading).
//
// Why a separate exe: engine.exe must stay a CONSOLE subsystem binary (scripts,
// CI and the HTTP test suites capture its output).  The launcher is its own
// SUBSYSTEM:WINDOWS process with no console window; it only shells out to
// `engine.exe models --json` / `info --json` / `serve --print-config --json`
// and `engine.exe serve ...` via CreateProcess, and talks to the running server
// through WinHTTP (/health, /shutdown).
//
// No third-party dependencies: plain Win32 controls + GDI custom drawing +
// WinHTTP.  commctrl32 is loaded dynamically only for the tooltip window class.
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commctrl.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <uxtheme.h>
#include <winhttp.h>
#include <tlhelp32.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <mutex>
#include <functional>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
std::wstring widen(const std::string& s) {
  if (s.empty()) return std::wstring();
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
  if (n <= 0) return std::wstring();
  std::wstring out(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), out.data(), n);
  return out;
}

std::string narrow(const std::wstring& s) {
  if (s.empty()) return std::string();
  const int n = WideCharToMultiByte(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0,
                                    nullptr, nullptr);
  if (n <= 0) return std::string();
  std::string out(static_cast<size_t>(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), out.data(), n, nullptr,
                      nullptr);
  return out;
}

std::wstring wfmt(const wchar_t* fmt, ...) {
  wchar_t buf[2048];
  va_list ap;
  va_start(ap, fmt);
  const int n = vswprintf(buf, 2048, fmt, ap);
  va_end(ap);
  if (n <= 0) return std::wstring();
  return std::wstring(buf, static_cast<size_t>(n));
}

std::wstring wnum(long long v) { return wfmt(L"%lld", v); }
std::wstring wnum_u(uint64_t v) { return wfmt(L"%llu", static_cast<unsigned long long>(v)); }

// Human readable size.  The UI says GB/MB (1024-based) because that is what
// users compare against the Windows task manager / nvidia-smi numbers.
std::wstring fmt_size(uint64_t bytes) {
  const double b = static_cast<double>(bytes);
  if (b >= 1073741824.0) return wfmt(L"%.2f GB", b / 1073741824.0);
  if (b >= 1048576.0) return wfmt(L"%.0f MB", b / 1048576.0);
  if (b >= 1024.0) return wfmt(L"%.1f KB", b / 1024.0);
  return wfmt(L"%.0f B", b);
}

std::wstring fmt_gib2(double gib) { return wfmt(L"%.2f GB", gib); }

std::wstring trim(const std::wstring& s) {
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == L' ' || s[a] == L'\t' || s[a] == L'\r' || s[a] == L'\n')) ++a;
  while (b > a && (s[b - 1] == L' ' || s[b - 1] == L'\t' || s[b - 1] == L'\r' || s[b - 1] == L'\n'))
    --b;
  return s.substr(a, b - a);
}

std::wstring lower_w(std::wstring s) {
  for (wchar_t& c : s)
    if (c >= L'A' && c <= L'Z') c = static_cast<wchar_t>(c - L'A' + L'a');
  return s;
}

bool parse_ll(const std::wstring& text, long long* out) {
  const std::wstring s = trim(text);
  if (s.empty()) return false;
  wchar_t* end = nullptr;
  const long long v = wcstoll(s.c_str(), &end, 10);
  if (end == nullptr || *end != L'\0') return false;
  if (out != nullptr) *out = v;
  return true;
}

bool parse_dd(const std::wstring& text, double* out) {
  const std::wstring s = trim(text);
  if (s.empty()) return false;
  wchar_t* end = nullptr;
  const double v = wcstod(s.c_str(), &end);
  if (end == nullptr || *end != L'\0') return false;
  if (out != nullptr) *out = v;
  return true;
}

std::wstring quote_arg(const std::wstring& s) { return L"\"" + s + L"\""; }

std::wstring join_path(const std::wstring& dir, const std::wstring& rel) {
  if (dir.empty()) return rel;
  if (dir.back() == L'\\' || dir.back() == L'/') return dir + rel;
  return dir + L"\\" + rel;
}

bool file_exists(const std::wstring& path) {
  const DWORD a = GetFileAttributesW(path.c_str());
  return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

std::wstring exe_dir() {
  wchar_t buf[4096];
  const DWORD n = GetModuleFileNameW(nullptr, buf, 4096);
  if (n == 0 || n >= 4096) return std::wstring();
  std::wstring s(buf, n);
  const size_t pos = s.find_last_of(L"\\/");
  return pos == std::wstring::npos ? std::wstring() : s.substr(0, pos);
}

std::string read_file_utf8(const std::wstring& path) {
  HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return std::string();
  std::string out;
  char buf[65536];
  DWORD got = 0;
  while (ReadFile(h, buf, sizeof(buf), &got, nullptr) && got > 0) out.append(buf, got);
  CloseHandle(h);
  return out;
}

bool write_file_utf8(const std::wstring& path, const std::string& text) {
  HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return false;
  DWORD written = 0;
  bool ok = true;
  size_t off = 0;
  while (off < text.size()) {
    const DWORD chunk = static_cast<DWORD>(std::min<size_t>(text.size() - off, 1u << 20));
    if (!WriteFile(h, text.data() + off, chunk, &written, nullptr) || written != chunk) {
      ok = false;
      break;
    }
    off += written;
  }
  CloseHandle(h);
  return ok;
}

// ---------------------------------------------------------------------------
// Minimal JSON parser (the launcher only reads JSON produced by engine.exe:
// `models --json`, `info --json`, `serve --print-config --json`).  It supports
// objects, arrays, strings (with \u escapes), numbers, booleans and null - no
// streaming, no depth limit beyond what a hostile file could force, but the
// input is our own engine's output.
// ---------------------------------------------------------------------------
struct JVal {
  enum class T { Null, Bool, Num, Str, Arr, Obj } type = T::Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<JVal> arr;
  std::vector<std::pair<std::string, JVal>> obj;

  const JVal* get(const char* key) const {
    if (type != T::Obj) return nullptr;
    for (const auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  bool is_num() const { return type == T::Num; }
  bool is_str() const { return type == T::Str; }
  bool is_arr() const { return type == T::Arr; }
  bool is_obj() const { return type == T::Obj; }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : s_(text) {}

  bool parse(JVal* out) {
    skip_ws();
    if (!value(out)) return false;
    skip_ws();
    return i_ >= s_.size();
  }

 private:
  bool value(JVal* out) {
    skip_ws();
    if (i_ >= s_.size()) return false;
    const char c = s_[i_];
    switch (c) {
      case '{': return object(out);
      case '[': return array(out);
      case '"': out->type = JVal::T::Str; return string(&out->str);
      case 't':
        if (s_.compare(i_, 4, "true") == 0) { i_ += 4; out->type = JVal::T::Bool; out->b = true; return true; }
        return false;
      case 'f':
        if (s_.compare(i_, 5, "false") == 0) { i_ += 5; out->type = JVal::T::Bool; out->b = false; return true; }
        return false;
      case 'n':
        if (s_.compare(i_, 4, "null") == 0) { i_ += 4; out->type = JVal::T::Null; return true; }
        return false;
      default: return number(out);
    }
  }

  bool number(JVal* out) {
    const size_t start = i_;
    if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
    while (i_ < s_.size() && ((s_[i_] >= '0' && s_[i_] <= '9') || s_[i_] == '.' || s_[i_] == 'e' ||
                              s_[i_] == 'E' || s_[i_] == '-' || s_[i_] == '+'))
      ++i_;
    if (i_ == start) return false;
    out->type = JVal::T::Num;
    out->num = std::strtod(s_.substr(start, i_ - start).c_str(), nullptr);
    return true;
  }

  bool string(std::string* out) {
    if (i_ >= s_.size() || s_[i_] != '"') return false;
    ++i_;
    out->clear();
    while (i_ < s_.size()) {
      const unsigned char c = static_cast<unsigned char>(s_[i_++]);
      if (c == '"') return true;
      if (c != '\\') { out->push_back(static_cast<char>(c)); continue; }
      if (i_ >= s_.size()) return false;
      const char esc = s_[i_++];
      switch (esc) {
        case '"': out->push_back('"'); break;
        case '\\': out->push_back('\\'); break;
        case '/': out->push_back('/'); break;
        case 'b': out->push_back('\b'); break;
        case 'f': out->push_back('\f'); break;
        case 'n': out->push_back('\n'); break;
        case 'r': out->push_back('\r'); break;
        case 't': out->push_back('\t'); break;
        case 'u': {
          unsigned cp = 0;
          if (!hex4(&cp)) return false;
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            // surrogate pair
            if (i_ + 1 < s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
              i_ += 2;
              unsigned lo = 0;
              if (!hex4(&lo)) return false;
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            }
          }
          append_utf8(out, cp);
          break;
        }
        default: return false;
      }
    }
    return false;
  }

  bool hex4(unsigned* out) {
    if (i_ + 4 > s_.size()) return false;
    unsigned v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = s_[i_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
      else return false;
    }
    *out = v;
    return true;
  }

  static void append_utf8(std::string* out, unsigned cp) {
    if (cp < 0x80) {
      out->push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  bool array(JVal* out) {
    out->type = JVal::T::Arr;
    ++i_;  // [
    skip_ws();
    if (i_ < s_.size() && s_[i_] == ']') { ++i_; return true; }
    while (true) {
      JVal v;
      if (!value(&v)) return false;
      out->arr.push_back(std::move(v));
      skip_ws();
      if (i_ >= s_.size()) return false;
      if (s_[i_] == ',') { ++i_; continue; }
      if (s_[i_] == ']') { ++i_; return true; }
      return false;
    }
  }

  bool object(JVal* out) {
    out->type = JVal::T::Obj;
    ++i_;  // {
    skip_ws();
    if (i_ < s_.size() && s_[i_] == '}') { ++i_; return true; }
    while (true) {
      skip_ws();
      std::string key;
      if (!string(&key)) return false;
      skip_ws();
      if (i_ >= s_.size() || s_[i_] != ':') return false;
      ++i_;
      JVal v;
      if (!value(&v)) return false;
      out->obj.emplace_back(std::move(key), std::move(v));
      skip_ws();
      if (i_ >= s_.size()) return false;
      if (s_[i_] == ',') { ++i_; continue; }
      if (s_[i_] == '}') { ++i_; return true; }
      return false;
    }
  }

  void skip_ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\r' || s_[i_] == '\n'))
      ++i_;
  }

  const std::string& s_;
  size_t i_ = 0;
};

const JVal* jget(const JVal* v, const char* key) { return v == nullptr ? nullptr : v->get(key); }

std::string jstr(const JVal* v, const char* key, const std::string& fallback = std::string()) {
  const JVal* x = jget(v, key);
  if (x == nullptr || !x->is_str()) return fallback;
  return x->str;
}

long long jint(const JVal* v, const char* key, long long fallback = 0) {
  const JVal* x = jget(v, key);
  if (x == nullptr || !x->is_num()) return fallback;
  return static_cast<long long>(x->num);
}

uint64_t juint(const JVal* v, const char* key, uint64_t fallback = 0) {
  const JVal* x = jget(v, key);
  if (x == nullptr || !x->is_num()) return fallback;
  if (x->num < 0) return fallback;
  return static_cast<uint64_t>(x->num);
}

bool jbool(const JVal* v, const char* key, bool fallback = false) {
  const JVal* x = jget(v, key);
  if (x == nullptr || x->type != JVal::T::Bool) return fallback;
  return x->b;
}

double jnum(const JVal* v, const char* key, double fallback = 0.0) {
  const JVal* x = jget(v, key);
  if (x == nullptr || !x->is_num()) return fallback;
  return x->num;
}

}  // namespace
// ---------------------------------------------------------------------------
// Shared state (declared early so every part of this translation unit sees it)
// ---------------------------------------------------------------------------
namespace {

constexpr wchar_t kAppTitle[] = L"本地推理引擎 · Launcher";
constexpr wchar_t kAppClass[] = L"EngineLauncherWindowV2";

HWND g_hwnd = nullptr;
std::wstring g_launcher_dir;
std::wstring g_engine;  // full path to engine.exe
int g_dpi = 96;

struct Palette {
  COLORREF bg, panel, panel2, text, dim, accent, accent_dim, danger, warn, border, input_bg,
      toggle_off, header_line;
};
Palette g_pal;
bool g_dark = true;
int g_scroll = 0;
int g_content_h = 0;   // total scrollable height (content coordinates)
int g_view_h = 0;      // viewport height (client height minus bottom bar)
int g_client_w = 0;
int g_client_h = 0;

HFONT g_font = nullptr, g_font_small = nullptr, g_font_bold = nullptr, g_font_title = nullptr;
HBRUSH g_br_bg = nullptr, g_br_panel = nullptr, g_br_panel2 = nullptr, g_br_input = nullptr,
       g_br_border = nullptr;

int S(int v) { return MulDiv(v, g_dpi, 96); }

bool system_is_dark() {
  HKEY key = nullptr;
  bool dark = true;  // fallback: dark (requested default when the registry can't be read)
  if (RegOpenKeyExW(HKEY_CURRENT_USER,
                    L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize", 0,
                    KEY_QUERY_VALUE, &key) == ERROR_SUCCESS) {
    DWORD value = 1, size = sizeof(value), type = 0;
    if (RegQueryValueExW(key, L"AppsUseLightTheme", nullptr, &type,
                         reinterpret_cast<LPBYTE>(&value), &size) == ERROR_SUCCESS &&
        type == REG_DWORD)
      dark = value == 0;
    RegCloseKey(key);
  }
  return dark;
}

// ---------------------------------------------------------------------------
// DPI awareness (Per-Monitor v2 when available; fall back to system aware)
// ---------------------------------------------------------------------------
void enable_dpi_awareness() {
  HMODULE user32 = GetModuleHandleW(L"user32.dll");
  if (user32 != nullptr) {
    using SetCtxFn = BOOL(WINAPI*)(HANDLE);
    using SetSysFn = BOOL(WINAPI*)(void);
    auto set_ctx = reinterpret_cast<SetCtxFn>(GetProcAddress(user32, "SetProcessDpiAwarenessContext"));
    if (set_ctx != nullptr) {
      // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == (HANDLE)-4
      if (set_ctx(reinterpret_cast<HANDLE>(-4))) return;
    }
    auto set_sys = reinterpret_cast<SetSysFn>(GetProcAddress(user32, "SetProcessDPIAware"));
    if (set_sys != nullptr && set_sys()) return;
  }
}

int query_dpi(HWND hwnd) {
  HMODULE user32 = GetModuleHandleW(L"user32.dll");
  if (user32 != nullptr) {
    using GetDpiFn = UINT(WINAPI*)(HWND);
    auto get_dpi = reinterpret_cast<GetDpiFn>(GetProcAddress(user32, "GetDpiForWindow"));
    if (get_dpi != nullptr && hwnd != nullptr) {
      const UINT d = get_dpi(hwnd);
      if (d >= 72 && d <= 480) return static_cast<int>(d);
    }
  }
  HDC screen = GetDC(nullptr);
  const int d = screen != nullptr ? GetDeviceCaps(screen, LOGPIXELSY) : 96;
  if (screen != nullptr) ReleaseDC(nullptr, screen);
  return (d >= 72 && d <= 480) ? d : 96;
}

// ---------------------------------------------------------------------------
// Theme / fonts / brushes
// ---------------------------------------------------------------------------
void destroy_theme_objects() {
  HBRUSH* brushes[] = {&g_br_bg, &g_br_panel, &g_br_panel2, &g_br_input, &g_br_border};
  for (HBRUSH* b : brushes) {
    if (*b != nullptr) DeleteObject(*b);
    *b = nullptr;
  }
  HFONT* fonts[] = {&g_font, &g_font_small, &g_font_bold, &g_font_title};
  for (HFONT* f : fonts) {
    if (*f != nullptr) DeleteObject(*f);
    *f = nullptr;
  }
}

HFONT make_font(int pt, int weight) {
  const int height = -MulDiv(pt, g_dpi, 72);
  HFONT f = CreateFontW(height, 0, 0, 0, weight, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                        DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei UI");
  if (f == nullptr) {
    f = CreateFontW(height, 0, 0, 0, weight, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                    OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                    DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
  }
  return f;
}

void init_theme(int theme_mode) {  // 0 = follow system, 1 = dark, 2 = light
  g_dark = (theme_mode == 1) ? true : (theme_mode == 2 ? false : system_is_dark());
  if (g_dark) {
    g_pal = {RGB(18, 20, 25),    RGB(27, 30, 37),    RGB(35, 39, 48),    RGB(235, 238, 244),
             RGB(150, 157, 170), RGB(96, 154, 255),  RGB(58, 104, 200),  RGB(240, 104, 104),
             RGB(236, 186, 90),  RGB(53, 59, 71),    RGB(21, 24, 30),    RGB(78, 86, 101),
             RGB(68, 76, 92)};
  } else {
    g_pal = {RGB(246, 247, 250),  RGB(255, 255, 255),  RGB(238, 241, 246), RGB(28, 31, 38),
             RGB(108, 114, 126),  RGB(36, 99, 220),    RGB(24, 70, 160),   RGB(196, 62, 62),
             RGB(168, 112, 12),   RGB(215, 220, 228),  RGB(255, 255, 255), RGB(186, 192, 203),
             RGB(226, 230, 237)};
  }
  destroy_theme_objects();
  g_br_bg = CreateSolidBrush(g_pal.bg);
  g_br_panel = CreateSolidBrush(g_pal.panel);
  g_br_panel2 = CreateSolidBrush(g_pal.panel2);
  g_br_input = CreateSolidBrush(g_pal.input_bg);
  g_br_border = CreateSolidBrush(g_pal.border);
  g_font = make_font(10, FW_NORMAL);
  g_font_small = make_font(9, FW_NORMAL);
  g_font_bold = make_font(10, FW_SEMIBOLD);
  g_font_title = make_font(13, FW_SEMIBOLD);
}

void set_dark_title_bar(HWND hwnd) {
  const BOOL dark = g_dark ? TRUE : FALSE;
  // 20 = DWMWA_USE_IMMERSIVE_DARK_MODE on current Windows 10/11 builds.
  DwmSetWindowAttribute(hwnd, 20, &dark, sizeof(dark));
  const COLORREF caption = g_pal.bg;
  DwmSetWindowAttribute(hwnd, 35 /*DWMWA_CAPTION_COLOR*/, &caption, sizeof(caption));
  const COLORREF border = g_pal.border;
  DwmSetWindowAttribute(hwnd, 34 /*DWMWA_BORDER_COLOR*/, &border, sizeof(border));
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
void fill_rect(HDC dc, const RECT& r, COLORREF c) {
  HBRUSH br = CreateSolidBrush(c);
  FillRect(dc, &r, br);
  DeleteObject(br);
}

void fill_rect_bg(HDC dc, const RECT& r, HBRUSH br) { FillRect(dc, &r, br); }

void frame_rect(HDC dc, const RECT& r, COLORREF c) {
  HBRUSH br = CreateSolidBrush(c);
  FrameRect(dc, &r, br);
  DeleteObject(br);
}

void rounded_fill(HDC dc, const RECT& r, int radius, COLORREF fill, COLORREF border) {
  HBRUSH br = CreateSolidBrush(fill);
  HPEN pen = CreatePen(PS_SOLID, 1, border);
  const HGDIOBJ old_br = SelectObject(dc, br);
  const HGDIOBJ old_pen = SelectObject(dc, pen);
  RoundRect(dc, r.left, r.top, r.right, r.bottom, radius * 2, radius * 2);
  SelectObject(dc, old_br);
  SelectObject(dc, old_pen);
  DeleteObject(br);
  DeleteObject(pen);
}

void hline(HDC dc, int x0, int x1, int y, COLORREF c) {
  HPEN pen = CreatePen(PS_SOLID, 1, c);
  const HGDIOBJ old_pen = SelectObject(dc, pen);
  MoveToEx(dc, x0, y, nullptr);
  LineTo(dc, x1, y);
  SelectObject(dc, old_pen);
  DeleteObject(pen);
}

void draw_text(HDC dc, const std::wstring& text, const RECT& r, UINT flags, COLORREF color,
               HFONT font) {
  const HGDIOBJ old_font = SelectObject(dc, font);
  SetBkMode(dc, TRANSPARENT);
  SetTextColor(dc, color);
  RECT rr = r;
  DrawTextW(dc, text.c_str(), static_cast<int>(text.size()), &rr, flags);
  SelectObject(dc, old_font);
}

std::wstring ellipsize(HDC dc, const std::wstring& text, int width, HFONT font) {
  const HGDIOBJ old_font = SelectObject(dc, font);
  SIZE sz{};
  GetTextExtentPoint32W(dc, text.c_str(), static_cast<int>(text.size()), &sz);
  SelectObject(dc, old_font);
  if (sz.cx <= width || text.empty()) return text;
  std::wstring s = text;
  while (!s.empty()) {
    s.pop_back();
    const HGDIOBJ f2 = SelectObject(dc, font);
    GetTextExtentPoint32W(dc, (s + L"…").c_str(), static_cast<int>(s.size() + 1), &sz);
    SelectObject(dc, f2);
    if (sz.cx <= width) return s + L"…";
  }
  return L"…";
}

int text_width(HDC dc, const std::wstring& text, HFONT font) {
  const HGDIOBJ old_font = SelectObject(dc, font);
  SIZE sz{};
  GetTextExtentPoint32W(dc, text.c_str(), static_cast<int>(text.size()), &sz);
  SelectObject(dc, old_font);
  return sz.cx;
}

// ---------------------------------------------------------------------------
// Label colour roles (WM_CTLCOLORSTATIC)
// ---------------------------------------------------------------------------
enum class Role { Normal, Dim, Accent, Danger, Warn };
std::map<HWND, Role> g_label_role;

void set_label_role(HWND h, Role r) {
  g_label_role[h] = r;
  if (h != nullptr) InvalidateRect(h, nullptr, TRUE);
}

Role role_of(HWND h) {
  const auto it = g_label_role.find(h);
  return it == g_label_role.end() ? Role::Normal : it->second;
}

COLORREF role_color(Role r) {
  switch (r) {
    case Role::Dim: return g_pal.dim;
    case Role::Accent: return g_pal.accent;
    case Role::Danger: return g_pal.danger;
    case Role::Warn: return g_pal.warn;
    case Role::Normal:
    default: return g_pal.text;
  }
}

// ---------------------------------------------------------------------------
// Toggle (switch) state.  BS_OWNERDRAW buttons store no check state on their
// own (BM_SETCHECK is ignored for them), so keep it here - the owner-draw code
// and the command handlers both read this map.
// ---------------------------------------------------------------------------
std::map<HWND, bool> g_toggle_state;

bool toggle_is_on(HWND h) {
  const auto it = g_toggle_state.find(h);
  return it != g_toggle_state.end() && it->second;
}

void toggle_set(HWND h, bool on) {
  if (h == nullptr) return;
  g_toggle_state[h] = on;
  InvalidateRect(h, nullptr, TRUE);
}

bool toggle_flip(HWND h) {
  const bool now = !toggle_is_on(h);
  toggle_set(h, now);
  return now;
}

// ---------------------------------------------------------------------------
// Tooltips (comctl32 is loaded dynamically: no extra import library needed)
// ---------------------------------------------------------------------------
HWND g_tip = nullptr;
std::vector<std::wstring*> g_tip_owned;
// control id -> tooltip text, dumped into launcher-ui-map.json so the
// acceptance test can verify the mandated risk wording without cross-process
// TTM_GETTEXT (WM_USER messages with pointers are not marshalled).
std::map<int, std::wstring> g_tip_registry;
bool g_comctl_ready = false;

bool ensure_comctl32() {
  static bool tried = false;
  static bool ok = false;
  if (tried) return ok;
  tried = true;
  HMODULE mod = LoadLibraryW(L"comctl32.dll");
  if (mod == nullptr) return ok = false;
  using InitFn = BOOL(WINAPI*)(LPINITCOMMONCONTROLSEX);
  auto init = reinterpret_cast<InitFn>(GetProcAddress(mod, "InitCommonControlsEx"));
  if (init != nullptr) {
    INITCOMMONCONTROLSEX icc{};
    icc.dwSize = sizeof(icc);
    icc.dwICC = ICC_BAR_CLASSES | ICC_WIN95_CLASSES | ICC_STANDARD_CLASSES;
    init(&icc);
  }
  ok = true;
  return ok;
}

void init_tooltips(HWND owner) {
  if (!ensure_comctl32()) return;
  g_tip = CreateWindowExW(WS_EX_TOPMOST, TOOLTIPS_CLASSW, nullptr,
                          WS_POPUP | TTS_ALWAYSTIP | TTS_NOPREFIX, CW_USEDEFAULT, CW_USEDEFAULT,
                          CW_USEDEFAULT, CW_USEDEFAULT, owner, nullptr, GetModuleHandleW(nullptr),
                          nullptr);
  if (g_tip == nullptr) return;
  SetWindowPos(g_tip, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
  SendMessageW(g_tip, TTM_SETMAXTIPWIDTH, 0, S(520));
  SendMessageW(g_tip, TTM_SETDELAYTIME, TTDT_INITIAL, MAKELPARAM(S(500), 0));
}

void add_tip(HWND control, const std::wstring& text) {
  if (g_tip == nullptr || control == nullptr || text.empty()) return;
  std::wstring* owned = new std::wstring(text);
  g_tip_owned.push_back(owned);
  TOOLINFOW ti{};
  // Classic comctl32 (no v6 manifest in this build) rejects a TOOLINFO larger
  // than its V2 struct: lpReserved made sizeof() 72 and TTM_ADDTOOLW returned 0.
  // V2 size (up to lParam) is accepted by both classic and v6 comctl32.
#ifdef TTTOOLINFOW_V2_SIZE
  ti.cbSize = TTTOOLINFOW_V2_SIZE;
#else
  ti.cbSize = static_cast<UINT>(sizeof(ti) - sizeof(void*));
#endif
  ti.uFlags = TTF_IDISHWND | TTF_SUBCLASS;
  ti.hwnd = g_hwnd;
  ti.uId = reinterpret_cast<UINT_PTR>(control);
  ti.lpszText = const_cast<wchar_t*>(owned->c_str());
  if (SendMessageW(g_tip, TTM_ADDTOOLW, 0, reinterpret_cast<LPARAM>(&ti)) != 0)
    g_tip_registry[GetDlgCtrlID(control)] = text;
}

// ---------------------------------------------------------------------------
// Subprocess helpers (`engine.exe models --json`, `info --json`, ...)
// ---------------------------------------------------------------------------
struct ProcResult {
  bool launched = false;
  DWORD exit_code = 0;
  std::string out;  // stdout+stderr, UTF-8
};

ProcResult run_capture(const std::wstring& exe, const std::wstring& args_utf16,
                       const std::wstring& work_dir, DWORD timeout_ms = 60000,
                       const std::function<void(const std::string&)>& on_chunk = nullptr) {
  ProcResult result;
  if (exe.empty()) return result;
  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  HANDLE read_pipe = nullptr, write_pipe = nullptr;
  if (!CreatePipe(&read_pipe, &write_pipe, &sa, 1 << 20)) return result;
  SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);
  HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
  si.wShowWindow = SW_HIDE;
  si.hStdOutput = write_pipe;
  si.hStdError = write_pipe;
  si.hStdInput = (nul == INVALID_HANDLE_VALUE) ? nullptr : nul;
  PROCESS_INFORMATION pi{};
  std::wstring cmd = quote_arg(exe) + L" " + args_utf16;
  const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                                 nullptr, work_dir.empty() ? nullptr : work_dir.c_str(), &si, &pi);
  CloseHandle(write_pipe);
  if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
  if (!ok) {
    CloseHandle(read_pipe);
    return result;
  }
  result.launched = true;
  CloseHandle(pi.hThread);
  const ULONGLONG start = GetTickCount64();
  for (;;) {
    DWORD avail = 0;
    if (PeekNamedPipe(read_pipe, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) {
      char buf[65536];
      DWORD got = 0;
      if (ReadFile(read_pipe, buf, sizeof(buf), &got, nullptr) && got > 0) {
        result.out.append(buf, got);
        if (on_chunk) on_chunk(std::string(buf, got));
        continue;
      }
    }
    if (WaitForSingleObject(pi.hProcess, 15) == WAIT_OBJECT_0) break;
    if (timeout_ms != 0 && GetTickCount64() - start > timeout_ms) break;
  }
  // drain
  for (;;) {
    DWORD avail = 0;
    if (!PeekNamedPipe(read_pipe, nullptr, 0, nullptr, &avail, nullptr) || avail == 0) break;
    char buf[65536];
    DWORD got = 0;
    if (!ReadFile(read_pipe, buf, sizeof(buf), &got, nullptr) || got == 0) break;
    result.out.append(buf, got);
  }
  if (on_chunk) on_chunk(std::string());  // flush signal
  WaitForSingleObject(pi.hProcess, 5000);
  GetExitCodeProcess(pi.hProcess, &result.exit_code);
  CloseHandle(pi.hProcess);
  CloseHandle(read_pipe);
  return result;
}

std::string engine_capture(const std::wstring& args, DWORD timeout_ms = 60000) {
  const ProcResult r = run_capture(g_engine, args, g_launcher_dir, timeout_ms);
  return r.out;
}

// ---------------------------------------------------------------------------
// HTTP helpers (WinHTTP): GET /health, POST /shutdown
// ---------------------------------------------------------------------------
struct HttpResponse {
  bool ok = false;
  int status = 0;
  std::string body;
};

HttpResponse http_request(int port, const wchar_t* method, const wchar_t* path,
                          DWORD timeout_ms, const wchar_t* extra_headers = nullptr) {
  HttpResponse r;
  if (port <= 0) return r;
  HINTERNET session = WinHttpOpen(L"engine-launcher/2.0", WINHTTP_ACCESS_TYPE_NO_PROXY,
                                  WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (session == nullptr) return r;
  WinHttpSetTimeouts(session, static_cast<int>(timeout_ms), static_cast<int>(timeout_ms),
                     static_cast<int>(timeout_ms), static_cast<int>(timeout_ms));
  HINTERNET connect = WinHttpConnect(session, L"127.0.0.1", static_cast<INTERNET_PORT>(port), 0);
  if (connect == nullptr) {
    WinHttpCloseHandle(session);
    return r;
  }
  HINTERNET request = WinHttpOpenRequest(connect, method, path, nullptr, WINHTTP_NO_REFERER,
                                         WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
  if (request != nullptr) {
    if (WinHttpSendRequest(request, extra_headers, extra_headers == nullptr ? 0 : -1L, nullptr, 0,
                           0, 0) &&
        WinHttpReceiveResponse(request, nullptr)) {
      DWORD code = 0, size = sizeof(code);
      WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                          WINHTTP_HEADER_NAME_BY_INDEX, &code, &size, WINHTTP_NO_HEADER_INDEX);
      r.status = static_cast<int>(code);
      DWORD avail = 0;
      while (WinHttpQueryDataAvailable(request, &avail) && avail > 0) {
        std::string chunk(avail, '\0');
        DWORD read = 0;
        if (!WinHttpReadData(request, chunk.data(), avail, &read) || read == 0) break;
        r.body.append(chunk.data(), read);
      }
      r.ok = r.status >= 200 && r.status < 300;
    }
    WinHttpCloseHandle(request);
  }
  WinHttpCloseHandle(connect);
  WinHttpCloseHandle(session);
  return r;
}

bool health_ok(int port, DWORD timeout_ms = 1200) {
  // Connection: close is deliberate: the engine's HTTP server serves one
  // connection at a time and blocks reading a kept-alive socket, so a pooled
  // probe connection would make the next client (browser / test) time out.
  return http_request(port, L"GET", L"/health", timeout_ms, L"Connection: close\r\n").ok;
}

}  // namespace
// ---------------------------------------------------------------------------
// Control IDs (stable: build/test_launcher.py drives the UI through these)
// ---------------------------------------------------------------------------
namespace {

enum : int {
  IDC_MODEL_COMBO = 100,
  IDC_MODEL_AUTO = 101,
  IDC_REFRESH = 102,
  IDC_MODEL_INFO = 103,
  IDC_SCAN_STATUS = 104,
  IDC_GPU_SLIDER = 110,
  IDC_GPU_AUTO = 111,
  IDC_GPU_TEXT = 112,
  IDC_CTX_SLIDER = 120,
  IDC_CTX_TEXT = 121,
  IDC_CTX_EDIT = 122,
  IDC_CTX_AUTO = 123,
  IDC_KV_COMBO = 130,
  IDC_KV_TEXT = 131,
  IDC_SSD_TOGGLE = 140,
  IDC_SSD_NOTE = 141,
  IDC_DEV_ROUTE = 142,
  IDC_LAYER_GRAPH = 143,
  IDC_CPU_INT8 = 144,
  IDC_WIN_PRIORITY = 145,
  IDC_THINKING_LEN = 146,
  IDC_VRAM_AUTO = 147,       // opt-in post-load GPU-layer promotion
  IDC_VRAM_AUTO_WARN = 148,  // red risk note, only visible while it is on
  IDC_THREADS_COMBO = 150,
  IDC_WIN_AFFINITY = 149,
  IDC_THREADS_HINT = 151,
  IDC_PORT_EDIT = 160,
  IDC_HOST_EDIT = 161,
  IDC_AUTO_BROWSER = 162,
  IDC_PRESET_FAST = 170,
  IDC_PRESET_LOWVRAM = 171,
  IDC_PRESET_BALANCED = 172,
  IDC_ADV_TOGGLE = 180,
  IDC_ADV_SEARCH = 181,
  IDC_ADV_COUNT = 182,
  IDC_START = 190,
  IDC_STOP = 191,
  IDC_OPEN_PAGE = 192,
  IDC_OPEN_LOG = 193,
  IDC_STATUS = 200,
  IDC_URL = 201,
  IDC_LOG_INFO = 202,
  IDC_ARGS_SUMMARY = 203,
  IDC_ENGINE_PLACEMENT = 204,
  IDC_SCHEMA_INFO = 205,
  IDC_MODEL_TITLE = 206,
  IDC_FOOTER = 207,
  IDC_EST_SUMMARY = 208,
  // Dynamic controls of the advanced panel: IDC_ADV_BASE + option index.
  IDC_ADV_BASE = 1000,
};

// ---------------------------------------------------------------------------
// Engine configuration schema (from `engine.exe serve --print-config --json`)
// ---------------------------------------------------------------------------
struct OptDef {
  std::string key, cli, env, type, def, help;
  bool takes_value = false;
  bool invert_env = false;
  double minv = 0, maxv = 0;
  std::string value;      // effective value at launcher start (default/env)
  std::string source;     // default / env / config / cli
  bool has_cli = false;
};
std::vector<OptDef> g_opts;
std::map<std::string, int> g_opt_index;
bool g_schema_loaded = false;
std::wstring g_schema_error;

const OptDef* find_opt(const std::string& key) {
  const auto it = g_opt_index.find(key);
  return it == g_opt_index.end() ? nullptr : &g_opts[static_cast<size_t>(it->second)];
}

// Current UI value per option key.  Booleans keep `b`; everything else `text`.
struct OptState {
  std::string text;
  bool b = false;
  bool is_bool = false;
  bool dirty = false;  // changed by the user (differs from the start-up value)
};
std::map<std::string, OptState> g_opt_state;

std::string opt_text(const std::string& key, const std::string& fallback = std::string()) {
  const auto it = g_opt_state.find(key);
  if (it == g_opt_state.end()) return fallback;
  return it->second.is_bool ? (it->second.b ? "true" : "false") : it->second.text;
}

long long opt_int(const std::string& key, long long fallback) {
  const auto it = g_opt_state.find(key);
  if (it == g_opt_state.end()) return fallback;
  if (it->second.is_bool) return it->second.b ? 1 : 0;
  wchar_t* end = nullptr;
  const std::wstring w = widen(it->second.text);
  const long long v = wcstoll(w.c_str(), &end, 10);
  if (end == w.c_str()) return fallback;
  return v;
}

double opt_double(const std::string& key, double fallback) {
  const auto it = g_opt_state.find(key);
  if (it == g_opt_state.end()) return fallback;
  wchar_t* end = nullptr;
  const std::wstring w = widen(it->second.text);
  const double v = wcstod(w.c_str(), &end);
  if (end == w.c_str()) return fallback;
  return v;
}

bool opt_bool(const std::string& key, bool fallback) {
  const auto it = g_opt_state.find(key);
  if (it == g_opt_state.end()) return fallback;
  if (it->second.is_bool) return it->second.b;
  const std::wstring w = lower_w(trim(widen(it->second.text)));
  if (w == L"1" || w == L"true" || w == L"yes" || w == L"on") return true;
  if (w == L"0" || w == L"false" || w == L"no" || w == L"off") return false;
  return fallback;
}

bool opt_dirty(const std::string& key) {
  const auto it = g_opt_state.find(key);
  return it != g_opt_state.end() && it->second.dirty;
}

// ---------------------------------------------------------------------------
// Discovered models (`engine.exe models --json`) and per-model metadata
// (`engine.exe info --json`)
// ---------------------------------------------------------------------------
struct ModelEntry {
  std::wstring path, name, type, quant, params, drive;
  uint64_t size_bytes = 0;
  bool mtp = false;
};
std::vector<ModelEntry> g_models;
int g_model_sel = -1;

struct ModelMeta {
  bool loaded = false;
  bool valid = false;
  std::string path, arch, validation;
  int n_layer = 0, hidden = 0, ffn = 0;
  uint64_t vocab = 0, ctx_train = 0;
  std::string attn_variant;
  int n_head = 0, n_head_kv = 0, klen = 0, vlen = 0;
  int kv_layers = 0;
  int n_expert = 0, n_expert_used = 0, expert_ffn = 0;
  uint64_t total_bytes = 0, fixed_bytes = 0, expert_bytes = 0;
  std::vector<uint64_t> expert_per_layer;  // index = moe layer slot (0..n-1)
  uint64_t kv_per_token[3] = {0, 0, 0};    // f32 / q8_0 / q4_0
  int mtp_layers = 0;
  bool vram_valid = false, host_valid = false;
  uint64_t vram_total = 0, vram_free = 0;
  uint64_t host_total = 0, host_free = 0;
};
ModelMeta g_meta;
std::wstring g_meta_error;

// ---------------------------------------------------------------------------
// UI state
// ---------------------------------------------------------------------------
struct UiState {
  int model_index = -1;
  bool model_auto = true;    // first combo item: --model auto (largest model)
  bool gpu_auto = true;      // gpu_layers = -1
  int gpu_layers = 0;        // explicit layer count (0..n_moe)
  // Context length is freely editable: any positive integer 1..kCtxMax.
  bool ctx_auto = false;     // auto = derive the largest length that fits free VRAM
  int ctx_value = 4096;      // manual value (mirrors the edit box)
  bool ctx_invalid = false;  // edit box currently holds something that is not 1..kCtxMax
  int kv_sel = 0;            // 0 f32 / 1 q8_0 / 2 q4_0
  bool ssd = false;
  bool dev_route = true;
  bool layer_graph_auto = true;
  // Opt-in: after load, promote one more GPU layer from measured free VRAM.
  // Default OFF: decode-time free VRAM drops to ~80 MiB (driver paging risk).
  bool vram_auto_layers = false;
  int cpu_int8 = 1;
  int win_priority = 1;
  bool unlimited_tokens = true;
  bool threads_auto = true;  // true = do not pass --threads (engine default)
  int threads = 12;          // explicit fallback / shown default (2 x physical cores)
  bool win_affinity = true;  // --win-affinity 2 (SMT pair pinning, engine default)
  int port = 7030;
  std::wstring host = L"127.0.0.1";
  bool auto_browser = true;
  bool advanced_open = false;
  std::wstring adv_filter;
  std::wstring preset_note;
  bool busy = false;  // loading model metadata
};
UiState g_ui;

// Arbitrary context length: the edit box is the precise input; the slider is a
// log-scale helper covering the whole legal range 1..1,048,576 (the engine's own
// config bound).  Log scale keeps useful resolution where people actually pick
// values (512..131072) instead of wasting 99% of the travel above 300k.
constexpr int kCtxMax = 1 << 20;
constexpr int kCtxSliderSteps = 1000;
int ctx_from_slider(int pos) {
  const double steps = static_cast<double>(kCtxSliderSteps);
  const double t = static_cast<double>(std::max(0, std::min(kCtxSliderSteps, pos))) / steps;
  const double v = std::pow(static_cast<double>(kCtxMax), t);
  return std::max(1, std::min(kCtxMax, static_cast<int>(std::llround(v))));
}
int ctx_to_slider(int value) {
  if (value <= 1) return 0;
  const double steps = static_cast<double>(kCtxSliderSteps);
  const double v = static_cast<double>(std::max(1, std::min(kCtxMax, value)));
  const double p = std::log(v) / std::log(static_cast<double>(kCtxMax)) * steps;
  return std::max(0, std::min(kCtxSliderSteps, static_cast<int>(std::llround(p))));
}
const wchar_t* const kKvNames[] = {L"f32（默认，精度最高）", L"q8_0（省 4 倍显存）",
                                   L"q4_0（省 8 倍显存）"};
const wchar_t* const kKvShort[] = {L"f32", L"q8_0", L"q4_0"};
const char* const kKvKeys[] = {"f32", "q8_0", "q4_0"};

int logical_cpus() {
  SYSTEM_INFO si{};
  GetSystemInfo(&si);
  return si.dwNumberOfProcessors > 0 ? static_cast<int>(si.dwNumberOfProcessors) : 8;
}

// The engine derives the auto thread count from the physical core count:
// 1.5x physical cores (measured optimum on this host: 6 physical / 12 logical
// -> 9 workers); "3/4 of logical" is the fallback when topology is unknown.
int default_thread_count() {
  int physical = 0;
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
  if (len > 0) {
    std::vector<unsigned char> buf(len);
    if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                         reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                                             buf.data()),
                                         &len)) {
      // The MinGW define of SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX can be
      // bigger than the 48-byte entries Windows returns (union padding), so walk
      // the buffer with the relationship/size header instead of sizeof().
      size_t off = 0;
      const size_t total = static_cast<size_t>(len);
      while (off + 8 <= total) {
        int relationship = 0;
        DWORD entry_size = 0;
        std::memcpy(&relationship, buf.data() + off, sizeof(relationship));
        std::memcpy(&entry_size, buf.data() + off + 4, sizeof(entry_size));
        if (entry_size == 0) break;
        if (relationship == static_cast<int>(RelationProcessorCore)) ++physical;
        off += entry_size;
      }
    }
  }
  const int logical = logical_cpus();
  int threads = 0;
  if (physical > 0) {
    // Engine default (threads=0/auto + win_affinity=2): 2 x physical cores when
    // SMT siblings exist (this host: 6C/12T -> 12), otherwise the old 1.5x.
    threads = logical > physical ? physical * 2 : physical + physical / 2;
  } else {
    threads = logical * 3 / 4;
  }
  if (threads < 1) threads = 1;
  if (threads > logical) threads = logical;
  return threads;
}

// ---------------------------------------------------------------------------
// Byte accounting mirrors of src/planner.cpp (estimates shown in the UI; the
// engine's own decider stays authoritative and its decision is echoed from the
// engine log once the model starts loading).
// ---------------------------------------------------------------------------
uint64_t activation_bytes_est(int rows) {
  const int hidden = g_meta.hidden > 0 ? g_meta.hidden : 4096;
  const int n_head = g_meta.n_head > 0 ? g_meta.n_head : 32;
  const int head_dim = g_meta.klen > 0 ? g_meta.klen : 128;
  const int n_kv = g_meta.n_head_kv > 0 ? g_meta.n_head_kv : n_head;
  const int inter = g_meta.ffn > 0 ? g_meta.ffn : hidden * 4;
  const uint64_t vocab = g_meta.vocab;
  const uint64_t r = rows > 0 ? static_cast<uint64_t>(rows) : 1;
  uint64_t per_row = static_cast<uint64_t>(hidden) * 4 * 6;
  per_row += static_cast<uint64_t>(n_head) * static_cast<uint64_t>(head_dim) * 4 * 2;
  per_row += static_cast<uint64_t>(n_kv) * static_cast<uint64_t>(head_dim) * 4 * 2;
  per_row += static_cast<uint64_t>(inter) * 4 * 3;
  per_row += vocab * 4;
  return r * per_row + (2ull << 20);
}

uint64_t host_reserve_bytes() {
  const uint64_t v = static_cast<uint64_t>(opt_int("host_margin_mb", 0)) << 20;
  if (v != 0) return v;
  const uint64_t gib = 1ull << 30;
  const uint64_t by_frac = g_meta.host_total / 16;
  return by_frac > gib ? by_frac : gib;
}

struct Estimate {
  bool valid = false;
  int kind = 0;  // 0 = gpu, 1 = vram+ram split, 2 = ssd stream
  int n_moe = 0;
  bool dense = false;
  bool mla_kv_forced_f32 = false;
  uint64_t vram_total = 0, vram_free = 0, vram_reserve = 0, vram_usable = 0;
  uint64_t host_total = 0, host_free = 0, host_reserve = 0, host_usable = 0;
  uint64_t fixed = 0, experts = 0, kv = 0, workspace = 0;
  uint64_t need_fixed = 0;        // fixed + kv + workspace
  uint64_t need_all = 0;          // need_fixed + experts
  uint64_t weights_on_gpu = 0;    // expert bytes resident in VRAM
  uint64_t weights_on_host = 0;   // expert bytes on the host (CPU path)
  uint64_t vram_need = 0;         // need_fixed + weights_on_gpu
  uint64_t host_need = 0;         // expert bytes that must live in host RAM
  int gpu_layers = 0;             // shown / explicit
  int gpu_layers_auto = 0;        // auto estimate
  int cpu_layers = 0;
  bool over_usable = false;
  int max_fit_layers = 0;
  int ctx_used = 0;
  int auto_ctx = 0;
  int ctx_fit = 0;   // largest ctx that fits the VRAM budget
  bool ctx_auto = false;
  int kv_type_used = 0;
  uint64_t kv_per_token = 0;
};

int auto_ctx_estimate(uint64_t vram_free, uint64_t reserve, int kv_type) {
  if (!g_meta.loaded || g_meta.kv_layers <= 0) return 0;
  const uint64_t kv_bpt = g_meta.kv_per_token[kv_type];
  int model_ctx = g_meta.ctx_train > 0
                      ? static_cast<int>(std::min<int64_t>(static_cast<int64_t>(g_meta.ctx_train),
                                                           1 << 20))
                      : (1 << 20);
  if (kv_bpt == 0) return model_ctx;
  const int cap = static_cast<int>(opt_int("prefill_rows", 512));
  const int want = std::max(1, model_ctx);
  const uint64_t weights = g_meta.fixed_bytes;
  int seq = want;
  int fit = want;
  for (int it = 0; it < 4; ++it) {
    const int rows = std::max(1, std::min(seq, cap > 0 ? cap : 1));
    const uint64_t act = activation_bytes_est(rows);
    const uint64_t avail =
        vram_free > weights + act + reserve ? vram_free - weights - act - reserve : 0;
    fit = static_cast<int>(avail / kv_bpt);
    fit = std::min(fit, want);
    if (fit >= seq) break;
    seq = std::max(1, fit);
  }
  return std::max(0, std::min(want, fit));
}

Estimate compute_estimate() {
  Estimate e;
  if (!g_meta.loaded) return e;
  e.valid = true;
  e.n_moe = static_cast<int>(g_meta.expert_per_layer.size());
  e.dense = e.n_moe == 0;
  const uint64_t margin = static_cast<uint64_t>(std::max<long long>(0, opt_int("vram_margin_mb", 1200))) << 20;
  e.workspace = static_cast<uint64_t>(std::max<long long>(0, opt_int("workspace_mb", 768))) << 20;
  const uint64_t budget = static_cast<uint64_t>(std::max<long long>(0, opt_int("vram_budget_mb", 0))) << 20;
  e.vram_total = g_meta.vram_total;
  e.vram_free = budget != 0 ? budget : g_meta.vram_free;
  e.vram_reserve = margin + e.workspace;
  e.vram_usable = e.vram_free > e.vram_reserve ? e.vram_free - e.vram_reserve : 0;
  e.host_total = g_meta.host_total;
  e.host_free = g_meta.host_free;
  e.host_reserve = host_reserve_bytes();
  e.host_usable = e.host_free > e.host_reserve ? e.host_free - e.host_reserve : 0;

  e.fixed = g_meta.fixed_bytes;
  e.experts = g_meta.expert_bytes;
  int kv_type = static_cast<int>(g_ui.kv_sel);
  if (g_meta.attn_variant == "mla" && kv_type != 0) {
    kv_type = 0;
    e.mla_kv_forced_f32 = true;
  }
  e.kv_type_used = kv_type;
  e.kv_per_token = g_meta.kv_per_token[kv_type];
  e.ctx_auto = g_ui.ctx_auto;
  e.auto_ctx = auto_ctx_estimate(e.vram_free, e.vram_reserve, kv_type);
  e.ctx_fit = e.auto_ctx;
  e.ctx_used = e.ctx_auto ? e.auto_ctx : g_ui.ctx_value;
  if (e.ctx_used <= 0) e.ctx_used = 4096;
  e.kv = e.kv_per_token * static_cast<uint64_t>(e.ctx_used);
  e.need_fixed = e.fixed + e.kv + e.workspace;
  e.need_all = e.need_fixed + e.experts;

  // Largest feasible prefix of GPU-resident expert layers: walk the real
  // per-layer expert bytes (not an average) and stop when the projected VRAM
  // footprint would exceed the usable budget.  This is the "自动" estimate the
  // slider shows; the engine's measured placement is echoed from its log.
  {
    uint64_t acc = e.need_fixed;
    for (int i = 0; i < e.n_moe; ++i) {
      const uint64_t next = acc + g_meta.expert_per_layer[static_cast<size_t>(i)];
      if (next > e.vram_usable) break;
      acc = next;
      e.max_fit_layers = i + 1;
    }
  }
  e.gpu_layers_auto = e.max_fit_layers;
  e.gpu_layers = e.dense ? 0 : (g_ui.gpu_auto ? e.gpu_layers_auto : g_ui.gpu_layers);
  if (!e.dense) e.gpu_layers = std::max(0, std::min(e.gpu_layers, e.n_moe));
  e.cpu_layers = e.n_moe - e.gpu_layers;
  for (int i = 0; i < e.gpu_layers; ++i)
    e.weights_on_gpu += g_meta.expert_per_layer[static_cast<size_t>(i)];
  e.weights_on_host = e.experts - e.weights_on_gpu;
  e.vram_need = e.need_fixed + e.weights_on_gpu;
  e.host_need = e.weights_on_host;
  e.over_usable = e.vram_need > e.vram_usable;

  // Placement decision, mirroring placement::decide().
  if (e.need_all <= e.vram_usable) {
    e.kind = 0;
  } else if (!e.dense && e.need_fixed <= e.vram_usable) {
    const uint64_t left = e.vram_usable - e.need_fixed;
    const uint64_t need_host = e.experts > left ? e.experts - left : 0;
    e.kind = (need_host <= e.host_usable) ? 1 : 2;
  } else {
    e.kind = 2;
  }
  return e;
}

Estimate g_est;

// ---------------------------------------------------------------------------
// Option defaults that are mirrored into the UI state (avoids hard coding)
// ---------------------------------------------------------------------------
void sync_state_from_schema() {
  const int schema_threads = static_cast<int>(opt_int("threads", 0));
  g_ui.threads_auto = schema_threads <= 0;  // engine default = auto
  g_ui.threads = schema_threads > 0 ? schema_threads : default_thread_count();
  g_ui.win_affinity = opt_int("win_affinity", 2) == 2;  // engine default: SMT pairs
  g_ui.port = static_cast<int>(opt_int("serve_port", 7030));
  g_ui.host = widen(opt_text("serve_host", "127.0.0.1"));
  g_ui.kv_sel = 0;
  const std::wstring kv = lower_w(widen(opt_text("kv_type", "f32")));
  if (kv == L"q8_0") g_ui.kv_sel = 1;
  else if (kv == L"q4_0") g_ui.kv_sel = 2;
  g_ui.dev_route = opt_bool("dev_route", true);
  g_ui.layer_graph_auto = opt_bool("layer_graph_auto", true);
  g_ui.vram_auto_layers = opt_bool("vram_auto_layers", false);  // default OFF
  g_ui.cpu_int8 = static_cast<int>(opt_int("cpu_int8", 1));
  g_ui.win_priority = static_cast<int>(opt_int("win_priority", 1));
  g_ui.unlimited_tokens = opt_int("default_max_tokens", 1048576) >= 1048576;
  g_ui.ctx_auto = false;
  g_ui.ctx_value = 4096;
  g_ui.ctx_invalid = false;
  g_ui.gpu_auto = true;
  g_ui.gpu_layers = 0;
}

// ---------------------------------------------------------------------------
// Launch specification: CLI arguments + config-file overrides
// ---------------------------------------------------------------------------
const char* const kCuratedKeys[] = {
    "model",          "serve_host",     "serve_port",     "ctx",
    "kv_type",        "gpu_layers",     "threads",        "ssd_stream",
    "dev_route",      "layer_graph_auto", "vram_auto_layers", "cpu_int8",
    "win_priority",   "win_affinity",
    "default_max_tokens"};

bool is_curated_key(const std::string& key) {
  for (const char* k : kCuratedKeys)
    if (key == k) return true;
  return false;
}

struct LaunchSpec {
  std::wstring cmdline;
  std::wstring summary;
  std::string cfg_text;
  std::wstring cfg_path;
  std::wstring log_path;
  std::wstring warn_text;
  bool has_warn = false;
  int advanced_overrides = 0;
  int model_index = -1;
};

std::wstring current_model_path() {
  if (g_ui.model_auto) {
    if (!g_models.empty()) return g_models.front().path;
    return std::wstring();
  }
  if (g_ui.model_index >= 0 && g_ui.model_index < static_cast<int>(g_models.size()))
    return g_models[static_cast<size_t>(g_ui.model_index)].path;
  return std::wstring();
}

std::wstring current_model_label() {
  if (g_ui.model_auto) return L"自动（最大模型）";
  if (g_ui.model_index >= 0 && g_ui.model_index < static_cast<int>(g_models.size()))
    return g_models[static_cast<size_t>(g_ui.model_index)].name;
  return L"（未选择）";
}

LaunchSpec build_launch_spec(bool write_cfg = true) {
  LaunchSpec spec;
  spec.cfg_path = join_path(g_launcher_dir, L"launcher-engine.cfg");
  spec.log_path = join_path(g_launcher_dir, L"launcher-engine.log");
  const std::wstring model = current_model_path();

  std::wstring cmd = quote_arg(g_engine) + L" serve";
  if (g_ui.model_auto) cmd += L" --model auto";
  else cmd += L" --model " + quote_arg(model);
  cmd += L" --host " + quote_arg(g_ui.host);
  cmd += L" --port " + wnum(g_ui.port);
  const int ctx = g_ui.ctx_auto ? 0 : g_ui.ctx_value;
  cmd += L" --ctx " + wnum(ctx);
  cmd += std::wstring(L" --kv-type ") + kKvShort[g_ui.kv_sel];
  cmd += L" --gpu-layers " + wnum(g_ui.gpu_auto ? -1 : g_ui.gpu_layers);
  // Auto threads: pass nothing, the engine computes its default from its
  // win_affinity/SMT topology (12 on a 6C/12T host).  Explicit N overrides it.
  if (!g_ui.threads_auto) cmd += L" --threads " + wnum(g_ui.threads);
  cmd += L" --win-affinity " + wnum(g_ui.win_affinity ? 2 : 0);
  cmd += L" --cpu-int8 " + wnum(g_ui.cpu_int8);
  cmd += L" --win-priority " + wnum(g_ui.win_priority);
  cmd += L" --default-max-tokens " + wnum(g_ui.unlimited_tokens ? 1048576 : 2048);
  if (g_ui.dev_route) cmd += L" --dev-route";
  if (g_ui.layer_graph_auto) cmd += L" --layer-graph-auto";
  // Boolean with explicit value: the console parser does not consume a
  // separate "1" token, so the "=1" spelling is the safe CLI form.
  if (g_ui.vram_auto_layers) cmd += L" --vram-auto-layers=1";
  if (g_ui.ssd) cmd += L" --ssd-stream";

  // Advanced panel: emit every changed option into the config file (this is the
  // universal escape hatch: some engine options only exist as negative switches
  // or have no CLI form at all, and the config file outranks the environment).
  std::string cfg;
  cfg += "# generated by build/launcher.exe at launch - advanced panel overrides\n";
  int overrides = 0;
  for (const auto& od : g_opts) {
    if (is_curated_key(od.key)) continue;
    if (od.key == "config") continue;  // handled separately below
    const auto it = g_opt_state.find(od.key);
    if (it == g_opt_state.end() || !it->second.dirty) continue;
    std::string value;
    if (it->second.is_bool) {
      value = it->second.b ? "true" : "false";
    } else {
      value = trim(widen(it->second.text)).empty() ? std::string() : it->second.text;
      if (od.key == "ssd_cache" || od.key == "model_dir" || od.key == "kernel_cache" ||
          od.key == "nvrtc_path")
        value = narrow(trim(widen(it->second.text)));
    }
    if (od.key == "print_config" || od.key == "kernel_calibrate") {
      spec.has_warn = true;
      spec.warn_text += widen(od.key) + L" 会让引擎打印/校准后立即退出；";
    }
    cfg += od.key + " = " + value + "\n";
    ++overrides;
  }
  spec.advanced_overrides = overrides;
  // A user-supplied extra config file is merged first, our overrides win.
  const std::string extra = opt_text("config");
  if (!extra.empty()) {
    const std::string extra_text = read_file_utf8(widen(extra));
    if (!extra_text.empty()) cfg = "# merged from " + extra + "\n" + extra_text + "\n" + cfg;
  }
  spec.cfg_text = cfg;
  if (overrides > 0 || !extra.empty()) {
    if (write_cfg) write_file_utf8(spec.cfg_path, cfg);
    cmd += L" --config " + quote_arg(spec.cfg_path);
  }

  spec.cmdline = cmd;
  spec.model_index = g_ui.model_auto ? 0 : g_ui.model_index;

  std::wstring ctx_text;
  if (ctx == 0)
    ctx_text = g_est.auto_ctx > 0 ? (L"自动→" + wnum(g_est.auto_ctx)) : std::wstring(L"自动");
  else
    ctx_text = wnum(ctx);
  spec.summary = L"模型 " + current_model_label() + L" · 端口 " + wnum(g_ui.port) + L" · ctx " +
                 ctx_text + L" · KV " +
                 widen(kKvKeys[g_ui.kv_sel]) + L" · GPU层 " +
                 (g_ui.gpu_auto ? std::wstring(L"自动") : wnum(g_ui.gpu_layers)) + L" · 线程 " +
                 (g_ui.threads_auto ? std::wstring(L"自动") : wnum(g_ui.threads)) +
                 L" · SMT绑核 " + (g_ui.win_affinity ? std::wstring(L"开") : std::wstring(L"关")) +
                 L" · 高级覆盖 " + wnum(overrides) + L" 项";
  if (g_ui.vram_auto_layers) spec.summary += L" · VRAM 自动加层";
  return spec;
}

// Coverage evidence for the "全部 CLI 选项" requirement: written next to the
// launcher so build/test_launcher.py can prove 100% mapping without relying on
// window enumeration.
std::wstring g_ui_map_path;
std::wstring g_effective_args_path;

}  // namespace
// ---------------------------------------------------------------------------
// Main control handles
// ---------------------------------------------------------------------------
namespace {

HWND g_model_combo = nullptr;
HWND g_refresh_btn = nullptr;
HWND g_scan_status = nullptr;
HWND g_model_info = nullptr;
HWND g_engine_placement = nullptr;
HWND g_gpu_slider = nullptr;
HWND g_gpu_auto = nullptr;
HWND g_gpu_text = nullptr;
HWND g_ctx_slider = nullptr;
HWND g_ctx_edit = nullptr;
HWND g_ctx_auto = nullptr;
HWND g_ctx_text = nullptr;
bool g_ctx_sync = false;  // suppress EN_CHANGE while update_ctx_controls writes the box
HWND g_kv_combo = nullptr;
HWND g_kv_text = nullptr;
HWND g_ssd_toggle = nullptr;
HWND g_ssd_note = nullptr;
HWND g_dev_route = nullptr;
HWND g_layer_graph = nullptr;
HWND g_cpu_int8 = nullptr;
HWND g_win_prio = nullptr;
HWND g_thinking_len = nullptr;
HWND g_vram_auto = nullptr;
HWND g_vram_auto_warn = nullptr;
HWND g_threads_combo = nullptr;
HWND g_win_affinity = nullptr;
HWND g_threads_hint = nullptr;
HWND g_port_edit = nullptr;
HWND g_host_edit = nullptr;
HWND g_auto_browser = nullptr;
HWND g_preset_fast = nullptr;
HWND g_preset_low = nullptr;
HWND g_preset_bal = nullptr;
HWND g_adv_toggle = nullptr;
HWND g_adv_search = nullptr;
HWND g_adv_count = nullptr;
HWND g_start = nullptr;
HWND g_stop = nullptr;
HWND g_open_page = nullptr;
HWND g_open_log = nullptr;
HWND g_status = nullptr;
HWND g_url = nullptr;
HWND g_log_info = nullptr;
HWND g_args_summary = nullptr;
HWND g_est_summary = nullptr;
HWND g_schema_info = nullptr;
HWND g_footer = nullptr;

struct AdvCell {
  int opt_index = 0;
  HWND label = nullptr;
  HWND ctrl = nullptr;
  bool created = false;
};
std::vector<AdvCell> g_adv_cells;
std::map<UINT, int> g_adv_id_to_opt;  // control id -> option index

// Painted section headers / separators (content coordinates, scrolled).
struct PaintedHeader {
  int y = 0;
  int h = 0;
  std::wstring title;
  std::wstring hint;
};
std::vector<PaintedHeader> g_painted;

// ---------------------------------------------------------------------------
// Widget creation helpers
// ---------------------------------------------------------------------------
std::map<HWND, int> g_ctrl_font_pt;  // font size (pt) per control, for DPI changes

int font_pt_of(HFONT f) {
  if (f == g_font_small) return 9;
  if (f == g_font_title) return 13;
  if (f == g_font_bold) return 11;
  return 10;
}

HWND mk_static(HWND parent, const std::wstring& text, int id, Role role, DWORD extra_style,
               HFONT font) {
  HWND h = CreateWindowExW(0, L"STATIC", text.c_str(),
                           WS_CHILD | WS_VISIBLE | SS_LEFT | extra_style, 0, 0, 0, 0, parent,
                           reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr, nullptr);
  if (h == nullptr) return nullptr;
  SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
  g_label_role[h] = role;
  g_ctrl_font_pt[h] = font_pt_of(font);
  return h;
}

std::map<HWND, bool> g_edit_controls;  // edit boxes get a custom 1px frame

HWND mk_edit(HWND parent, const std::wstring& text, int id, DWORD extra_style = 0) {
  HWND h = CreateWindowExW(0, L"EDIT", text.c_str(),
                           WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL | ES_LEFT | extra_style, 0, 0,
                           0, 0, parent, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                           nullptr, nullptr);
  if (h == nullptr) return nullptr;
  SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(g_font), TRUE);
  g_ctrl_font_pt[h] = 10;
  g_edit_controls[h] = true;
  return h;
}

HWND mk_button(HWND parent, const std::wstring& text, int id) {
  HWND h = CreateWindowExW(0, L"BUTTON", text.c_str(),
                           WS_CHILD | WS_VISIBLE | BS_OWNERDRAW | WS_TABSTOP, 0, 0, 0, 0, parent,
                           reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr, nullptr);
  if (h == nullptr) return nullptr;
  SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(g_font_bold), TRUE);
  g_ctrl_font_pt[h] = 10;
  return h;
}

HWND mk_toggle(HWND parent, const std::wstring& text, int id) {
  HWND h = CreateWindowExW(0, L"BUTTON", text.c_str(),
                           WS_CHILD | WS_VISIBLE | BS_OWNERDRAW | WS_TABSTOP, 0, 0, 0, 0, parent,
                           reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr, nullptr);
  if (h == nullptr) return nullptr;
  SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(g_font), TRUE);
  g_ctrl_font_pt[h] = 10;
  return h;
}

// Combo boxes are *native* controls (CBS_DROPDOWNLIST): Windows handles the
// popup list, mouse hit-testing and keyboard (Up/Down/Enter/Esc).  We only
// owner-draw the items (WM_DRAWITEM) to get dark colours; no subclassing.
// A real user typically leaves keyboard focus on the model combo after picking
// an item; a native CBS_DROPDOWNLIST would then consume WM_MOUSEWHEEL and change
// the selection instead of scrolling the page.  While closed we swallow the
// wheel and forward it to the main window (the page scroll host); while the
// drop-down list is open the system default still scrolls the list.
std::map<HWND, WNDPROC> g_combo_old_proc;

LRESULT CALLBACK combo_wheel_proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
  WNDPROC old = nullptr;
  auto it = g_combo_old_proc.find(h);
  if (it != g_combo_old_proc.end()) old = it->second;
  if (msg == WM_MOUSEWHEEL || msg == WM_MOUSEHWHEEL) {
    const bool dropped = SendMessageW(h, CB_GETDROPPEDSTATE, 0, 0) != 0;
    if (!dropped) {
      if (msg == WM_MOUSEWHEEL && g_hwnd != nullptr)
        SendMessageW(g_hwnd, WM_MOUSEWHEEL, wp, lp);  // scroll the page
      return 0;  // never let a closed combo change its selection via the wheel
    }
    // Open list: scroll the drop-down list, but never move the selection with
    // the wheel (the user must click/Enter; Esc closes without changes).
    if (msg == WM_MOUSEWHEEL) {
      COMBOBOXINFO ci;
      std::memset(&ci, 0, sizeof(ci));
      ci.cbSize = sizeof(ci);
      if (GetComboBoxInfo(h, &ci) && ci.hwndList != nullptr)
        SendMessageW(ci.hwndList, WM_MOUSEWHEEL, wp, lp);
    }
    return 0;
  }
  return old != nullptr ? CallWindowProcW(old, h, msg, wp, lp)
                        : DefWindowProcW(h, msg, wp, lp);
}

void install_combo_wheel_guard(HWND h) {
  if (h == nullptr || g_combo_old_proc.count(h) != 0) return;
  WNDPROC old = reinterpret_cast<WNDPROC>(SetWindowLongPtrW(
      h, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(combo_wheel_proc)));
  if (old != nullptr) g_combo_old_proc[h] = old;
}

// The drop-down list is a separate native "ComboLBox" window.  Its default
// WM_MOUSEWHEEL handling may move the combo selection; a wheel over the open
// list must only scroll the list (selection changes only on click/Enter/Esc).
std::map<HWND, WNDPROC> g_combo_list_old;
std::map<HWND, HWND> g_combo_list_owner;

LRESULT CALLBACK combo_list_wheel_proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
  WNDPROC old = nullptr;
  auto it = g_combo_list_old.find(h);
  if (it != g_combo_list_old.end()) old = it->second;
  if (msg == WM_MOUSEWHEEL) {
    const int delta = GET_WHEEL_DELTA_WPARAM(wp);
    if (delta != 0) {
      HWND combo = nullptr;
      auto oit = g_combo_list_owner.find(h);
      if (oit != g_combo_list_owner.end()) combo = oit->second;
      int item_h = S(24);
      if (combo != nullptr) {
        const int ih = static_cast<int>(SendMessageW(combo, CB_GETITEMHEIGHT, 0, 0));
        if (ih > 0) item_h = ih;
      }
      RECT rc;
      GetClientRect(h, &rc);
      const int visible = std::max(1, static_cast<int>((rc.bottom - rc.top) / std::max(1, item_h)));
      const int count = static_cast<int>(SendMessageW(h, LB_GETCOUNT, 0, 0));
      const int top = static_cast<int>(SendMessageW(h, LB_GETTOPINDEX, 0, 0));
      const int step = delta > 0 ? -3 : 3;
      const int max_top = std::max(0, count - visible);
      const int ntop = std::max(0, std::min(max_top, top + step));
      if (ntop != top) SendMessageW(h, LB_SETTOPINDEX, static_cast<WPARAM>(ntop), 0);
    }
    return 0;  // never change the selection with the wheel
  }
  if (msg == WM_MOUSEHWHEEL) return 0;
  return old != nullptr ? CallWindowProcW(old, h, msg, wp, lp)
                        : DefWindowProcW(h, msg, wp, lp);
}

void install_combo_list_wheel_guard(HWND combo) {
  if (combo == nullptr) return;
  COMBOBOXINFO ci;
  std::memset(&ci, 0, sizeof(ci));
  ci.cbSize = sizeof(ci);
  if (!GetComboBoxInfo(combo, &ci) || ci.hwndList == nullptr) return;
  HWND list = ci.hwndList;
  if (g_combo_list_old.count(list) != 0) return;
  WNDPROC old = reinterpret_cast<WNDPROC>(SetWindowLongPtrW(
      list, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(combo_list_wheel_proc)));
  if (old != nullptr) {
    g_combo_list_old[list] = old;
    g_combo_list_owner[list] = combo;
  }
}

std::set<HWND> g_combo_controls;

bool is_combo_control(HWND h) { return h != nullptr && g_combo_controls.count(h) != 0; }

HWND mk_combo(HWND parent, int id) {
  HWND h = CreateWindowExW(0, L"COMBOBOX", nullptr,
                           WS_CHILD | WS_VISIBLE | WS_VSCROLL | CBS_DROPDOWNLIST |
                               CBS_OWNERDRAWFIXED | CBS_HASSTRINGS | WS_TABSTOP,
                           0, 0, S(240), S(30), parent,
                           reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr, nullptr);
  if (h == nullptr) return nullptr;
  SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(g_font), TRUE);
  SendMessageW(h, CB_SETITEMHEIGHT, static_cast<WPARAM>(-1), S(24));
  SendMessageW(h, CB_SETITEMHEIGHT, 0, S(24));
  g_ctrl_font_pt[h] = 10;
  g_combo_controls.insert(h);
  install_combo_wheel_guard(h);
  return h;
}

// ---------------------------------------------------------------------------
// (native combobox item painting lives in draw_combo_item / WM_DRAWITEM below)
// Custom slider control (dark themed, no comctl32 trackbar dependency)
// ---------------------------------------------------------------------------
constexpr wchar_t kSliderClass[] = L"EngineSliderV2";
constexpr UINT WM_UI_SLIDER = WM_APP + 20;      // parent notification (wparam=id, lparam=pos)
constexpr UINT WM_UI_SLIDER_SET = WM_APP + 21;  // external set (wparam=pos)

struct SliderData {
  int min_v = 0;
  int max_v = 1;
  int pos = 0;
  int ticks = 0;
  int id = 0;
};

void slider_notify(HWND h, SliderData* d) {
  SendMessageW(GetParent(h), WM_UI_SLIDER, static_cast<WPARAM>(d->id), static_cast<LPARAM>(d->pos));
  InvalidateRect(h, nullptr, TRUE);
}

void slider_paint(HWND h, SliderData* d) {
  PAINTSTRUCT ps;
  HDC dc = BeginPaint(h, &ps);
  RECT rc;
  GetClientRect(h, &rc);
  fill_rect(dc, rc, g_pal.bg);
  const bool enabled = IsWindowEnabled(h) != FALSE;
  const int thumb_r = S(9);
  const int left = rc.left + thumb_r;
  const int right = rc.right - thumb_r;
  const int cy = (rc.top + rc.bottom) / 2;
  const int track_h = S(6);
  RECT track{left, cy - track_h / 2, right, cy + track_h / 2};
  const COLORREF track_c = enabled ? g_pal.toggle_off : g_pal.border;
  rounded_fill(dc, track, track_h / 2, g_pal.panel2, track_c);
  const int span = d->max_v > d->min_v ? d->max_v - d->min_v : 1;
  const int span_px = right - left;
  int pos_px = left + static_cast<int>(static_cast<long long>(span_px) * (d->pos - d->min_v) / span);
  if (pos_px < left) pos_px = left;
  if (pos_px > right) pos_px = right;
  if (pos_px > left) {
    RECT fillr{left, track.top, pos_px, track.bottom};
    rounded_fill(dc, fillr, track_h / 2, enabled ? g_pal.accent : g_pal.accent_dim,
                 enabled ? g_pal.accent : g_pal.accent_dim);
  }
  // tick dots (context presets)
  if (d->ticks > 1) {
    for (int i = 0; i < d->ticks; ++i) {
      const int x = left + static_cast<int>(static_cast<long long>(span_px) * i / (d->ticks - 1));
      RECT dot{x - S(1), cy + S(9), x + S(2), cy + S(12)};
      fill_rect(dc, dot, enabled ? g_pal.dim : g_pal.border);
    }
  }
  RECT thumb{pos_px - thumb_r, cy - thumb_r, pos_px + thumb_r, cy + thumb_r};
  rounded_fill(dc, thumb, thumb_r, enabled ? g_pal.accent : g_pal.toggle_off,
               enabled ? g_pal.accent : g_pal.border);
  if (GetFocus() == h) {
    RECT focus{thumb.left - S(2), thumb.top - S(2), thumb.right + S(2), thumb.bottom + S(2)};
    HBRUSH br = CreateSolidBrush(g_pal.text);
    DrawFocusRect(dc, &focus);
    DeleteObject(br);
  }
  EndPaint(h, &ps);
}

void slider_set_from_x(HWND h, SliderData* d, int x) {
  RECT rc;
  GetClientRect(h, &rc);
  const int left = rc.left + S(9);
  const int right = rc.right - S(9);
  const int span_px = right - left;
  if (span_px <= 0) return;
  int rel = x - left;
  if (rel < 0) rel = 0;
  if (rel > span_px) rel = span_px;
  const int span = d->max_v > d->min_v ? d->max_v - d->min_v : 1;
  const int nv = d->min_v + static_cast<int>(static_cast<long long>(rel) * span / span_px);
  if (nv != d->pos) {
    d->pos = nv;
    slider_notify(h, d);
  }
}

LRESULT CALLBACK slider_proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
  SliderData* d = reinterpret_cast<SliderData*>(GetWindowLongPtrW(h, GWLP_USERDATA));
  switch (msg) {
    case WM_CREATE: {
      auto* sd = new SliderData();
      sd->id = GetDlgCtrlID(h);
      SetWindowLongPtrW(h, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(sd));
      return 0;
    }
    case WM_DESTROY: {
      delete d;
      SetWindowLongPtrW(h, GWLP_USERDATA, 0);
      return 0;
    }
    case WM_NCDESTROY:
      return DefWindowProcW(h, msg, wp, lp);
    case WM_ERASEBKGND:
      return 1;
    case WM_PAINT:
      if (d != nullptr) slider_paint(h, d);
      return 0;
    case WM_LBUTTONDOWN:
      if (d != nullptr && IsWindowEnabled(h)) {
        SetFocus(h);
        SetCapture(h);
        slider_set_from_x(h, d, static_cast<int>(static_cast<short>(LOWORD(lp))));
      }
      return 0;
    case WM_MOUSEMOVE:
      if (d != nullptr && GetCapture() == h) slider_set_from_x(h, d, static_cast<int>(static_cast<short>(LOWORD(lp))));
      return 0;
    case WM_LBUTTONUP:
      if (GetCapture() == h) ReleaseCapture();
      return 0;
    case WM_MOUSEWHEEL:
      if (d != nullptr && IsWindowEnabled(h)) {
        const int step = GET_WHEEL_DELTA_WPARAM(wp) > 0 ? 1 : -1;
        const int nv = std::max(d->min_v, std::min(d->max_v, d->pos + step));
        if (nv != d->pos) {
          d->pos = nv;
          slider_notify(h, d);
        }
      }
      return 0;
    case WM_GETDLGCODE:
      return DLGC_WANTARROWS;
    case WM_KEYDOWN:
      if (d != nullptr && IsWindowEnabled(h)) {
        int nv = d->pos;
        if (wp == VK_LEFT) --nv;
        else if (wp == VK_RIGHT) ++nv;
        else if (wp == VK_HOME) nv = d->min_v;
        else if (wp == VK_END) nv = d->max_v;
        else return 0;
        nv = std::max(d->min_v, std::min(d->max_v, nv));
        if (nv != d->pos) {
          d->pos = nv;
          slider_notify(h, d);
        }
      }
      return 0;
    case WM_SETFOCUS:
    case WM_KILLFOCUS:
    case WM_ENABLE:
      InvalidateRect(h, nullptr, TRUE);
      return 0;
    case WM_UI_SLIDER_SET:
      if (d != nullptr) {
        const int nv = std::max(d->min_v, std::min(d->max_v, static_cast<int>(wp)));
        if (nv != d->pos) {
          d->pos = nv;
          slider_notify(h, d);
        }
      }
      return 0;
    default:
      break;
  }
  return DefWindowProcW(h, msg, wp, lp);
}

void register_slider_class() {
  static bool done = false;
  if (done) return;
  done = true;
  WNDCLASSEXW wc{};
  wc.cbSize = sizeof(wc);
  wc.lpfnWndProc = slider_proc;
  wc.hInstance = GetModuleHandleW(nullptr);
  wc.hCursor = LoadCursorW(nullptr, IDC_HAND);
  wc.lpszClassName = kSliderClass;
  wc.hbrBackground = nullptr;
  RegisterClassExW(&wc);
}

HWND mk_slider(HWND parent, int id) {
  register_slider_class();
  HWND h = CreateWindowExW(0, kSliderClass, nullptr, WS_CHILD | WS_VISIBLE | WS_TABSTOP, 0, 0, 0,
                           0, parent, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr,
                           nullptr);
  return h;
}

void slider_range(HWND h, int min_v, int max_v) {
  auto* d = reinterpret_cast<SliderData*>(GetWindowLongPtrW(h, GWLP_USERDATA));
  if (d == nullptr) return;
  if (max_v < min_v) max_v = min_v;
  d->min_v = min_v;
  d->max_v = max_v;
  d->pos = std::max(min_v, std::min(max_v, d->pos));
  InvalidateRect(h, nullptr, TRUE);
}

void slider_set(HWND h, int pos, bool notify) {
  auto* d = reinterpret_cast<SliderData*>(GetWindowLongPtrW(h, GWLP_USERDATA));
  if (d == nullptr) return;
  const int nv = std::max(d->min_v, std::min(d->max_v, pos));
  if (nv == d->pos) {
    InvalidateRect(h, nullptr, TRUE);
    return;
  }
  d->pos = nv;
  if (notify) slider_notify(h, d);
  else InvalidateRect(h, nullptr, TRUE);
}

int slider_get(HWND h) {
  auto* d = reinterpret_cast<SliderData*>(GetWindowLongPtrW(h, GWLP_USERDATA));
  return d == nullptr ? 0 : d->pos;
}

void slider_ticks(HWND h, int n) {
  auto* d = reinterpret_cast<SliderData*>(GetWindowLongPtrW(h, GWLP_USERDATA));
  if (d != nullptr) d->ticks = n;
}

// ---------------------------------------------------------------------------
// Enum hints for schema-generated combo boxes
// ---------------------------------------------------------------------------
const std::vector<std::wstring>* enum_values_for(const std::string& key) {
  static const std::map<std::string, std::vector<std::wstring>> kEnums = {
      {"placement_policy", {L"auto", L"gpu", L"split", L"ssd"}},
      {"router", {L"topk_softmax", L"sigmoid_topk"}},
      {"mla_q_layout", {L"global", L"global_pe", L"perhead"}},
      {"mla_kv_layout", {L"global", L"global_vk", L"perhead", L"perhead_vk"}},
  };
  const auto it = kEnums.find(key);
  return it == kEnums.end() ? nullptr : &it->second;
}

// ---------------------------------------------------------------------------
// Advanced panel: one control per option of the engine schema
// ---------------------------------------------------------------------------
bool is_curated_key_pub(const std::string& key) { return is_curated_key(key); }

void create_advanced_controls() {
  for (size_t i = 0; i < g_opts.size(); ++i) {
    const OptDef& od = g_opts[i];
    if (is_curated_key(od.key)) continue;
    AdvCell cell;
    cell.opt_index = static_cast<int>(i);
    std::wstring label = od.cli.empty() ? widen(od.key) : (L"--" + widen(od.cli));
    cell.label = mk_static(g_hwnd, label, 0, Role::Dim, 0, g_font_small);
    const int id = IDC_ADV_BASE + static_cast<int>(g_adv_cells.size());
    const std::wstring tooltip =
        L"--" + (od.cli.empty() ? widen(od.key) : widen(od.cli)) + L"  (config: " +
        widen(od.key) + L", " + widen(od.type) + L", 默认 " + widen(od.def) + L")\n" +
        widen(od.help);
    if (od.type == "bool") {
      cell.ctrl = mk_toggle(g_hwnd, L"", id);
    } else if (od.type == "string" && enum_values_for(od.key) != nullptr) {
      cell.ctrl = mk_combo(g_hwnd, id);
      for (const auto& v : *enum_values_for(od.key))
        SendMessageW(cell.ctrl, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(v.c_str()));
      const std::wstring cur = widen(opt_text(od.key));
      int sel = 0;
      for (size_t k = 0; k < enum_values_for(od.key)->size(); ++k)
        if ((*enum_values_for(od.key))[k] == cur) sel = static_cast<int>(k);
      SendMessageW(cell.ctrl, CB_SETCURSEL, static_cast<WPARAM>(sel), 0);
    } else {
      cell.ctrl = mk_edit(g_hwnd, widen(opt_text(od.key)), id);
    }
    if (cell.ctrl == nullptr) continue;
    // initial check state for booleans
    if (od.type == "bool") {
      const auto it = g_opt_state.find(od.key);
      const bool on = it != g_opt_state.end() && it->second.b;
      toggle_set(cell.ctrl, on);
    }
    add_tip(cell.label, tooltip);
    add_tip(cell.ctrl, tooltip);
    g_adv_id_to_opt[static_cast<UINT>(id)] = static_cast<int>(i);
    cell.created = true;
    g_adv_cells.push_back(cell);
  }
}

bool adv_matches(const OptDef& od, const std::wstring& filter_lower) {
  if (filter_lower.empty()) return true;
  const std::wstring hay = lower_w(widen(od.key + " --" + od.cli + " " + od.help + " " + od.def));
  return hay.find(filter_lower) != std::wstring::npos;
}

// Update the `visible` flag of every advanced cell from the search box.
int update_adv_visibility() {
  const std::wstring filter = lower_w(trim(g_ui.adv_filter));
  int shown = 0;
  for (auto& cell : g_adv_cells) {
    if (!cell.created) continue;
    const OptDef& od = g_opts[static_cast<size_t>(cell.opt_index)];
    const bool match = adv_matches(od, filter);
    if (cell.label != nullptr) ShowWindow(cell.label, match ? SW_SHOW : SW_HIDE);
    if (cell.ctrl != nullptr) ShowWindow(cell.ctrl, match ? SW_SHOW : SW_HIDE);
    if (match) ++shown;
  }
  return shown;
}

// ---------------------------------------------------------------------------
// Model metadata loading (`engine.exe info --json`)
// ---------------------------------------------------------------------------
std::string extract_json_object(const std::string& text) {
  // Engine output may mix streaming "[scan] ..." progress lines (stderr) with
  // the final JSON object on stdout.  The JSON object starts at a line start.
  size_t pos = text.rfind("\n{");
  if (pos != std::string::npos) return text.substr(pos + 1);
  pos = text.find('{');
  return pos == std::string::npos ? std::string() : text.substr(pos);
}

bool parse_model_meta(const std::string& json, ModelMeta* out) {
  JVal root;
  JsonParser parser(json);
  if (!parser.parse(&root) || !root.is_obj()) return false;
  ModelMeta m;
  m.loaded = true;
  m.path = jstr(&root, "model");
  m.arch = jstr(&root, "arch");
  m.validation = jstr(&root, "validation", "?");
  m.n_layer = static_cast<int>(jint(&root, "n_layer"));
  m.hidden = static_cast<int>(jint(&root, "hidden"));
  m.ffn = static_cast<int>(jint(&root, "ffn"));
  m.vocab = juint(&root, "vocab");
  m.ctx_train = juint(&root, "ctx_train");
  m.kv_layers = static_cast<int>(jint(&root, "kv_layers"));
  m.mtp_layers = static_cast<int>(jint(&root, "mtp_layers"));
  const JVal* attn = jget(&root, "attn");
  m.attn_variant = jstr(attn, "variant");
  m.n_head = static_cast<int>(jint(attn, "n_head"));
  m.n_head_kv = static_cast<int>(jint(attn, "n_head_kv"));
  m.klen = static_cast<int>(jint(attn, "key_length"));
  m.vlen = static_cast<int>(jint(attn, "value_length"));
  const JVal* moe = jget(&root, "moe");
  m.n_expert = static_cast<int>(jint(moe, "n_expert"));
  m.n_expert_used = static_cast<int>(jint(moe, "n_expert_used"));
  m.expert_ffn = static_cast<int>(jint(moe, "expert_ffn_length"));
  const JVal* bytes = jget(&root, "bytes");
  m.total_bytes = juint(bytes, "total");
  m.fixed_bytes = juint(bytes, "fixed");
  m.expert_bytes = juint(bytes, "experts");
  const JVal* per = jget(&root, "expert_per_layer");
  if (per != nullptr && per->is_obj()) {
    for (const auto& kv : per->obj) {
      if (kv.second.is_num()) m.expert_per_layer.push_back(static_cast<uint64_t>(kv.second.num));
    }
  }
  const JVal* kvb = jget(&root, "kv_per_token");
  m.kv_per_token[0] = juint(kvb, "f32");
  m.kv_per_token[1] = juint(kvb, "q8_0");
  m.kv_per_token[2] = juint(kvb, "q4_0");
  const JVal* sys = jget(&root, "system");
  m.vram_valid = jbool(sys, "vram_valid");
  m.host_valid = jbool(sys, "host_valid");
  m.vram_total = juint(sys, "vram_total");
  m.vram_free = juint(sys, "vram_free");
  m.host_total = juint(sys, "host_total");
  m.host_free = juint(sys, "host_free");
  m.valid = m.validation == "PASS";
  *out = std::move(m);
  return true;
}

void update_model_info_labels();  // forward

void refresh_model_combo_labels();
void load_model_meta(bool update_ui) {
  const std::wstring path = current_model_path();
  if (path.empty() || g_engine.empty()) {
    g_meta = ModelMeta();
    g_meta_error = path.empty() ? L"（未选择模型）" : L"找不到 engine.exe";
    if (update_ui) update_model_info_labels();
    return;
  }
  if (update_ui) {
    if (g_model_info != nullptr) {
      SetWindowTextW(g_model_info, L"读取模型元数据中…");
      set_label_role(g_model_info, Role::Dim);
    }
    UpdateWindow(g_model_info);
  }
  const std::string out = engine_capture(L"info --json " + quote_arg(path), 30000);
  const std::string json = extract_json_object(out);
  ModelMeta m;
  if (!parse_model_meta(json, &m)) {
    g_meta = ModelMeta();
    g_meta_error = json.empty() ? L"engine info --json 无输出" : L"engine info --json 解析失败";
    if (update_ui) update_model_info_labels();
    return;
  }
  g_meta = m;
  g_meta_error.clear();
  if (g_ui.gpu_auto) g_ui.gpu_layers = 0;
  if (g_ui.gpu_layers < 0) g_ui.gpu_layers = 0;
  if (g_ui.gpu_layers > static_cast<int>(g_meta.expert_per_layer.size()))
    g_ui.gpu_layers = static_cast<int>(g_meta.expert_per_layer.size());
  if (update_ui) {
    update_model_info_labels();
    refresh_model_combo_labels();
  }
}

void update_model_info_labels() {
  if (g_model_info == nullptr) return;
  if (!g_meta.loaded) {
    SetWindowTextW(g_model_info, g_meta_error.empty() ? L"（未选择模型）" : g_meta_error.c_str());
    set_label_role(g_model_info, g_meta_error.empty() ? Role::Dim : Role::Danger);
    if (g_engine_placement != nullptr) SetWindowTextW(g_engine_placement, L"");
    return;
  }
  std::wstring text = L"架构 " + widen(g_meta.arch) + L" · " + wnum(g_meta.n_layer) + L" 层 · hidden " +
                      wnum(g_meta.hidden) + L" · ";
  if (g_meta.expert_per_layer.empty()) text += L"dense（无 MoE 专家层）";
  else text += L"MoE " + wnum(static_cast<int>(g_meta.expert_per_layer.size())) + L" 层 × " +
               wnum(g_meta.n_expert) + L" 专家（每 token " + wnum(g_meta.n_expert_used) + L" 个）";
  text += L" · 权重 " + fmt_size(g_meta.total_bytes) + L"（fixed " + fmt_size(g_meta.fixed_bytes) +
          L" + experts " + fmt_size(g_meta.expert_bytes) + L"） · KV 层 " +
          wnum(g_meta.kv_layers) + L" · 校验 " + widen(g_meta.validation);
  SetWindowTextW(g_model_info, text.c_str());
  set_label_role(g_model_info, g_meta.valid ? Role::Dim : Role::Danger);
}

// ---------------------------------------------------------------------------
// Estimates -> labels
// ---------------------------------------------------------------------------
extern HWND g_label_gpu;  // defined with the other main-panel labels below

// A dense hybrid model (Qwen3.5-9B) has no MoE expert layers at all: moe.n_expert
// is 0 / expert_per_layer is empty or all zero.  GPU/CPU expert splitting does
// not apply to it, so the GPU-layer controls are disabled with an explanation
// instead of looking broken.
bool model_is_dense() {
  if (!g_meta.loaded) return false;
  if (g_meta.n_expert <= 0) return true;
  if (g_meta.expert_per_layer.empty()) return true;
  for (uint64_t bytes : g_meta.expert_per_layer)
    if (bytes != 0) return false;
  return true;
}

void update_args_summary();

void update_gpu_controls() {
  if (g_gpu_slider == nullptr) return;
  const bool dense = model_is_dense();
  const int n_moe = static_cast<int>(g_meta.expert_per_layer.size());
  if (!g_meta.loaded) {
    EnableWindow(g_gpu_slider, FALSE);
    EnableWindow(g_gpu_auto, FALSE);
    if (g_vram_auto != nullptr) {
      EnableWindow(g_vram_auto, FALSE);
      if (g_vram_auto_warn != nullptr) ShowWindow(g_vram_auto_warn, SW_HIDE);
    }
    SetWindowTextW(g_gpu_text, L"（选择模型后显示显存估算）");
    set_label_role(g_gpu_text, Role::Dim);
    return;
  }
  if (dense) {
    // Nothing to split: disable + say why, and force the visible auto switch
    // OFF so it does not look like an unclickable checked checkbox.
    EnableWindow(g_gpu_slider, FALSE);
    EnableWindow(g_gpu_auto, FALSE);
    toggle_set(g_gpu_auto, false);
    slider_range(g_gpu_slider, 0, 1);
    slider_set(g_gpu_slider, 0, false);
    slider_ticks(g_gpu_slider, 0);
    if (g_label_gpu != nullptr) set_label_role(g_label_gpu, Role::Dim);
    SetWindowTextW(g_gpu_text,
                   L"该模型是稠密模型（无 MoE 专家层），全部权重都在显存，无需 GPU/CPU 分层");
    set_label_role(g_gpu_text, Role::Warn);
    // vram_auto_layers promotes MoE expert layers -> meaningless for dense.
    if (g_vram_auto != nullptr) {
      EnableWindow(g_vram_auto, FALSE);
      if (g_ui.vram_auto_layers) {
        g_ui.vram_auto_layers = false;
        update_args_summary();
      }
      toggle_set(g_vram_auto, false);
      if (g_vram_auto_warn != nullptr) ShowWindow(g_vram_auto_warn, SW_HIDE);
    }
    return;
  }
  // MoE model: re-enable the controls and show the live layer estimate.
  EnableWindow(g_gpu_slider, TRUE);
  EnableWindow(g_gpu_auto, TRUE);
  if (g_label_gpu != nullptr) set_label_role(g_label_gpu, Role::Normal);
  if (g_vram_auto != nullptr) EnableWindow(g_vram_auto, TRUE);
  toggle_set(g_gpu_auto, g_ui.gpu_auto);
  if (!g_ui.gpu_auto) g_ui.gpu_layers = std::max(0, std::min(g_ui.gpu_layers, n_moe));
  slider_range(g_gpu_slider, 0, n_moe);
  const int shown = g_ui.gpu_auto ? g_est.gpu_layers_auto : g_ui.gpu_layers;
  slider_set(g_gpu_slider, shown, false);
  slider_ticks(g_gpu_slider, 0);

  std::wstring text;
  if (g_ui.gpu_auto) text = L"自动（默认）：预计 GPU 专家层 ";
  else text = L"GPU 专家层 ";
  text += wnum(shown) + L"/" + wnum(n_moe);
  text += L" · 显存 ≈ " + fmt_size(g_est.vram_need) + L" / 可用 " + fmt_size(g_est.vram_usable);
  text += L"（显存共 " + fmt_size(g_est.vram_total) + L"，当前空闲 " + fmt_size(g_est.vram_free) +
          L"）";
  text += L" · CPU " + wnum(n_moe - shown) + L" 层 ≈ " + fmt_size(g_est.weights_on_host) +
          L" 主机内存";
  if (n_moe > 0) {
    const uint64_t avg = g_meta.expert_bytes / static_cast<uint64_t>(n_moe);
    text += L" · 每层 ≈ " + fmt_size(avg);
  }
  if (g_est.over_usable) {
    text += L"  ⚠ 超过可用显存，引擎实测会自动降到 " + wnum(g_est.max_fit_layers) + L" 层";
  }
  SetWindowTextW(g_gpu_text, text.c_str());
  set_label_role(g_gpu_text, g_est.over_usable ? Role::Danger : Role::Normal);
}

void update_ctx_controls() {
  if (g_ctx_slider == nullptr) return;
  const bool loaded = g_meta.loaded;
  const bool manual = loaded && !g_ui.ctx_auto;
  const int shown = g_ui.ctx_auto ? (g_est.auto_ctx > 0 ? g_est.auto_ctx : 4096) : g_ui.ctx_value;
  slider_range(g_ctx_slider, 0, kCtxSliderSteps);
  slider_ticks(g_ctx_slider, 0);
  slider_set(g_ctx_slider, ctx_to_slider(shown), false);
  EnableWindow(g_ctx_slider, manual ? TRUE : FALSE);
  if (g_ctx_edit != nullptr) {
    EnableWindow(g_ctx_edit, manual ? TRUE : FALSE);
    // Only rewrite the box when it does not already show the value we are about to
    // display (auto numbers always rewrite; user input is never disturbed mid-typing).
    wchar_t cur[32] = {0};
    GetWindowTextW(g_ctx_edit, cur, 32);
    long long cur_v = 0;
    const bool cur_ok = parse_ll(cur, &cur_v);
    if (!g_ui.ctx_invalid && (!manual || !cur_ok || cur_v != shown)) {
      g_ctx_sync = true;
      SetWindowTextW(g_ctx_edit, wnum(shown).c_str());
      g_ctx_sync = false;
    }
  }
  if (g_ctx_auto != nullptr) toggle_set(g_ctx_auto, g_ui.ctx_auto);
  if (!loaded) {
    SetWindowTextW(g_ctx_text, L"（选择模型后显示 KV 显存估算；可输入任意正整数，如 1000 / 12345）");
    set_label_role(g_ctx_text, Role::Dim);
    return;
  }
  if (g_ui.ctx_invalid) {
    SetWindowTextW(g_ctx_text,
                   L"⚠ 上下文长度必须是 1..1048576 的整数（任意值均可；当前输入不能启动）");
    set_label_role(g_ctx_text, Role::Danger);
    return;
  }
  std::wstring text;
  if (g_ui.ctx_auto) {
    text = L"自动：预计最多 " + wnum(g_est.ctx_used) + L" tokens（由空闲显存推导；按 KV " +
           kKvShort[g_est.kv_type_used] + std::wstring(L" 计算）");
  } else {
    text = L"手动 " + wnum(g_est.ctx_used) + L" tokens";
  }
  text += L" · KV 显存 ≈ " + fmt_size(g_est.kv) + L"（每 token " +
          fmt_size(g_est.kv_per_token) + L"）· 权重+KV+工作区 ≈ " +
          fmt_size(g_est.need_fixed) + L" / 可用 " + fmt_size(g_est.vram_usable);
  bool danger = g_est.need_fixed > g_est.vram_usable;
  if (!g_ui.ctx_auto && g_est.ctx_fit > 0 && g_est.ctx_used > g_est.ctx_fit) {
    text += L"  ⚠ 超过当前显存可容纳的 " + wnum(g_est.ctx_fit) +
            L" tokens：引擎实测放不下时会自动夹紧并打印 requested→actual 账目";
    danger = true;
  }
  if (g_est.mla_kv_forced_f32) text += L" · MLA 模型强制 f32 KV";
  text += L" · 可输入任意整数（1..1048576）";
  SetWindowTextW(g_ctx_text, text.c_str());
  set_label_role(g_ctx_text, danger ? Role::Danger : Role::Normal);
}

void update_kv_hint() {
  if (g_kv_text == nullptr) return;
  static const wchar_t* hints[] = {
      L"f32：4 B/元素（默认，精度最高）",
      L"q8_0：1.06 B/元素，省 4 倍显存；40 步生成可保持逐 token 一致，但有量化误差",
      L"q4_0：0.56 B/元素，省 8 倍显存；量化误差更大"};
  SetWindowTextW(g_kv_text, hints[std::max(0, std::min(2, g_ui.kv_sel))]);
  set_label_role(g_kv_text, Role::Dim);
}

void update_ssd_state() {
  if (g_ssd_toggle == nullptr) return;
  const bool need = g_est.valid && g_est.kind == 2;
  const bool can_check = need && !g_est.dense;
  if (!can_check && g_ui.ssd) {
    g_ui.ssd = false;
    toggle_set(g_ssd_toggle, false);
  }
  EnableWindow(g_ssd_toggle, can_check ? TRUE : FALSE);
  std::wstring note;
  Role role = Role::Dim;
  if (!g_est.valid) {
    note = L"（选择模型后判定是否需要 SSD 流式）";
  } else if (g_est.kind == 0) {
    note = L"模型可装入显存+内存（放置决策：gpu）。无需 SSD —— 已自动禁用并置灰";
  } else if (g_est.kind == 1) {
    note = L"模型可装入显存+内存（放置决策：gpu / vram+ram split）。无需 SSD —— 已自动禁用并置灰";
  } else if (g_est.dense) {
    note = L"dense 模型装不进显存，此构建没有 dense 主机/SSD 权重路径，SSD 流式必然失败";
    role = Role::Warn;
  } else {
    note = L"模型装不进显存+内存：可勾选 SSD 流式。注意：引擎报告「ssd 流式未实现」（接口占位），"
           L"勾选后引擎会报错退出";
    role = Role::Warn;
  }
  if (g_ui.ssd && can_check) {
    note = L"已请求 SSD 流式（--ssd-stream）。引擎报告：ssd 流式未实现 —— 启动后引擎会报错退出，"
           L"日志见 build/launcher-engine.log";
    role = Role::Danger;
  }
  SetWindowTextW(g_ssd_note, note.c_str());
  set_label_role(g_ssd_note, role);
}

void update_est_summary() {
  if (g_est_summary == nullptr) return;
  if (!g_est.valid) {
    SetWindowTextW(g_est_summary, L"");
    return;
  }
  if (g_est.dense) {
    // Dense model: no expert placement decision to make; show the full VRAM
    // formula the user asked for instead of "gpu/vram+ram split".
    std::wstring text = L"显存占用 = 全部权重 + KV + 工作区 ≈ " + fmt_size(g_est.need_all) +
                        L" / 可用 " + fmt_size(g_est.vram_usable);
    text += L"（显存共 " + fmt_size(g_est.vram_total) + L"，当前空闲 " +
            fmt_size(g_est.vram_free) + L"）";
    SetWindowTextW(g_est_summary, text.c_str());
    set_label_role(g_est_summary,
                   g_est.need_all > g_est.vram_usable ? Role::Danger : Role::Dim);
    return;
  }
  static const wchar_t* kinds[] = {L"gpu（全部常驻显存）", L"vram+ram split（GPU/CPU 分层）",
                                   L"ssd stream（需要磁盘流式，引擎未实现）"};
  std::wstring text = L"放置决策（估算，引擎实测为准）：" +
                      std::wstring(kinds[std::max(0, std::min(2, g_est.kind))]);
  text += L" · 显存占用 ≈ " + fmt_size(g_est.vram_need) + L" / 可用 " + fmt_size(g_est.vram_usable);
  text += L" · 主机内存 ≈ " + fmt_size(g_est.weights_on_host) + L" / 可用 " +
          fmt_size(g_est.host_usable);
  if (g_est.over_usable) text += L" · ⚠ 当前 GPU 层数超出可用显存";
  SetWindowTextW(g_est_summary, text.c_str());
  set_label_role(g_est_summary, g_est.over_usable ? Role::Danger
                                                 : (g_est.kind == 2 ? Role::Warn : Role::Dim));
}

void update_status_message(const std::wstring& text, Role role);
void update_buttons_state();
bool model_scan_running();

void refresh_estimates(bool update_controls) {
  g_est = compute_estimate();
  if (update_controls) {
    update_gpu_controls();
    update_ctx_controls();
    update_ssd_state();
  }
  update_est_summary();
}

std::string json_escape_utf8(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (unsigned char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", c);
          out += b;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Schema info + coverage map (evidence for "全部 CLI 选项")
// ---------------------------------------------------------------------------
std::wstring g_ui_map_note;

std::string ui_map_json() {
  std::string json = "{\n  \"generated_by\": \"build/launcher.exe\",\n";
  json += "  \"schema_options\": " + std::to_string(static_cast<int>(g_opts.size())) + ",\n";
  size_t main_count = 0, adv_count = 0;
  for (const auto& od : g_opts) {
    if (is_curated_key(od.key)) ++main_count;
    else ++adv_count;
  }
  json += "  \"main_panel_controls\": " + std::to_string(static_cast<int>(main_count)) + ",\n";
  json += "  \"advanced_controls\": " + std::to_string(static_cast<int>(adv_count)) + ",\n";
  json += "  \"status\": \"";
  if (main_count + adv_count == g_opts.size()) {
    json += std::to_string(static_cast<int>(main_count + adv_count)) + "/" +
            std::to_string(static_cast<int>(g_opts.size())) + " (100%: " +
            std::to_string(static_cast<int>(main_count)) + " main-panel + " +
            std::to_string(static_cast<int>(adv_count)) + " advanced-panel)";
  } else {
    json += "MISMATCH";
  }
  json += "\",\n  \"main_panel\": [\n";
  bool first = true;
  for (const auto& od : g_opts) {
    if (!is_curated_key(od.key)) continue;
    if (!first) json += ",\n";
    first = false;
    json += "    {\"key\": \"" + od.key + "\", \"cli\": \"" + (od.cli.empty() ? std::string("-") : od.cli) +
            "\", \"type\": \"" + od.type + "\", \"control\": \"main-panel\"}";
  }
  json += "\n  ],\n  \"advanced\": [\n";
  first = true;
  for (const auto& od : g_opts) {
    if (is_curated_key(od.key)) continue;
    if (!first) json += ",\n";
    first = false;
    json += "    {\"key\": \"" + od.key + "\", \"cli\": \"" + (od.cli.empty() ? std::string("-") : od.cli) +
            "\", \"type\": \"" + od.type + "\", \"control\": \"advanced-panel\"}";
  }
  json += "\n  ],\n  \"serve_only_flags\": [\n";
  json += "    {\"flag\": \"--model auto\", \"ui\": \"模型下拉框的第一项「自动（最大模型）」\"},\n";
  json += "    {\"flag\": \"--pick N\", \"ui\": \"模型下拉框（引擎 models 顺序；等价选择）\"},\n";
  json += "    {\"flag\": \"--open-browser\", \"ui\": \"服务分组「启动后自动打开对话页」\"},\n";
  json += "    {\"flag\": \"--out/--json/--tsv/--plain\", \"ui\": \"CLI 输出格式开关，GUI 不需要\"},\n";
  json += "    {\"flag\": \"--ids/--n (run)\", \"ui\": \"run 子命令专用，serve 无此参数\"}\n";
  json += "  ],\n  \"tooltips\": [\n";
  bool first_tip = true;
  for (const auto& kv : g_tip_registry) {
    if (!first_tip) json += ",\n";
    first_tip = false;
    json += "    {\"control_id\": " + std::to_string(kv.first) + ", \"text\": \"" +
            json_escape_utf8(narrow(kv.second)) + "\"}";
  }
  json += "\n  ]\n}\n";
  return json;
}

void write_ui_map() {
  if (g_ui_map_path.empty()) return;
  write_file_utf8(g_ui_map_path, ui_map_json());
}

void update_schema_info() {
  if (g_schema_info == nullptr) return;
  if (!g_schema_loaded) {
    SetWindowTextW(g_schema_info, g_schema_error.empty() ? L"引擎配置 schema 未加载"
                                                        : g_schema_error.c_str());
    set_label_role(g_schema_info, Role::Danger);
    return;
  }
  size_t main_count = 0;
  for (const auto& od : g_opts)
    if (is_curated_key(od.key)) ++main_count;
  const int total = static_cast<int>(g_opts.size());
  const int main_n = static_cast<int>(main_count);
  const int adv_n = total - main_n;
  const std::wstring text = L"引擎选项 " + wnum(total) +
                            L" 项（--print-config --json）· 主面板 " + wnum(main_n) +
                            L" 项 · 高级面板 " + wnum(adv_n) + L" 项 · 覆盖 " +
                            wnum(total) + L" 项（100%）";
  SetWindowTextW(g_schema_info, text.c_str());
  set_label_role(g_schema_info, Role::Dim);
}

}  // namespace
// ---------------------------------------------------------------------------
// Engine process state (lifecycle lives in the last part; declared here because
// the UI update helpers below need it)
// ---------------------------------------------------------------------------
namespace {

bool g_running = false;
bool g_starting = false;
HANDLE g_proc = nullptr;
int g_running_port = 0;
ULONGLONG g_start_tick = 0;
bool g_auto_open = true;
std::wstring g_boot_error;
std::wstring g_engine_placement_text;
std::wstring g_last_log_tail;

// Extra labels
HWND g_label_gpu = nullptr;
HWND g_label_ctx = nullptr;
HWND g_label_kv = nullptr;
HWND g_label_threads = nullptr;
HWND g_label_port = nullptr;
HWND g_label_host = nullptr;
HWND g_label_presets = nullptr;
HWND g_label_adv_search = nullptr;
HWND g_preset_note = nullptr;

// ---------------------------------------------------------------------------
// Owner-draw helpers
// ---------------------------------------------------------------------------
bool is_toggle_id(UINT id) {
  switch (id) {
    case IDC_GPU_AUTO:
    case IDC_CTX_AUTO:
    case IDC_SSD_TOGGLE:
    case IDC_DEV_ROUTE:
    case IDC_LAYER_GRAPH:
    case IDC_CPU_INT8:
    case IDC_WIN_PRIORITY:
    case IDC_THINKING_LEN:
    case IDC_VRAM_AUTO:
    case IDC_WIN_AFFINITY:
    case IDC_AUTO_BROWSER:
      return true;
    default: break;
  }
  if (id >= IDC_ADV_BASE) {
    const auto it = g_adv_id_to_opt.find(id);
    if (it != g_adv_id_to_opt.end()) {
      const OptDef& od = g_opts[static_cast<size_t>(it->second)];
      return od.type == "bool";
    }
  }
  return false;
}

bool is_checkbox_style(UINT id) {
  if (id == IDC_GPU_AUTO || id == IDC_CTX_AUTO) return true;
  if (id < IDC_ADV_BASE) return false;
  const auto it = g_adv_id_to_opt.find(id);
  return it != g_adv_id_to_opt.end();
}

// toggle_is_on / toggle_set / toggle_flip live in the theme part: BS_OWNERDRAW
// buttons do not store a check state, the map does.

void draw_check_marks(HDC dc, RECT box) {
  HPEN pen = CreatePen(PS_SOLID, S(2), RGB(255, 255, 255));
  const HGDIOBJ old = SelectObject(dc, pen);
  const int x0 = box.left + (box.right - box.left) / 5;
  const int y0 = box.top + (box.bottom - box.top) / 2;
  const int x1 = box.left + (box.right - box.left) * 2 / 5;
  const int y1 = box.bottom - (box.bottom - box.top) / 4;
  const int x2 = box.right - (box.right - box.left) / 5;
  const int y2 = box.top + (box.bottom - box.top) / 4;
  MoveToEx(dc, x0, y0, nullptr);
  LineTo(dc, x1, y1);
  LineTo(dc, x2, y2);
  SelectObject(dc, old);
  DeleteObject(pen);
}

void draw_toggle_item(const DRAWITEMSTRUCT* dis) {
  HDC dc = dis->hDC;
  RECT r = dis->rcItem;
  const bool on = toggle_is_on(dis->hwndItem);
  const bool enabled = IsWindowEnabled(dis->hwndItem) != FALSE;
  const bool checkbox = is_checkbox_style(dis->CtlID);
  fill_rect(dc, r, g_pal.bg);
  const int h = r.bottom - r.top;
  const std::wstring text = [&] {
    wchar_t buf[256] = {0};
    GetWindowTextW(dis->hwndItem, buf, 256);
    return std::wstring(buf);
  }();
  if (checkbox) {
    const int box = S(18);
    RECT br{r.left + S(2), r.top + (h - box) / 2, r.left + S(2) + box, r.top + (h + box) / 2};
    rounded_fill(dc, br, S(4), on ? (enabled ? g_pal.accent : g_pal.accent_dim) : g_pal.input_bg,
                 enabled ? (on ? g_pal.accent : g_pal.border) : g_pal.border);
    if (on) draw_check_marks(dc, br);
    RECT tr{br.right + S(8), r.top, r.right, r.bottom};
    draw_text(dc, text, tr, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS,
              enabled ? (on ? g_pal.text : g_pal.dim) : g_pal.dim, g_font);
    return;
  }
  const int sw = S(42), sh = S(22);
  RECT track{r.left + S(2), r.top + (h - sh) / 2, r.left + S(2) + sw, r.top + (h + sh) / 2};
  const COLORREF track_c = !enabled ? g_pal.border : (on ? g_pal.accent : g_pal.toggle_off);
  rounded_fill(dc, track, sh / 2, track_c, track_c);
  const int knob = sh - S(4);
  const int knob_x = on ? track.right - knob - S(2) : track.left + S(2);
  RECT kr{knob_x, track.top + S(2), knob_x + knob, track.bottom - S(2)};
  const COLORREF knob_c = enabled ? (on ? RGB(255, 255, 255) : g_pal.dim) : g_pal.dim;
  rounded_fill(dc, kr, knob / 2, knob_c, knob_c);
  RECT tr{track.right + S(10), r.top, r.right, r.bottom};
  // Enabled switches keep the normal text colour: a greyed label means the
  // switch is disabled by the live estimate (e.g. SSD when the model fits).
  draw_text(dc, text, tr, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS,
            enabled ? g_pal.text : g_pal.dim, g_font);
}

void draw_button_item(const DRAWITEMSTRUCT* dis) {
  HDC dc = dis->hDC;
  const RECT r = dis->rcItem;
  const bool disabled = (dis->itemState & ODS_DISABLED) != 0;
  const bool pressed = (dis->itemState & ODS_SELECTED) != 0;
  const bool focused = (dis->itemState & ODS_FOCUS) != 0;
  const UINT id = dis->CtlID;
  wchar_t raw[512] = {0};
  GetWindowTextW(dis->hwndItem, raw, 512);
  const std::wstring text(raw);
  fill_rect(dc, r, g_pal.bg);
  const bool primary = (id == IDC_START);
  const bool danger = (id == IDC_STOP);
  COLORREF fill = g_pal.panel2;
  COLORREF border = g_pal.border;
  COLORREF color = g_pal.text;
  if (primary) {
    fill = disabled ? g_pal.panel2 : (pressed ? g_pal.accent_dim : g_pal.accent);
    border = fill;
    color = disabled ? g_pal.dim : RGB(255, 255, 255);
  } else if (danger) {
    fill = disabled ? g_pal.panel2 : (pressed ? RGB(190, 70, 70) : g_pal.danger);
    border = fill;
    color = disabled ? g_pal.dim : RGB(255, 255, 255);
  } else {
    fill = pressed ? g_pal.panel : g_pal.panel2;
    border = (id == IDC_ADV_TOGGLE || id == IDC_PRESET_FAST || id == IDC_PRESET_LOWVRAM ||
              id == IDC_PRESET_BALANCED)
                 ? g_pal.border
                 : g_pal.border;
  }
  RECT rr = r;
  rr.right -= 1;
  rr.bottom -= 1;
  rounded_fill(dc, rr, S(6), fill, border);
  draw_text(dc, text, rr, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS, color,
            primary || danger ? g_font_bold : g_font);
  if (focused && !disabled) {
    RECT fr{rr.left + S(3), rr.top + S(3), rr.right - S(3), rr.bottom - S(3)};
    DrawFocusRect(dc, &fr);
  }
}

void draw_combo_item(const DRAWITEMSTRUCT* dis) {
  HDC dc = dis->hDC;
  const RECT r = dis->rcItem;
  const bool selected = (dis->itemState & ODS_SELECTED) != 0;
  std::wstring text;
  if (dis->itemID != static_cast<UINT>(-1)) {
    const int len = static_cast<int>(SendMessageW(dis->hwndItem, CB_GETLBTEXTLEN,
                                                  static_cast<WPARAM>(dis->itemID), 0));
    if (len > 0) {
      std::vector<wchar_t> buf(static_cast<size_t>(len) + 2, L'\0');
      SendMessageW(dis->hwndItem, CB_GETLBTEXT, static_cast<WPARAM>(dis->itemID),
                   reinterpret_cast<LPARAM>(buf.data()));
      text.assign(buf.data(), static_cast<size_t>(len));
    }
  } else {
    const int sel = static_cast<int>(SendMessageW(dis->hwndItem, CB_GETCURSEL, 0, 0));
    if (sel >= 0) {
      wchar_t buf[512] = {0};
      SendMessageW(dis->hwndItem, CB_GETLBTEXT, static_cast<WPARAM>(sel),
                   reinterpret_cast<LPARAM>(buf));
      text = buf;
    }
  }
  fill_rect(dc, r, selected ? g_pal.accent_dim : g_pal.input_bg);
  RECT tr = r;
  tr.left += S(6);
  tr.right -= S(6);
  draw_text(dc, text, tr, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS,
            selected ? RGB(255, 255, 255) : g_pal.text, g_font);
  if (dis->itemID != static_cast<UINT>(-1) && (dis->itemState & ODS_FOCUS) != 0) {
    RECT fr = r;
    DrawFocusRect(dc, &fr);
  }
}

void draw_item(const DRAWITEMSTRUCT* dis) {
  if (dis == nullptr) return;
  if (dis->CtlType == ODT_COMBOBOX) {
    draw_combo_item(dis);
    return;
  }
  if (is_toggle_id(dis->CtlID)) {
    draw_toggle_item(dis);
    return;
  }
  draw_button_item(dis);
}

// ---------------------------------------------------------------------------
// Control creation
// ---------------------------------------------------------------------------
void sync_advanced_from_state();
void update_args_summary();
void refresh_estimates(bool update_controls);

void create_main_controls() {
  g_label_gpu = mk_static(g_hwnd, L"GPU 专家层数", 0, Role::Normal, 0, g_font);
  g_gpu_auto = mk_toggle(g_hwnd, L"自动（引擎自适应放置）", IDC_GPU_AUTO);
  g_gpu_slider = mk_slider(g_hwnd, IDC_GPU_SLIDER);
  g_gpu_text = mk_static(g_hwnd, L"", IDC_GPU_TEXT, Role::Dim, 0, g_font_small);
  g_label_ctx = mk_static(g_hwnd, L"上下文长度", 0, Role::Normal, 0, g_font);
  g_ctx_edit = mk_edit(g_hwnd, wnum(g_ui.ctx_value), IDC_CTX_EDIT, ES_NUMBER);
  g_ctx_auto = mk_toggle(g_hwnd, L"自动（按空闲显存推导）", IDC_CTX_AUTO);
  g_ctx_slider = mk_slider(g_hwnd, IDC_CTX_SLIDER);
  g_ctx_text = mk_static(g_hwnd, L"", IDC_CTX_TEXT, Role::Dim, 0, g_font_small);
  g_label_kv = mk_static(g_hwnd, L"KV 量化", 0, Role::Normal, 0, g_font);
  g_kv_combo = mk_combo(g_hwnd, IDC_KV_COMBO);
  SetWindowTheme(g_kv_combo, g_dark ? L"DarkMode_Explorer" : nullptr, nullptr);
  g_kv_text = mk_static(g_hwnd, L"", IDC_KV_TEXT, Role::Dim, 0, g_font_small);
  g_est_summary = mk_static(g_hwnd, L"", IDC_EST_SUMMARY, Role::Dim, 0, g_font_small);

  g_model_combo = mk_combo(g_hwnd, IDC_MODEL_COMBO);
  SetWindowTheme(g_model_combo, g_dark ? L"DarkMode_Explorer" : nullptr, nullptr);
  g_refresh_btn = mk_button(g_hwnd, L"重新扫描（后台）", IDC_REFRESH);
  g_model_info = mk_static(g_hwnd, L"", IDC_MODEL_INFO, Role::Dim, SS_ENDELLIPSIS, g_font_small);
  g_engine_placement = mk_static(g_hwnd, L"", IDC_ENGINE_PLACEMENT, Role::Dim, SS_ENDELLIPSIS,
                                  g_font_small);
  g_scan_status = mk_static(g_hwnd, L"", IDC_SCAN_STATUS, Role::Dim, SS_ENDELLIPSIS,
                            g_font_small);

  g_label_threads = mk_static(g_hwnd, L"CPU 线程数", 0, Role::Normal, 0, g_font);
  g_threads_combo = mk_combo(g_hwnd, IDC_THREADS_COMBO);
  SetWindowTheme(g_threads_combo, g_dark ? L"DarkMode_Explorer" : nullptr, nullptr);
  SendMessageW(g_threads_combo, CB_ADDSTRING, 0,
               reinterpret_cast<LPARAM>(L"自动（跟随引擎默认）"));
  {
    const int max_threads = std::max(8, logical_cpus() * 2);
    for (int n = 1; n <= max_threads; ++n) {
      const std::wstring text = wnum(n);
      SendMessageW(g_threads_combo, CB_ADDSTRING, 0,
                   reinterpret_cast<LPARAM>(text.c_str()));
    }
  }
  g_threads_hint = mk_static(g_hwnd, L"", IDC_THREADS_HINT, Role::Dim, SS_ENDELLIPSIS,
                             g_font_small);

  g_ssd_toggle = mk_toggle(g_hwnd, L"SSD 加速（实验）", IDC_SSD_TOGGLE);
  g_dev_route = mk_toggle(g_hwnd, L"设备端路由 dev_route", IDC_DEV_ROUTE);
  g_layer_graph = mk_toggle(g_hwnd, L"逐层 CUDA Graph", IDC_LAYER_GRAPH);
  g_cpu_int8 = mk_toggle(g_hwnd, L"CPU int8 GEMV", IDC_CPU_INT8);
  g_win_prio = mk_toggle(g_hwnd, L"高优先级 win_priority", IDC_WIN_PRIORITY);
  g_thinking_len = mk_toggle(g_hwnd, L"不限思考/回复长度", IDC_THINKING_LEN);
  g_vram_auto = mk_toggle(g_hwnd, L"显存自动加层 vram_auto_layers", IDC_VRAM_AUTO);
  g_win_affinity = mk_toggle(g_hwnd, L"SMT 成对绑核 win_affinity", IDC_WIN_AFFINITY);
  g_vram_auto_warn = mk_static(g_hwnd, L"低显存余量风险：decode 期间可能仅剩 ~80 MiB", IDC_VRAM_AUTO_WARN,
                               Role::Danger, 0, g_font_small);
  ShowWindow(g_vram_auto_warn, SW_HIDE);
  g_ssd_note = mk_static(g_hwnd, L"", IDC_SSD_NOTE, Role::Dim, 0, g_font_small);
  g_label_presets = mk_static(g_hwnd, L"性能预设", 0, Role::Normal, 0, g_font);
  g_preset_fast = mk_button(g_hwnd, L"最快", IDC_PRESET_FAST);
  g_preset_low = mk_button(g_hwnd, L"最省显存", IDC_PRESET_LOWVRAM);
  g_preset_bal = mk_button(g_hwnd, L"平衡", IDC_PRESET_BALANCED);
  g_preset_note = mk_static(g_hwnd, L"", 0, Role::Dim, SS_ENDELLIPSIS, g_font_small);

  g_label_port = mk_static(g_hwnd, L"端口", 0, Role::Normal, 0, g_font);
  g_port_edit = mk_edit(g_hwnd, wnum(g_ui.port), IDC_PORT_EDIT, ES_NUMBER);
  g_label_host = mk_static(g_hwnd, L"监听地址", 0, Role::Normal, 0, g_font);
  g_host_edit = mk_edit(g_hwnd, g_ui.host, IDC_HOST_EDIT);
  g_auto_browser = mk_toggle(g_hwnd, L"启动后自动打开对话页", IDC_AUTO_BROWSER);

  g_label_adv_search = mk_static(g_hwnd, L"搜索", 0, Role::Normal, 0, g_font);
  g_adv_toggle = mk_button(g_hwnd, L"▶ 高级：全部引擎参数（展开）", IDC_ADV_TOGGLE);
  g_adv_count = mk_static(g_hwnd, L"", IDC_ADV_COUNT, Role::Dim, SS_ENDELLIPSIS, g_font_small);
  g_adv_search = mk_edit(g_hwnd, L"", IDC_ADV_SEARCH);
  g_footer = mk_static(g_hwnd, L"", IDC_FOOTER, Role::Dim, SS_ENDELLIPSIS, g_font_small);

  g_start = mk_button(g_hwnd, L"启动引擎", IDC_START);
  g_stop = mk_button(g_hwnd, L"停止", IDC_STOP);
  g_open_page = mk_button(g_hwnd, L"打开对话页", IDC_OPEN_PAGE);
  g_open_log = mk_button(g_hwnd, L"打开日志", IDC_OPEN_LOG);
  g_status = mk_static(g_hwnd, L"就绪", IDC_STATUS, Role::Normal, SS_ENDELLIPSIS, g_font_bold);
  g_url = mk_static(g_hwnd, L"", IDC_URL, Role::Accent, SS_ENDELLIPSIS, g_font_small);
  g_log_info = mk_static(g_hwnd, L"", IDC_LOG_INFO, Role::Dim, SS_ENDELLIPSIS, g_font_small);
  g_args_summary = mk_static(g_hwnd, L"", IDC_ARGS_SUMMARY, Role::Dim, SS_ENDELLIPSIS,
                             g_font_small);
  g_schema_info = mk_static(g_hwnd, L"", IDC_SCHEMA_INFO, Role::Dim, SS_ENDELLIPSIS,
                            g_font_small);

  // KV combo contents
  for (const wchar_t* kv : kKvNames)
    SendMessageW(g_kv_combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(kv));
  SendMessageW(g_kv_combo, CB_SETCURSEL, static_cast<WPARAM>(g_ui.kv_sel), 0);

  // Tooltips: the engine's own option help where a mapping exists.
  auto tip_for = [](const char* key, const std::wstring& fallback) -> std::wstring {
    const OptDef* od = find_opt(key);
    if (od == nullptr) return fallback;
    std::wstring s = L"--" + (od->cli.empty() ? widen(od->key) : widen(od->cli)) + L"  (config: " +
                     widen(od->key) + L", 默认 " + widen(od->def) + L")\n" + widen(od->help);
    return s;
  };
  add_tip(g_model_combo, L"模型下拉框：第一项“自动（最大模型）”映射 --model auto；其余项来自 engine.exe models --json");
  add_tip(g_refresh_btn, L"后台重新全盘扫描（engine.exe models --refresh）："
                         L"枚举所有固定盘/移动盘、跳过系统目录，扫描结果写入 "
                         L"engine-models-cache.json；扫描期间可继续用下拉里的已有项");
  add_tip(g_label_gpu, tip_for("gpu_layers", L""));
  add_tip(g_gpu_slider, tip_for("gpu_layers", L""));
  add_tip(g_gpu_auto, tip_for("gpu_layers", L"") + L"\n自动 = 引擎按实测空闲显存逐层放置（--gpu-layers -1）");
  add_tip(g_label_ctx, tip_for("ctx", L""));
  add_tip(g_ctx_slider, tip_for("ctx", L""));
  add_tip(g_ctx_edit,
          L"直接输入任意上下文长度（1..1048576 tokens，KV 按它分配）。"
          L"引擎只在显存物理放不下时才夹紧，并在日志打印 requested→actual 账目；"
          L"/props 的 n_ctx_requested / actual_ctx 也可查询。");
  add_tip(g_ctx_auto, L"勾选后由空闲显存自动推导最大可用 ctx（等价 --ctx 0）。");
  add_tip(g_label_kv, tip_for("kv_type", L""));
  add_tip(g_kv_combo, tip_for("kv_type", L""));
  add_tip(g_label_threads, tip_for("threads", L""));
  add_tip(g_threads_combo, tip_for("threads", L"") +
                               L"\n\n默认「自动」不传 --threads，引擎按自身 win_affinity/SMT "
                               L"拓扑计算默认线程数（本机 6C/12T -> 12 线程）。显式数字会覆盖引擎默认。");
  add_tip(g_win_affinity,
          L"按 SMT 兄弟对绑核（--win-affinity 2），实测 CPU 专家层快 8–12%；关闭用于排查/对比"
          L"（--win-affinity 0，交给系统调度）。引擎默认开启。");
  add_tip(g_ssd_toggle, tip_for("ssd_stream", L""));
  add_tip(g_ssd_note, tip_for("ssd_stream", L""));
  add_tip(g_dev_route, tip_for("dev_route", L""));
  add_tip(g_layer_graph, tip_for("layer_graph_auto", L""));
  add_tip(g_cpu_int8, tip_for("cpu_int8", L""));
  add_tip(g_win_prio, tip_for("win_priority", L""));
  add_tip(g_thinking_len, tip_for("default_max_tokens", L""));
  const std::wstring vram_auto_tip =
      L"装载后按实测显存自动多放 1 个 GPU 层；可省 ~0.7 ms/token，但 decode 期间显存余量会低至 ~80 MiB，极端情况下触发驱动分页导致严重变慢。建议只在确认没有其他程序占用显存时开启。serve 场景下（ctx≥4096）不建议开启：实测 35B 会在 CUDA graph 捕获时 OOM。";
  add_tip(g_vram_auto, vram_auto_tip);
  add_tip(g_vram_auto_warn, vram_auto_tip);
  add_tip(g_label_port, tip_for("serve_port", L""));
  add_tip(g_port_edit, tip_for("serve_port", L""));
  add_tip(g_label_host, tip_for("serve_host", L""));
  add_tip(g_host_edit, tip_for("serve_host", L""));
  add_tip(g_auto_browser, L"启动后等服务就绪，自动用默认浏览器打开 http://<host>:<port>/（对应于引擎的 --open-browser）");
  add_tip(g_preset_fast, L"一组常用开关组合：全部显存 + f32 KV + 高优先级 + 逐层 CUDA Graph");
  add_tip(g_preset_low, L"一组常用开关组合：KV q4_0 + GPU 专家层 0（全部走 CPU）+ ctx 4096");
  add_tip(g_preset_bal, L"一组常用开关组合：KV q8_0 + 自动放置 + ctx 8192");
  add_tip(g_adv_toggle, L"展开/收起由 engine.exe serve --print-config --json 动态生成的全部高级参数");
  add_tip(g_adv_search, L"按配置键 / CLI 名 / 说明过滤高级参数");
  add_tip(g_start, L"启动 engine.exe serve（命令行与日志见 window 下方状态区）");
  add_tip(g_stop, L"POST /shutdown 让引擎优雅退出；超时则终止该子进程");
  add_tip(g_open_page,
          L"打开 http://<host>:<port>/ 的对话页。首次打开会加载 llama.cpp 的界面；发出消息后首字有 0.2–0.5s 延迟是正常的（模型 prefill），不是卡死。");
  add_tip(g_open_log, L"用记事本打开 build/launcher-engine.log");
}

void create_all_controls() {
  create_main_controls();
  if (g_schema_loaded) create_advanced_controls();
  // hide advanced rows until the panel is expanded
  for (auto& cell : g_adv_cells) {
    if (cell.label != nullptr) ShowWindow(cell.label, SW_HIDE);
    if (cell.ctrl != nullptr) ShowWindow(cell.ctrl, SW_HIDE);
  }
  if (g_adv_search != nullptr) ShowWindow(g_adv_search, SW_HIDE);
  if (g_label_adv_search != nullptr) ShowWindow(g_label_adv_search, SW_HIDE);
  if (g_preset_note != nullptr) ShowWindow(g_preset_note, SW_HIDE);
  update_schema_info();
  write_ui_map();
}

// ---------------------------------------------------------------------------
// State -> controls
// ---------------------------------------------------------------------------
void sync_advanced_from_state() {
  for (auto& cell : g_adv_cells) {
    if (!cell.created || cell.ctrl == nullptr) continue;
    const OptDef& od = g_opts[static_cast<size_t>(cell.opt_index)];
    const auto it = g_opt_state.find(od.key);
    if (it == g_opt_state.end()) continue;
    if (od.type == "bool") {
      toggle_set(cell.ctrl, it->second.b);
    } else if (od.type == "string" && enum_values_for(od.key) != nullptr) {
      int sel = 0;
      const std::wstring cur = widen(it->second.text);
      for (size_t k = 0; k < enum_values_for(od.key)->size(); ++k)
        if ((*enum_values_for(od.key))[k] == cur) sel = static_cast<int>(k);
      SendMessageW(cell.ctrl, CB_SETCURSEL, static_cast<WPARAM>(sel), 0);
    } else {
      SetWindowTextW(cell.ctrl, widen(it->second.text).c_str());
    }
  }
}

void sync_controls_from_state() {
  if (g_model_combo != nullptr) {
    const int sel = g_ui.model_auto ? 0 : (g_ui.model_index >= 0 ? g_ui.model_index + 1 : 0);
    SendMessageW(g_model_combo, CB_SETCURSEL, static_cast<WPARAM>(sel), 0);
  }
  toggle_set(g_gpu_auto, g_ui.gpu_auto);
  slider_set(g_gpu_slider, g_ui.gpu_auto ? g_est.gpu_layers_auto : g_ui.gpu_layers, false);
  update_ctx_controls();
  SendMessageW(g_kv_combo, CB_SETCURSEL, static_cast<WPARAM>(g_ui.kv_sel), 0);
  toggle_set(g_ssd_toggle, g_ui.ssd);
  toggle_set(g_dev_route, g_ui.dev_route);
  toggle_set(g_layer_graph, g_ui.layer_graph_auto);
  toggle_set(g_cpu_int8, g_ui.cpu_int8 != 0);
  toggle_set(g_win_prio, g_ui.win_priority != 0);
  toggle_set(g_thinking_len, g_ui.unlimited_tokens);
  toggle_set(g_vram_auto, g_ui.vram_auto_layers);
  ShowWindow(g_vram_auto_warn, g_ui.vram_auto_layers ? SW_SHOW : SW_HIDE);
  toggle_set(g_auto_browser, g_ui.auto_browser);
  if (g_ui.threads_auto) {
    SendMessageW(g_threads_combo, CB_SETCURSEL, 0, 0);
  } else if (g_ui.threads >= 1 &&
             g_ui.threads <= static_cast<int>(SendMessageW(g_threads_combo, CB_GETCOUNT, 0, 0)) - 1) {
    SendMessageW(g_threads_combo, CB_SETCURSEL, static_cast<WPARAM>(g_ui.threads), 0);
  } else {
    SendMessageW(g_threads_combo, CB_SETCURSEL, 0, 0);
  }
  toggle_set(g_win_affinity, g_ui.win_affinity);
  SetWindowTextW(g_port_edit, wnum(g_ui.port).c_str());
  SetWindowTextW(g_host_edit, g_ui.host.c_str());
  const std::wstring threads_hint =
      g_ui.threads_auto
          ? std::wstring(L"自动：不传 --threads，跟随引擎默认（本机 6C/12T → 12 线程）")
          : (L"显式 " + wnum(g_ui.threads) + L" 线程（覆盖引擎默认；物理核 " +
             wnum(default_thread_count() / 2) + L"）");
  SetWindowTextW(g_threads_hint, threads_hint.c_str());
  sync_advanced_from_state();
  update_kv_hint();
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------
void apply_preset(int kind) {
  if (kind == 0) {  // 最快
    g_ui.gpu_auto = true;
    g_ui.ctx_auto = false;
    g_ui.ctx_value = 4096;
    g_ui.ctx_invalid = false;
    g_ui.kv_sel = 0;
    g_ui.dev_route = true;
    g_ui.layer_graph_auto = true;
    g_ui.vram_auto_layers = false;
    g_ui.cpu_int8 = 1;
    g_ui.win_priority = 1;
    g_ui.unlimited_tokens = true;
    g_ui.threads_auto = true;
    g_ui.win_affinity = true;
    g_ui.ssd = false;
    g_ui.preset_note = L"已应用「最快」：全部权重进显存（自动分层）、f32 KV、高优先级、逐层 CUDA Graph";
  } else if (kind == 1) {  // 最省显存
    g_ui.gpu_auto = false;
    g_ui.gpu_layers = 0;
    g_ui.ctx_auto = false;
    g_ui.ctx_value = 4096;
    g_ui.ctx_invalid = false;
    g_ui.kv_sel = 2;
    g_ui.dev_route = true;
    g_ui.layer_graph_auto = true;
    g_ui.vram_auto_layers = false;
    g_ui.cpu_int8 = 1;
    g_ui.win_priority = 1;
    g_ui.unlimited_tokens = true;
    g_ui.threads_auto = true;
    g_ui.win_affinity = true;
    g_ui.ssd = false;
    g_ui.preset_note = L"已应用「最省显存」：KV q4_0、GPU 专家层 0（MoE 专家全部走 CPU）、ctx 4096";
  } else {  // 平衡
    g_ui.gpu_auto = true;
    g_ui.ctx_auto = false;
    g_ui.ctx_value = 8192;
    g_ui.ctx_invalid = false;
    g_ui.kv_sel = 1;
    g_ui.dev_route = true;
    g_ui.layer_graph_auto = true;
    g_ui.vram_auto_layers = false;
    g_ui.cpu_int8 = 1;
    g_ui.win_priority = 1;
    g_ui.unlimited_tokens = true;
    g_ui.threads_auto = true;
    g_ui.win_affinity = true;
    g_ui.ssd = false;
    g_ui.preset_note = L"已应用「平衡」：自动放置、KV q8_0、ctx 8192";
  }
  sync_controls_from_state();
  refresh_estimates(true);
  update_args_summary();
  SetWindowTextW(g_preset_note, g_ui.preset_note.c_str());
  ShowWindow(g_preset_note, SW_SHOW);
  update_status_message(g_ui.preset_note, Role::Accent);
}

// ---------------------------------------------------------------------------
// Status / buttons / args summary
// ---------------------------------------------------------------------------
void update_status_message(const std::wstring& text, Role role) {
  if (g_status == nullptr) return;
  SetWindowTextW(g_status, text.c_str());
  set_label_role(g_status, role);
}

void update_url_line() {
  if (g_url == nullptr) return;
  if (g_running && g_running_port > 0) {
    const std::wstring url = L"对话页：http://" + g_ui.host + L":" + wnum(g_running_port) + L"/";
    SetWindowTextW(g_url, url.c_str());
    set_label_role(g_url, Role::Accent);
  } else {
    SetWindowTextW(g_url,
                   L"↓ 向下滚动查看「服务」「高级（全部引擎参数）」；悬停任意控件可看引擎原始说明");
    set_label_role(g_url, Role::Dim);
  }
}

void update_buttons_state() {
  const bool has_model = !current_model_path().empty();
  if (g_start != nullptr)
    EnableWindow(g_start, (!g_running && !g_starting && has_model && !g_ui.busy) ? TRUE : FALSE);
  if (g_stop != nullptr) EnableWindow(g_stop, (g_running || g_starting) ? TRUE : FALSE);
  if (g_open_page != nullptr) EnableWindow(g_open_page, g_running ? TRUE : FALSE);
  if (g_refresh_btn != nullptr)
    EnableWindow(g_refresh_btn,
                 (!g_running && !g_starting && !model_scan_running()) ? TRUE : FALSE);
  if (g_model_combo != nullptr)
    EnableWindow(g_model_combo, (!g_running && !g_starting) ? TRUE : FALSE);
}

void update_args_summary() {
  if (g_args_summary == nullptr) return;
  const LaunchSpec spec = build_launch_spec(false);
  SetWindowTextW(g_args_summary, (L"启动参数：" + spec.summary).c_str());
  if (g_log_info != nullptr) {
    const std::wstring log_line = L"引擎日志：" + spec.log_path + L"     配置覆盖：" +
                                  spec.cfg_path;
    SetWindowTextW(g_log_info, log_line.c_str());
  }
}

void persist_effective_args() {
  const LaunchSpec spec = build_launch_spec(true);
  const std::wstring path = join_path(g_launcher_dir, L"launcher-effective-args.txt");
  std::string text = narrow(spec.cmdline) + "\n";
  text += "# summary: " + narrow(spec.summary) + "\n";
  text += "# advanced overrides: " + std::to_string(spec.advanced_overrides) + "\n";
  write_file_utf8(path, text);
  if (g_args_summary != nullptr) {
    SetWindowTextW(g_args_summary, (L"启动参数：" + spec.summary).c_str());
  }
}

// ---------------------------------------------------------------------------
// Model list loading
// ---------------------------------------------------------------------------
void update_model_combo_selection() {
  if (g_model_combo == nullptr) return;
  const int sel = g_ui.model_auto ? 0 : (g_ui.model_index >= 0 ? g_ui.model_index + 1 : 0);
  SendMessageW(g_model_combo, CB_SETCURSEL, static_cast<WPARAM>(sel), 0);
}

// A model larger than 80% of (VRAM + host RAM) cannot be layered fast: most
// experts/weights end up on the CPU or need SSD streaming.  Warn in the drop-down
// so picking it does not look like "the engine is just slow".
bool oversized_model(const ModelEntry& m) {
  if (!g_meta.loaded || g_meta.vram_total == 0 || g_meta.host_total == 0) return false;
  const double total = static_cast<double>(g_meta.vram_total + g_meta.host_total);
  return total > 0 && static_cast<double>(m.size_bytes) > total * 0.8;
}

std::wstring model_combo_label(const ModelEntry& m) {
  wchar_t prefix[96] = {0};
  const double gb = static_cast<double>(m.size_bytes) / 1e9;
  std::wstring type = L"dense";
  if (m.type.find(L"MoE") != std::wstring::npos || m.type.find(L"moe") != std::wstring::npos)
    type = L"MoE";
  std::wstring drive = m.drive;
  if (drive.empty() && m.path.size() >= 2 && m.path[1] == L':')
    drive = m.path.substr(0, 2);
  std::swprintf(prefix, 96, L"[%ls] %.2f GB · %ls · ", drive.empty() ? L"?" : drive.c_str(),
                gb, type.c_str());
  std::wstring label = prefix + m.name;
  if (oversized_model(m)) {
    wchar_t warn[96] = {0};
    std::swprintf(warn, 96, L"（⚠ %.1fG 超过显存+内存 80%%，会非常慢）", gb);
    label += warn;
  }
  if (m.params == L"9B" && gb < 10.0)
    label += L"（推荐，日常对话最快）";
  else if (gb >= 15.0)
    label += L"（最强，但慢）";
  else if (gb >= 4.0)
    label += L"（中等）";
  return label;
}

// Rebuild the drop-down labels (warning depends on the selected model's VRAM/RAM
// readings, which arrive with load_model_meta()).
void refresh_model_combo_labels() {
  if (g_model_combo == nullptr) return;
  const int cur = static_cast<int>(SendMessageW(g_model_combo, CB_GETCURSEL, 0, 0));
  SendMessageW(g_model_combo, CB_RESETCONTENT, 0, 0);
  SendMessageW(g_model_combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"自动（最大模型）"));
  for (const auto& m : g_models) {
    const std::wstring label = model_combo_label(m);
    SendMessageW(g_model_combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(label.c_str()));
  }
  if (cur >= 0) SendMessageW(g_model_combo, CB_SETCURSEL, static_cast<WPARAM>(cur), 0);
  else update_model_combo_selection();
}

bool apply_models_json(const std::string& out, bool preserve_selection) {
  const std::string json = extract_json_object(out);
  JVal root;
  JsonParser parser(json);
  if (json.empty() || !parser.parse(&root) || !root.is_obj()) return false;
  std::string source = jstr(&root, "source");
  std::wstring old_path;
  if (preserve_selection) old_path = current_model_path();
  std::vector<ModelEntry> models;
  const JVal* arr = jget(&root, "models");
  if (arr != nullptr && arr->is_arr()) {
    for (const auto& m : arr->arr) {
      ModelEntry e;
      e.path = widen(jstr(&m, "path"));
      e.name = widen(jstr(&m, "name"));
      e.type = widen(jstr(&m, "type"));
      e.quant = widen(jstr(&m, "quant"));
      e.params = widen(jstr(&m, "params"));
      e.drive = widen(jstr(&m, "drive"));
      e.size_bytes = juint(&m, "size_bytes");
      e.mtp = jbool(&m, "mtp");
      if (!e.path.empty()) models.push_back(std::move(e));
    }
  }
  g_models = std::move(models);
  SendMessageW(g_model_combo, CB_RESETCONTENT, 0, 0);
  SendMessageW(g_model_combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"自动（最大模型）"));
  for (const auto& m : g_models) {
    const std::wstring label = model_combo_label(m);
    SendMessageW(g_model_combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(label.c_str()));
  }
  if (g_models.empty()) {
    g_ui.model_index = -1;
    g_ui.model_auto = false;
    update_model_combo_selection();
    g_meta = ModelMeta();
    if (g_model_info != nullptr) {
      SetWindowTextW(g_model_info, L"没有扫描到模型（全盘扫描结果为空）");
      set_label_role(g_model_info, Role::Danger);
    }
    refresh_estimates(true);
    update_buttons_state();
    return false;
  }
  bool restored = false;
  if (preserve_selection) {
    for (int i = 0; i < static_cast<int>(g_models.size()); ++i) {
      if (!old_path.empty() && g_models[static_cast<size_t>(i)].path == old_path) {
        g_ui.model_auto = false;
        g_ui.model_index = i;
        restored = true;
        break;
      }
    }
  }
  static bool default_model_applied = false;
  if (!preserve_selection || !restored) {
    if (!default_model_applied) {
      default_model_applied = true;
      if (g_ui.model_auto) {
        int preferred = -1, fallback = -1;
        for (int i = 0; i < static_cast<int>(g_models.size()); ++i) {
          const ModelEntry& m = g_models[static_cast<size_t>(i)];
          const double gb = static_cast<double>(m.size_bytes) / 1e9;
          if (m.params == L"9B" && gb < 10.0) {
            preferred = i;
            break;
          }
          if (fallback < 0 && gb >= 4.0 && gb < 15.0) fallback = i;
        }
        const int pick = preferred >= 0 ? preferred : fallback;
        if (pick >= 0) {
          g_ui.model_auto = false;
          g_ui.model_index = pick;
        }
      }
    } else if (preserve_selection && !restored) {
      // Keep the old index in range; never jump to a different model silently.
      if (g_ui.model_index >= static_cast<int>(g_models.size()))
        g_ui.model_index = static_cast<int>(g_models.size()) - 1;
      if (g_ui.model_index < 0 && !g_ui.model_auto) g_ui.model_index = 0;
    }
  }
  if (g_ui.model_index >= static_cast<int>(g_models.size())) g_ui.model_index = 0;
  if (!g_ui.model_auto && g_ui.model_index < 0) g_ui.model_index = 0;
  update_model_combo_selection();
  load_model_meta(true);
  refresh_estimates(true);
  update_buttons_state();
  update_args_summary();
  (void)source;
  return true;
}

// ---- background full-disk scan -------------------------------------------
struct ScanJob {
  std::thread thread;
  std::mutex mu;
  bool running = false;
  bool done = false;
  bool ok = false;
  std::string output;
  std::string pending;
  std::string progress;
  std::string error;
  int drives_done = 0;
  int drives_total = 0;
  int found = 0;
  int exit_code = 0;
};
ScanJob g_scan_job;

bool model_scan_running() { return g_scan_job.running; }

bool scan_line_int(const std::string& line, const char* key, int* out) {
  const std::string k = std::string(key) + "=";
  const size_t p = line.find(k);
  if (p == std::string::npos) return false;
  int v = 0;
  size_t i = p + k.size();
  while (i < line.size() && line[i] >= '0' && line[i] <= '9') {
    v = v * 10 + (line[i] - '0');
    ++i;
  }
  if (i == p + k.size()) return false;
  *out = v;
  return true;
}

void scan_worker() {
  const std::wstring args = L"models --refresh --progress --json";
  const ProcResult r = run_capture(
      g_engine, args, g_launcher_dir, 15 * 60 * 1000,
      [](const std::string& chunk) {
        if (chunk.empty()) return;
        std::lock_guard<std::mutex> lock(g_scan_job.mu);
        g_scan_job.output += chunk;
        g_scan_job.pending += chunk;
        size_t pos = 0;
        while ((pos = g_scan_job.pending.find('\n')) != std::string::npos) {
          const std::string line = g_scan_job.pending.substr(0, pos);
          g_scan_job.pending.erase(0, pos + 1);
          if (line.rfind("[scan] ", 0) != 0) continue;
          if (line.find("[scan] found ") == 0) ++g_scan_job.found;
          else if (line.find("[scan] progress ") == 0) {
            int a = 0, b = 0, c = 0;
            if (scan_line_int(line, "drives_done", &a)) g_scan_job.drives_done = a;
            if (scan_line_int(line, "drives_total", &b)) g_scan_job.drives_total = b;
            if (scan_line_int(line, "found", &c)) g_scan_job.found = c;
            if (b > 0) {
              wchar_t buf[128];
              std::swprintf(buf, 128, L"扫描中 %d/%d 盘 · 已找到 %d 个模型",
                            g_scan_job.drives_done, b, c);
              g_scan_job.progress = narrow(buf);
            }
          } else if (line.find("[scan] found ") == 0) {
            wchar_t buf[128];
            std::swprintf(buf, 128, L"扫描中 · 已找到 %d 个模型", g_scan_job.found);
            g_scan_job.progress = narrow(buf);
          }
        }
        if (g_scan_job.output.size() > (8u << 20)) g_scan_job.output.resize(8u << 20);
      });
  std::lock_guard<std::mutex> lock(g_scan_job.mu);
  g_scan_job.exit_code = static_cast<int>(r.exit_code);
  g_scan_job.ok = r.exit_code == 0 && r.out.find("\"models\"") != std::string::npos;
  if (!g_scan_job.ok) {
    g_scan_job.error = r.out.empty() ? "engine models --refresh 没有输出"
                                     : "engine models --refresh 失败（exit " +
                                           std::to_string(r.exit_code) + "）";
  }
  g_scan_job.output = r.out.empty() ? g_scan_job.output : r.out;
  g_scan_job.running = false;
  g_scan_job.done = true;
  if (g_hwnd != nullptr) PostMessageW(g_hwnd, WM_APP + 70, 0, 0);
}

void start_background_model_scan() {
  if (g_scan_job.running) return;
  if (g_scan_job.thread.joinable()) g_scan_job.thread.join();
  {
    std::lock_guard<std::mutex> lock(g_scan_job.mu);
    g_scan_job.running = true;
    g_scan_job.done = false;
    g_scan_job.ok = false;
    g_scan_job.output.clear();
    g_scan_job.pending.clear();
    g_scan_job.error.clear();
    g_scan_job.progress = narrow(L"正在扫描全盘…（约 1–3 分钟，可先用下拉里的已有项）");
    g_scan_job.drives_done = 0;
    g_scan_job.drives_total = 0;
    g_scan_job.found = 0;
  }
  if (g_refresh_btn != nullptr) {
    SetWindowTextW(g_refresh_btn, L"扫描中…");
    EnableWindow(g_refresh_btn, FALSE);
  }
  if (g_scan_status != nullptr)
    SetWindowTextW(g_scan_status, L"正在扫描全盘…（约 1–3 分钟，可先用下拉里的已有项）");
  if (!g_running && !g_starting)
    update_status_message(L"正在扫描全盘…（约 1–3 分钟，可先用下拉里的已有项）", Role::Accent);
  g_scan_job.thread = std::thread(scan_worker);
  SetTimer(g_hwnd, 2, 400, nullptr);
}

void poll_model_scan() {
  if (!g_scan_job.running && !g_scan_job.done) return;
  std::string progress, output, error;
  int found = 0, done_drives = 0, total_drives = 0;
  bool done = false, ok = false;
  {
    std::lock_guard<std::mutex> lock(g_scan_job.mu);
    progress = g_scan_job.progress;
    found = g_scan_job.found;
    done_drives = g_scan_job.drives_done;
    total_drives = g_scan_job.drives_total;
    done = g_scan_job.done;
    ok = g_scan_job.ok;
    output = g_scan_job.output;
    error = g_scan_job.error;
    if (done) g_scan_job.done = false;
  }
  if (!done) {
    if (!progress.empty() && g_scan_status != nullptr)
      SetWindowTextW(g_scan_status, widen(progress).c_str());
    if (!g_running && !g_starting && !progress.empty())
      update_status_message(widen(progress), Role::Accent);
    return;
  }
  KillTimer(g_hwnd, 2);
  if (g_scan_job.thread.joinable()) g_scan_job.thread.join();
  if (g_refresh_btn != nullptr) {
    SetWindowTextW(g_refresh_btn, L"重新扫描（后台）");
    EnableWindow(g_refresh_btn, (!g_running && !g_starting) ? TRUE : FALSE);
  }
  size_t model_count = g_models.size();
  if (ok && apply_models_json(output, true)) {
    model_count = g_models.size();
    const std::wstring done_text = L"扫描完成：" + std::to_wstring(model_count) + L" 个模型 · " +
                                   std::to_wstring(total_drives > 0 ? total_drives : done_drives) +
                                   L" 个盘 · 缓存已更新";
    if (g_scan_status != nullptr) SetWindowTextW(g_scan_status, done_text.c_str());
    if (!g_running && !g_starting) update_status_message(done_text, Role::Normal);
  } else {
    if (g_scan_status != nullptr)
      SetWindowTextW(g_scan_status,
                     error.empty() ? L"全盘扫描失败（沿用缓存/快速扫描结果）"
                                   : widen(error).c_str());
    if (!g_running && !g_starting)
      update_status_message(L"全盘扫描失败（沿用缓存/快速扫描结果）", Role::Danger);
  }
  update_buttons_state();
}

bool load_models() {
  if (g_engine.empty()) {
    update_status_message(L"找不到 engine.exe（放到 build/ 或设置 ENGINE_EXE）", Role::Danger);
    return false;
  }
  const std::string out = engine_capture(L"models --json", 120000);
  const bool loaded = apply_models_json(out, false);
  if (!loaded) {
    if (g_scan_status != nullptr)
      SetWindowTextW(g_scan_status, L"没有缓存：正在后台全盘扫描…");
  }
  // Fast cache list is on screen already; refresh the full-disk inventory in the
  // background (`ENGINE_LAUNCHER_NO_AUTO_SCAN=1` disables this for tests).
  const char* no_auto = std::getenv("ENGINE_LAUNCHER_NO_AUTO_SCAN");
  if (no_auto == nullptr || *no_auto == '\0') start_background_model_scan();
  return loaded;
}

void on_model_selection_changed() {
  if (g_model_combo == nullptr) return;
  const int sel = static_cast<int>(SendMessageW(g_model_combo, CB_GETCURSEL, 0, 0));
  if (sel < 0) return;
  if (sel == 0) {
    g_ui.model_auto = true;
    g_ui.model_index = -1;
  } else {
    g_ui.model_auto = false;
    g_ui.model_index = sel - 1;
  }
  if (g_engine_placement != nullptr) SetWindowTextW(g_engine_placement, L"");
  g_engine_placement_text.clear();
  load_model_meta(true);
  refresh_estimates(true);
  update_buttons_state();
  update_args_summary();
}

}  // namespace
// ---------------------------------------------------------------------------
// Layout: a single scrollable page + a fixed bottom status bar
// ---------------------------------------------------------------------------
namespace {

constexpr int kMarginPx = 24;
constexpr int kBarHeightPx = 176;

int g_bar_h = 0;
bool g_scroll_visible = false;
// 1px frames drawn by the parent around the borderless edit boxes (content coords)
std::vector<RECT> g_edit_frames;
std::vector<RECT> g_slider_frames;

int layout_bottom() {
  const int M = S(kMarginPx);
  g_bar_h = S(kBarHeightPx);
  const int barTop = g_client_h - g_bar_h;
  g_view_h = barTop;
  const int btnY = barTop + S(14);
  const int btnH = S(38);
  int x = M;
  auto place = [&](HWND h, int w) {
    if (h != nullptr) MoveWindow(h, x, btnY, w, btnH, TRUE);
    x += w + S(10);
  };
  place(g_start, S(112));
  place(g_stop, S(96));
  place(g_open_page, S(136));
  place(g_open_log, S(112));
  const int lineX = M;
  const int lineW = g_client_w - 2 * M;
  if (g_status != nullptr) MoveWindow(g_status, lineX, btnY + btnH + S(10), lineW, S(24), TRUE);
  if (g_url != nullptr) MoveWindow(g_url, lineX, btnY + btnH + S(36), lineW, S(20), TRUE);
  if (g_log_info != nullptr) MoveWindow(g_log_info, lineX, btnY + btnH + S(58), lineW, S(18), TRUE);
  if (g_args_summary != nullptr)
    MoveWindow(g_args_summary, lineX, btnY + btnH + S(78), lineW, S(18), TRUE);
  if (g_schema_info != nullptr)
    MoveWindow(g_schema_info, lineX, btnY + btnH + S(98), lineW, S(18), TRUE);
  return barTop;
}

// Measure/apply the scrollable content.  `apply` moves the child windows;
// the return value is the content height in pixels.
int layout_content(bool apply, bool scroll_visible) {
  const int M = S(kMarginPx);
  const int contentW = g_client_w - 2 * M - (scroll_visible ? S(20) : 0);
  if (apply) {
    g_painted.clear();
    g_edit_frames.clear();
    g_slider_frames.clear();
  }
  int y = S(8);

  auto header = [&](const std::wstring& title, const std::wstring& hint) {
    if (apply) g_painted.push_back({y, S(34), title, hint});
    y += S(34) + S(6);
  };
  auto place = [&](HWND h, int px, int py, int w, int hh) {
    if (!apply || h == nullptr) return;
    const int top = py - g_scroll;
    const int height = std::max(1, hh);
    // Child windows are not clipped to the scroll viewport by Windows, so hide
    // whatever is outside it instead of letting it paint over the bottom bar.
    const bool visible = (top + height > 0) && (top < g_view_h);
    if (!visible) {
      ShowWindow(h, SW_HIDE);
      return;
    }
    MoveWindow(h, px, top, std::max(1, w), height, TRUE);
    ShowWindow(h, SW_SHOW);
    if (g_edit_controls.find(h) != g_edit_controls.end() ||
        (px >= 0 && GetDlgCtrlID(h) == IDC_ADV_SEARCH)) {
      RECT fr{px - 1, top - 1, px + std::max(1, w) + 1, top + height + 1};
      g_edit_frames.push_back(fr);
    }
  };
  auto show = [&](HWND h, bool visible) {
    if (!apply || h == nullptr) return;
    ShowWindow(h, visible ? SW_SHOW : SW_HIDE);
  };

  // ---- 模型 ----
  header(L"模型", L"全盘扫描（固定盘/移动盘）· 缓存 engine-models-cache.json");
  {
    const int rowH = S(28);
    const int refreshW = S(96);
    const int comboW = contentW - refreshW - S(10);
    show(g_model_combo, true);
    place(g_model_combo, M, y, comboW, rowH);
    place(g_refresh_btn, M + comboW + S(10), y, refreshW, rowH);
    y += rowH + S(6);
    place(g_model_info, M, y, contentW, S(22));
    y += S(22) + S(2);
    place(g_engine_placement, M, y, contentW, S(20));
    y += S(20);
    place(g_scan_status, M, y, contentW, S(20));
    y += S(20) + S(12);
  }

  // ---- 显存与上下文 ----
  header(L"显存与上下文", L"拖动滑块时实时估算：显存 / 主机内存 / KV 缓存");
  {
    const int labelW = S(140);
    const int rowH = S(26);
    // GPU layers
    place(g_label_gpu, M, y + S(3), labelW, S(22));
    place(g_gpu_auto, M + labelW, y, S(216), rowH);
    place(g_gpu_slider, M + labelW + S(226), y, contentW - labelW - S(226), rowH);
    y += rowH + S(4);
    place(g_gpu_text, M, y, contentW, S(22));
    y += S(22) + S(10);
    // context: [label] [editable number] [auto] [log-scale slider]
    place(g_label_ctx, M, y + S(3), labelW, S(22));
    place(g_ctx_edit, M + labelW, y, S(118), rowH);
    place(g_ctx_auto, M + labelW + S(128), y, S(228), rowH);
    place(g_ctx_slider, M + labelW + S(366), y, contentW - labelW - S(366), rowH);
    y += rowH + S(4);
    place(g_ctx_text, M, y, contentW, S(22));
    y += S(22) + S(10);
    // KV type
    place(g_label_kv, M, y + S(4), labelW, S(22));
    place(g_kv_combo, M + labelW, y, S(236), rowH);
    place(g_kv_text, M + labelW + S(246), y + S(4), contentW - labelW - S(246), S(22));
    y += rowH + S(8);
    place(g_est_summary, M, y, contentW, S(22));
    y += S(22) + S(18);
  }

  // ---- 性能开关 ----
  header(L"性能开关", L"每个开关都注明引擎参数名；悬停查看引擎原始说明");
  {
    const int labelW = S(140);
    const int rowH = S(26);
    place(g_label_threads, M, y + S(3), labelW, S(22));
    place(g_threads_combo, M + labelW, y, S(190), rowH);
    place(g_threads_hint, M + labelW + S(200), y + S(4), contentW - labelW - S(200), S(22));
    y += rowH + S(10);

    const int gap = S(16);
    const int cellW = (contentW - gap) / 2;
    const int cellH = S(32);
    HWND toggles[] = {g_ssd_toggle, g_dev_route,   g_layer_graph,
                      g_cpu_int8,   g_win_prio,    g_thinking_len,
                      g_vram_auto,  g_win_affinity};
    const int toggle_count = static_cast<int>(sizeof(toggles) / sizeof(toggles[0]));
    for (int i = 0; i < toggle_count; ++i) {
      const int col = i % 2, row = i / 2;
      place(toggles[i], M + col * (cellW + gap), y + row * cellH, cellW, S(28));
    }
    y += ((toggle_count + 1) / 2) * cellH;
    // Red risk note: only while the opt-in VRAM auto-layer promotion is ON.
    show(g_vram_auto_warn, g_ui.vram_auto_layers);
    if (g_ui.vram_auto_layers) {
      place(g_vram_auto_warn, M, y, contentW, S(20));
      y += S(20) + S(4);
    }
    place(g_ssd_note, M, y, contentW, S(22));
    y += S(22) + S(10);

    place(g_label_presets, M, y + S(4), labelW, S(22));
    place(g_preset_fast, M + labelW, y, S(96), S(28));
    place(g_preset_low, M + labelW + S(106), y, S(112), S(28));
    place(g_preset_bal, M + labelW + S(228), y, S(96), S(28));
    place(g_preset_note, M + labelW + S(336), y + S(4), contentW - labelW - S(336), S(22));
    y += S(28) + S(18);
  }

  // ---- 服务 ----
  header(L"服务", L"HTTP 服务参数（修改后需重启引擎生效）");
  {
    const int rowH = S(26);
    place(g_label_port, M, y + S(3), S(56), S(22));
    place(g_port_edit, M + S(58), y, S(88), rowH);
    place(g_label_host, M + S(166), y + S(3), S(88), S(22));
    place(g_host_edit, M + S(256), y, S(180), rowH);
    place(g_auto_browser, M + S(452), y, S(260), rowH);
    y += rowH + S(18);
  }

  // ---- 高级（可折叠，schema 驱动） ----
  {
    header(L"高级", L"由 engine.exe serve --print-config --json 动态生成；改动写入 launcher-engine.cfg");
    const int toggleW = S(470);
    show(g_adv_toggle, true);
    place(g_adv_toggle, M, y, toggleW, S(32));
    place(g_adv_count, M + toggleW + S(12), y + S(7), contentW - toggleW - S(12), S(22));
    y += S(32) + S(6);
    if (g_ui.advanced_open) {
      show(g_label_adv_search, true);
      show(g_adv_search, true);
      place(g_label_adv_search, M, y + S(3), S(48), S(22));
      place(g_adv_search, M + S(52), y, contentW - S(52), S(26));
      y += S(26) + S(8);
      const int gap = S(16);
      const int cellW = (contentW - gap) / 2;
      const int cellH = S(30);
      const std::wstring filter = lower_w(trim(g_ui.adv_filter));
      int shown = 0;
      for (auto& cell : g_adv_cells) {
        if (!cell.created) continue;
        const OptDef& od = g_opts[static_cast<size_t>(cell.opt_index)];
        const bool match = g_ui.advanced_open && adv_matches(od, filter);
        show(cell.label, match);
        show(cell.ctrl, match);
        if (!match) continue;
        const int col = shown % 2, row = shown / 2;
        const int cellX = M + col * (cellW + gap);
        place(cell.label, cellX, y + row * cellH + S(1), S(184), S(26));
        place(cell.ctrl, cellX + S(188), y + row * cellH, cellW - S(188), S(26));
        ++shown;
      }
      if (apply) {
        const std::wstring count = L"匹配 " + wnum(shown) + L" / " +
                                   wnum(static_cast<int>(g_adv_cells.size())) + L" 项" +
                                   (filter.empty() ? L"" : (L"（过滤：“" + g_ui.adv_filter + L"”）"));
        SetWindowTextW(g_adv_count, count.c_str());
      }
      y += ((shown + 1) / 2) * cellH + S(4);
    } else {
      show(g_label_adv_search, false);
      show(g_adv_search, false);
      for (auto& cell : g_adv_cells) {
        show(cell.label, false);
        show(cell.ctrl, false);
      }
      if (apply) {
        const std::wstring count = g_schema_loaded
                                       ? (L"共 " + wnum(static_cast<int>(g_adv_cells.size())) +
                                          L" 项引擎参数（收起）")
                                       : g_schema_error;
        SetWindowTextW(g_adv_count, count.c_str());
      }
    }
    y += S(4);
    show(g_footer, true);
    place(g_footer, M, y, contentW, S(22));
    y += S(22) + S(16);
  }
  return y;
}

void layout() {
  if (g_hwnd == nullptr) return;
  RECT rc;
  GetClientRect(g_hwnd, &rc);
  g_client_w = rc.right;
  g_client_h = rc.bottom;
  layout_bottom();
  bool scroll = g_scroll_visible;
  int height = 0;
  for (int iter = 0; iter < 4; ++iter) {
    const int h = layout_content(false, scroll);
    const bool need = h > g_view_h;
    if (need != scroll) {
      scroll = need;
      continue;
    }
    height = h;
    break;
  }
  g_content_h = height;
  g_scroll_visible = scroll;
  const int max_scroll = std::max(0, g_content_h - g_view_h);
  if (g_scroll > max_scroll) g_scroll = max_scroll;
  if (g_scroll < 0) g_scroll = 0;
  // apply positions
  layout_content(true, scroll);
  ShowScrollBar(g_hwnd, SB_VERT, scroll ? TRUE : FALSE);
  SCROLLINFO si{};
  si.cbSize = sizeof(si);
  si.fMask = SIF_RANGE | SIF_PAGE | SIF_POS;
  si.nMin = 0;
  si.nMax = std::max(1, g_content_h);
  si.nPage = static_cast<UINT>(std::max(1, g_view_h));
  si.nPos = g_scroll;
  SetScrollInfo(g_hwnd, SB_VERT, &si, TRUE);
  InvalidateRect(g_hwnd, nullptr, FALSE);
}

void scroll_by(int delta) {
  const int max_scroll = std::max(0, g_content_h - g_view_h);
  int next = g_scroll + delta;
  if (next < 0) next = 0;
  if (next > max_scroll) next = max_scroll;
  if (next == g_scroll) return;
  g_scroll = next;
  layout();
  UpdateWindow(g_hwnd);
}

// ---------------------------------------------------------------------------
// Painting the page background, section headers and the bottom bar
// ---------------------------------------------------------------------------
void paint_root(HDC dc) {
  RECT rc;
  GetClientRect(g_hwnd, &rc);
  FillRect(dc, &rc, g_br_bg);
  const int M = S(kMarginPx);
  const int contentW = g_client_w - 2 * M - (g_scroll_visible ? S(20) : 0);
  for (const auto& h : g_painted) {
    const int y = h.y - g_scroll;
    if (y + h.h < 0 || y > g_view_h) continue;
    RECT accent{M, y + S(9), M + S(3), y + S(27)};
    fill_rect(dc, accent, g_pal.accent);
    draw_text(dc, h.title, RECT{M + S(12), y + S(2), M + S(400), y + S(32)},
              DT_LEFT | DT_VCENTER | DT_SINGLELINE, g_pal.text, g_font_title);
    if (!h.hint.empty()) {
      RECT tr{M + S(220), y + S(8), M + contentW, y + S(28)};
      draw_text(dc, h.hint, tr, DT_RIGHT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS,
                g_pal.dim, g_font_small);
    }
    hline(dc, M, M + contentW, y + S(31), g_pal.border);
  }
  // borderless edit boxes: 1px rounded frame drawn by the parent
  for (const RECT& fr : g_edit_frames) {
    if (fr.bottom < 0 || fr.top > g_view_h) continue;
    rounded_fill(dc, fr, S(4), g_pal.bg, g_pal.border);
  }
  const int barTop = g_client_h - g_bar_h;
  RECT bar{0, barTop, g_client_w, g_client_h};
  FillRect(dc, &bar, g_br_panel);
  hline(dc, 0, g_client_w, barTop, g_pal.border);
  RECT accentLine{0, barTop, S(3), g_client_h};
  fill_rect(dc, accentLine, g_pal.accent);
}

// ---------------------------------------------------------------------------
// Model meta reload helper used by the engine placement echo
// ---------------------------------------------------------------------------
void update_engine_placement_label(const std::wstring& text, Role role) {
  g_engine_placement_text = text;
  if (g_engine_placement == nullptr) return;
  SetWindowTextW(g_engine_placement, text.c_str());
  set_label_role(g_engine_placement, role);
}

// Read the tail of the engine log and surface the engine's own placement line
// and the last error line in the UI (no silent failures).
void read_engine_log_feedback() {
  const std::wstring log_path = join_path(g_launcher_dir, L"launcher-engine.log");
  const std::string raw = read_file_utf8(log_path);
  if (raw.empty()) return;
  const size_t begin = raw.size() > 24000 ? raw.size() - 24000 : 0;
  const std::string tail = raw.substr(begin);
  const std::wstring wide = widen(tail);
  // placement line
  size_t pos = wide.find(L"[placement] gpu_moe_layers=");
  if (pos != std::wstring::npos) {
    size_t end = wide.find(L'\n', pos);
    if (end == std::wstring::npos) end = wide.size();
    update_engine_placement_label(L"引擎实测（engine 日志）：" + wide.substr(pos, end - pos),
                                  Role::Accent);
  } else {
    pos = wide.find(L"[placement] decision:");
    if (pos != std::wstring::npos) {
      size_t end = wide.find(L'\n', pos);
      if (end == std::wstring::npos) end = wide.size();
      update_engine_placement_label(L"引擎实测：" + wide.substr(pos, end - pos), Role::Accent);
    }
  }
  // last error line
  std::wstring err;
  size_t scan = 0;
  while (true) {
    const size_t p = wide.find(L"error", scan);
    if (p == std::wstring::npos) break;
    size_t end = wide.find(L'\n', p);
    if (end == std::wstring::npos) end = wide.size();
    err = wide.substr(p, std::min<size_t>(end - p, 220));
    scan = end;
  }
  if (!err.empty()) g_last_log_tail = err;
}

void open_url(const std::wstring& url) {
  if (url.empty()) return;
  ShellExecuteW(g_hwnd, L"open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

void open_path_shell(const std::wstring& path) {
  if (path.empty()) return;
  ShellExecuteW(g_hwnd, L"open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Existing-instance detection: a second "启动引擎" must never silently start a
// second engine on the same port (double click / two launcher windows).
// ---------------------------------------------------------------------------
std::vector<DWORD> engine_process_pids(DWORD exclude_pid) {
  std::vector<DWORD> pids;
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snap == INVALID_HANDLE_VALUE) return pids;
  PROCESSENTRY32W pe{};
  pe.dwSize = sizeof(pe);
  const DWORD me = GetCurrentProcessId();
  if (Process32FirstW(snap, &pe)) {
    do {
      std::wstring name = pe.szExeFile;
      for (wchar_t& c : name) c = static_cast<wchar_t>(std::towlower(c));
      if (name != L"engine.exe") continue;
      if (pe.th32ProcessID == me || pe.th32ProcessID == exclude_pid) continue;
      pids.push_back(pe.th32ProcessID);
    } while (Process32NextW(snap, &pe));
  }
  CloseHandle(snap);
  return pids;
}

// PID owning a LISTENING TCP socket on `port` (iphlpapi loaded dynamically so the
// launcher's link libraries stay unchanged).  0 = nobody listening yet.
DWORD tcp_listener_pid(int port) {
  struct Row {
    DWORD state, local_addr, local_port, remote_addr, remote_port, pid;
  };
  struct Table {
    DWORD count;
    Row rows[1];
  };
  using Fn = DWORD(WINAPI*)(PVOID, PDWORD, BOOL, ULONG, ULONG, ULONG);
  HMODULE iphlp = LoadLibraryW(L"iphlpapi.dll");
  if (iphlp == nullptr) return 0;
  Fn get_table = reinterpret_cast<Fn>(GetProcAddress(iphlp, "GetExtendedTcpTable"));
  DWORD pid = 0;
  if (get_table != nullptr) {
    DWORD size = 0;
    get_table(nullptr, &size, FALSE, 2 /*AF_INET*/, 3 /*OWNER_PID_LISTENER*/, 0);
    if (size > 0) {
      std::vector<unsigned char> buf(size);
      if (get_table(buf.data(), &size, FALSE, 2, 3, 0) == 0) {
        const Table* t = reinterpret_cast<const Table*>(buf.data());
        for (DWORD i = 0; i < t->count; ++i) {
          const DWORD lp = t->rows[i].local_port;
          const int p = static_cast<int>(((lp & 0xFF) << 8) | ((lp >> 8) & 0xFF));
          if (p == port && t->rows[i].state == 2 /*LISTEN*/) {
            pid = t->rows[i].pid;
            break;
          }
        }
      }
    }
  }
  FreeLibrary(iphlp);
  return pid;
}

bool process_is_engine(DWORD pid) {
  HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
  if (h == nullptr) return false;
  wchar_t path[1024] = {0};
  DWORD n = 1024;
  bool ok = QueryFullProcessImageNameW(h, 0, path, &n) != FALSE;
  CloseHandle(h);
  if (!ok) return false;
  std::wstring p = path;
  for (wchar_t& c : p) c = static_cast<wchar_t>(std::towlower(c));
  return p.size() >= 10 && p.compare(p.size() - 10, 10, L"engine.exe") == 0;
}

std::wstring existing_engine_model_hint() {
  const std::string raw =
      read_file_utf8(join_path(g_launcher_dir, L"launcher-engine.log"));
  const std::wstring w = widen(raw.substr(0, std::min<size_t>(raw.size(), 4000)));
  const size_t pos = w.find(L"--model \"");
  if (pos == std::wstring::npos) return L"未知";
  const size_t begin = pos + 9;
  const size_t end = w.find(L'"', begin);
  if (end == std::wstring::npos || end <= begin) return L"未知";
  std::wstring path = w.substr(begin, end - begin);
  const size_t slash = path.find_last_of(L"\\/");
  return slash == std::wstring::npos ? path : path.substr(slash + 1);
}

void terminate_pid(DWORD pid) {
  HANDLE h = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, pid);
  if (h == nullptr) return;
  TerminateProcess(h, 1);
  WaitForSingleObject(h, 3000);
  CloseHandle(h);
}

// Returns true when it is OK to proceed with a fresh start.  If another engine
// is detected it asks the user (unless a test env var auto-decides) and stops it.
bool resolve_existing_engine_before_start() {
  const DWORD own_child = g_proc != nullptr ? GetProcessId(g_proc) : 0;
  std::vector<DWORD> pids = engine_process_pids(own_child);
  const DWORD listener = tcp_listener_pid(g_ui.port);
  if (listener != 0 && listener != own_child) {
    bool dup = false;
    for (DWORD p : pids)
      if (p == listener) dup = true;
    if (!dup) pids.push_back(listener);
  }
  if (pids.empty()) return true;

  std::wstring pid_text;
  for (size_t i = 0; i < pids.size(); ++i) {
    if (i != 0) pid_text += L", ";
    pid_text += std::to_wstring(pids[i]);
  }
  const std::wstring model_hint = existing_engine_model_hint();
  const std::wstring msg =
      L"检测到引擎已在运行（PID " + pid_text + L"，模型 " + model_hint +
      L"）。\n\n是 = 停止它并重新启动\n否 = 取消（保留当前正在运行的引擎）";
  const char* auto_stop = std::getenv("ENGINE_LAUNCHER_AUTO_STOP_EXISTING");
  const char* no_dialog = std::getenv("ENGINE_LAUNCHER_NO_DIALOG");
  bool stop = auto_stop != nullptr && std::strcmp(auto_stop, "1") == 0;
  if (!stop) {
    if (no_dialog != nullptr && std::strcmp(no_dialog, "1") == 0) {
      update_status_message(L"检测到引擎已在运行（PID " + pid_text + L"）；已取消启动",
                            Role::Danger);
      return false;
    }
    const int rc = MessageBoxW(g_hwnd, msg.c_str(), L"引擎已在运行",
                               MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2);
    stop = rc == IDYES;
  }
  if (!stop) return false;
  for (DWORD pid : pids)
    if (process_is_engine(pid)) terminate_pid(pid);
  for (int i = 0; i < 20; ++i) {
    if (tcp_listener_pid(g_ui.port) == 0) break;
    Sleep(100);
  }
  update_status_message(L"已停止已有引擎，正在重新启动…", Role::Normal);
  update_buttons_state();
  return true;
}

void start_engine() {
  if (g_proc != nullptr) {
    // Same launcher clicked again while its own engine is running/loading.
    update_status_message(L"引擎已在运行/启动中，已忽略重复启动", Role::Warn);
    return;
  }
  if (g_engine.empty()) {
    MessageBoxW(g_hwnd, L"找不到 engine.exe（应位于 build/engine.exe）", L"启动失败",
                MB_ICONERROR);
    return;
  }
  if (g_ui.busy) return;
  long long port_v = 0;
  if (g_port_edit != nullptr) {
    wchar_t buf[64] = {0};
    GetWindowTextW(g_port_edit, buf, 64);
    if (!parse_ll(buf, &port_v) || port_v < 1 || port_v > 65535) {
      MessageBoxW(g_hwnd, L"端口必须是 1..65535 之间的数字", L"启动失败", MB_ICONERROR);
      return;
    }
    g_ui.port = static_cast<int>(port_v);
  }
  if (current_model_path().empty() && !g_ui.model_auto) {
    MessageBoxW(g_hwnd, L"请先选择模型", L"启动失败", MB_ICONINFORMATION);
    return;
  }
  if (!g_ui.ctx_auto) {
    long long ctx_v = 0;
    wchar_t buf[32] = {0};
    if (g_ctx_edit != nullptr) GetWindowTextW(g_ctx_edit, buf, 32);
    if (!parse_ll(buf, &ctx_v) || ctx_v < 1 || ctx_v > kCtxMax) {
      MessageBoxW(g_hwnd,
                  L"上下文长度必须是 1..1048576 之间的整数（任意值都可以，例如\n"
                  L"1 / 1000 / 12345 / 32768 / 131072 / 262144）。\n"
                  L"输入框标红时不能启动；留空或非法字符也会被拒绝。",
                  L"启动失败", MB_ICONERROR);
      return;
    }
    g_ui.ctx_value = static_cast<int>(ctx_v);
    g_ui.ctx_invalid = false;
  }
  if (!resolve_existing_engine_before_start()) return;
  if (g_ui.ssd) {
    const int rc = MessageBoxW(g_hwnd,
                L"已勾选 SSD 流式。引擎报告：ssd 流式未实现（接口占位），启动后引擎会报错退出。\n"
                L"日志：build/launcher-engine.log\n\n是否仍要启动？",
                L"SSD 流式未实现", MB_ICONWARNING | MB_YESNO);
    if (rc != IDYES) return;
  }
  if (g_est.over_usable && !g_ui.gpu_auto) {
    const int rc = MessageBoxW(
        g_hwnd,
        (L"当前 GPU 专家层 = " + wnum(g_ui.gpu_layers) + L"，预计显存 " +
         fmt_size(g_est.vram_need) + L" 超过可用 " + fmt_size(g_est.vram_usable) +
         L"。\n引擎会实测降层，可能更慢或 OOM。\n\n仍要启动？")
            .c_str(),
        L"显存不足", MB_ICONWARNING | MB_YESNO);
    if (rc != IDYES) return;
  }
  LaunchSpec spec = build_launch_spec(true);
  if (spec.has_warn) {
    const int rc = MessageBoxW(g_hwnd, (spec.warn_text + L"\n是否仍要启动？").c_str(),
                               L"高级选项提示", MB_ICONWARNING | MB_YESNO);
    if (rc != IDYES) return;
  }
  if (g_engine_placement != nullptr) SetWindowTextW(g_engine_placement, L"");
  g_last_log_tail.clear();

  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  HANDLE log = CreateFileW(spec.log_path.c_str(), GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
  if (log == INVALID_HANDLE_VALUE) {
    MessageBoxW(g_hwnd, L"无法创建引擎日志文件 build/launcher-engine.log", L"启动失败",
                MB_ICONERROR);
    return;
  }
  {
    // Record the exact command line in the engine log before the engine starts
    // (the acceptance test greps this line; the engine's own output follows).
    const std::string header =
        "# launcher command line (exact):\r\n# " + narrow(spec.cmdline) +
        "\r\n# advanced config overrides: " + std::to_string(spec.advanced_overrides) +
        (spec.advanced_overrides > 0 ? ("\r\n# config file: " + narrow(spec.cfg_path)) : "") +
        "\r\n# ---- engine output below ----\r\n";
    DWORD wrote = 0;
    WriteFile(log, header.data(), static_cast<DWORD>(header.size()), &wrote, nullptr);
  }
  HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
  si.wShowWindow = SW_HIDE;
  si.hStdOutput = log;
  si.hStdError = log;
  si.hStdInput = (nul == INVALID_HANDLE_VALUE) ? nullptr : nul;
  PROCESS_INFORMATION pi{};
  std::wstring cmd = spec.cmdline;
  const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                                 nullptr, g_launcher_dir.empty() ? nullptr : g_launcher_dir.c_str(),
                                 &si, &pi);
  CloseHandle(log);
  if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
  if (!ok) {
    const DWORD err = GetLastError();
    MessageBoxW(g_hwnd, (L"启动 engine.exe 失败（错误码 " + wnum(static_cast<long long>(err)) + L"）")
                            .c_str(),
                L"启动失败", MB_ICONERROR);
    return;
  }
  CloseHandle(pi.hThread);
  g_proc = pi.hProcess;
  g_starting = true;
  g_running = false;
  g_running_port = g_ui.port;
  g_start_tick = GetTickCount64();
  g_auto_open = g_ui.auto_browser;
  update_status_message(L"模型加载中… 0s", Role::Accent);
  update_url_line();
  update_buttons_state();
  persist_effective_args();
  SetTimer(g_hwnd, 1, 250, nullptr);
}

void pump_for(DWORD total_ms) {
  MSG msg;
  const ULONGLONG until = GetTickCount64() + total_ms;
  while (GetTickCount64() < until) {
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
    Sleep(20);
  }
}

void stop_engine() {
  if (g_proc == nullptr) return;
  update_status_message(L"正在停止…", Role::Warn);
  UpdateWindow(g_status);
  if (g_running_port > 0)
    http_request(g_running_port, L"POST", L"/shutdown", 1500, L"Connection: close\r\n");
  bool exited = false;
  for (int i = 0; i < 60; ++i) {
    if (WaitForSingleObject(g_proc, 0) == WAIT_OBJECT_0) {
      exited = true;
      break;
    }
    pump_for(100);
  }
  if (!exited) {
    TerminateProcess(g_proc, 0);
    WaitForSingleObject(g_proc, 3000);
  }
  CloseHandle(g_proc);
  g_proc = nullptr;
  g_running = false;
  g_starting = false;
  g_running_port = 0;
  KillTimer(g_hwnd, 1);
  update_status_message(L"已停止", Role::Normal);
  update_url_line();
  update_buttons_state();
}

bool port_serves_our_model(int port) {
  // /health being reachable is not enough: another engine may own the port.
  // Compare /props.model_path with the model this window is launching.
  const std::wstring path = current_model_path();
  if (path.empty()) return true;
  std::wstring base = path;
  const size_t slash = base.find_last_of(L"\\/");
  if (slash != std::wstring::npos) base = base.substr(slash + 1);
  const std::string needle = narrow(base);
  if (needle.empty()) return true;
  const HttpResponse r = http_request(port, L"GET", L"/props", 1200);
  if (!r.ok) return true;  // cannot check (do not block a normal start)
  return r.body.find(needle) != std::string::npos;
}

void on_timer() {
  if (g_proc == nullptr) return;
  DWORD code = 0;
  const bool exited = WaitForSingleObject(g_proc, 0) == WAIT_OBJECT_0;
  if (exited) {
    GetExitCodeProcess(g_proc, &code);
    read_engine_log_feedback();
    CloseHandle(g_proc);
    g_proc = nullptr;
    g_running = false;
    g_starting = false;
    g_running_port = 0;
    KillTimer(g_hwnd, 1);
    std::wstring msg = L"引擎已退出（代码 " + wnum(static_cast<long long>(code)) + L"）";
    if (!g_last_log_tail.empty()) msg += L" · " + g_last_log_tail;
    update_status_message(msg, code == 0 ? Role::Normal : Role::Danger);
    update_url_line();
    update_buttons_state();
    return;
  }
  if (g_starting) {
    const ULONGLONG elapsed = (GetTickCount64() - g_start_tick) / 1000;
    if (health_ok(g_running_port, 400)) {
      if (!port_serves_our_model(g_running_port)) {
        // Somebody else's engine owns the port; do not claim "运行中".
        TerminateProcess(g_proc, 1);
        WaitForSingleObject(g_proc, 3000);
        CloseHandle(g_proc);
        g_proc = nullptr;
        g_running = false;
        g_starting = false;
        KillTimer(g_hwnd, 1);
        update_status_message(
            L"端口 " + wnum(g_running_port) +
                L" 已被别的引擎占用（不是本窗口启动的模型），请先停止它或换端口",
            Role::Danger);
        update_url_line();
        update_buttons_state();
        g_running_port = 0;
        return;
      }
      g_starting = false;
      g_running = true;
      read_engine_log_feedback();
      update_status_message(
          L"运行中 · 加载耗时 " + wnum(static_cast<long long>(elapsed)) + L"s", Role::Accent);
      update_url_line();
      update_buttons_state();
      if (g_auto_open) {
        open_url(L"http://" + g_ui.host + L":" + wnum(g_running_port) + L"/");
        g_auto_open = false;
      }
      SetTimer(g_hwnd, 1, 1000, nullptr);
    } else {
      update_status_message(L"模型加载中… " + wnum(static_cast<long long>(elapsed)) + L"s",
                            Role::Accent);
    }
    if ((elapsed % 2) == 0) read_engine_log_feedback();
  } else if (g_running) {
    static int ticks = 0;
    if (++ticks % 3 == 0) read_engine_log_feedback();
  }
}

}  // namespace
// ---------------------------------------------------------------------------
// Schema loading (engine is the single source of truth for the option list)
// ---------------------------------------------------------------------------
namespace {

bool load_schema() {
  g_opts.clear();
  g_opt_index.clear();
  g_opt_state.clear();
  g_schema_loaded = false;
  if (g_engine.empty()) {
    g_schema_error = L"看不到 engine.exe：无法读取 --print-config --json";
    return false;
  }
  const std::string out = engine_capture(L"serve --print-config --json", 30000);
  const std::string json = extract_json_object(out);
  JVal root;
  JsonParser parser(json);
  if (json.empty() || !parser.parse(&root) || !root.is_obj()) {
    g_schema_error = L"engine serve --print-config --json 解析失败";
    return false;
  }
  const JVal* arr = jget(&root, "options");
  if (arr == nullptr || !arr->is_arr()) {
    g_schema_error = L"engine --print-config --json 缺少 options 数组";
    return false;
  }
  for (const auto& item : arr->arr) {
    OptDef d;
    d.key = jstr(&item, "key");
    d.cli = jstr(&item, "cli");
    d.env = jstr(&item, "env");
    d.type = jstr(&item, "type", "string");
    d.def = jstr(&item, "def");
    d.help = jstr(&item, "help");
    d.takes_value = jbool(&item, "takes_value", true);
    d.invert_env = jbool(&item, "invert_env", false);
    d.minv = jnum(&item, "min", 0);
    d.maxv = jnum(&item, "max", 0);
    d.value = jstr(&item, "value");
    if (item.get("value") != nullptr && item.get("value")->type == JVal::T::Bool)
      d.value = item.get("value")->b ? "true" : "false";
    d.source = jstr(&item, "source", "default");
    d.has_cli = !d.cli.empty();
    if (d.key.empty()) continue;
    g_opt_index[d.key] = static_cast<int>(g_opts.size());
    g_opts.push_back(std::move(d));
  }
  if (g_opts.empty()) {
    g_schema_error = L"engine --print-config --json 返回 0 个选项";
    return false;
  }
  for (const auto& od : g_opts) {
    OptState st;
    st.is_bool = (od.type == "bool");
    st.b = (od.value == "true" || od.value == "1");
    st.text = od.value;
    g_opt_state[od.key] = std::move(st);
  }
  g_schema_loaded = true;
  g_schema_error.clear();
  return true;
}

// ---------------------------------------------------------------------------
// Font rebuild for DPI changes
// ---------------------------------------------------------------------------
void reload_fonts() {
  HFONT* list[] = {&g_font, &g_font_small, &g_font_bold, &g_font_title};
  for (HFONT* f : list) {
    if (*f != nullptr) DeleteObject(*f);
    *f = nullptr;
  }
  g_font = make_font(10, FW_NORMAL);
  g_font_small = make_font(9, FW_NORMAL);
  g_font_bold = make_font(10, FW_SEMIBOLD);
  g_font_title = make_font(13, FW_SEMIBOLD);
}

void apply_font_to_control(HWND h, int pt) {
  HFONT f = g_font;
  if (pt <= 9) f = g_font_small;
  else if (pt >= 13) f = g_font_title;
  else if (pt == 11) f = g_font_bold;
  SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(f), TRUE);
}

// ---------------------------------------------------------------------------
// Command handling helpers
// ---------------------------------------------------------------------------
// Native combobox dropdown height.  A dropped-list combo's window height is
// the dropped list height, so the list only shows a couple of rows if the
// closed control is 30 px tall.  CBN_DROPDOWN is sent *before* the list is
// shown, so growing the window here is the documented way to control it;
// CBN_CLOSEUP restores the compact height.  This keeps the control's layout
// footprint small while giving the native popup a usable height.
void resize_combo_dropdown(HWND combo, bool open) {
  if (combo == nullptr) return;
  RECT rc;
  GetWindowRect(combo, &rc);
  POINT tl{rc.left, rc.top};
  ScreenToClient(g_hwnd, &tl);
  const int width = std::max(1, static_cast<int>(rc.right - rc.left));
  const int height = open ? S(300) : S(30);
  SetWindowPos(combo, nullptr, tl.x, tl.y, width, height, SWP_NOZORDER | SWP_NOACTIVATE);
}
// ---------------------------------------------------------------------------
bool value_equals_baseline(const OptDef& od, const OptState& st) {
  if (od.type == "bool") {
    const bool baseline = (od.value == "true");
    return st.b == baseline;
  }
  const std::wstring a = lower_w(trim(widen(st.text)));
  const std::wstring b = lower_w(trim(widen(od.value)));
  if (a == b) return true;
  if (od.type == "int" || od.type == "float") {
    double x = 0, y = 0;
    if (parse_dd(a, &x) && parse_dd(b, &y)) return x == y;
  }
  return false;
}

void on_advanced_control_changed(int ctrl_id, int code) {
  const auto it = g_adv_id_to_opt.find(static_cast<UINT>(ctrl_id));
  if (it == g_adv_id_to_opt.end()) return;
  const OptDef& od = g_opts[static_cast<size_t>(it->second)];
  OptState& st = g_opt_state[od.key];
  HWND ctrl = GetDlgItem(g_hwnd, ctrl_id);
  if (ctrl == nullptr) return;
  if (od.type == "bool") {
    st.is_bool = true;
    st.b = toggle_flip(ctrl);
  } else if (od.type == "string" && enum_values_for(od.key) != nullptr) {
    const int sel = static_cast<int>(SendMessageW(ctrl, CB_GETCURSEL, 0, 0));
    if (sel >= 0) {
      const auto& vals = *enum_values_for(od.key);
      if (sel < static_cast<int>(vals.size())) st.text = narrow(vals[static_cast<size_t>(sel)]);
    }
  } else {
    const int len = GetWindowTextLengthW(ctrl);
    std::vector<wchar_t> buf(static_cast<size_t>(len) + 2, L'\0');
    GetWindowTextW(ctrl, buf.data(), static_cast<int>(buf.size()));
    st.text = narrow(trim(std::wstring(buf.data(), static_cast<size_t>(len))));
  }
  st.dirty = !value_equals_baseline(od, st);
  // Options that change the estimate must refresh it immediately.
  if (od.key == "vram_margin_mb" || od.key == "workspace_mb" ||
      od.key == "vram_budget_mb" || od.key == "host_margin_mb" ||
      od.key == "prefill_rows") {
    refresh_estimates(true);
  }
  update_args_summary();
  (void)code;
}

void on_edit_changed(int ctrl_id) {
  if (ctrl_id == IDC_CTX_EDIT) {
    if (g_ctx_sync) return;
    wchar_t buf[32] = {0};
    GetWindowTextW(g_ctx_edit, buf, 32);
    long long v = 0;
    const bool ok = parse_ll(buf, &v) && v >= 1 && v <= kCtxMax;
    g_ui.ctx_invalid = !ok;
    if (ok) {
      g_ui.ctx_value = static_cast<int>(v);
      g_ui.ctx_auto = false;
    }
    refresh_estimates(true);
    update_args_summary();
    return;
  }
  if (ctrl_id == IDC_PORT_EDIT) {
    wchar_t buf[64] = {0};
    GetWindowTextW(g_port_edit, buf, 64);
    long long v = 0;
    if (parse_ll(buf, &v) && v >= 1 && v <= 65535) g_ui.port = static_cast<int>(v);
    update_args_summary();
    return;
  }
  if (ctrl_id == IDC_HOST_EDIT) {
    // NOTE: this also fires while the control is being created (its initial
    // text triggers EN_CHANGE before the global handle is assigned), so read it
    // through the ID and never accept an empty host.
    HWND edit = g_host_edit != nullptr ? g_host_edit : GetDlgItem(g_hwnd, IDC_HOST_EDIT);
    wchar_t buf[256] = {0};
    if (edit != nullptr) GetWindowTextW(edit, buf, 256);
    const std::wstring host = trim(buf);
    if (!host.empty()) g_ui.host = host;
    update_args_summary();
    return;
  }
  if (ctrl_id == IDC_ADV_SEARCH) {
    wchar_t buf[256] = {0};
    GetWindowTextW(g_adv_search, buf, 256);
    g_ui.adv_filter = buf;
    layout();
    return;
  }
  if (ctrl_id >= IDC_ADV_BASE) {
    const auto it = g_adv_id_to_opt.find(static_cast<UINT>(ctrl_id));
    if (it != g_adv_id_to_opt.end() &&
        g_opts[static_cast<size_t>(it->second)].type != "bool")
      on_advanced_control_changed(ctrl_id, 0);
  }
}

void on_command(int id, int code, HWND ctrl) {
  if ((code == CBN_DROPDOWN || code == CBN_CLOSEUP) && is_combo_control(ctrl)) {
    if (code == CBN_DROPDOWN) install_combo_list_wheel_guard(ctrl);
    resize_combo_dropdown(ctrl, code == CBN_DROPDOWN);
    return;
  }
  if (id >= IDC_ADV_BASE) {
    on_advanced_control_changed(id, code);
    return;
  }
  switch (id) {
    case IDC_MODEL_COMBO:
      if (code == CBN_SELCHANGE) on_model_selection_changed();
      return;
    case IDC_KV_COMBO:
      if (code == CBN_SELCHANGE) {
        const int sel = static_cast<int>(SendMessageW(g_kv_combo, CB_GETCURSEL, 0, 0));
        g_ui.kv_sel = std::max(0, std::min(2, sel));
        update_kv_hint();
        refresh_estimates(true);
        update_args_summary();
      }
      return;
    case IDC_CTX_AUTO:
      if (code == BN_CLICKED) {
        g_ui.ctx_auto = toggle_flip(g_ctx_auto);
        g_ui.ctx_invalid = false;
        refresh_estimates(true);
        update_args_summary();
      }
      return;
    case IDC_GPU_AUTO:
      if (code == BN_CLICKED) {
        g_ui.gpu_auto = toggle_flip(g_gpu_auto);
        if (!g_ui.gpu_auto) g_ui.gpu_layers = slider_get(g_gpu_slider);
        refresh_estimates(true);
        update_args_summary();
      }
      return;
    case IDC_SSD_TOGGLE:
      if (code == BN_CLICKED) {
        g_ui.ssd = toggle_flip(g_ssd_toggle);
        update_ssd_state();
        update_args_summary();
      }
      return;
    case IDC_DEV_ROUTE:
      g_ui.dev_route = toggle_flip(g_dev_route);
      update_args_summary();
      return;
    case IDC_LAYER_GRAPH:
      g_ui.layer_graph_auto = toggle_flip(g_layer_graph);
      update_args_summary();
      return;
    case IDC_CPU_INT8:
      g_ui.cpu_int8 = toggle_flip(g_cpu_int8) ? 1 : 0;
      update_args_summary();
      return;
    case IDC_WIN_PRIORITY:
      g_ui.win_priority = toggle_flip(g_win_prio) ? 1 : 0;
      update_args_summary();
      return;
    case IDC_THINKING_LEN:
      g_ui.unlimited_tokens = toggle_flip(g_thinking_len);
      update_args_summary();
      return;
    case IDC_VRAM_AUTO:
      g_ui.vram_auto_layers = toggle_flip(g_vram_auto);
      if (g_vram_auto_warn != nullptr)
        ShowWindow(g_vram_auto_warn, g_ui.vram_auto_layers ? SW_SHOW : SW_HIDE);
      update_args_summary();
      layout();  // the red risk note appears/disappears (page height changes)
      return;
    case IDC_WIN_AFFINITY:
      g_ui.win_affinity = toggle_flip(g_win_affinity);
      update_args_summary();
      return;
    case IDC_THREADS_COMBO:
      if (code == CBN_SELCHANGE) {
        const int sel = static_cast<int>(
            SendMessageW(g_threads_combo, CB_GETCURSEL, 0, 0));
        g_ui.threads_auto = (sel <= 0);
        if (!g_ui.threads_auto) g_ui.threads = sel;
        const std::wstring hint =
            g_ui.threads_auto
                ? std::wstring(L"自动：不传 --threads，跟随引擎默认（本机 6C/12T → 12 线程）")
                : (L"显式 " + wnum(g_ui.threads) + L" 线程（覆盖引擎默认）");
        SetWindowTextW(g_threads_hint, hint.c_str());
        update_args_summary();
      }
      return;
    case IDC_AUTO_BROWSER:
      g_ui.auto_browser = toggle_flip(g_auto_browser);
      return;
    case IDC_REFRESH:
      start_background_model_scan();
      return;
    case IDC_PRESET_FAST: apply_preset(0); return;
    case IDC_PRESET_LOWVRAM: apply_preset(1); return;
    case IDC_PRESET_BALANCED: apply_preset(2); return;
    case IDC_ADV_TOGGLE: {
      g_ui.advanced_open = !g_ui.advanced_open;
      SetWindowTextW(g_adv_toggle,
                     g_ui.advanced_open ? L"▼ 高级：全部引擎参数（收起）"
                                        : L"▶ 高级：全部引擎参数（展开）");
      layout();
      if (g_ui.advanced_open) {
        // Scroll so the advanced section starts near the top of the viewport.
        int target = std::max(0, g_content_h - g_view_h);
        for (const auto& h : g_painted) {
          if (h.title == L"高级") {
            target = std::max(0, h.y - S(8));
            break;
          }
        }
        scroll_by(target - g_scroll);
      }
      UpdateWindow(g_hwnd);
      return;
    }
    case IDC_START:
      if (code == BN_CLICKED) start_engine();
      return;
    case IDC_STOP:
      if (code == BN_CLICKED) stop_engine();
      return;
    case IDC_OPEN_PAGE:
      if (code == BN_CLICKED)
        open_url(L"http://" + g_ui.host + L":" + wnum(g_running_port > 0 ? g_running_port : g_ui.port) + L"/");
      return;
    case IDC_OPEN_LOG:
      if (code == BN_CLICKED)
        open_path_shell(join_path(g_launcher_dir, L"launcher-engine.log"));
      return;
    default: break;
  }
  if (code == EN_CHANGE) on_edit_changed(id);
}

// ---------------------------------------------------------------------------
// Window procedure
// ---------------------------------------------------------------------------
bool is_bottom_bar_control(HWND h) {
  return h == g_status || h == g_url || h == g_log_info || h == g_args_summary ||
         h == g_schema_info;
}

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  switch (msg) {
    case WM_CREATE: {
      g_hwnd = hwnd;
      set_dark_title_bar(hwnd);
      init_tooltips(hwnd);
      create_all_controls();
      layout();
      return 0;
    }
    case WM_SIZE:
      if (g_hwnd != nullptr) layout();
      return 0;
    case WM_GETMINMAXINFO: {
      auto* mmi = reinterpret_cast<MINMAXINFO*>(lp);
      if (mmi != nullptr) {
        mmi->ptMinTrackSize.x = S(980);
        mmi->ptMinTrackSize.y = S(720);
      }
      return 0;
    }
    case WM_ERASEBKGND:
      return 1;
    case WM_PAINT: {
      PAINTSTRUCT ps;
      HDC dc = BeginPaint(hwnd, &ps);
      paint_root(dc);
      EndPaint(hwnd, &ps);
      return 0;
    }
    case WM_CTLCOLORSTATIC: {
      HDC dc = reinterpret_cast<HDC>(wp);
      HWND ctl = reinterpret_cast<HWND>(lp);
      SetBkMode(dc, TRANSPARENT);
      SetTextColor(dc, role_color(role_of(ctl)));
      return reinterpret_cast<LRESULT>(is_bottom_bar_control(ctl) ? g_br_panel : g_br_bg);
    }
    case WM_CTLCOLOREDIT:
    case WM_CTLCOLORLISTBOX: {
      HDC dc = reinterpret_cast<HDC>(wp);
      SetBkColor(dc, g_pal.input_bg);
      SetTextColor(dc, g_pal.text);
      return reinterpret_cast<LRESULT>(g_br_input);
    }
    case WM_CTLCOLORBTN: {
      HDC dc = reinterpret_cast<HDC>(wp);
      SetBkColor(dc, g_pal.bg);
      SetTextColor(dc, g_pal.text);
      return reinterpret_cast<LRESULT>(g_br_bg);
    }
    case WM_DRAWITEM:
      draw_item(reinterpret_cast<const DRAWITEMSTRUCT*>(lp));
      return TRUE;
    case WM_MEASUREITEM: {
      auto* mis = reinterpret_cast<MEASUREITEMSTRUCT*>(lp);
      if (mis != nullptr) {
        mis->itemHeight = S(24);
        return TRUE;
      }
      return 0;
    }
    case WM_COMMAND:
      on_command(LOWORD(wp), HIWORD(wp), reinterpret_cast<HWND>(lp));
      return 0;
    case WM_UI_SLIDER: {
      const int id = static_cast<int>(wp);
      const int pos = static_cast<int>(lp);
      if (id == IDC_GPU_SLIDER) {
        g_ui.gpu_auto = false;
        toggle_set(g_gpu_auto, false);
        const int n = static_cast<int>(g_meta.expert_per_layer.size());
        g_ui.gpu_layers = std::max(0, std::min(n, pos));
        refresh_estimates(true);
        update_args_summary();
      } else if (id == IDC_CTX_SLIDER) {
        g_ui.ctx_auto = false;
        g_ui.ctx_invalid = false;
        g_ui.ctx_value = ctx_from_slider(pos);
        refresh_estimates(true);
        update_args_summary();
      }
      return 0;
    }
    case WM_VSCROLL: {
      SCROLLINFO si{};
      si.cbSize = sizeof(si);
      si.fMask = SIF_ALL;
      GetScrollInfo(hwnd, SB_VERT, &si);
      int pos = si.nPos;
      switch (LOWORD(wp)) {
        case SB_LINEUP: pos -= S(30); break;
        case SB_LINEDOWN: pos += S(30); break;
        case SB_PAGEUP: pos -= g_view_h; break;
        case SB_PAGEDOWN: pos += g_view_h; break;
        case SB_THUMBTRACK:
        case SB_THUMBPOSITION: pos = si.nTrackPos; break;
        case SB_TOP: pos = 0; break;
        case SB_BOTTOM: pos = si.nMax; break;
        default: return 0;
      }
      const int max_scroll = std::max(0, g_content_h - g_view_h);
      pos = std::max(0, std::min(max_scroll, pos));
      if (pos != g_scroll) {
        g_scroll = pos;
        layout();
        UpdateWindow(hwnd);
      }
      return 0;
    }
    case WM_MOUSEWHEEL: {
      const int delta = GET_WHEEL_DELTA_WPARAM(wp);
      scroll_by(-delta / WHEEL_DELTA * S(60));
      return 0;
    }
    case WM_MOUSEHWHEEL:  // no horizontally scrollable content
      return 0;
    case WM_TIMER:
      if (wp == 1) on_timer();
      else if (wp == 2) poll_model_scan();
      return 0;
    case WM_APP + 70:
      poll_model_scan();
      return 0;
    // Test/automation hooks: read/write the scroll offset directly so
    // build/test_launcher.py can take deterministic screenshots.
    case WM_APP + 30:
      return g_scroll;
    case WM_APP + 31: {
      const int max_scroll = std::max(0, g_content_h - g_view_h);
      int target = static_cast<int>(lp);
      target = std::max(0, std::min(max_scroll, target));
      if (target != g_scroll) {
        g_scroll = target;
        layout();
        UpdateWindow(hwnd);
      }
      return g_scroll;
    }
    case WM_APP + 40: {  // automation: internal toggle state (0/1)
      HWND c = GetDlgItem(g_hwnd, static_cast<int>(wp));
      return c == nullptr ? -1 : (toggle_is_on(c) ? 1 : 0);
    }
    case WM_DPICHANGED: {
      const int dpi = static_cast<int>(HIWORD(wp));
      if (dpi >= 72 && dpi <= 480) {
        g_dpi = dpi;
        reload_fonts();
        for (const auto& kv : g_ctrl_font_pt) apply_font_to_control(kv.first, kv.second);
        const RECT* r = reinterpret_cast<const RECT*>(lp);
        if (r != nullptr)
          SetWindowPos(hwnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top,
                       SWP_NOZORDER | SWP_NOACTIVATE);
        layout();
      }
      return 0;
    }
    case WM_CLOSE: {
      if (g_proc != nullptr) {
        const int rc = MessageBoxW(hwnd, L"引擎仍在运行，是否停止引擎并退出？", L"本地推理引擎",
                                   MB_ICONQUESTION | MB_YESNO);
        if (rc != IDYES) return 0;
        stop_engine();
      }
      DestroyWindow(hwnd);
      return 0;
    }
    case WM_DESTROY:
      if (g_proc != nullptr) stop_engine();
      KillTimer(hwnd, 1);
      PostQuitMessage(0);
      return 0;
    default: break;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
std::wstring find_engine() {
  if (const wchar_t* env = _wgetenv(L"ENGINE_EXE"); env != nullptr && *env != L'\0') {
    if (file_exists(env)) return env;
  }
  const std::wstring candidates[] = {
      join_path(g_launcher_dir, L"engine.exe"),
      join_path(g_launcher_dir, L"build\\engine.exe"),
      join_path(g_launcher_dir, L"..\\build\\engine.exe"),
      L"engine.exe",
  };
  for (const auto& c : candidates)
    if (file_exists(c)) return c;
  wchar_t found[MAX_PATH];
  if (SearchPathW(nullptr, L"engine.exe", nullptr, MAX_PATH, found, nullptr) > 0) return found;
  return std::wstring();
}

}  // namespace

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int show) {
  enable_dpi_awareness();
  g_dpi = query_dpi(nullptr);
  if (g_dpi < 72 || g_dpi > 480) g_dpi = 96;

  int theme_mode = 0;  // 0 follow system, 1 dark, 2 light
  int autostart = 0;
  int cli_port = 0;
  int cli_ctx = -1;
  bool cli_auto_browser = true;
  std::wstring cli_engine;
  int argc = 0;
  LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
  for (int i = 1; argv != nullptr && i < argc; ++i) {
    const std::wstring a = argv[i];
    auto next = [&](const wchar_t* name) -> std::wstring {
      if (i + 1 < argc) return argv[++i];
      MessageBoxW(nullptr, (std::wstring(L"缺少参数值：") + name).c_str(), L"启动参数",
                  MB_ICONERROR);
      return std::wstring();
    };
    if (a == L"--theme") {
      const std::wstring v = lower_w(next(L"--theme"));
      if (v == L"dark") theme_mode = 1;
      else if (v == L"light") theme_mode = 2;
      else theme_mode = 0;
    } else if (a == L"--autostart") {
      const std::wstring v = next(L"--autostart");
      long long n = 0;
      if (parse_ll(v, &n) && n >= 1 && n <= 1000) autostart = static_cast<int>(n);
    } else if (a == L"--port") {
      const std::wstring v = next(L"--port");
      long long n = 0;
      if (parse_ll(v, &n) && n >= 1 && n <= 65535) cli_port = static_cast<int>(n);
    } else if (a == L"--ctx") {
      const std::wstring v = next(L"--ctx");
      long long n = 0;
      if (parse_ll(v, &n) && n >= 0 && n <= kCtxMax) cli_ctx = static_cast<int>(n);
    } else if (a == L"--no-browser") {
      cli_auto_browser = false;
    } else if (a == L"--engine") {
      cli_engine = next(L"--engine");
    }
  }
  if (argv != nullptr) LocalFree(argv);

  g_launcher_dir = exe_dir();
  if (g_launcher_dir.empty()) {
    wchar_t cwd[4096];
    if (GetCurrentDirectoryW(4096, cwd) > 0) g_launcher_dir = cwd;
  }
  g_ui_map_path = join_path(g_launcher_dir, L"launcher-ui-map.json");
  g_effective_args_path = join_path(g_launcher_dir, L"launcher-effective-args.txt");
  g_engine = file_exists(cli_engine) ? cli_engine : find_engine();

  init_theme(theme_mode);
  load_schema();
  sync_state_from_schema();
  if (cli_port > 0) g_ui.port = cli_port;
  if (cli_ctx >= 0) {
    if (cli_ctx == 0) {
      g_ui.ctx_auto = true;
    } else {
      g_ui.ctx_auto = false;
      g_ui.ctx_value = cli_ctx;
    }
    g_ui.ctx_invalid = false;
  }
  g_ui.auto_browser = cli_auto_browser;

  WNDCLASSEXW wc{};
  wc.cbSize = sizeof(wc);
  wc.style = CS_HREDRAW | CS_VREDRAW;
  wc.lpfnWndProc = wnd_proc;
  wc.hInstance = instance;
  wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
  wc.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
  wc.hbrBackground = nullptr;
  wc.lpszClassName = kAppClass;
  if (RegisterClassExW(&wc) == 0) return 1;

  const int width = S(1080), height = S(880);
  const int x = (GetSystemMetrics(SM_CXSCREEN) - width) / 2;
  const int y = (GetSystemMetrics(SM_CYSCREEN) - height) / 2;
  // WS_EX_COMPOSITED double-buffers the whole page (parent + child controls)
  // so scrolling with the wheel/scrollbar does not flicker; WS_CLIPCHILDREN
  // keeps the parent from painting over its children, and WM_ERASEBKGND
  // returns 1 (all background work happens in WM_PAINT).
  HWND hwnd = CreateWindowExW(WS_EX_COMPOSITED, kAppClass, kAppTitle,
                              WS_OVERLAPPEDWINDOW | WS_VSCROLL | WS_CLIPCHILDREN, x, y, width,
                              height, nullptr, nullptr, instance, nullptr);
  if (hwnd == nullptr) return 1;
  set_dark_title_bar(hwnd);
  ShowWindow(hwnd, show == 0 ? SW_SHOWNORMAL : show);
  UpdateWindow(hwnd);

  sync_controls_from_state();
  update_schema_info();
  update_buttons_state();
  update_args_summary();
  update_status_message(g_engine.empty() ? L"找不到 engine.exe：无法启动服务"
                                         : L"就绪：选择模型后点「启动引擎」",
                        g_engine.empty() ? Role::Danger : Role::Normal);
  update_url_line();
  layout();
  const bool have_models = load_models();
  if (autostart > 0 && have_models && autostart <= static_cast<int>(g_models.size())) {
    g_ui.model_auto = false;
    g_ui.model_index = autostart - 1;
    update_model_combo_selection();
    load_model_meta(true);
    refresh_estimates(true);
    update_args_summary();
    start_engine();
  } else {
    update_buttons_state();
    update_args_summary();
    layout();
    UpdateWindow(hwnd);
  }

  MSG msg;
  while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
    TranslateMessage(&msg);
    DispatchMessageW(&msg);
  }
  return 0;
}

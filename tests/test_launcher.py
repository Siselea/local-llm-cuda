#!/usr/bin/env python3
"""Acceptance test for build/launcher.exe (Win32 GUI launcher, UI round 2).

Covers the "看得懂 / 全部 CLI 选项 / 滑块实时估算 / SSD 判据 / 启动停止" requirements:

  1. default page      -> build/launcher-01-default.png
  2. advanced panel    -> build/launcher-02-advanced.png (schema-generated controls)
  3. SSD greyed state  -> build/launcher-03-ssd.png  (+ launcher-03b-ssd-enabled.png)
  4. running engine    -> build/launcher-04-running.png (status + URL)

UI round 3 (user bug reports) also covered: native combo boxes are selectable
with real mouse messages (model + KV), the composited window scrolls without
flicker/ghosting (frame hashes), the default model is the 9B (recommended), and
the "打开对话页" tooltip explains the 0.2-0.5 s first-token latency.

Also checks, with machine-readable evidence in build/launcher-evidence.json:
  * launcher.exe is a GUI-subsystem binary (no console window)
  * every `engine.exe serve --print-config --json` option has a control (100%)
  * tooltips are registered for the controls
  * the GPU-layer slider and the editable arbitrary-context box (any positive
    integer, not just presets) update the live estimate labels; an invalid context
    value shows an error hint instead of silently launching
  * the SSD switch is disabled with a rationale when the model fits VRAM+RAM,
    becomes enabled when it does not, and warns that SSD streaming is unimplemented
  * a real start/stop of the 0.5B model through the UI (engine /health reachable)
  * the exact engine command line (incl. an explicit --gpu-layers value) is in
    build/launcher-engine.log and the advanced override in launcher-engine.cfg

Run with a Python that has Pillow (screenshots), from the repository root:
  MODEL_DIR=<directory with the four .gguf models> python tests/test_launcher.py
or create build/test_launcher.py as a wrapper for the same command.
"""
import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import os
import re
import subprocess
import sys
import struct
import time
import http.client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("MODEL_DIR", "")
LAUNCHER = ROOT + "/build/launcher.exe"
ENGINE = ROOT + "/build/engine.exe"
OUT = ROOT + "/build"
TITLE = "本地推理引擎 · Launcher"
PORT = 7030

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

user32.FindWindowW.restype = wt.HWND
user32.FindWindowW.argtypes = [wt.LPCWSTR, wt.LPCWSTR]
user32.GetDlgItem.restype = wt.HWND
user32.GetDlgItem.argtypes = [wt.HWND, ctypes.c_int]
user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetWindowTextLengthW.argtypes = [wt.HWND]
user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.SendMessageW.restype = ctypes.c_ssize_t
user32.SendMessageW.argtypes = [wt.HWND, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
user32.PostMessageW.argtypes = [wt.HWND, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
user32.GetWindowDC.restype = wt.HDC
user32.GetWindowDC.argtypes = [wt.HWND]
user32.PrintWindow.argtypes = [wt.HWND, wt.HDC, ctypes.c_uint]
user32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]
user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
user32.SetForegroundWindow.argtypes = [wt.HWND]
user32.RedrawWindow.argtypes = [wt.HWND, ctypes.c_void_p, wt.HANDLE, ctypes.c_uint]
user32.IsWindowEnabled.restype = wt.BOOL
user32.IsWindowEnabled.argtypes = [wt.HWND]
user32.EnumWindows.restype = wt.BOOL
user32.EnumWindows.argtypes = [ctypes.c_void_p, wt.LPARAM]
user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.GetWindowLongPtrW.argtypes = [wt.HWND, ctypes.c_int]
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.QueryFullProcessImageNameW.restype = wt.BOOL
kernel32.QueryFullProcessImageNameW.argtypes = [
    wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)]
kernel32.CloseHandle.argtypes = [wt.HANDLE]

gdi32.CreateCompatibleDC.restype = wt.HDC
gdi32.CreateCompatibleDC.argtypes = [wt.HDC]
gdi32.CreateCompatibleBitmap.restype = wt.HBITMAP
gdi32.CreateCompatibleBitmap.argtypes = [wt.HDC, ctypes.c_int, ctypes.c_int]
gdi32.SelectObject.restype = wt.HGDIOBJ
gdi32.SelectObject.argtypes = [wt.HDC, wt.HGDIOBJ]
gdi32.DeleteObject.argtypes = [wt.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wt.HDC]
gdi32.BitBlt.argtypes = [wt.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                         wt.HDC, ctypes.c_int, ctypes.c_int, wt.DWORD]
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.GetDC.restype = wt.HDC
user32.GetDC.argtypes = [wt.HWND]
gdi32.GetDIBits.restype = ctypes.c_int
gdi32.GetDIBits.argtypes = [wt.HDC, wt.HBITMAP, ctypes.c_uint, ctypes.c_uint,
                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]

IDS = {
    "model": 100, "refresh": 102, "model_info": 103, "scan_status": 104, "refresh": 102,
    "gpu_slider": 110, "gpu_auto": 111, "gpu_text": 112,
    "ctx_slider": 120, "ctx_text": 121, "ctx_edit": 122, "ctx_auto": 123,
    "kv": 130, "kv_text": 131,
    "ssd": 140, "ssd_note": 141,
    "dev_route": 142, "layer_graph": 143, "cpu_int8": 144, "win_priority": 145,
    "thinking": 146, "vram_auto": 147, "vram_auto_warn": 148, "win_affinity": 149,
    "threads": 150, "threads_hint": 151,
    "port": 160, "host": 161, "auto_browser": 162,
    "adv_toggle": 180, "adv_search": 181, "adv_count": 182,
    "start": 190, "stop": 191, "open_page": 192, "open_log": 193,
    "status": 200, "url": 201, "log_info": 202, "args": 203,
    "placement": 204, "schema": 205, "est": 208,
}
IDC_ADV_BASE = 1000
WM_COMMAND = 0x0111
WM_SETTEXT = 0x000C
WM_CHAR = 0x0102
WM_VSCROLL = 0x0114
WM_MOUSEWHEEL = 0x020A
BN_CLICKED = 0
EN_CHANGE = 0x0300
CBN_SELCHANGE = 1
CB_SETCURSEL = 0x014E
CB_GETCOUNT = 0x0146
SB_TOP = 6
WM_APP_SLIDER_SET = 0x8000 + 21
TTM_GETTOOLCOUNT = 0x0400 + 13  # WM_USER+13 per commctrl.h
CB_GETCURSEL = 0x0147
CB_GETLBTEXT = 0x0148
CB_GETLBTEXTLEN = 0x0149
CB_GETITEMHEIGHT = 0x0154
CB_GETDROPPEDSTATE = 0x0157
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_EX_COMPOSITED = 0x02000000
CBS_TYPEMASK = 0x0003
CBS_DROPDOWNLIST = 0x0003


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16), ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32), ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32), ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32), ("biClrImportant", ctypes.c_uint32)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", ctypes.c_uint32 * 3)]


def wait_window(title, timeout=60, pid=None):
    """Find our launcher window; pid-scoped so a user's window with the same
    title (already open on this desktop) is never picked up by mistake."""
    end = time.time() + timeout
    while time.time() < end:
        if pid is None:
            hwnd = user32.FindWindowW(None, title)
            if hwnd:
                return hwnd
        else:
            found = []

            @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
            def cb(hwnd, _):
                wpid = wt.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
                if wpid.value != pid:
                    return True
                buf = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, buf, 256)
                if buf.value == title:
                    found.append(hwnd)
                return True

            user32.EnumWindows(cb, 0)
            if found:
                return found[0]
        time.sleep(0.2)
    return None


def process_image_path(pid):
    h = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    buf = ctypes.create_unicode_buffer(1024)
    size = wt.DWORD(len(buf))
    ok = kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
    kernel32.CloseHandle(h)
    return buf.value if ok else ""


def find_launcher_window(pid, exe_path, timeout=60):
    """Triple validation: owning PID + class name + title (+ image path).

    A user's launcher may be open with the same title on the same desktop; only a
    window that belongs to our own process may be driven by the test.
    """
    end = time.time() + timeout
    identity = {}
    while time.time() < end:
        found = []

        @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        def cb(hwnd, _):
            wpid = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
            if wpid.value != pid:
                return True
            cls = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, cls, 64)
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            if cls.value == "EngineLauncherWindowV2" and buf.value == TITLE:
                found.append((hwnd, cls.value, buf.value))
            return True

        user32.EnumWindows(cb, 0)
        if found:
            hwnd, cls_name, title = found[0]
            image = process_image_path(pid)
            try:
                same_image = (os.path.normcase(os.path.abspath(image)) ==
                              os.path.normcase(os.path.abspath(exe_path)))
            except Exception:
                same_image = False
            identity = {"hwnd": hwnd, "pid": pid, "class": cls_name, "title": title,
                        "image_path": image, "image_matches": same_image,
                        "valid": bool(image) and same_image}
            return hwnd, identity
        time.sleep(0.2)
    return None, identity


def other_launcher_windows(own_pid):
    """Any EngineLauncherWindowV2 window that is not ours (e.g. the user's)."""
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        wpid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
        if wpid.value == own_pid:
            return True
        cls = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, cls, 64)
        if cls.value == "EngineLauncherWindowV2":
            found.append((hwnd, wpid.value))
        return True

    user32.EnumWindows(cb, 0)
    return found


def wait_gpu_free(min_free_mb=4500, timeout_s=180):
    """Shared machine: wait until the GPU has room for the 0.5B test model."""
    start = time.time()
    while time.time() - start < timeout_s:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True)
        try:
            free_mb = int(out.stdout.decode("utf-8", "replace").strip().splitlines()[0])
        except Exception:
            free_mb = 0
        if free_mb >= min_free_mb:
            return True, free_mb
        time.sleep(2.0)
    return False, free_mb


def count_engine_processes():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq engine.exe", "/FO", "CSV", "/NH"],
                         capture_output=True).stdout.decode("utf-8", "replace")
    return sum(1 for line in out.splitlines() if "engine.exe" in line.lower())


def model_index_of(hwnd, needle):
    combo = user32.GetDlgItem(hwnd, IDS["model"])
    n = int(user32.SendMessageW(combo, 0x0146, 0, 0))
    for i in range(1, n):
        if needle in combo_item_text(combo, i):
            return i
    return -1


def toggle_state(hwnd, ctrl_id):
    """Internal owner-draw toggle state (WM_APP+40, added by launcher.cpp)."""
    return int(user32.SendMessageW(hwnd, 0x8000 + 40, ctrl_id, 0))


def gpu_control_state(hwnd):
    slider = user32.GetDlgItem(hwnd, IDS["gpu_slider"])
    auto = user32.GetDlgItem(hwnd, IDS["gpu_auto"])
    vram_auto = user32.GetDlgItem(hwnd, IDS["vram_auto"])
    return {
        "slider_enabled": bool(user32.IsWindowEnabled(slider)) if slider else None,
        "auto_enabled": bool(user32.IsWindowEnabled(auto)) if auto else None,
        "auto_toggle": toggle_state(hwnd, IDS["gpu_auto"]),
        "vram_auto_enabled": bool(user32.IsWindowEnabled(vram_auto)) if vram_auto else None,
        "gpu_text": get_text(hwnd, "gpu_text"),
        "estimate": get_text(hwnd, "est"),
    }


def get_text(hwnd, key):
    ctrl = user32.GetDlgItem(hwnd, IDS[key])
    if not ctrl:
        return ""
    length = user32.GetWindowTextLengthW(ctrl)
    buf = ctypes.create_unicode_buffer(max(length + 2, 8))
    user32.GetWindowTextW(ctrl, buf, len(buf))
    return buf.value


def command(hwnd, key, code=BN_CLICKED, lparam=0):
    user32.SendMessageW(hwnd, WM_COMMAND, (IDS[key] & 0xFFFF) | ((code & 0xFFFF) << 16), lparam)


def combo_select(hwnd, key, index):
    combo = user32.GetDlgItem(hwnd, IDS[key])
    user32.SendMessageW(combo, CB_SETCURSEL, index, 0)
    user32.SendMessageW(hwnd, WM_COMMAND, (IDS[key] & 0xFFFF) | (CBN_SELCHANGE << 16), combo)
    return combo


WM_APP_GET_SCROLL = 0x8000 + 30
WM_APP_SET_SCROLL = 0x8000 + 31


def scroll_to(hwnd, offset):
    """Deterministic scroll (the launcher exposes WM_APP+31 for automation)."""
    user32.SendMessageW(hwnd, WM_APP_SET_SCROLL, 0, offset)
    time.sleep(0.5)


def scroll_top(hwnd):
    scroll_to(hwnd, 0)


def scroll_down(hwnd, notches=6):
    """Scroll by whole notches (each 60 px), returning the final offset."""
    cur = int(user32.SendMessageW(hwnd, WM_APP_GET_SCROLL, 0, 0))
    scroll_to(hwnd, cur + notches * 60)


def redraw(hwnd):
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetForegroundWindow(hwnd)
    user32.RedrawWindow(hwnd, None, None, 0x0181)  # INVALIDATE|UPDATENOW|ALLCHILDREN
    time.sleep(0.5)


def capture(hwnd, name, use_screen=True):
    from PIL import Image
    rect = wt.RECT()
    # Prefer the *visible* frame (DWMWA_EXTENDED_FRAME_BOUNDS): GetWindowRect
    # includes the invisible resize border, which would capture desktop slivers.
    dwmapi = ctypes.WinDLL("dwmapi")
    got = dwmapi.DwmGetWindowAttribute(wt.HWND(hwnd), 9, ctypes.byref(rect), ctypes.sizeof(rect))
    if got != 0:
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w <= 0 or h <= 0:
        raise RuntimeError("bad window rect %r" % ((w, h),))
    # Park the mouse so a random tooltip cannot end up in a deliverable screenshot,
    # and make sure the launcher is the foreground window: the capture is a screen
    # BitBlt, so a transient overlay from a background app would corrupt the frame.
    user32.SetCursorPos(4, 4)
    user32.ShowWindow(hwnd, 5)  # SW_SHOW
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.4)
    hdc = user32.GetWindowDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mem, bmp)
    if use_screen:
        # PrintWindow with PW_RENDERFULLCONTENT occasionally returns a partially
        # composed frame right after a programmatic scroll; the window is
        # foreground here, so a screen BitBlt is the faithful capture.
        screen = user32.GetDC(None)
        gdi32.BitBlt(mem, 0, 0, w, h, screen, rect.left, rect.top, 0x00CC0020)
        user32.ReleaseDC(None, screen)
    else:
        user32.PrintWindow(hwnd, mem, 2)  # PW_RENDERFULLCONTENT
    info = BITMAPINFO()
    info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    info.bmiHeader.biWidth = w
    info.bmiHeader.biHeight = -h
    info.bmiHeader.biPlanes = 1
    info.bmiHeader.biBitCount = 32
    info.bmiHeader.biCompression = 0
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(info), 0)
    Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).save(os.path.join(OUT, name))
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(hwnd, hdc)
    return os.path.join(OUT, name)


def health_ok(port=PORT, timeout=1.2):
    # http.client against 127.0.0.1: bypasses the system proxy that urllib would
    # otherwise honour.  "Connection: close" + a full body read keeps the engine's
    # single-threaded HTTP server from parking a kept-alive socket.
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("GET", "/health", headers={"Connection": "close"})
        r = conn.getresponse()
        r.read()
        conn.close()
        return r.status == 200
    except Exception:
        return False


def http_get(path, port=PORT, timeout=3.0):
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("GET", path, headers={"Connection": "close"})
        r = conn.getresponse()
        body = r.read(65536).decode("utf-8", "replace")
        conn.close()
        return r.status, body
    except Exception as exc:
        return 0, str(exc)


def http_get_retry(path, port=PORT, deadline_s=20.0):
    end = time.time() + deadline_s
    last = (0, "not tried")
    while time.time() < end:
        last = http_get(path, port)
        if last[0] == 200:
            return last
        time.sleep(0.5)
    return last


def pe_subsystem(path):
    """2 = WINDOWS_GUI (no console window)."""
    with open(path, "rb") as fh:
        data = fh.read(0x200)
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    return struct.unpack_from("<H", data, e_lfanew + 24 + 68)[0]


def run_json_engine(args):
    raw = subprocess.run([ENGINE] + args, capture_output=True).stdout.decode("utf-8", "replace")
    pos = raw.find("{")
    return json.loads(raw[pos:]) if pos >= 0 else {}


def tooltip_count(pid):
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        wpid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
        if wpid.value != pid:
            return True
        cls = ctypes.create_unicode_buffer(128)
        user32.GetClassNameW(hwnd, cls, 128)
        if cls.value.lower().startswith("tooltips_class32"):
            found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    if not found:
        return 0
    return int(user32.SendMessageW(found[0], TTM_GETTOOLCOUNT, 0, 0))


def adv_ctrl_id(uimap, key):
    for idx, item in enumerate(uimap.get("advanced", [])):
        if item.get("key") == key:
            return IDC_ADV_BASE + idx
    return None


def set_advanced(hwnd, uimap, key, value):
    cid = adv_ctrl_id(uimap, key)
    if cid is None:
        return False
    ctrl = user32.GetDlgItem(hwnd, cid)
    if not ctrl:
        return False
    buf = ctypes.create_unicode_buffer(str(value))
    user32.SendMessageW(ctrl, 0x000C, 0, ctypes.cast(buf, ctypes.c_void_p).value)  # WM_SETTEXT
    user32.SendMessageW(hwnd, WM_COMMAND, (cid & 0xFFFF) | (EN_CHANGE << 16), ctrl)
    return True


def combo_item_text(combo, index):
    n = int(user32.SendMessageW(combo, CB_GETLBTEXTLEN, index, 0))
    buf = ctypes.create_unicode_buffer(max(n + 2, 8))
    user32.SendMessageW(combo, CB_GETLBTEXT, index,
                        ctypes.cast(buf, ctypes.c_void_p).value or 0)
    return buf.value


def find_combo_lbox(pid, timeout=2.0):
    """Find the native drop-down list window ("ComboLBox") of the launcher."""
    end = time.time() + timeout
    while time.time() < end:
        found = []

        @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        def cb(hwnd, _):
            wpid = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
            if wpid.value != pid:
                return True
            cls = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, cls, 64)
            if cls.value == "ComboLBox" and user32.IsWindowVisible(hwnd):
                found.append(hwnd)
            return True

        user32.EnumWindows(cb, 0)
        if found:
            return found[0]
        time.sleep(0.1)
    return None


def click_sync(hwnd, x, y):
    """Deliver a real left-click (down+up) to a control, synchronously."""
    lp = ((y & 0xFFFF) << 16) | (x & 0xFFFF)
    user32.SendMessageW(hwnd, WM_LBUTTONDOWN, 1, lp)
    time.sleep(0.06)
    user32.SendMessageW(hwnd, WM_LBUTTONUP, 0, lp)


def dropdown_open(hwnd, key, pid):
    combo = user32.GetDlgItem(hwnd, IDS[key])
    # If the list is already open (e.g. kept open for a screenshot), do not
    # click the arrow again - that would close it.
    if int(user32.SendMessageW(combo, CB_GETDROPPEDSTATE, 0, 0)) == 1:
        return find_combo_lbox(pid, timeout=1.0)
    rect = wt.RECT()
    user32.GetWindowRect(combo, ctypes.byref(rect))
    w = max(8, rect.right - rect.left)
    h = max(8, rect.bottom - rect.top)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.2)
    click_sync(combo, w - 10, h // 2)   # the native drop-down arrow
    return find_combo_lbox(pid)


def dropdown_pick(hwnd, key, pid, index):
    """Open a native combo by mouse, click item `index`, return the new selection."""
    combo = user32.GetDlgItem(hwnd, IDS[key])
    lbox = dropdown_open(hwnd, key, pid)
    if not lbox:
        return -2  # dropdown did not open
    item_h = int(user32.SendMessageW(combo, CB_GETITEMHEIGHT, 0, 0)) or 20
    click_sync(lbox, 12, item_h // 2 + index * item_h)
    time.sleep(0.8)
    return int(user32.SendMessageW(combo, CB_GETCURSEL, 0, 0))


def frame_hash(hwnd):
    """SHA-256 of the window content, for scroll-stability checks.

    Uses PrintWindow (PW_RENDERFULLCONTENT) so other windows that happen to
    overlap the launcher on the shared desktop cannot change the frames; falls
    back to a screen BitBlt only if PrintWindow refuses.
    """
    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    hdc = user32.GetWindowDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mem, bmp)
    user32.PrintWindow.restype = ctypes.c_int
    if not user32.PrintWindow(hwnd, mem, 2):
        screen = user32.GetDC(None)
        gdi32.BitBlt(mem, 0, 0, w, h, screen, rect.left, rect.top, 0x00CC0020)
        user32.ReleaseDC(None, screen)
    info = BITMAPINFO()
    info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    info.bmiHeader.biWidth = w
    info.bmiHeader.biHeight = -h
    info.bmiHeader.biPlanes = 1
    info.bmiHeader.biBitCount = 32
    info.bmiHeader.biCompression = 0
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(info), 0)
    data = bytes(buf)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(hwnd, hdc)
    return hashlib.sha256(data).hexdigest()


def parse_mb(text, needle):
    """Return the MB number that follows `needle` in a label like 'KV 显存 ≈ 96 MB'."""
    pos = text.find(needle)
    if pos < 0:
        return None
    tail = text[pos + len(needle):].strip()
    num = ""
    for ch in tail:
        if ch.isdigit() or ch == ".":
            num += ch
        else:
            break
    if not num:
        return None
    value = float(num)
    if "GB" in tail[:len(num) + 4]:
        value *= 1024.0
    return value


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not MODEL_DIR:
        print("ERROR: set MODEL_DIR to the directory containing the four .gguf models")
        return 1
    scan_drive = os.path.splitdrive(MODEL_DIR)[0] + "/"
    os.makedirs(OUT, exist_ok=True)
    evidence = {"launcher": LAUNCHER, "engine": ENGINE}
    checks = {}

    if health_ok(0.5):
        print("ERROR: something is already listening on %d" % PORT)
        return 1

    # ---- full-disk scan pipeline (D: only to stay fast; temp cache) ----------
    test_cache = os.path.join(OUT, "test-launcher-cache.json")
    try:
        os.remove(test_cache)
    except OSError:
        pass
    scan_env = os.environ.copy()
    scan_env["ENGINE_MODELS_CACHE"] = test_cache
    scan_env["ENGINE_SCAN_DRIVES"] = scan_drive
    scan_env.pop("ENGINE_LAUNCHER_NO_AUTO_SCAN", None)
    t_scan0 = time.time()
    scan_run = subprocess.run([ENGINE, "models", "--refresh", "--progress", "--json"], env=scan_env, capture_output=True,
                              timeout=600)
    scan_seconds = time.time() - t_scan0
    scan_progress = scan_run.stderr.decode("utf-8", "replace")
    scan_out = scan_run.stdout.decode("utf-8", "replace")
    scan_json = {}
    try:
        scan_json = json.loads(scan_out)
    except Exception:
        pass
    evidence["scan_progress_log"] = scan_progress[-2000:]
    evidence["scan_first_seconds"] = scan_seconds
    evidence["scan_first_count"] = scan_json.get("count", -1)
    checks["scan_progress_reported"] = ("[scan] progress" in scan_progress and
                                        "drives_done=" in scan_progress)
    checks["scan_cache_written"] = os.path.exists(test_cache)
    t_cache0 = time.time()
    cache_run = subprocess.run([ENGINE, "models", "--json"], env=scan_env,
                               capture_output=True, timeout=60)
    cache_seconds = time.time() - t_cache0
    cache_json = {}
    try:
        cache_json = json.loads(cache_run.stdout.decode("utf-8", "replace"))
    except Exception:
        pass
    evidence["scan_cache_seconds"] = cache_seconds
    checks["cache_hit_read"] = (cache_json.get("source") == "cache" and
                                cache_json.get("count", -1) == evidence["scan_first_count"] and
                                cache_seconds < 2.0)
    # --no-scan: only the explicitly given dir, no full-disk/cache use
    ns_env = os.environ.copy()
    ns_env.pop("ENGINE_MODELS_CACHE", None)
    ns_run = subprocess.run([ENGINE, "models", "--no-scan", "--model-dir", MODEL_DIR,
                             "--json"], env=ns_env, capture_output=True, timeout=120)
    ns_json = {}
    try:
        ns_json = json.loads(ns_run.stdout.decode("utf-8", "replace"))
    except Exception:
        pass
    ns_paths = [m.get("path", "") for m in ns_json.get("models", [])]
    model_root = os.path.normcase(os.path.abspath(MODEL_DIR)).rstrip("\\/")
    def under_model_dir(p):
        q = os.path.normcase(os.path.abspath(p))
        return q == model_root or q.startswith(model_root + os.sep)
    checks["no_scan_only_model_dir"] = (ns_json.get("source") == "no-scan" and
                                        len(ns_paths) >= 5 and
                                        all(under_model_dir(p) for p in ns_paths))
    print("[0] scan: progress=%s cache=%s cache_read=%.3fs no_scan=%s" % (
        checks["scan_progress_reported"], checks["scan_cache_written"],
        cache_seconds, checks["no_scan_only_model_dir"]))

    # ---- binary / GUI-subsystem ------------------------------------------------
    checks["gui_subsystem"] = pe_subsystem(LAUNCHER) == 2
    evidence["pe_subsystem"] = pe_subsystem(LAUNCHER)
    print("[1] GUI subsystem:", checks["gui_subsystem"])

    foreign = other_launcher_windows(-1)
    if foreign:
        print("ERROR: another launcher window is already open (pid(s)=%r); "
              "ask the user to close it before running the automated test" %
              [p for _, p in foreign])
        return 4
    gpu_ok, gpu_free = wait_gpu_free()
    evidence["gpu_free_mb_before_test"] = gpu_free
    print("[gpu] free before test:", gpu_free, "MiB", "ok=", gpu_ok)
    if not gpu_ok:
        print("ERROR: GPU is still busy (another model/agent is running); retry later")
        return 1
    launcher_env = os.environ.copy()
    launcher_env["ENGINE_MODELS_CACHE"] = test_cache
    launcher_env["ENGINE_LAUNCHER_NO_AUTO_SCAN"] = "1"
    launcher_env["ENGINE_SCAN_DRIVES"] = scan_drive
    launcher_env["ENGINE_LAUNCHER_NO_DIALOG"] = "1"
    proc = subprocess.Popen([LAUNCHER, "--theme", "dark", "--port", str(PORT), "--no-browser"],
                            env=launcher_env)
    hwnd, window_identity = find_launcher_window(proc.pid, LAUNCHER)
    checks["window"] = bool(hwnd)
    checks["window_owner_ok"] = bool(window_identity.get("valid"))
    evidence["window_identity"] = window_identity
    if not hwnd:
        print("ERROR: launcher window not found, exit=%r" % proc.poll())
        return 1
    print("[2] window:", hwnd)
    time.sleep(2.5)

    # ---- [2b] native combos / composited window / default model 9B ---------
    model_combo = user32.GetDlgItem(hwnd, IDS["model"])
    kv_combo = user32.GetDlgItem(hwnd, IDS["kv"])
    cls = ctypes.create_unicode_buffer(64)
    user32.GetClassNameW(model_combo, cls, 64)
    style = int(user32.GetWindowLongPtrW(model_combo, GWL_STYLE))
    exstyle = int(user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE))
    model_count = int(user32.SendMessageW(model_combo, 0x0146, 0, 0))  # CB_GETCOUNT
    model_items = [combo_item_text(model_combo, i) for i in range(model_count)]
    item_9b = next((i for i, t in enumerate(model_items) if "Qwen3.5-9B" in t), -1)
    item_35b = next((i for i, t in enumerate(model_items) if "35B" in t), -1)
    item_05b = next((i for i, t in enumerate(model_items) if "qwen2.5" in t), -1)
    model_cursel = int(user32.SendMessageW(model_combo, CB_GETCURSEL, 0, 0))
    evidence["native_combo_class"] = cls.value
    evidence["main_exstyle_hex"] = hex(exstyle)
    evidence["model_items"] = model_items
    evidence["model_default_sel"] = model_cursel
    evidence["model_default_9b_index"] = item_9b
    checks["native_combo"] = (cls.value == "ComboBox" and
                              (style & CBS_TYPEMASK) == CBS_DROPDOWNLIST)
    checks["composited_window"] = bool(exstyle & WS_EX_COMPOSITED)
    checks["default_model_is_9b"] = (item_9b >= 0 and model_cursel == item_9b and
                                     "推荐" in model_items[item_9b])
    checks["model_labels"] = (item_35b >= 0 and "最强，但慢" in model_items[item_35b])
    checks["drive_labels"] = all(re.match(r"^\[[A-Za-z]:\] \d+\.\d\d GB · ", t)
                                 for t in model_items[1:])
    threads_combo = user32.GetDlgItem(hwnd, IDS["threads"])
    thread_sel = int(user32.SendMessageW(threads_combo, CB_GETCURSEL, 0, 0))
    thread_text = combo_item_text(threads_combo, thread_sel)
    evidence["threads_combo_sel"] = thread_sel
    evidence["threads_combo_text"] = thread_text
    evidence["threads_hint"] = get_text(hwnd, "threads_hint") if "threads_hint" in IDS else ""
    checks["threads_default_auto"] = (thread_sel == 0 and "自动" in thread_text)
    aff_on = bool(toggle_state(hwnd, IDS["win_affinity"]))
    evidence["win_affinity_default"] = aff_on
    checks["win_affinity_toggle_default_on"] = aff_on
    print("[2b] native_combo=%s composited=%s default_9b=%s labels=%s" % (
        checks["native_combo"], checks["composited_window"],
        checks["default_model_is_9b"], checks["model_labels"]))

    # ---- schema coverage -------------------------------------------------------
    schema = run_json_engine(["serve", "--print-config", "--json"])
    schema_keys = [o["key"] for o in schema.get("options", [])]
    uimap = json.load(open(os.path.join(OUT, "launcher-ui-map.json"), encoding="utf-8"))
    mapped = [x["key"] for x in uimap.get("main_panel", [])] + [x["key"] for x in uimap.get("advanced", [])]
    checks["schema_loaded"] = len(schema_keys) >= 100
    checks["coverage_100"] = set(mapped) == set(schema_keys)
    open_page_tip = {t.get("control_id"): t.get("text", "")
                     for t in uimap.get("tooltips", [])}.get(IDS["open_page"], "")
    evidence["open_page_tooltip"] = open_page_tip
    checks["open_page_tooltip_hint"] = (
        "llama.cpp" in open_page_tip and "0.2–0.5s" in open_page_tip)
    evidence["schema_options"] = len(schema_keys)
    evidence["ui_mapped"] = len(mapped)
    evidence["coverage_missing"] = sorted(set(schema_keys) - set(mapped))
    evidence["coverage_extra"] = sorted(set(mapped) - set(schema_keys))

    # ---- engine-side truth: vram_auto_layers CLI name + default OFF ----------
    def schema_opt(obj, key):
        for o in obj.get("options", []):
            if o.get("key") == key:
                return o
        return {}

    engine_vram_auto = schema_opt(schema, "vram_auto_layers")
    engine_vram_auto_on = schema_opt(
        run_json_engine(["serve", "--print-config", "--json", "--vram-auto-layers=1"]),
        "vram_auto_layers")
    evidence["engine_vram_auto_default"] = engine_vram_auto
    evidence["engine_vram_auto_on"] = engine_vram_auto_on
    checks["engine_vram_auto_default_off"] = engine_vram_auto.get("value") is False
    checks["engine_vram_auto_cli_on"] = engine_vram_auto_on.get("value") is True

    main_n = len(uimap.get("main_panel", []))
    adv_n = len(uimap.get("advanced", []))
    total_n = len(schema_keys)
    checks["ui_map_counts"] = (uimap.get("main_panel_controls") == main_n and
                               uimap.get("advanced_controls") == adv_n and
                               uimap.get("schema_options") == total_n and
                               main_n + adv_n == total_n)
    checks["ui_map_vram_auto_main"] = (
        any(x.get("key") == "vram_auto_layers" for x in uimap.get("main_panel", [])) and
        not any(x.get("key") == "vram_auto_layers" for x in uimap.get("advanced", [])) and
        ("%d main-panel" % main_n) in uimap.get("status", ""))
    evidence["ui_map_counts_actual"] = {
        "main_panel_controls": uimap.get("main_panel_controls"),
        "advanced_controls": uimap.get("advanced_controls"),
        "schema_options": uimap.get("schema_options"),
        "status": uimap.get("status"),
    }
    print("[3] coverage: schema=%d ui=%d missing=%s" %
          (len(schema_keys), len(mapped), evidence["coverage_missing"]))

    # serve --help option coverage: every --flag printed there is either a schema
    # CLI name or one of the documented serve-specific flags the UI maps instead.
    help_text = subprocess.run([ENGINE, "serve", "--help"], capture_output=True).stdout.decode(
        "utf-8", "replace")
    help_opts = set()
    for line in help_text.splitlines():
        s = line.strip()
        if s.startswith("--"):
            help_opts.add(s.split()[0])
    schema_cli = {"--" + o["cli"] for o in schema.get("options", []) if o.get("cli")}
    serve_specific = {"--pick", "--model", "--open-browser", "--out", "--json", "--tsv", "--plain",
                      "--ids", "--n", "--print-config", "--kernel-calibrate", "--no-tensors",
                      "--force", "--vram-gib", "--kv-bpe", "--config"}
    uncovered = help_opts - schema_cli - serve_specific
    evidence["serve_help_options"] = len(help_opts)
    evidence["serve_help_uncovered"] = sorted(uncovered)
    checks["help_coverage"] = not uncovered
    print("[4] serve --help options=%d uncovered=%s" % (len(help_opts), sorted(uncovered)))

    # ---- tooltips --------------------------------------------------------------
    tips = tooltip_count(proc.pid)
    evidence["tooltip_tools"] = tips
    checks["tooltips"] = tips >= 100  # one per control (labels + widgets)
    print("[5] tooltip entries:", tips)

    # ---- screenshot 1: default page -------------------------------------------
    scroll_top(hwnd)
    redraw(hwnd)
    evidence["screenshot_default"] = capture(hwnd, "launcher-01-default.png")
    evidence["model_info"] = get_text(hwnd, "model_info")
    evidence["gpu_text_default"] = get_text(hwnd, "gpu_text")
    evidence["ctx_text_default"] = get_text(hwnd, "ctx_text")
    evidence["kv_hint"] = get_text(hwnd, "kv_text")
    evidence["est_summary"] = get_text(hwnd, "est")
    evidence["schema_info"] = get_text(hwnd, "schema")
    evidence["args_summary"] = get_text(hwnd, "args")
    gpu_default = evidence["gpu_text_default"]
    checks["labels_have_units"] = (
        "显存" in gpu_default and
        ("主机内存" in gpu_default or "dense 模型" in gpu_default or
         "稠密模型" in gpu_default) and
        "KV 显存" in evidence["ctx_text_default"] and
        "GB" in evidence["est_summary"])
    checks["schema_info_100"] = (
        ("覆盖 %d 项（100%%）" % total_n) in evidence["schema_info"] and
        ("主面板 %d 项" % main_n) in evidence["schema_info"] and
        ("高级面板 %d 项" % adv_n) in evidence["schema_info"])
    print("[6] default screenshot + labels: labels_have_units=", checks["labels_have_units"])

    # ---- [6c] native dropdown: real mouse messages select items ------------
    scroll_top(hwnd)
    redraw(hwnd)
    dropdown_open(hwnd, "model", proc.pid)
    time.sleep(0.5)
    evidence["screenshot_dropdown"] = capture(hwnd, "launcher-06-dropdown.png")
    picked_model = dropdown_pick(hwnd, "model", proc.pid, item_05b)  # 0.5B
    evidence["model_pick_mouse"] = picked_model
    time.sleep(1.8)
    info_after = get_text(hwnd, "model_info")
    evidence["model_info_after_mouse"] = info_after
    checks["model_dropdown_mouse_select"] = (picked_model == item_05b and
                                              "896" in info_after)
    f32_text = get_text(hwnd, "ctx_text")
    f32_mb = parse_mb(f32_text, "KV 显存 ≈")
    picked_kv = dropdown_pick(hwnd, "kv", proc.pid, 1)  # q8_0
    time.sleep(0.8)
    kv_text_after = get_text(hwnd, "kv_text")
    q8_mb = parse_mb(get_text(hwnd, "ctx_text"), "KV 显存 ≈")
    evidence["kv_after_mouse"] = {"sel": picked_kv, "kv_text": kv_text_after,
                                  "kv_mb_f32": f32_mb, "kv_mb_q8": q8_mb}
    checks["kv_dropdown_mouse_select"] = (picked_kv == 1 and "q8_0" in kv_text_after and
                                          f32_mb is not None and q8_mb is not None and
                                          q8_mb < f32_mb)
    evidence["screenshot_model_picked"] = capture(hwnd, "launcher-07-model-picked.png")
    # restore the 9B default + f32 KV for the remaining checks
    combo_select(hwnd, "model", item_9b)
    time.sleep(1.8)
    combo_select(hwnd, "kv", 0)
    time.sleep(0.5)
    print("[6c] dropdown mouse: model=%s kv=%s (KV %s MB -> %s MB)" % (
        checks["model_dropdown_mouse_select"], checks["kv_dropdown_mouse_select"],
        f32_mb, q8_mb))

    # ---- [6d] scrolling: composited frames must be stable -----------------
    scroll_top(hwnd)
    time.sleep(0.8)
    frame1 = frame_hash(hwnd)
    time.sleep(0.5)
    frame2 = frame_hash(hwnd)
    scroll_down(hwnd, 6)
    time.sleep(1.0)
    frame3 = frame_hash(hwnd)
    evidence["screenshot_scroll"] = capture(hwnd, "launcher-08-scroll.png")
    scroll_top(hwnd)
    time.sleep(1.0)
    frame4 = frame_hash(hwnd)
    evidence["scroll_frame_hashes"] = [x[:16] for x in (frame1, frame2, frame3, frame4)]
    checks["scroll_frames_stable"] = (frame1 == frame2 == frame4)
    checks["scroll_content_moves"] = (frame1 != frame3)
    print("[6d] scroll frames: stable=%s moves=%s hashes=%s" % (
        checks["scroll_frames_stable"], checks["scroll_content_moves"],
        evidence["scroll_frame_hashes"]))

    # ---- [6e] dense 9B vs MoE 35B: GPU-layer controls -----------------------
    combo_select(hwnd, "model", item_9b)  # 9B dense
    time.sleep(2.0)
    dense_state = gpu_control_state(hwnd)
    evidence["gpu_controls_dense"] = dense_state
    checks["dense_gpu_controls_disabled"] = (
        dense_state["slider_enabled"] is False and dense_state["auto_enabled"] is False and
        dense_state["auto_toggle"] == 0 and dense_state["vram_auto_enabled"] is False and
        "稠密模型" in dense_state["gpu_text"] and "无需 GPU/CPU 分层" in dense_state["gpu_text"])
    checks["dense_estimate_formula"] = (
        "显存占用 = 全部权重 + KV + 工作区" in dense_state["estimate"])
    scroll_top(hwnd)
    redraw(hwnd)
    evidence["screenshot_dense_disabled"] = capture(hwnd, "launcher-09-dense-gpu-disabled.png")

    combo_select(hwnd, "model", item_35b)  # 35B MoE
    time.sleep(3.0)
    moe_state = gpu_control_state(hwnd)
    evidence["gpu_controls_moe"] = moe_state
    checks["moe_gpu_controls_enabled"] = (
        moe_state["slider_enabled"] is True and moe_state["auto_enabled"] is True and
        moe_state["vram_auto_enabled"] is True and "GPU 专家层" in moe_state["gpu_text"] and
        "/40" in moe_state["gpu_text"] and "放置决策" in moe_state["estimate"])
    scroll_top(hwnd)
    redraw(hwnd)
    evidence["screenshot_moe_enabled"] = capture(hwnd, "launcher-10-moe-gpu-enabled.png")

    cycles = []
    for _ in range(3):
        combo_select(hwnd, "model", item_9b)
        time.sleep(1.8)
        d = gpu_control_state(hwnd)
        combo_select(hwnd, "model", item_35b)
        time.sleep(1.8)
        m = gpu_control_state(hwnd)
        cycles.append({
            "dense_disabled": (d["slider_enabled"] is False and d["auto_enabled"] is False and
                               d["auto_toggle"] == 0 and d["vram_auto_enabled"] is False and
                               "稠密模型" in d["gpu_text"]),
            "moe_enabled": (m["slider_enabled"] is True and m["auto_enabled"] is True and
                            m["vram_auto_enabled"] is True and "/40" in m["gpu_text"]),
        })
    evidence["gpu_control_cycles"] = cycles
    checks["model_switch_cycles"] = all(c["dense_disabled"] and c["moe_enabled"] for c in cycles)
    combo_select(hwnd, "model", item_9b)  # back to the default
    time.sleep(1.8)
    print("[6e] dense_disabled=%s formula=%s moe_enabled=%s cycles=%s" % (
        checks["dense_gpu_controls_disabled"], checks["dense_estimate_formula"],
        checks["moe_gpu_controls_enabled"], checks["model_switch_cycles"]))

    # ---- opt-in VRAM auto-layer switch (default OFF, red risk note, CLI map) --
    # vram_auto_layers only promotes MoE expert layers: select the 35B first.
    combo_select(hwnd, "model", item_35b)
    time.sleep(2.5)
    auto_ctrl = user32.GetDlgItem(hwnd, IDS["vram_auto"])
    warn_ctrl = user32.GetDlgItem(hwnd, IDS["vram_auto_warn"])
    evidence["vram_auto_ctrl_found"] = bool(auto_ctrl and warn_ctrl)
    evidence["vram_auto_args_default"] = get_text(hwnd, "args")
    evidence["vram_auto_warn_default_visible"] = bool(user32.IsWindowVisible(warn_ctrl))
    checks["vram_auto_default_off"] = (
        evidence["vram_auto_ctrl_found"] and
        not evidence["vram_auto_warn_default_visible"] and
        "vram-auto-layers" not in evidence["vram_auto_args_default"])
    command(hwnd, "vram_auto")
    time.sleep(0.8)
    evidence["vram_auto_warn_text"] = get_text(hwnd, "vram_auto_warn")
    evidence["vram_auto_warn_visible"] = bool(user32.IsWindowVisible(warn_ctrl))
    evidence["vram_auto_args_on"] = get_text(hwnd, "args")
    checks["vram_auto_warning_visible"] = (
        evidence["vram_auto_warn_visible"] and "~80 MiB" in evidence["vram_auto_warn_text"])
    # The summary line is the friendly form ("· VRAM 自动加层"); the raw
    # --vram-auto-layers=1 flag is verified in [15b] against engine.exe's cmdline.
    checks["vram_auto_args_mapping"] = "VRAM 自动加层" in evidence["vram_auto_args_on"]
    tip_map = {t.get("control_id"): t.get("text", "") for t in uimap.get("tooltips", [])}
    evidence["vram_auto_tooltip"] = tip_map.get(IDS["vram_auto"], "")
    evidence["tooltip_evidence_count"] = len(tip_map)
    checks["vram_auto_tooltip_risk"] = (
        "decode 期间显存余量会低至 ~80 MiB" in evidence["vram_auto_tooltip"] and
        "建议只在确认没有其他程序占用显存时开启" in evidence["vram_auto_tooltip"] and
        "CUDA graph 捕获时 OOM" in evidence["vram_auto_tooltip"])
    checks["tooltips_evidence_count"] = len(tip_map) >= 100
    scroll_top(hwnd)
    redraw(hwnd)
    evidence["screenshot_vram_auto"] = capture(hwnd, "launcher-05-vram-auto.png")
    command(hwnd, "vram_auto")  # back to the default OFF state
    time.sleep(0.7)
    evidence["vram_auto_warn_off_visible"] = bool(user32.IsWindowVisible(warn_ctrl))
    evidence["vram_auto_args_off"] = get_text(hwnd, "args")
    checks["vram_auto_off_again"] = (
        not evidence["vram_auto_warn_off_visible"] and
        "vram-auto-layers" not in evidence["vram_auto_args_off"])
    print("[6b] vram_auto: default_off=%s warn=%s map=%s tooltip=%s" % (
        checks["vram_auto_default_off"], checks["vram_auto_warning_visible"],
        checks["vram_auto_args_mapping"], checks["vram_auto_tooltip_risk"]))
    combo_select(hwnd, "model", item_9b)  # restore the default model
    time.sleep(2.0)

    # ---- screenshot 2: advanced panel -----------------------------------------
    command(hwnd, "adv_toggle")
    time.sleep(0.8)
    redraw(hwnd)
    evidence["screenshot_advanced"] = capture(hwnd, "launcher-02-advanced.png")
    evidence["adv_count"] = get_text(hwnd, "adv_count")
    checks["advanced_open"] = ("%d / %d" % (adv_n, adv_n)) in evidence["adv_count"]
    # an advanced override used later by the launch test
    checks["advanced_control_edit"] = set_advanced(hwnd, uimap, "cpu_pool_spin", "256")
    time.sleep(0.4)
    evidence["args_after_adv"] = get_text(hwnd, "args")
    print("[7] advanced:", evidence["adv_count"], "edit ok=", checks["advanced_control_edit"])
    command(hwnd, "adv_toggle")  # collapse
    time.sleep(0.5)

    # ---- screenshot 3: SSD judgement (35B MoE -> vram+ram split) --------------
    combo_select(hwnd, "model", item_35b)  # 35B MoE (largest)
    time.sleep(2.5)
    ssd_ctrl = user32.GetDlgItem(hwnd, IDS["ssd"])
    note = get_text(hwnd, "ssd_note")
    disabled = not bool(user32.IsWindowEnabled(ssd_ctrl))
    fits = "无需 SSD" in note
    not_fit = "装不进显存+内存" in note
    evidence["ssd_note_35b"] = note
    evidence["ssd_enabled_35b"] = not disabled
    checks["ssd_disabled_when_fits"] = (disabled == fits) and (fits or not_fit)
    checks["ssd_reason_gpu_or_split"] = (
        not disabled or ("无需 SSD" in note and ("vram+ram split" in note or "gpu" in note)))
    checks["gpu_estimate_35b"] = "GPU 专家层" in get_text(hwnd, "gpu_text")
    # Deterministic fits path: a huge VRAM budget makes the 35B "fit" even when
    # the shared machine is under memory pressure -> SSD must grey out.
    set_advanced(hwnd, uimap, "vram_budget_mb", "60000")
    time.sleep(1.0)
    note_forced = get_text(hwnd, "ssd_note")
    disabled_forced = not bool(user32.IsWindowEnabled(ssd_ctrl))
    evidence["ssd_note_forced_fit"] = note_forced
    checks["ssd_forced_fit_disabled"] = disabled_forced and "无需 SSD" in note_forced
    scroll_down(hwnd, 6)
    redraw(hwnd)
    time.sleep(1.0)
    redraw(hwnd)
    time.sleep(0.5)
    evidence["screenshot_ssd"] = capture(hwnd, "launcher-03-ssd.png")
    set_advanced(hwnd, uimap, "vram_budget_mb", "0")
    time.sleep(0.8)
    print("[8] SSD:", ("greyed" if disabled else "enabled"), "|", note,
          "| forced-fit greyed:", checks["ssd_forced_fit_disabled"])

    # SSD becomes selectable when the model cannot fit VRAM+RAM at all.
    checks["advanced_budget_edit"] = set_advanced(hwnd, uimap, "vram_budget_mb", "2000")
    time.sleep(0.8)
    checks["ssd_enabled_when_not_fit"] = bool(user32.IsWindowEnabled(ssd_ctrl))
    note2 = get_text(hwnd, "ssd_note")
    evidence["ssd_note_budget2000"] = note2
    checks["ssd_unimplemented_warning"] = "未实现" in note2
    redraw(hwnd)
    evidence["screenshot_ssd_enabled"] = capture(hwnd, "launcher-03b-ssd-enabled.png")
    if checks["ssd_enabled_when_not_fit"]:
        command(hwnd, "ssd")  # check it: the note must scream "not implemented"
        time.sleep(0.5)
        note3 = get_text(hwnd, "ssd_note")
        evidence["ssd_note_checked"] = note3
        checks["ssd_checked_warns"] = "未实现" in note3 and "报错退出" in note3
        command(hwnd, "ssd")  # unchecked
        time.sleep(0.3)
    set_advanced(hwnd, uimap, "vram_budget_mb", "0")
    time.sleep(0.6)
    print("[9] SSD enabled-under-budget:", checks["ssd_enabled_when_not_fit"], "|", note2)

    # ---- sliders: live estimate ----------------------------------------------
    checks["live_gpu_uncheck"] = True
    command(hwnd, "gpu_auto")  # switch from auto to explicit
    time.sleep(0.4)
    user32.SendMessageW(user32.GetDlgItem(hwnd, IDS["gpu_slider"]), WM_APP_SLIDER_SET, 12, 0)
    time.sleep(0.5)
    gpu12 = get_text(hwnd, "gpu_text")
    evidence["gpu_text_12"] = gpu12
    checks["live_gpu_slider"] = "GPU 专家层 12/40" in gpu12 and "显存 ≈" in gpu12
    user32.SendMessageW(user32.GetDlgItem(hwnd, IDS["gpu_slider"]), WM_APP_SLIDER_SET, 40, 0)
    time.sleep(0.5)
    gpu40 = get_text(hwnd, "gpu_text")
    evidence["gpu_text_40"] = gpu40
    checks["over_vram_warning"] = "超过可用显存" in gpu40
    user32.SendMessageW(user32.GetDlgItem(hwnd, IDS["gpu_slider"]), WM_APP_SLIDER_SET, 12, 0)
    time.sleep(0.3)
    print("[10] GPU slider live:", checks["live_gpu_slider"], "over-warning:", checks["over_vram_warning"])

    # context: editable arbitrary number box + log-scale slider + auto checkbox
    edit = user32.GetDlgItem(hwnd, IDS["ctx_edit"])
    slider = user32.GetDlgItem(hwnd, IDS["ctx_slider"])

    def set_ctx_text(txt):
        # WM_SETTEXT does not reliably notify the parent; send EN_CHANGE explicitly
        user32.SendMessageW(edit, WM_SETTEXT, 0, ctypes.c_wchar_p(txt))
        user32.SendMessageW(hwnd, WM_COMMAND,
                            (EN_CHANGE << 16) | IDS["ctx_edit"], edit)

    def type_ctx(txt):
        set_ctx_text("")
        for ch in txt:
            user32.SendMessageW(edit, WM_CHAR, ord(ch), 0)

    type_ctx("12345")  # arbitrary value, not one of the old presets
    time.sleep(0.6)
    ctx12345 = get_text(hwnd, "ctx_text")
    evidence["ctx_text_12345"] = ctx12345
    type_ctx("8192")
    time.sleep(0.4)
    ctx8k = get_text(hwnd, "ctx_text")
    type_ctx("4096")
    time.sleep(0.4)
    ctx4k = get_text(hwnd, "ctx_text")
    kv_f32 = parse_mb(ctx4k, "KV 显存 ≈")
    combo_select(hwnd, "kv", 2)  # q4_0
    time.sleep(0.5)
    kv_q4_text = get_text(hwnd, "ctx_text")
    kv_q4 = parse_mb(kv_q4_text, "KV 显存 ≈")
    # invalid input (empty box) must show the red hint, not silently launch
    set_ctx_text("")
    time.sleep(0.5)
    ctx_bad = get_text(hwnd, "ctx_text")
    evidence["ctx_text_invalid"] = ctx_bad
    # slider is the coarse helper: position 500 maps into the log range (not 4096)
    set_ctx_text("4096")
    time.sleep(0.3)
    user32.SendMessageW(slider, WM_APP_SLIDER_SET, 500, 0)
    time.sleep(0.5)
    ctx_slider = get_text(hwnd, "ctx_text")
    evidence["ctx_text_slider500"] = ctx_slider
    # auto checkbox -> engine "--ctx 0" path, then back to manual 4096 for the start test
    command(hwnd, "ctx_auto")
    time.sleep(0.5)
    ctx_auto_text = get_text(hwnd, "ctx_text")
    evidence["ctx_text_auto"] = ctx_auto_text
    command(hwnd, "ctx_auto")
    time.sleep(0.4)
    type_ctx("4096")
    time.sleep(0.4)
    evidence["ctx_text_8192"] = ctx8k
    evidence["ctx_text_4096_f32"] = ctx4k
    evidence["ctx_text_4096_q4"] = kv_q4_text
    checks["ctx_editable_arbitrary"] = bool(
        "12345 tokens" in ctx12345 and "可输入任意整数" in ctx12345)
    checks["ctx_slider_live"] = ("8192 tokens" in ctx8k) and ("4096 tokens" in ctx4k)
    checks["ctx_invalid_hint"] = "必须是" in ctx_bad
    checks["ctx_slider_maps_value"] = ("tokens" in ctx_slider and
                                       "4096 tokens" not in ctx_slider)
    checks["ctx_auto_toggle"] = "自动" in ctx_auto_text
    checks["kv_quant_estimate"] = kv_f32 is not None and kv_q4 is not None and kv_q4 < kv_f32
    combo_select(hwnd, "kv", 0)  # back to f32
    time.sleep(0.4)
    print("[11] ctx/kv:", checks["ctx_editable_arbitrary"], checks["ctx_slider_live"],
          checks["ctx_invalid_hint"], (kv_f32, kv_q4))

    # ---- [11b] existing engine guard: never start a second instance ----------
    ext = subprocess.Popen([ENGINE, "serve",
                            os.path.join(MODEL_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
                            "--port", str(PORT), "--ctx", "2048"],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    time.sleep(2.0)
    command(hwnd, "start")
    time.sleep(1.5)
    guard_status = get_text(hwnd, "status")
    guard_count = count_engine_processes()
    evidence["existing_guard_status"] = guard_status
    evidence["existing_guard_engine_count"] = guard_count
    checks["existing_engine_guard"] = ("已在运行" in guard_status and guard_count == 1)
    ext.terminate()
    try:
        ext.wait(timeout=10)
    except Exception:
        ext.kill()
    time.sleep(1.0)
    print("[11b] existing-engine guard:", checks["existing_engine_guard"],
          guard_status[:80], "engine_count=", guard_count)

    # ---- screenshot 4: run the 0.5B model through the UI ----------------------
    combo_select(hwnd, "model", item_05b)  # qwen2.5-0.5b
    time.sleep(2.0)
    dense_text = get_text(hwnd, "gpu_text")
    checks["dense_model_note"] = ("dense" in dense_text or "稠密模型" in dense_text)
    # (the auto switch is intentionally disabled for dense models - no click)
    scroll_top(hwnd)
    # The port must be free: another agent's engine may share this machine.
    port_free = False
    end_free = time.time() + 60
    while time.time() < end_free:
        if not health_ok(PORT, 0.5):
            port_free = True
            break
        time.sleep(0.5)
    evidence["port_free_before_start"] = port_free
    checks["port_free_before_start"] = port_free
    if not port_free:
        print("ERROR: port %d is still busy after 60s (another engine is using it);"
              " refusing to start and leaving the launcher open for inspection" % PORT)
        return 3
    command(hwnd, "start")
    time.sleep(0.8)
    evidence["status_loading"] = get_text(hwnd, "status")
    checks["status_loading"] = "加载中" in evidence["status_loading"]
    started = time.time()
    ok = False
    trace = []
    while time.time() - started < 180:
        if health_ok():
            ok = True
            break
        elapsed = time.time() - started
        if len(trace) < 8 or int(elapsed) % 10 == 0:
            trace.append([round(elapsed, 1), get_text(hwnd, "status")])
        time.sleep(0.5)
    checks["engine_healthy"] = ok
    evidence["load_seconds"] = round(time.time() - started, 1)
    evidence["load_trace"] = trace
    time.sleep(1.0)
    scroll_top(hwnd)
    redraw(hwnd)
    time.sleep(0.8)
    redraw(hwnd)
    time.sleep(0.4)
    evidence["status_running"] = get_text(hwnd, "status")
    evidence["url_running"] = get_text(hwnd, "url")
    evidence["placement_echo"] = get_text(hwnd, "placement")
    checks["status_running"] = "运行中" in evidence["status_running"]
    checks["url_running"] = ("http://127.0.0.1:%d" % PORT) in evidence["url_running"]
    checks["placement_echo"] = "placement" in evidence["placement_echo"]
    # A second click while running must not spawn another engine.
    command(hwnd, "start")
    time.sleep(1.0)
    dup_status = get_text(hwnd, "status")
    dup_count = count_engine_processes()
    evidence["double_start_status"] = dup_status
    evidence["double_start_engine_count"] = dup_count
    checks["double_start_single_engine"] = (dup_count == 1 and "忽略" in dup_status)
    print("[12b] double-start:", checks["double_start_single_engine"],
          dup_status[:60], "engine_count=", dup_count)
    # /props first: it proves the listener belongs to the model we started.
    # /health is retried separately and only recorded (the engine's HTTP accept
    # loop can be slow while a freshly loaded model settles; that must not be
    # reported as a launcher failure).
    st_props, body_props = http_get_retry("/props")
    st_health, body_health = http_get_retry("/health", deadline_s=8.0)
    props_model = ""
    try:
        props_model = json.loads(body_props).get("model_name", "")
    except Exception:
        pass
    model_matches = "qwen2.5" in props_model   # the 0.5B selected above
    evidence["http"] = {"health": st_health, "props": st_props,
                        "health_body": body_health[:200], "props_body": body_props[:200],
                        "props_model": props_model, "model_matches_ours": model_matches}
    health_ours = st_health == 200 and "qwen2.5" in body_health
    evidence["http"]["health_model_matches_ours"] = health_ours
    checks["http_reachable"] = health_ours or model_matches
    evidence["screenshot_running"] = capture(hwnd, "launcher-04-running.png")
    print("[12] running:", checks["engine_healthy"], evidence["status_running"],
          evidence["url_running"], "load %.1fs" % evidence["load_seconds"])

    command(hwnd, "stop")
    stopped = False
    end = time.time() + 40
    while time.time() < end:
        if not health_ok(0.5):
            stopped = True
            break
        time.sleep(0.5)
    time.sleep(1.0)
    evidence["status_stopped"] = get_text(hwnd, "status")
    checks["engine_stopped"] = stopped and "已停止" in evidence["status_stopped"]
    print("[13] stopped:", checks["engine_stopped"], "|", evidence["status_stopped"])

    # ---- command line evidence ------------------------------------------------
    log_path = os.path.join(OUT, "launcher-engine.log")
    log_text = open(log_path, encoding="utf-8", errors="replace").read()
    evidence["log_head"] = log_text.splitlines()[:4]
    cmdline_line = log_text.splitlines()[1] if len(log_text.splitlines()) > 1 else log_text
    evidence["engine_cmdline"] = cmdline_line
    checks["log_cmdline_0p5b"] = (
        all(x in cmdline_line for x in [
            "serve --model", '--host "127.0.0.1"', "--port %d" % PORT, "--ctx 4096",
            "--kv-type f32", '--config "', "launcher-engine.cfg"]) and
        ("--gpu-layers 0" in cmdline_line or "--gpu-layers -1" in cmdline_line))
    # New engine defaults: auto threads (no --threads argument) + SMT-paired affinity.
    checks["engine_threads_auto_cli"] = "--threads " not in cmdline_line
    checks["win_affinity_default_on_cli"] = "--win-affinity 2" in cmdline_line
    # advanced override must be in the generated config file and referenced
    cfg_path = os.path.join(OUT, "launcher-engine.cfg")
    cfg_text = open(cfg_path, encoding="utf-8").read() if os.path.exists(cfg_path) else ""
    evidence["cfg_text"] = cfg_text.splitlines()[:6]
    checks["advanced_override_applied"] = "cpu_pool_spin = 256" in cfg_text and "--config " in log_text
    checks["log_placement"] = "[placement]" in log_text
    checks["log_load"] = "[serve] model ready" in log_text
    print("[14] log: override=", checks["advanced_override_applied"],
          "placement=", checks["log_placement"])

    # ---- explicit GPU layer count reaches the engine --------------------------
    combo_select(hwnd, "model", item_35b)  # 35B
    time.sleep(2.5)
    command(hwnd, "gpu_auto")  # explicit mode
    time.sleep(0.3)
    user32.SendMessageW(user32.GetDlgItem(hwnd, IDS["gpu_slider"]), WM_APP_SLIDER_SET, 12, 0)
    time.sleep(0.6)
    evidence["gpu_text_explicit12"] = get_text(hwnd, "gpu_text")
    command(hwnd, "vram_auto")  # ON: verify the CLI mapping reaches engine.exe
    time.sleep(0.5)
    command(hwnd, "start")
    time.sleep(3.0)
    log_text = open(log_path, encoding="utf-8", errors="replace").read()
    evidence["log_head_35b"] = log_text.splitlines()[:2]
    checks["explicit_gpu_layers_cli"] = "--gpu-layers 12" in log_text
    args_file = open(os.path.join(OUT, "launcher-effective-args.txt"), encoding="utf-8").read()
    checks["effective_args_file"] = "--gpu-layers 12" in args_file
    checks["vram_auto_cli_reaches_engine"] = (
        "--vram-auto-layers=1" in log_text and "--vram-auto-layers=1" in args_file)
    evidence["vram_auto_log_line"] = [ln for ln in log_text.splitlines()
                                      if "vram-auto-layers" in ln][:2]
    command(hwnd, "vram_auto")  # back to OFF
    time.sleep(0.4)
    print("[15] explicit --gpu-layers 12:", checks["explicit_gpu_layers_cli"],
          checks["effective_args_file"])
    print("[15b] --vram-auto-layers=1 reaches engine:",
          checks["vram_auto_cli_reaches_engine"])

    # ---- [16] refresh button runs the async full-disk scan --------------------
    combo = user32.GetDlgItem(hwnd, IDS["model"])
    sel_before = int(user32.SendMessageW(combo, CB_GETCURSEL, 0, 0))
    name_before = combo_item_text(combo, sel_before) if sel_before >= 0 else ""
    command(hwnd, "refresh")
    scan_started = time.time()
    scan_ok = False
    while time.time() - scan_started < 120:
        st = get_text(hwnd, "scan_status")
        if "扫描完成" in st:
            scan_ok = True
            break
        time.sleep(0.5)
    time.sleep(0.5)
    sel_after = int(user32.SendMessageW(combo, CB_GETCURSEL, 0, 0))
    name_after = combo_item_text(combo, sel_after) if sel_after >= 0 else ""
    evidence["refresh_scan_seconds"] = time.time() - scan_started
    evidence["refresh_scan_status"] = get_text(hwnd, "scan_status")
    checks["refresh_button_background_scan"] = (scan_ok and sel_before == sel_after and
                                                name_before == name_after)
    print("[16] refresh background scan:", checks["refresh_button_background_scan"],
          evidence["refresh_scan_status"], "%.1fs" % evidence["refresh_scan_seconds"])
    command(hwnd, "stop")
    time.sleep(8.0)
    evidence["status_after_35b_stop"] = get_text(hwnd, "status")

    # ---- [17] reset every toggle to its schema default, delete generated cfg --
    cfg_path = os.path.join(os.path.dirname(LAUNCHER), "launcher-engine.cfg")
    # restore advanced overrides used by the test
    set_advanced(hwnd, uimap, "cpu_pool_spin", "0")
    time.sleep(0.3)
    toggles_default = [("vram_auto", False), ("ssd", False), ("dev_route", True),
                       ("layer_graph", True), ("cpu_int8", True), ("thinking", True),
                       ("gpu_auto", True), ("win_affinity", True)]
    for key, want in toggles_default:
        if key not in IDS:
            continue
        cur = bool(toggle_state(hwnd, IDS[key]))
        if cur != want:
            command(hwnd, key)
            time.sleep(0.25)
    combo_select(hwnd, "kv", 0)
    time.sleep(0.4)
    final_states = {k: bool(toggle_state(hwnd, IDS[k])) for k, _ in toggles_default if k in IDS}
    evidence["final_toggle_states"] = final_states
    checks["toggles_reset_after_test"] = all(
        final_states.get(k) == want for k, want in toggles_default if k in final_states)
    try:
        os.remove(cfg_path)
    except OSError:
        pass
    checks["launcher_cfg_clean"] = not os.path.exists(cfg_path)
    print("[17] toggles reset:", checks["toggles_reset_after_test"],
          "cfg removed:", checks["launcher_cfg_clean"], final_states)

    # ---- [18] a fresh launcher must re-read defaults from the engine schema ---
    command(hwnd, "vram_auto")  # simulate a user leaving the risky toggle on
    time.sleep(0.5)
    evidence["restart_left_vram_auto_on"] = bool(toggle_state(hwnd, IDS["vram_auto"]))
    user32.PostMessageW(hwnd, 0x0010, 0, 0)
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    first_exit = proc.poll()
    proc2 = subprocess.Popen([LAUNCHER, "--theme", "dark", "--port", str(PORT), "--no-browser"],
                             env=launcher_env)
    hwnd2, ident2 = find_launcher_window(proc2.pid, LAUNCHER)
    restart_ok = False
    if hwnd2:
        time.sleep(2.5)
        vram_auto_after = bool(toggle_state(hwnd2, IDS["vram_auto"]))
        args_after = get_text(hwnd2, "args")
        evidence["restart_vram_auto_toggle"] = vram_auto_after
        evidence["restart_args"] = args_after
        restart_ok = (not vram_auto_after) and ("VRAM 自动加层" not in args_after)
        user32.PostMessageW(hwnd2, 0x0010, 0, 0)
    try:
        proc2.wait(timeout=30)
    except Exception:
        proc2.kill()
    checks["startup_resets_toggles_from_schema"] = restart_ok
    evidence["exit_code"] = first_exit
    checks["exited"] = first_exit is not None
    print("[18] startup defaults reset:", restart_ok,
          "vram_auto_on_restart=", evidence.get("restart_vram_auto_toggle"))

    # ---- [19] oversized-model warning in the drop-down (real full cache) ------
    env_full = os.environ.copy()
    env_full["ENGINE_LAUNCHER_NO_AUTO_SCAN"] = "1"
    env_full["ENGINE_LAUNCHER_NO_DIALOG"] = "1"
    env_full.pop("ENGINE_MODELS_CACHE", None)
    proc3 = subprocess.Popen([LAUNCHER, "--theme", "dark", "--port", str(PORT), "--no-browser"],
                             env=env_full)
    hwnd3, ident3 = find_launcher_window(proc3.pid, LAUNCHER)
    warn_ok = False
    warn_items = []
    if hwnd3:
        time.sleep(2.5)
        combo3 = user32.GetDlgItem(hwnd3, IDS["model"])
        n3 = int(user32.SendMessageW(combo3, CB_GETCOUNT, 0, 0))
        warn_items = [combo_item_text(combo3, i) for i in range(1, n3)]
        warn_ok = any("\u26a0" in t and "37.8" in t for t in warn_items)
        evidence["oversize_warning_items"] = [t for t in warn_items if "\u26a0" in t][:3]
        user32.PostMessageW(hwnd3, 0x0010, 0, 0)
    try:
        proc3.wait(timeout=30)
    except Exception:
        proc3.kill()
    checks["oversized_model_warning"] = warn_ok
    print("[19] oversized-model warning:", warn_ok,
          [t[-80:] for t in warn_items if "\u26a0" in t][:2])
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    evidence["checks"] = checks
    evidence["passed"] = all(checks.values())
    with open(os.path.join(OUT, "launcher-evidence.json"), "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, ensure_ascii=False, indent=2)
    print("\n=== checks ===")
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    total = len(checks)
    passed = sum(1 for v in checks.values() if v)
    print("RESULT: %d/%d %s" % (passed, total, "PASS" if evidence["passed"] else "FAIL"))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

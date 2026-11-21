import re

MACROS = r'''
// ---------------------------------------------------------------------------
// Cached environment lookups for the decode hot path.
// std::getenv() measures 2.13 us/call on this host (the CRT scans the whole
// environment block), and run_layer+run_moe+the qwen35 layer path evaluate ~30 of them
// per layer, i.e. >2 ms/token of pure getenv.  The process environment cannot change
// after start-up, so each call site caches its own answer once.  The returned values are
// identical to the uncached ones, so behaviour stays bit-exact.
#define ENGINE_ENV_ON(name) \
  ([]() -> bool { static const bool v = std::getenv(name) != nullptr; return v; }())
#define ENGINE_ENV_STR(name) \
  ([]() -> const char* { static const char* const v = std::getenv(name); return v; }())
// ---------------------------------------------------------------------------

'''

def patch_file(path, start_marker_names, insert_before, min_line=1, skip_ranges=()):
    s = open(path, encoding='utf-8').read()
    lines = s.split('\n')
    out = []
    changed = 0
    for i, ln in enumerate(lines):
        n = i + 1
        if n < min_line or any(a <= n <= b for a, b in skip_ranges):
            out.append(ln)
            continue
        o = ln
        # bool-ish forms
        ln = re.sub(r'std::getenv\("([A-Z0-9_]+)"\) != nullptr', r'ENGINE_ENV_ON("\1")', ln)
        ln = re.sub(r'std::getenv\("([A-Z0-9_]+)"\) == nullptr', r'!ENGINE_ENV_ON("\1")', ln)
        ln = re.sub(r'!std::getenv\("([A-Z0-9_]+)"\)', r'!ENGINE_ENV_ON("\1")', ln)
        # pointer-capture forms
        ln = re.sub(r'const char\* (\w+) = std::getenv\("([A-Z0-9_]+)"\)',
                    r'const char* \1 = ENGINE_ENV_STR("\2")', ln)
        # bare truthiness (only where it is a complete condition or in && / || chains)
        ln = re.sub(r'if \(std::getenv\("([A-Z0-9_]+)"\)\)', r'if (ENGINE_ENV_ON("\1"))', ln)
        ln = re.sub(r'if \(std::getenv\("([A-Z0-9_]+)"\) &&', r'if (ENGINE_ENV_ON("\1") &&', ln)
        ln = re.sub(r'&& std::getenv\("([A-Z0-9_]+)"\) == nullptr', r'&& !ENGINE_ENV_ON("\1")', ln)
        ln = re.sub(r'std::getenv\("([A-Z0-9_]+)"\) != nullptr \|\|', r'ENGINE_ENV_ON("\1") ||', ln)
        if ln != o:
            changed += 1
        out.append(ln)
    s2 = '\n'.join(out)
    assert insert_before in s2, insert_before
    s2 = s2.replace(insert_before, MACROS + insert_before, 1) if insert_before else s2
    open(path, 'w', encoding='utf-8', newline='').write(s2)
    return changed

# runner.cpp: everything from run_layer onwards; keep the load-time and module-build
# getenvs and the prof_acc init block untouched.
n1 = patch_file('src/runner.cpp', None, 'namespace runner {\n',
                min_line=1450, skip_ranges=[(1466, 1500)])
print('runner.cpp lines changed:', n1)

# qwen35.cpp: hot per-layer sites (attention/ffn dump lambdas + graph debug flag).
n2 = patch_file('src/qwen35.cpp', None, '', min_line=270)
print('qwen35.cpp lines changed:', n2)

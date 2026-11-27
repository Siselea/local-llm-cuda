// Numeric probe: is the batched (verify-shaped) forward numerically identical to the
// single-token (decode-shaped) forward on the SAME prefix state?
//
// Speculative decoding (MTP) is lossless only if the target model's arg-max computed by
// the (k+1)-row verify batch equals the arg-max of the single-token greedy path.  This
// tool measures that difference directly, plus the recurrent-state drift that a
// batch verify leaves behind, without touching the engine:
//
//   A) prefill(prompt) -> forward({p}, prompt_len)            : decode-shaped logits
//   B) prefill(prompt) -> forward({p,x,y}, prompt_len)         : batch-shaped row-0 logits
//   C) prefill -> fwd({a}) ; fwd({b}) ; fwd({c})              : sequential state, then logits
//   D) prefill -> fwd({a,b}) ; fwd({c})                       : batch state, then logits
//
// A vs B isolates the "verify batch vs decode" numeric gap at the decision point.
// C vs D isolates the recurrent-state drift a batched advance leaves in conv/rec state.
//
// Usage:
//   mtp_numprobe <model.gguf> [--ids "760,6511,314,9338,369"] [--ctx 512]
//
// Knobs (env) worth sweeping: ENGINE_PREFILL_F16=0, ENGINE_NO_GEMV=1,
// ENGINE_NO_DP4A_GEMV=1, ENGINE_KV_TYPE / --kv-type q8_0.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/config.h"
#include "engine/runner.h"

namespace {

std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> ids;
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    ids.push_back(std::atoi(s.substr(p, q - p).c_str()));
    p = q + 1;
  }
  return ids;
}

runner::Container detect_container(const std::string& path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) return runner::Container::Safetensors;
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f) {
    char magic[4] = {0, 0, 0, 0};
    const size_t got = std::fread(magic, 1, 4, f);
    std::fclose(f);
    if (got == 4 && std::memcmp(magic, "GGUF", 4) == 0) return runner::Container::Gguf;
  }
  return runner::Container::Gguf;
}

int argmax_of(const float* row, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (row[i] > row[best]) best = i;
  return best;
}

int first_diff_ids(const std::vector<int>& a, const std::vector<int>& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i]) return static_cast<int>(i);
  return a.size() == b.size() ? -1 : static_cast<int>(n);
}

void print_stream_diff(const char* label, const std::vector<int>& a, const std::vector<int>& b) {
  const int d = first_diff_ids(a, b);
  if (d < 0) {
    std::printf("%-40s IDENTICAL (%zu tokens)\n", label, a.size());
    return;
  }
  std::printf("%-40s DIFFERS at %d (a=%d b=%d, len %zu/%zu)\n", label, d,
              d < static_cast<int>(a.size()) ? a[static_cast<size_t>(d)] : -1,
              d < static_cast<int>(b.size()) ? b[static_cast<size_t>(d)] : -1, a.size(),
              b.size());
}

void top2(const float* row, int n, int* i1, int* i2, float* v1, float* v2) {
  *i1 = *i2 = 0;
  *v1 = *v2 = -1e30f;
  for (int i = 0; i < n; ++i) {
    const float v = row[i];
    if (v > *v1) {
      *v2 = *v1; *i2 = *i1;
      *v1 = v; *i1 = i;
    } else if (v > *v2) {
      *v2 = v; *i2 = i;
    }
  }
}

struct Diff {
  double max_abs = 0, mean_abs = 0, rel_l2 = 0;
  int argmax_a = -1, argmax_b = -1;
  float gap_a = 0, gap_b = 0;
};

Diff compare(const float* a, const float* b, int n) {
  Diff d;
  double ssd = 0, ssa = 0, sa = 0;
  for (int i = 0; i < n; ++i) {
    const double e = static_cast<double>(a[i]) - b[i];
    d.max_abs = std::max(d.max_abs, std::fabs(e));
    sa += std::fabs(e);
    ssd += e * e;
    ssa += static_cast<double>(a[i]) * a[i];
  }
  d.mean_abs = sa / n;
  d.rel_l2 = ssa > 0 ? std::sqrt(ssd / ssa) : std::sqrt(ssd);
  int a1, a2, b1, b2;
  float av1, av2, bv1, bv2;
  top2(a, n, &a1, &a2, &av1, &av2);
  top2(b, n, &b1, &b2, &bv1, &bv2);
  d.argmax_a = a1;
  d.argmax_b = b1;
  d.gap_a = av1 - av2;
  d.gap_b = bv1 - bv2;
  return d;
}

void print_diff(const char* label, const Diff& d, int vocab) {
  std::printf("%-34s argmax %6d vs %6d  %s | max|d|=%.6g mean|d|=%.3g rel_l2=%.3g "
              "| gap %.4g vs %.4g (top2 %6d)\n",
              label, d.argmax_a, d.argmax_b,
              d.argmax_a == d.argmax_b ? "SAME" : "DIFF", d.max_abs, d.mean_abs, d.rel_l2,
              d.gap_a, d.gap_b, 0);
  (void)vocab;
}

}  // namespace

int main(int argc, char** argv) {
  engine::config::Config& cfg = engine::config::Config::get();
  std::string ids_arg = "760,6511,314,9338,369";
  int ctx = 512;
  int n_gen = 180;
  std::vector<std::string> filtered;
  filtered.push_back(argv[0]);
  try {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto need = [&](const char* what) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + what);
        return argv[++i];
      };
      if (a == "--ids") ids_arg = need("--ids");
      else if (a == "--ctx") ctx = std::atoi(need("--ctx").c_str());
      else if (a == "--n-gen") n_gen = std::atoi(need("--n-gen").c_str());
      else filtered.push_back(a);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
  std::vector<char*> cargv;
  for (auto& s : filtered) cargv.push_back(const_cast<char*>(s.c_str()));
  std::vector<std::string> positional, unparsed;
  std::string cfg_err;
  if (!cfg.parse_command_line(static_cast<int>(cargv.size()), cargv.data(), 1, &positional,
                              &unparsed, &cfg_err)) {
    std::fprintf(stderr, "config error: %s\n", cfg_err.c_str());
    return 2;
  }
  if (positional.empty()) {
    std::fprintf(stderr, "usage: mtp_numprobe <model.gguf> [--ids ...] [--ctx N]\n");
    return 2;
  }
  const std::string model = positional[0];
  try {
    runner::DenseRunner r(model, ctx, detect_container(model));
    const int vocab = r.vocab();
    const std::vector<int> prompt = parse_ids(ids_arg);
    if (prompt.size() < 2) throw std::runtime_error("need >= 2 prompt ids");
    const int P = static_cast<int>(prompt.size());

    // Pick a continuation that exercises the model without needing drafts: reuse the
    // prompt's own last token and its arg-max (a real text path is not required for a
    // numerics probe).
    auto prefill = [&](std::vector<float>* last) {
      std::vector<float> out = r.forward(prompt, 0, /*last_row_only=*/true);
      if (last) *last = out;
    };

    std::vector<float> l_pf;
    prefill(&l_pf);
    const int p0 = argmax_of(l_pf.data(), vocab);
    const int p1 = p0;
    const int p2 = p0;

    // ---- A: decode-shaped single-token forward at position P ----
    prefill(nullptr);
    std::vector<float> logits = r.forward({p0}, P, /*last_row_only=*/false);
    std::vector<float> A(logits.begin(), logits.end());

    // ---- B: verify-shaped batch forward at position P (spec_snapshot_ + quantized
    // multi-row head, exactly what the accept rule sees); row 0 = same token ----
    prefill(nullptr);
    std::vector<int> row_argmax;
    std::vector<float> row_gap;
    std::vector<float> B =
        r.debug_verify_batch({p0, p1, p2}, P, &row_argmax, &row_gap);

    std::printf("model: %s  prompt=%d ids  vocab=%d  pending=%d\n", model.c_str(), P,
                vocab, p0);
    if (B.empty()) {
      std::printf("verify-shaped batch forward unavailable (no quantized lm head)\n");
    } else {
      print_diff("A decode n=1 vs B batch n=3", compare(A.data(), B.data(), vocab), vocab);
      std::printf("%-34s row_argmax=", "verify rows");
      for (int v : row_argmax) std::printf("%d ", v);
      std::printf(" row_gaps=");
      for (float v : row_gap) std::printf("%.4f ", v);
      std::printf("\n");
    }

    // ---- C: sequential advance a,b then c (all n=1) ----
    prefill(nullptr);
    r.forward({p0}, P, true);
    r.forward({p1}, P + 1, true);
    std::vector<float> logitsC = r.forward({p2}, P + 2, true);

    // ---- D: batched advance [a,b] then c (n=1) ----
    prefill(nullptr);
    r.forward({p0, p1}, P, true);
    std::vector<float> logitsD = r.forward({p2}, P + 2, true);

    print_diff("C seq-state vs D batch-state", compare(logitsC.data(), logitsD.data(), vocab),
               vocab);

    // ---- E: 2-token batch row 0 vs single-token (n=2 path, k=1 verify shape) ----
    prefill(nullptr);
    std::vector<float> logitsE2 = r.forward({p0}, P, false);
    prefill(nullptr);
    std::vector<float> logitsE = r.debug_verify_batch({p0, p1}, P, nullptr, nullptr);
    if (!logitsE.empty())
      print_diff("E decode n=1 vs batch n=2 (row0)",
                 compare(logitsE2.data(), logitsE.data(), vocab), vocab);

    // ---- greedy-stream comparison -------------------------------------------
    // plain   : n=1 decode forward per token (the reference greedy stream)
    // batchg  : the same token stream advanced with the *verify-shaped* batch path
    //           (3-row verify, then roll back to row 0) => "greedy under batch numerics"
    // spec    : the engine's MTP speculative decoder (must equal `batchg` if the
    //           accept/rollback logic is faithful; equals `plain` only if the batch
    //           numerics equal the decode numerics)
    if (n_gen > 0) {
      std::printf("\n== greedy streams over %d tokens ==\n", n_gen);
      std::vector<int> plain, batchg, spec;
      // plain
      r.forward(prompt, 0, true);
      {
        int pos = P;
        int pending = p0;
        for (int i = 0; i < n_gen; ++i) {
          plain.push_back(pending);
          std::vector<float> lg = r.forward({pending}, pos, true);
          pending = argmax_of(lg.data(), vocab);
          ++pos;
        }
      }
      // batch-numerics greedy (needs one rollback slot; prefix_state_save allocates it)
      const bool slot_ok = r.prefix_state_supported() ? r.prefix_state_save(0) : false;
      if (slot_ok) {
        r.forward(prompt, 0, true);
        int pos = P;
        int pending = p0;
        for (int i = 0; i < n_gen; ++i) {
          batchg.push_back(pending);
          std::vector<int> ra;
          std::vector<float> rg;
          std::vector<float> lg =
              r.debug_verify_batch({pending, pending, pending}, pos, &ra, &rg, /*restore=*/0);
          if (lg.empty() || ra.empty()) break;
          pending = ra[0];
          ++pos;
        }
      } else {
        std::printf("(skip batchg: no rollback slot / not a hybrid model)\n");
      }
      // MTP speculative stream
      if (r.mtp_available()) {
        runner::DenseRunner::SpecStats st;
        spec = r.spec_generate(prompt, n_gen, &st);
        std::printf("spec: k=%d rounds=%d alpha=%.3f\n", st.k, st.rounds,
                    st.drafted > 0 ? static_cast<double>(st.accepted) / st.drafted : 0.0);
      } else {
        std::printf("(spec: MTP not available; run with ENGINE_MTP=1)\n");
      }
      print_stream_diff("plain vs spec", plain, spec);
      if (!batchg.empty()) {
        print_stream_diff("plain vs batchgreedy(batch numerics)", plain, batchg);
        if (!spec.empty())
          print_stream_diff("spec vs batchgreedy (accept/rollback logic)", spec, batchg);
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}

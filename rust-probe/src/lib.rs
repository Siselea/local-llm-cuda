//! Rust AVX2 Q4_K row-dot prototype (feasibility test for "would Rust be faster?").
//!
//! This is a line-by-line port of `avx2_dot_q4k()` from `src/cpu_matvec.cpp`
//! (same block layout, same nibble order, same FMA order, same 4 accumulators and
//! the same prefetch pattern), exposed through a C ABI so the existing C++ engine
//! can link it and compare outputs/timing on identical weights.
//!
//! Q4_K block (144 bytes / 256 elements):
//!   +0  d    f16      `+2  dmin f16      +4  scales[12]      +16 qs[128]
//! For g in 0..4: q = qs + 32*g holds 32 bytes -> 64 nibbles: low nibbles of byte
//! l..l+7 feed sub-block 2g, high nibbles feed sub-block 2g+1 (x offset 16 bytes
//! apart in the activation vector).

#![allow(clippy::missing_safety_doc)]

use core::arch::x86_64::*;

#[inline(always)]
fn half_to_float(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = (h >> 10) as u32 & 0x1F;
    let mant = (h as u32) & 0x3FF;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let v = (mant as f32) * 5.960_464_5e-08; // 2^-24
        return if sign != 0 { -v } else { v };
    }
    if exp == 0x1F {
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    let e = exp as i32 - 15 + 127;
    f32::from_bits(sign | ((e as u32) << 23) | (mant << 13))
}

#[inline(always)]
unsafe fn scale_min(sub: usize, s: *const u8) -> (i32, i32) {
    // must match scale_min() in src/cpu_matvec.cpp / q4k_scale_min() in kernels.h
    if sub < 4 {
        ((*s.add(sub) & 63) as i32, ((*s.add(sub + 4)) & 63) as i32)
    } else {
        let sc = ((*s.add(sub + 4) & 0xF) as i32) | (((*s.add(sub - 4) >> 2) & 0x30) as i32);
        let mn = ((*s.add(sub + 4) >> 4) as i32) | (((*s.add(sub) >> 2) & 0x30) as i32);
        (sc, mn)
    }
}

#[inline(always)]
unsafe fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm_add_ps(lo, hi);
    let lo = _mm_hadd_ps(lo, lo);
    let lo = _mm_hadd_ps(lo, lo);
    _mm_cvtss_f32(lo)
}

/// One quantized row x activation vector -> dot product.  Mirrors the C++ kernel
/// instruction for instruction (same summation order => bit-identical result).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q4k_dot_row(row: *const u8, x: *const f32, n_in: usize) -> f32 {
    let nb = n_in / 256;
    let row_bytes = nb * 144;
    for i in (0..row_bytes).step_by(64) {
        _mm_prefetch(row.add(i) as *const i8, _MM_HINT_T0);
    }
    for i in (0..row_bytes).step_by(64) {
        _mm_prefetch(row.add(row_bytes + i) as *const i8, _MM_HINT_T1);
    }
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let c15 = _mm256_set1_epi32(0xF);
    for b in 0..nb {
        let p = row.add(b * 144);
        let d = half_to_float(*(p as *const u16));
        let dmin = half_to_float(*(p.add(2) as *const u16));
        let scales = p.add(4);
        let qs = p.add(16);
        let mut scf = [0f32; 8];
        let mut mnf = [0f32; 8];
        for sub in 0..8 {
            let (sc, mn) = scale_min(sub, scales);
            scf[sub] = d * sc as f32;
            mnf[sub] = dmin * mn as f32;
        }
        let xb = x.add(b * 256);
        for g in 0..4 {
            let q = qs.add(g * 32);
            let s0 = _mm256_set1_ps(scf[2 * g]);
            let m0 = _mm256_set1_ps(mnf[2 * g]);
            let s1 = _mm256_set1_ps(scf[2 * g + 1]);
            let m1 = _mm256_set1_ps(mnf[2 * g + 1]);
            let x0 = xb.add(g * 64);
            for l in (0..32).step_by(16) {
                let raw0 = _mm_loadl_epi64(q.add(l) as *const __m128i);
                let raw1 = _mm_loadl_epi64(q.add(l + 8) as *const __m128i);
                let lo0 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw0), c15);
                let hi0 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw0), 4);
                let lo1 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw1), c15);
                let hi1 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw1), 4);
                let f00 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo0), m0);
                let f01 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi0), m1);
                let f10 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo1), m0);
                let f11 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi1), m1);
                a0 = _mm256_fmadd_ps(f00, _mm256_loadu_ps(x0.add(l)), a0);
                a1 = _mm256_fmadd_ps(f01, _mm256_loadu_ps(x0.add(32 + l)), a1);
                a2 = _mm256_fmadd_ps(f10, _mm256_loadu_ps(x0.add(l + 8)), a2);
                a3 = _mm256_fmadd_ps(f11, _mm256_loadu_ps(x0.add(32 + l + 8)), a3);
            }
        }
    }
    a0 = _mm256_add_ps(a0, a1);
    a2 = _mm256_add_ps(a2, a3);
    hsum256(_mm256_add_ps(a0, a2))
}

/// y[n_out] = W[n_out][n_in] * x[n_in] for row-major Q4_K blocks (single thread).
#[no_mangle]
pub unsafe extern "C" fn rust_q4k_matvec(
    w: *const u8,
    n_out: i32,
    n_in: i32,
    x: *const f32,
    y: *mut f32,
) {
    let row_bytes = (n_in as usize / 256) * 144;
    for r in 0..n_out as usize {
        let v = q4k_dot_row(w.add(r * row_bytes), x, n_in as usize);
        *y.add(r) = v;
    }
}

/// Single row dot, C ABI (for the in-process A/B against the C++ kernel).
#[no_mangle]
pub unsafe extern "C" fn rust_q4k_dot_row(
    row: *const u8,
    x: *const f32,
    n_in: i32,
    out: *mut f32,
) {
    *out = q4k_dot_row(row, x, n_in as usize);
}

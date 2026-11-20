// Rust kernel library for the Rust-vs-C++ CPU kernel comparison (rust-probe/).
//
// Built as a *staticlib* so no Rust linker setup is needed on this machine (there is
// no MSVC `link.exe` and no MinGW gcc; the repo builds C++ with the pip zig toolchain,
// which is a complete clang+lld+mingw environment).  The C++ harness links this
// archive and calls the kernels through extern "C", so both languages then run in the
// SAME process, on the SAME pinned thread team, over the SAME rotating copies --
// the only variable left is the kernel code itself.
//
//   rustc --target x86_64-pc-windows-gnu --crate-type staticlib \
//         -C opt-level=3 -C target-cpu=native -C panic=abort \
//         rust_kernels.rs -o librust_kernels.a
//
// The kernels are a 1:1 transcription of src/cpu_matvec.cpp's avx2_dot_q4k /
// avx2_dot_q6k (same header parsing, scale/min math, nibble unpack order, 4-FMA
// accumulator chains and horizontal sum tree).
#![allow(clippy::missing_safety_doc)]

use std::arch::x86_64::*;

#[inline(always)]
unsafe fn half_to_float(h: u16) -> f32 {
    let sign = ((h & 0x8000u16) as u32) << 16;
    let mut exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let v = (mant as f32) * 5.9604644775390625e-08f32;
        return if sign != 0 { -v } else { v };
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F80_0000u32 | (mant << 13));
    }
    exp += 112;
    f32::from_bits(sign | (exp << 23) | (mant << 13))
}

#[inline(always)]
unsafe fn scale_min(sub: i32, s: *const u8, sc: &mut i32, mn: &mut i32) {
    if sub < 4 {
        let su = sub as usize;
        *sc = (*s.add(su) & 63) as i32;
        *mn = (*s.add(su + 4) & 63) as i32;
    } else {
        let su = sub as usize;
        *sc = ((*s.add(su + 4) & 0x0F) | ((*s.add(su - 4) >> 2) & 0x30)) as i32;
        *mn = ((*s.add(su + 4) >> 4) | ((*s.add(su) >> 2) & 0x30)) as i32;
    }
}

#[inline(always)]
unsafe fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let mut lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    _mm_cvtss_f32(lo)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4k_impl(row: *const u8, x: *const f32, n_in: usize) -> f32 {
    let nb = n_in / 256;
    let row_bytes = nb * 144;
    let mut i = 0usize;
    while i < row_bytes {
        _mm_prefetch(row.add(i) as *const i8, _MM_HINT_T0);
        i += 64;
    }
    let mut i = 0usize;
    while i < row_bytes {
        _mm_prefetch(row.add(row_bytes + i) as *const i8, _MM_HINT_T1);
        i += 64;
    }
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let c15 = _mm256_set1_epi32(0xF);
    for b in 0..nb {
        let p = row.add(b * 144);
        let d = half_to_float((p as *const u16).read_unaligned());
        let dmin = half_to_float((p.add(2) as *const u16).read_unaligned());
        let scales = p.add(4);
        let qs = p.add(16);
        let mut scf = [0f32; 8];
        let mut mnf = [0f32; 8];
        for sub in 0..8 {
            let (mut sc, mut mn) = (0i32, 0i32);
            scale_min(sub as i32, scales, &mut sc, &mut mn);
            scf[sub] = d * (sc as f32);
            mnf[sub] = dmin * (mn as f32);
        }
        let xb = x.add(b * 256);
        for g in 0..4 {
            let q = qs.add(g * 32);
            let s0 = _mm256_set1_ps(scf[2 * g]);
            let m0 = _mm256_set1_ps(mnf[2 * g]);
            let s1 = _mm256_set1_ps(scf[2 * g + 1]);
            let m1 = _mm256_set1_ps(mnf[2 * g + 1]);
            let x0 = xb.add(g * 64);
            let mut l = 0usize;
            while l < 32 {
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
                l += 16;
            }
        }
    }
    let s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    hsum256(s)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q6k_impl(row: *const u8, x: *const f32, n_in: usize) -> f32 {
    let nb = n_in / 256;
    let row_bytes = nb * 210;
    let mut i = 0usize;
    while i < row_bytes {
        _mm_prefetch(row.add(i) as *const i8, _MM_HINT_T0);
        i += 64;
    }
    let mut i = 0usize;
    while i < row_bytes {
        _mm_prefetch(row.add(row_bytes + i) as *const i8, _MM_HINT_T1);
        i += 64;
    }
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let c15 = _mm256_set1_epi32(0xF);
    let c3 = _mm256_set1_epi32(3);
    let c32 = _mm256_set1_epi32(32);
    for b in 0..nb {
        let p = row.add(b * 210);
        let d = half_to_float((p.add(208) as *const u16).read_unaligned());
        let sc = p.add(192) as *const i8;
        let xb = x.add(b * 256);
        let mut sfl = [0f32; 16];
        for i in 0..16 {
            sfl[i] = d * (*sc.add(i) as f32);
        }
        for n2 in 0..2usize {
            let ql = p.add(n2 * 64);
            let qh = p.add(128 + n2 * 32);
            let x0 = xb.add(n2 * 128);
            let mut l = 0usize;
            while l < 32 {
                let sfi = n2 * 8 + (l >> 4);
                let l1 = _mm_loadl_epi64(ql.add(l) as *const __m128i);
                let l2 = _mm_loadl_epi64(ql.add(l + 32) as *const __m128i);
                let h8 = _mm_loadl_epi64(qh.add(l) as *const __m128i);
                let h = _mm256_cvtepu8_epi32(h8);
                let lo1 = _mm256_cvtepu8_epi32(l1);
                let lo2 = _mm256_cvtepu8_epi32(l2);
                let q1 = _mm256_sub_epi32(
                    _mm256_or_si256(
                        _mm256_and_si256(lo1, c15),
                        _mm256_slli_epi32(_mm256_and_si256(h, c3), 4),
                    ),
                    c32,
                );
                let q2 = _mm256_sub_epi32(
                    _mm256_or_si256(
                        _mm256_and_si256(lo2, c15),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 2), c3), 4),
                    ),
                    c32,
                );
                let q3 = _mm256_sub_epi32(
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi32(lo1, 4), c15),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 4), c3), 4),
                    ),
                    c32,
                );
                let q4 = _mm256_sub_epi32(
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi32(lo2, 4), c15),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 6), c3), 4),
                    ),
                    c32,
                );
                a0 = _mm256_fmadd_ps(
                    _mm256_mul_ps(_mm256_set1_ps(sfl[sfi + 0]), _mm256_cvtepi32_ps(q1)),
                    _mm256_loadu_ps(x0.add(l)),
                    a0,
                );
                a1 = _mm256_fmadd_ps(
                    _mm256_mul_ps(_mm256_set1_ps(sfl[sfi + 2]), _mm256_cvtepi32_ps(q2)),
                    _mm256_loadu_ps(x0.add(l + 32)),
                    a1,
                );
                a2 = _mm256_fmadd_ps(
                    _mm256_mul_ps(_mm256_set1_ps(sfl[sfi + 4]), _mm256_cvtepi32_ps(q3)),
                    _mm256_loadu_ps(x0.add(l + 64)),
                    a2,
                );
                a3 = _mm256_fmadd_ps(
                    _mm256_mul_ps(_mm256_set1_ps(sfl[sfi + 6]), _mm256_cvtepi32_ps(q4)),
                    _mm256_loadu_ps(x0.add(l + 96)),
                    a3,
                );
                l += 8;
            }
        }
    }
    let s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    hsum256(s)
}

// ---------------------------------------------------------------- C ABI exports
#[no_mangle]
pub unsafe extern "C" fn rust_dot_q4k(row: *const u8, x: *const f32, n_in: usize) -> f32 {
    dot_q4k_impl(row, x, n_in)
}

#[no_mangle]
pub unsafe extern "C" fn rust_dot_q6k(row: *const u8, x: *const f32, n_in: usize) -> f32 {
    dot_q6k_impl(row, x, n_in)
}

// ---------------------------------------------------------------- tuned Q4_K v2
// The naive 1:1 transcription above measures 47% slower than the clang build: LLVM
// leaves the `for g in 0..4` group loop rolled, so scf[]/mnf[] stay in memory and the
// loop ends up with 44 stack ops per block (verified in the emitted asm).  This
// version unrolls the 4 groups and the 2 halves by hand with constant array indices
// (same instruction sequence and same 4-accumulator mapping as the C++ kernel), which
// is what the C++ compiler does on its own.
macro_rules! q4k_half {
    ($q:expr, $x0:expr, $l:expr, $s0:expr, $m0:expr, $s1:expr, $m1:expr,
     $a0:ident, $a1:ident, $a2:ident, $a3:ident, $c15:ident) => {{
        let raw0 = _mm_loadl_epi64(($q).add($l) as *const __m128i);
        let raw1 = _mm_loadl_epi64(($q).add($l + 8) as *const __m128i);
        let lo0 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw0), $c15);
        let hi0 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw0), 4);
        let lo1 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw1), $c15);
        let hi1 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw1), 4);
        let f00 = _mm256_fmsub_ps($s0, _mm256_cvtepi32_ps(lo0), $m0);
        let f01 = _mm256_fmsub_ps($s1, _mm256_cvtepi32_ps(hi0), $m1);
        let f10 = _mm256_fmsub_ps($s0, _mm256_cvtepi32_ps(lo1), $m0);
        let f11 = _mm256_fmsub_ps($s1, _mm256_cvtepi32_ps(hi1), $m1);
        $a0 = _mm256_fmadd_ps(f00, _mm256_loadu_ps(($x0).add($l)), $a0);
        $a1 = _mm256_fmadd_ps(f01, _mm256_loadu_ps(($x0).add(32 + $l)), $a1);
        $a2 = _mm256_fmadd_ps(f10, _mm256_loadu_ps(($x0).add($l + 8)), $a2);
        $a3 = _mm256_fmadd_ps(f11, _mm256_loadu_ps(($x0).add(32 + $l + 8)), $a3);
    }};
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4k_v2(row: *const u8, x: *const f32, n_in: usize) -> f32 {
    let nb = n_in / 256;
    let row_bytes = nb * 144;
    let mut i = 0usize;
    while i < row_bytes {
        _mm_prefetch(row.add(i) as *const i8, _MM_HINT_T0);
        i += 64;
    }
    let mut i = 0usize;
    while i < row_bytes {
        _mm_prefetch(row.add(row_bytes + i) as *const i8, _MM_HINT_T1);
        i += 64;
    }
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let c15 = _mm256_set1_epi32(0xF);
    for b in 0..nb {
        let p = row.add(b * 144);
        let d = half_to_float((p as *const u16).read_unaligned());
        let dmin = half_to_float((p.add(2) as *const u16).read_unaligned());
        let scales = p.add(4);
        let qs = p.add(16);
        let xb = x.add(b * 256);
        let mut scf = [0f32; 8];
        let mut mnf = [0f32; 8];
        for sub in 0..8 {
            let (mut sc, mut mn) = (0i32, 0i32);
            scale_min(sub as i32, scales, &mut sc, &mut mn);
            scf[sub] = d * (sc as f32);
            mnf[sub] = dmin * (mn as f32);
        }
        // group 0
        {
            let q = qs;
            let x0 = xb;
            let s0 = _mm256_set1_ps(scf[0]);
            let m0 = _mm256_set1_ps(mnf[0]);
            let s1 = _mm256_set1_ps(scf[1]);
            let m1 = _mm256_set1_ps(mnf[1]);
            q4k_half!(q, x0, 0usize, s0, m0, s1, m1, a0, a1, a2, a3, c15);
            q4k_half!(q, x0, 16usize, s0, m0, s1, m1, a0, a1, a2, a3, c15);
        }
        // group 1
        {
            let q = qs.add(32);
            let x0 = xb.add(64);
            let s0 = _mm256_set1_ps(scf[2]);
            let m0 = _mm256_set1_ps(mnf[2]);
            let s1 = _mm256_set1_ps(scf[3]);
            let m1 = _mm256_set1_ps(mnf[3]);
            q4k_half!(q, x0, 0usize, s0, m0, s1, m1, a0, a1, a2, a3, c15);
            q4k_half!(q, x0, 16usize, s0, m0, s1, m1, a0, a1, a2, a3, c15);
        }
        // group 2
        {
            let q = qs.add(64);
            let x0 = xb.add(128);
            let s0 = _mm256_set1_ps(scf[4]);
            let m0 = _mm256_set1_ps(mnf[4]);
            let s1 = _mm256_set1_ps(scf[5]);
            let m1 = _mm256_set1_ps(mnf[5]);
            q4k_half!(q, x0, 0usize, s0, m0, s1, m1, a0, a1, a2, a3, c15);
            q4k_half!(q, x0, 16usize, s0, m0, s1, m1, a0, a1, a2, a3, c15);
        }
        // group 3
        {
            let q = qs.add(96);
            let x0 = xb.add(192);
            let s0 = _mm256_set1_ps(scf[6]);
            let m0 = _mm256_set1_ps(mnf[6]);
            let s1 = _mm256_set1_ps(scf[7]);
            let m1 = _mm256_set1_ps(mnf[7]);
            q4k_half!(q, x0, 0usize, s0, m0, s1, m1, a0, a1, a2, a3, c15);
            q4k_half!(q, x0, 16usize, s0, m0, s1, m1, a0, a1, a2, a3, c15);
        }
    }
    let s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    hsum256(s)
}

#[no_mangle]
pub unsafe extern "C" fn rust_dot_q4k_v2(row: *const u8, x: *const f32, n_in: usize) -> f32 {
    dot_q4k_v2(row, x, n_in)
}

// ---------------------------------------------------------------- Q4_K v3
// v3 removes the scf[]/mnf[] arrays entirely (named scalars + a tuple-returning
// scale_min) so no scalar can be spilled; groups and halves stay hand-unrolled.
#[inline(always)]
unsafe fn scale_min2(sub: i32, s: *const u8) -> (i32, i32) {
    if sub < 4 {
        let su = sub as usize;
        ((*s.add(su) & 63) as i32, (*s.add(su + 4) & 63) as i32)
    } else {
        let su = sub as usize;
        (
            ((*s.add(su + 4) & 0x0F) | ((*s.add(su - 4) >> 2) & 0x30)) as i32,
            ((*s.add(su + 4) >> 4) | ((*s.add(su) >> 2) & 0x30)) as i32,
        )
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4k_v3(row: *const u8, x: *const f32, n_in: usize) -> f32 {
    let nb = n_in / 256;
    let row_bytes = nb * 144;
    let mut i = 0usize;
    while i < row_bytes {
        _mm_prefetch(row.add(i) as *const i8, _MM_HINT_T0);
        i += 64;
    }
    let mut i = 0usize;
    while i < row_bytes {
        _mm_prefetch(row.add(row_bytes + i) as *const i8, _MM_HINT_T1);
        i += 64;
    }
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let c15 = _mm256_set1_epi32(0xF);
    for b in 0..nb {
        let p = row.add(b * 144);
        let d = half_to_float((p as *const u16).read_unaligned());
        let dmin = half_to_float((p.add(2) as *const u16).read_unaligned());
        let scales = p.add(4);
        let qs = p.add(16);
        let xb = x.add(b * 256);
        let (sc0, mn0) = scale_min2(0, scales);
        let (sc1, mn1) = scale_min2(1, scales);
        let (sc2, mn2) = scale_min2(2, scales);
        let (sc3, mn3) = scale_min2(3, scales);
        let (sc4, mn4) = scale_min2(4, scales);
        let (sc5, mn5) = scale_min2(5, scales);
        let (sc6, mn6) = scale_min2(6, scales);
        let (sc7, mn7) = scale_min2(7, scales);
        let s0v = _mm256_set1_ps(d * (sc0 as f32));
        let m0v = _mm256_set1_ps(dmin * (mn0 as f32));
        let s1v = _mm256_set1_ps(d * (sc1 as f32));
        let m1v = _mm256_set1_ps(dmin * (mn1 as f32));
        let s2v = _mm256_set1_ps(d * (sc2 as f32));
        let m2v = _mm256_set1_ps(dmin * (mn2 as f32));
        let s3v = _mm256_set1_ps(d * (sc3 as f32));
        let m3v = _mm256_set1_ps(dmin * (mn3 as f32));
        let s4v = _mm256_set1_ps(d * (sc4 as f32));
        let m4v = _mm256_set1_ps(dmin * (mn4 as f32));
        let s5v = _mm256_set1_ps(d * (sc5 as f32));
        let m5v = _mm256_set1_ps(dmin * (mn5 as f32));
        let s6v = _mm256_set1_ps(d * (sc6 as f32));
        let m6v = _mm256_set1_ps(dmin * (mn6 as f32));
        let s7v = _mm256_set1_ps(d * (sc7 as f32));
        let m7v = _mm256_set1_ps(dmin * (mn7 as f32));
        q4k_half!(qs, xb, 0usize, s0v, m0v, s1v, m1v, a0, a1, a2, a3, c15);
        q4k_half!(qs, xb, 16usize, s0v, m0v, s1v, m1v, a0, a1, a2, a3, c15);
        q4k_half!(qs.add(32), xb.add(64), 0usize, s2v, m2v, s3v, m3v, a0, a1, a2, a3, c15);
        q4k_half!(qs.add(32), xb.add(64), 16usize, s2v, m2v, s3v, m3v, a0, a1, a2, a3, c15);
        q4k_half!(qs.add(64), xb.add(128), 0usize, s4v, m4v, s5v, m5v, a0, a1, a2, a3, c15);
        q4k_half!(qs.add(64), xb.add(128), 16usize, s4v, m4v, s5v, m5v, a0, a1, a2, a3, c15);
        q4k_half!(qs.add(96), xb.add(192), 0usize, s6v, m6v, s7v, m7v, a0, a1, a2, a3, c15);
        q4k_half!(qs.add(96), xb.add(192), 16usize, s6v, m6v, s7v, m7v, a0, a1, a2, a3, c15);
    }
    let s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    hsum256(s)
}

#[no_mangle]
pub unsafe extern "C" fn rust_dot_q4k_v3(row: *const u8, x: *const f32, n_in: usize) -> f32 {
    dot_q4k_v3(row, x, n_in)
}

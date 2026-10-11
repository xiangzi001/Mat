#if defined(__AVX512__)||defined(__AVX2__)
inline void __transpose4x4_pd(
    __m256d v0, __m256d v1, __m256d v2, __m256d v3,
    __m256d& r0, __m256d& r1, __m256d& r2, __m256d& r3)
{
    __m256d t0 = _mm256_unpacklo_pd(v0, v1);
    __m256d t1 = _mm256_unpackhi_pd(v0, v1);
    __m256d t2 = _mm256_unpacklo_pd(v2, v3);
    __m256d t3 = _mm256_unpackhi_pd(v2, v3);
    r0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    r2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    r3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}
#endif


#ifdef __AVX512__

template<>
constexpr int __micro_kernel_nr<double, double, double>() {
    return 12;
}

template<>
constexpr int __micro_kernel_mr<double, double, double>() {
    return 16;
}

template<>
inline void __pack_A_block<double, double>(
    const double* __restrict A,
    size_t ldA, size_t mc, size_t kc,
    double* __restrict Ap)
{
    constexpr int MR = 16;

    for (size_t ip = 0; ip * MR < mc; ++ip) {
        const int mm = std::min(MR, (int)(mc - ip * MR));
        double* dst = Ap + ip * kc * MR;

        if (mm == MR) {
            for (size_t p = 0; p < kc; ++p) {
                const double* src = A + p * ldA + ip * MR;
                _mm512_store_pd(dst, _mm512_loadu_pd(src));
                _mm512_store_pd(dst + 8, _mm512_loadu_pd(src + 8));
                dst += MR;
            }
        }
        else {
            const __mmask8 m0 = (mm >= 8) ? (__mmask8)0xFF
                : (__mmask8)((1u << mm) - 1u);
            const __mmask8 m1 = (mm > 8) ? (__mmask8)((1u << (mm - 8)) - 1u)
                : (__mmask8)0;
            for (size_t p = 0; p < kc; ++p) {
                const double* src = A + p * ldA + ip * MR;
                _mm512_store_pd(dst, _mm512_maskz_loadu_pd(m0, src));
                _mm512_store_pd(dst + 8, _mm512_maskz_loadu_pd(m1, src + 8));
                dst += MR;
            }
        }
    }
}
template<>
inline void __pack_AT_block<double, double>(
    const double* __restrict Vd,
    size_t ldV, size_t pc,
    size_t mc, size_t kc,
    double* __restrict Ap)
{
    constexpr size_t MR = 16;

    for (size_t ip = 0; ip * MR < mc; ++ip) {
        const size_t mm = std::min(MR, mc - ip * MR);
        double* dst = Ap + ip * kc * MR;
        const double* base = Vd + (ip * MR) * ldV + pc;

        size_t q = 0;
        if (mm == MR) {
            for (; q + 4 <= kc; q += 4) {
                __m256d v0 = _mm256_loadu_pd(base + 0 * ldV + q);
                __m256d v1 = _mm256_loadu_pd(base + 1 * ldV + q);
                __m256d v2 = _mm256_loadu_pd(base + 2 * ldV + q);
                __m256d v3 = _mm256_loadu_pd(base + 3 * ldV + q);
                __m256d v4 = _mm256_loadu_pd(base + 4 * ldV + q);
                __m256d v5 = _mm256_loadu_pd(base + 5 * ldV + q);
                __m256d v6 = _mm256_loadu_pd(base + 6 * ldV + q);
                __m256d v7 = _mm256_loadu_pd(base + 7 * ldV + q);
                __m256d v8 = _mm256_loadu_pd(base + 8 * ldV + q);
                __m256d v9 = _mm256_loadu_pd(base + 9 * ldV + q);
                __m256d v10 = _mm256_loadu_pd(base + 10 * ldV + q);
                __m256d v11 = _mm256_loadu_pd(base + 11 * ldV + q);
                __m256d v12 = _mm256_loadu_pd(base + 12 * ldV + q);
                __m256d v13 = _mm256_loadu_pd(base + 13 * ldV + q);
                __m256d v14 = _mm256_loadu_pd(base + 14 * ldV + q);
                __m256d v15 = _mm256_loadu_pd(base + 15 * ldV + q);

                __m256d r0, r1, r2, r3, r4, r5, r6, r7;
                __m256d r8, r9, r10, r11, r12, r13, r14, r15;
                __transpose4x4_pd(v0, v1, v2, v3, r0, r1, r2, r3);
                __transpose4x4_pd(v4, v5, v6, v7, r4, r5, r6, r7);
                __transpose4x4_pd(v8, v9, v10, v11, r8, r9, r10, r11);
                __transpose4x4_pd(v12, v13, v14, v15, r12, r13, r14, r15);

                _mm256_storeu_pd(dst + (q + 0) * MR + 0, r0);
                _mm256_storeu_pd(dst + (q + 0) * MR + 4, r4);
                _mm256_storeu_pd(dst + (q + 0) * MR + 8, r8);
                _mm256_storeu_pd(dst + (q + 0) * MR + 12, r12);

                _mm256_storeu_pd(dst + (q + 1) * MR + 0, r1);
                _mm256_storeu_pd(dst + (q + 1) * MR + 4, r5);
                _mm256_storeu_pd(dst + (q + 1) * MR + 8, r9);
                _mm256_storeu_pd(dst + (q + 1) * MR + 12, r13);

                _mm256_storeu_pd(dst + (q + 2) * MR + 0, r2);
                _mm256_storeu_pd(dst + (q + 2) * MR + 4, r6);
                _mm256_storeu_pd(dst + (q + 2) * MR + 8, r10);
                _mm256_storeu_pd(dst + (q + 2) * MR + 12, r14);

                _mm256_storeu_pd(dst + (q + 3) * MR + 0, r3);
                _mm256_storeu_pd(dst + (q + 3) * MR + 4, r7);
                _mm256_storeu_pd(dst + (q + 3) * MR + 8, r11);
                _mm256_storeu_pd(dst + (q + 3) * MR + 12, r15);
            }
        }
        for (; q < kc; ++q) {
            const double* src = base + q;
            for (size_t i = 0; i < mm; ++i) dst[q * MR + i] = src[i * ldV];
            for (size_t i = mm; i < MR; ++i) dst[q * MR + i] = 0.0;
        }
    }
}

template<>
inline void __pack_B_block<double, double>(
    const double* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    double* __restrict Bp)
{
    constexpr int NR = 12;

    for (size_t jp = 0; jp * NR < nc; ++jp) {
        const int nn = std::min(NR, (int)(nc - jp * NR));
        double* dst = Bp + jp * kc * NR;
        const double* Bcol = B + (jp * NR) * ldB;

        if (nn == NR) {
            size_t p = 0;
            for (; p + 4 <= kc; p += 4) {
                __m256d v0 = _mm256_loadu_pd(Bcol + 0 * ldB + p);
                __m256d v1 = _mm256_loadu_pd(Bcol + 1 * ldB + p);
                __m256d v2 = _mm256_loadu_pd(Bcol + 2 * ldB + p);
                __m256d v3 = _mm256_loadu_pd(Bcol + 3 * ldB + p);
                __m256d v4 = _mm256_loadu_pd(Bcol + 4 * ldB + p);
                __m256d v5 = _mm256_loadu_pd(Bcol + 5 * ldB + p);
                __m256d v6 = _mm256_loadu_pd(Bcol + 6 * ldB + p);
                __m256d v7 = _mm256_loadu_pd(Bcol + 7 * ldB + p);
                __m256d v8 = _mm256_loadu_pd(Bcol + 8 * ldB + p);
                __m256d v9 = _mm256_loadu_pd(Bcol + 9 * ldB + p);
                __m256d v10 = _mm256_loadu_pd(Bcol + 10 * ldB + p);
                __m256d v11 = _mm256_loadu_pd(Bcol + 11 * ldB + p);

                __m256d r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;
                __transpose4x4_pd(v0, v1, v2, v3, r0, r1, r2, r3);
                __transpose4x4_pd(v4, v5, v6, v7, r4, r5, r6, r7);
                __transpose4x4_pd(v8, v9, v10, v11, r8, r9, r10, r11);

                _mm256_storeu_pd(dst + (p + 0) * NR + 0, r0);
                _mm256_storeu_pd(dst + (p + 0) * NR + 4, r4);
                _mm256_storeu_pd(dst + (p + 0) * NR + 8, r8);

                _mm256_storeu_pd(dst + (p + 1) * NR + 0, r1);
                _mm256_storeu_pd(dst + (p + 1) * NR + 4, r5);
                _mm256_storeu_pd(dst + (p + 1) * NR + 8, r9);

                _mm256_storeu_pd(dst + (p + 2) * NR + 0, r2);
                _mm256_storeu_pd(dst + (p + 2) * NR + 4, r6);
                _mm256_storeu_pd(dst + (p + 2) * NR + 8, r10);

                _mm256_storeu_pd(dst + (p + 3) * NR + 0, r3);
                _mm256_storeu_pd(dst + (p + 3) * NR + 4, r7);
                _mm256_storeu_pd(dst + (p + 3) * NR + 8, r11);
            }

            for (; p < kc; ++p)
                for (int j = 0; j < NR; ++j)
                    dst[p * NR + j] = Bcol[j * ldB + p];
        }
        else {
            for (int j = 0; j < nn; ++j) {
                const double* col = Bcol + j * ldB;
                for (size_t p = 0; p < kc; ++p)
                    dst[p * NR + j] = col[p];
            }
            for (int j = nn; j < NR; ++j)
                for (size_t p = 0; p < kc; ++p)
                    dst[p * NR + j] = 0.0;
        }
    }
}

template<>
inline void __pack_BT_block<double, double>(
    const double* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    double* __restrict Bp)
{
    constexpr int NR = 12;

    for (size_t jp = 0; jp* NR < nc; ++jp) {
        const int nn = (int)((nc - jp * NR < (size_t)NR) ? (nc - jp * NR) : (size_t)NR);
        double* dst = Bp + jp * kc * NR;
        const double* Bcol = B + jp * NR;

        if (nn == NR) {
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                _mm512_storeu_pd(dst, _mm512_loadu_pd(src));
                _mm256_storeu_pd(dst + 8, _mm256_loadu_pd(src + 8));
                dst += NR;
            }
        }
        else {
            const __mmask8 m0 = (nn >= 8) ? (__mmask8)0xFF
                : (nn > 0) ? (__mmask8)((1u << nn) - 1u) : (__mmask8)0;
            const int rem = (nn > 8) ? (nn - 8) : 0;
            const __m256i m1 = _mm256_set_epi64x(
                rem > 3 ? -1LL : 0LL, rem > 2 ? -1LL : 0LL,
                rem > 1 ? -1LL : 0LL, rem > 0 ? -1LL : 0LL);
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                _mm512_storeu_pd(dst, _mm512_maskz_loadu_pd(m0, src));
                _mm256_storeu_pd(dst + 8, _mm256_maskload_pd(src + 8, m1));
                dst += NR;
            }
        }
    }
}

template<>
inline void __pack_BT_block_d<double, double>(
    const double* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    const double* __restrict d,
    double* __restrict Bp)
{
    constexpr int NR = 12;

    for (size_t jp = 0; jp* NR < nc; ++jp) {
        const int nn = (int)((nc - jp * NR < (size_t)NR) ? (nc - jp * NR) : (size_t)NR);
        double* dst = Bp + jp * kc * NR;
        const double* Bcol = B + jp * NR;

        if (nn == NR) {
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                __m512d d512 = _mm512_set1_pd(d[p]);
                __m256d d256 = _mm256_set1_pd(d[p]);
                _mm512_storeu_pd(dst, _mm512_mul_pd(_mm512_loadu_pd(src), d512));
                _mm256_storeu_pd(dst + 8, _mm256_mul_pd(_mm256_loadu_pd(src + 8), d256));
                dst += NR;
            }
        }
        else {
            const __mmask8 m0 = (nn >= 8) ? (__mmask8)0xFF
                : (nn > 0) ? (__mmask8)((1u << nn) - 1u) : (__mmask8)0;
            const int rem = (nn > 8) ? (nn - 8) : 0;
            const __m256i m1 = _mm256_set_epi64x(
                rem > 3 ? -1LL : 0LL, rem > 2 ? -1LL : 0LL,
                rem > 1 ? -1LL : 0LL, rem > 0 ? -1LL : 0LL);
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                __m512d d512 = _mm512_set1_pd(d[p]);
                __m256d d256 = _mm256_set1_pd(d[p]);
                _mm512_storeu_pd(dst,
                    _mm512_mul_pd(_mm512_maskz_loadu_pd(m0, src), d512));
                _mm256_storeu_pd(dst + 8,
                    _mm256_mul_pd(_mm256_maskload_pd(src + 8, m1), d256));
                dst += NR;
            }
        }
    }
}

static void __micro_kernel_mul_packed_full(
    const double* __restrict Ap, const double* __restrict Bp, double* __restrict C,
    size_t ldC, size_t k)
{
    __m512d c00 = _mm512_loadu_pd(C + 0 * ldC + 0);
    __m512d c01 = _mm512_loadu_pd(C + 0 * ldC + 8);
    __m512d c10 = _mm512_loadu_pd(C + 1 * ldC + 0);
    __m512d c11 = _mm512_loadu_pd(C + 1 * ldC + 8);
    __m512d c20 = _mm512_loadu_pd(C + 2 * ldC + 0);
    __m512d c21 = _mm512_loadu_pd(C + 2 * ldC + 8);
    __m512d c30 = _mm512_loadu_pd(C + 3 * ldC + 0);
    __m512d c31 = _mm512_loadu_pd(C + 3 * ldC + 8);
    __m512d c40 = _mm512_loadu_pd(C + 4 * ldC + 0);
    __m512d c41 = _mm512_loadu_pd(C + 4 * ldC + 8);
    __m512d c50 = _mm512_loadu_pd(C + 5 * ldC + 0);
    __m512d c51 = _mm512_loadu_pd(C + 5 * ldC + 8);
    __m512d c60 = _mm512_loadu_pd(C + 6 * ldC + 0);
    __m512d c61 = _mm512_loadu_pd(C + 6 * ldC + 8);
    __m512d c70 = _mm512_loadu_pd(C + 7 * ldC + 0);
    __m512d c71 = _mm512_loadu_pd(C + 7 * ldC + 8);
    __m512d c80 = _mm512_loadu_pd(C + 8 * ldC + 0);
    __m512d c81 = _mm512_loadu_pd(C + 8 * ldC + 8);
    __m512d c90 = _mm512_loadu_pd(C + 9 * ldC + 0);
    __m512d c91 = _mm512_loadu_pd(C + 9 * ldC + 8);
    __m512d c100 = _mm512_loadu_pd(C + 10 * ldC + 0);
    __m512d c101 = _mm512_loadu_pd(C + 10 * ldC + 8);
    __m512d c110 = _mm512_loadu_pd(C + 11 * ldC + 0);
    __m512d c111 = _mm512_loadu_pd(C + 11 * ldC + 8);

    for (size_t p = 0; p < k; ++p) {
        const __m512d a0 = _mm512_load_pd(Ap + p * 16);
        const __m512d a1 = _mm512_load_pd(Ap + p * 16 + 8);

        __m512d b;
        b = _mm512_set1_pd(Bp[p * 12 + 0]);
        c00 = _mm512_fmadd_pd(a0, b, c00);
        c01 = _mm512_fmadd_pd(a1, b, c01);
        b = _mm512_set1_pd(Bp[p * 12 + 1]);
        c10 = _mm512_fmadd_pd(a0, b, c10);
        c11 = _mm512_fmadd_pd(a1, b, c11);
        b = _mm512_set1_pd(Bp[p * 12 + 2]);
        c20 = _mm512_fmadd_pd(a0, b, c20);
        c21 = _mm512_fmadd_pd(a1, b, c21);
        b = _mm512_set1_pd(Bp[p * 12 + 3]);
        c30 = _mm512_fmadd_pd(a0, b, c30);
        c31 = _mm512_fmadd_pd(a1, b, c31);
        b = _mm512_set1_pd(Bp[p * 12 + 4]);
        c40 = _mm512_fmadd_pd(a0, b, c40);
        c41 = _mm512_fmadd_pd(a1, b, c41);
        b = _mm512_set1_pd(Bp[p * 12 + 5]);
        c50 = _mm512_fmadd_pd(a0, b, c50);
        c51 = _mm512_fmadd_pd(a1, b, c51);
        b = _mm512_set1_pd(Bp[p * 12 + 6]);
        c60 = _mm512_fmadd_pd(a0, b, c60);
        c61 = _mm512_fmadd_pd(a1, b, c61);
        b = _mm512_set1_pd(Bp[p * 12 + 7]);
        c70 = _mm512_fmadd_pd(a0, b, c70);
        c71 = _mm512_fmadd_pd(a1, b, c71);
        b = _mm512_set1_pd(Bp[p * 12 + 8]);
        c80 = _mm512_fmadd_pd(a0, b, c80);
        c81 = _mm512_fmadd_pd(a1, b, c81);
        b = _mm512_set1_pd(Bp[p * 12 + 9]);
        c90 = _mm512_fmadd_pd(a0, b, c90);
        c91 = _mm512_fmadd_pd(a1, b, c91);
        b = _mm512_set1_pd(Bp[p * 12 + 10]);
        c100 = _mm512_fmadd_pd(a0, b, c100);
        c101 = _mm512_fmadd_pd(a1, b, c101);
        b = _mm512_set1_pd(Bp[p * 12 + 11]);
        c110 = _mm512_fmadd_pd(a0, b, c110);
        c111 = _mm512_fmadd_pd(a1, b, c111);
    }

    _mm512_storeu_pd(C + 0 * ldC + 0, c00);
    _mm512_storeu_pd(C + 0 * ldC + 8, c01);
    _mm512_storeu_pd(C + 1 * ldC + 0, c10);
    _mm512_storeu_pd(C + 1 * ldC + 8, c11);
    _mm512_storeu_pd(C + 2 * ldC + 0, c20);
    _mm512_storeu_pd(C + 2 * ldC + 8, c21);
    _mm512_storeu_pd(C + 3 * ldC + 0, c30);
    _mm512_storeu_pd(C + 3 * ldC + 8, c31);
    _mm512_storeu_pd(C + 4 * ldC + 0, c40);
    _mm512_storeu_pd(C + 4 * ldC + 8, c41);
    _mm512_storeu_pd(C + 5 * ldC + 0, c50);
    _mm512_storeu_pd(C + 5 * ldC + 8, c51);
    _mm512_storeu_pd(C + 6 * ldC + 0, c60);
    _mm512_storeu_pd(C + 6 * ldC + 8, c61);
    _mm512_storeu_pd(C + 7 * ldC + 0, c70);
    _mm512_storeu_pd(C + 7 * ldC + 8, c71);
    _mm512_storeu_pd(C + 8 * ldC + 0, c80);
    _mm512_storeu_pd(C + 8 * ldC + 8, c81);
    _mm512_storeu_pd(C + 9 * ldC + 0, c90);
    _mm512_storeu_pd(C + 9 * ldC + 8, c91);
    _mm512_storeu_pd(C + 10 * ldC + 0, c100);
    _mm512_storeu_pd(C + 10 * ldC + 8, c101);
    _mm512_storeu_pd(C + 11 * ldC + 0, c110);
    _mm512_storeu_pd(C + 11 * ldC + 8, c111);
}

static void __micro_kernel_mul_packed_edge(
    const double* __restrict Ap, const double* __restrict Bp, double* __restrict C,
    size_t ldC, size_t k, int mr, int nr)
{
    const __mmask8 m0 = (mr >= 8) ? (__mmask8)0xFF
        : (__mmask8)((1u << mr) - 1u);
    const __mmask8 m1 = (mr >= 16) ? (__mmask8)0xFF
        : (mr > 8) ? (__mmask8)((1u << (mr - 8)) - 1u)
        : (__mmask8)0;

    __m512d c00 = _mm512_setzero_pd(), c01 = _mm512_setzero_pd();
    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        c00 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c01 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c10 = _mm512_setzero_pd(), c11 = _mm512_setzero_pd();
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        c10 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c11 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c20 = _mm512_setzero_pd(), c21 = _mm512_setzero_pd();
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        c20 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c21 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c30 = _mm512_setzero_pd(), c31 = _mm512_setzero_pd();
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        c30 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c31 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c40 = _mm512_setzero_pd(), c41 = _mm512_setzero_pd();
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        c40 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c41 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c50 = _mm512_setzero_pd(), c51 = _mm512_setzero_pd();
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        c50 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c51 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c60 = _mm512_setzero_pd(), c61 = _mm512_setzero_pd();
    if (6 < nr) {
        const size_t off = (size_t)(6) * ldC;
        c60 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c61 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c70 = _mm512_setzero_pd(), c71 = _mm512_setzero_pd();
    if (7 < nr) {
        const size_t off = (size_t)(7) * ldC;
        c70 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c71 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c80 = _mm512_setzero_pd(), c81 = _mm512_setzero_pd();
    if (8 < nr) {
        const size_t off = (size_t)(8) * ldC;
        c80 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c81 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c90 = _mm512_setzero_pd(), c91 = _mm512_setzero_pd();
    if (9 < nr) {
        const size_t off = (size_t)(9) * ldC;
        c90 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c91 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c100 = _mm512_setzero_pd(), c101 = _mm512_setzero_pd();
    if (10 < nr) {
        const size_t off = (size_t)(10) * ldC;
        c100 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c101 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c110 = _mm512_setzero_pd(), c111 = _mm512_setzero_pd();
    if (11 < nr) {
        const size_t off = (size_t)(11) * ldC;
        c110 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c111 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }

    for (size_t p = 0; p < k; ++p) {
        const __m512d a0 = _mm512_load_pd(Ap + p * 16);
        const __m512d a1 = _mm512_load_pd(Ap + p * 16 + 8);

        __m512d b;
        b = _mm512_set1_pd(Bp[p * 12 + 0]);
        c00 = _mm512_fmadd_pd(a0, b, c00);
        c01 = _mm512_fmadd_pd(a1, b, c01);
        b = _mm512_set1_pd(Bp[p * 12 + 1]);
        c10 = _mm512_fmadd_pd(a0, b, c10);
        c11 = _mm512_fmadd_pd(a1, b, c11);
        b = _mm512_set1_pd(Bp[p * 12 + 2]);
        c20 = _mm512_fmadd_pd(a0, b, c20);
        c21 = _mm512_fmadd_pd(a1, b, c21);
        b = _mm512_set1_pd(Bp[p * 12 + 3]);
        c30 = _mm512_fmadd_pd(a0, b, c30);
        c31 = _mm512_fmadd_pd(a1, b, c31);
        b = _mm512_set1_pd(Bp[p * 12 + 4]);
        c40 = _mm512_fmadd_pd(a0, b, c40);
        c41 = _mm512_fmadd_pd(a1, b, c41);
        b = _mm512_set1_pd(Bp[p * 12 + 5]);
        c50 = _mm512_fmadd_pd(a0, b, c50);
        c51 = _mm512_fmadd_pd(a1, b, c51);
        b = _mm512_set1_pd(Bp[p * 12 + 6]);
        c60 = _mm512_fmadd_pd(a0, b, c60);
        c61 = _mm512_fmadd_pd(a1, b, c61);
        b = _mm512_set1_pd(Bp[p * 12 + 7]);
        c70 = _mm512_fmadd_pd(a0, b, c70);
        c71 = _mm512_fmadd_pd(a1, b, c71);
        b = _mm512_set1_pd(Bp[p * 12 + 8]);
        c80 = _mm512_fmadd_pd(a0, b, c80);
        c81 = _mm512_fmadd_pd(a1, b, c81);
        b = _mm512_set1_pd(Bp[p * 12 + 9]);
        c90 = _mm512_fmadd_pd(a0, b, c90);
        c91 = _mm512_fmadd_pd(a1, b, c91);
        b = _mm512_set1_pd(Bp[p * 12 + 10]);
        c100 = _mm512_fmadd_pd(a0, b, c100);
        c101 = _mm512_fmadd_pd(a1, b, c101);
        b = _mm512_set1_pd(Bp[p * 12 + 11]);
        c110 = _mm512_fmadd_pd(a0, b, c110);
        c111 = _mm512_fmadd_pd(a1, b, c111);
    }

    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c00);
        _mm512_mask_storeu_pd(C + off + 8, m1, c01);
    }
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c10);
        _mm512_mask_storeu_pd(C + off + 8, m1, c11);
    }
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c20);
        _mm512_mask_storeu_pd(C + off + 8, m1, c21);
    }
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c30);
        _mm512_mask_storeu_pd(C + off + 8, m1, c31);
    }
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c40);
        _mm512_mask_storeu_pd(C + off + 8, m1, c41);
    }
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c50);
        _mm512_mask_storeu_pd(C + off + 8, m1, c51);
    }
    if (6 < nr) {
        const size_t off = (size_t)(6) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c60);
        _mm512_mask_storeu_pd(C + off + 8, m1, c61);
    }
    if (7 < nr) {
        const size_t off = (size_t)(7) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c70);
        _mm512_mask_storeu_pd(C + off + 8, m1, c71);
    }
    if (8 < nr) {
        const size_t off = (size_t)(8) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c80);
        _mm512_mask_storeu_pd(C + off + 8, m1, c81);
    }
    if (9 < nr) {
        const size_t off = (size_t)(9) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c90);
        _mm512_mask_storeu_pd(C + off + 8, m1, c91);
    }
    if (10 < nr) {
        const size_t off = (size_t)(10) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c100);
        _mm512_mask_storeu_pd(C + off + 8, m1, c101);
    }
    if (11 < nr) {
        const size_t off = (size_t)(11) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c110);
        _mm512_mask_storeu_pd(C + off + 8, m1, c111);
    }
}


static void __micro_kernel_nmul_packed_full(
    const double* __restrict Ap, const double* __restrict Bp, double* __restrict C,
    size_t ldC, size_t k)
{
    __m512d c00 = _mm512_loadu_pd(C + 0 * ldC + 0);
    __m512d c01 = _mm512_loadu_pd(C + 0 * ldC + 8);
    __m512d c10 = _mm512_loadu_pd(C + 1 * ldC + 0);
    __m512d c11 = _mm512_loadu_pd(C + 1 * ldC + 8);
    __m512d c20 = _mm512_loadu_pd(C + 2 * ldC + 0);
    __m512d c21 = _mm512_loadu_pd(C + 2 * ldC + 8);
    __m512d c30 = _mm512_loadu_pd(C + 3 * ldC + 0);
    __m512d c31 = _mm512_loadu_pd(C + 3 * ldC + 8);
    __m512d c40 = _mm512_loadu_pd(C + 4 * ldC + 0);
    __m512d c41 = _mm512_loadu_pd(C + 4 * ldC + 8);
    __m512d c50 = _mm512_loadu_pd(C + 5 * ldC + 0);
    __m512d c51 = _mm512_loadu_pd(C + 5 * ldC + 8);
    __m512d c60 = _mm512_loadu_pd(C + 6 * ldC + 0);
    __m512d c61 = _mm512_loadu_pd(C + 6 * ldC + 8);
    __m512d c70 = _mm512_loadu_pd(C + 7 * ldC + 0);
    __m512d c71 = _mm512_loadu_pd(C + 7 * ldC + 8);
    __m512d c80 = _mm512_loadu_pd(C + 8 * ldC + 0);
    __m512d c81 = _mm512_loadu_pd(C + 8 * ldC + 8);
    __m512d c90 = _mm512_loadu_pd(C + 9 * ldC + 0);
    __m512d c91 = _mm512_loadu_pd(C + 9 * ldC + 8);
    __m512d c100 = _mm512_loadu_pd(C + 10 * ldC + 0);
    __m512d c101 = _mm512_loadu_pd(C + 10 * ldC + 8);
    __m512d c110 = _mm512_loadu_pd(C + 11 * ldC + 0);
    __m512d c111 = _mm512_loadu_pd(C + 11 * ldC + 8);

    for (size_t p = 0; p < k; ++p) {
        const __m512d a0 = _mm512_load_pd(Ap + p * 16);
        const __m512d a1 = _mm512_load_pd(Ap + p * 16 + 8);

        __m512d b;
        b = _mm512_set1_pd(Bp[p * 12 + 0]);
        c00 = _mm512_fnmadd_pd(a0, b, c00);
        c01 = _mm512_fnmadd_pd(a1, b, c01);
        b = _mm512_set1_pd(Bp[p * 12 + 1]);
        c10 = _mm512_fnmadd_pd(a0, b, c10);
        c11 = _mm512_fnmadd_pd(a1, b, c11);
        b = _mm512_set1_pd(Bp[p * 12 + 2]);
        c20 = _mm512_fnmadd_pd(a0, b, c20);
        c21 = _mm512_fnmadd_pd(a1, b, c21);
        b = _mm512_set1_pd(Bp[p * 12 + 3]);
        c30 = _mm512_fnmadd_pd(a0, b, c30);
        c31 = _mm512_fnmadd_pd(a1, b, c31);
        b = _mm512_set1_pd(Bp[p * 12 + 4]);
        c40 = _mm512_fnmadd_pd(a0, b, c40);
        c41 = _mm512_fnmadd_pd(a1, b, c41);
        b = _mm512_set1_pd(Bp[p * 12 + 5]);
        c50 = _mm512_fnmadd_pd(a0, b, c50);
        c51 = _mm512_fnmadd_pd(a1, b, c51);
        b = _mm512_set1_pd(Bp[p * 12 + 6]);
        c60 = _mm512_fnmadd_pd(a0, b, c60);
        c61 = _mm512_fnmadd_pd(a1, b, c61);
        b = _mm512_set1_pd(Bp[p * 12 + 7]);
        c70 = _mm512_fnmadd_pd(a0, b, c70);
        c71 = _mm512_fnmadd_pd(a1, b, c71);
        b = _mm512_set1_pd(Bp[p * 12 + 8]);
        c80 = _mm512_fnmadd_pd(a0, b, c80);
        c81 = _mm512_fnmadd_pd(a1, b, c81);
        b = _mm512_set1_pd(Bp[p * 12 + 9]);
        c90 = _mm512_fnmadd_pd(a0, b, c90);
        c91 = _mm512_fnmadd_pd(a1, b, c91);
        b = _mm512_set1_pd(Bp[p * 12 + 10]);
        c100 = _mm512_fnmadd_pd(a0, b, c100);
        c101 = _mm512_fnmadd_pd(a1, b, c101);
        b = _mm512_set1_pd(Bp[p * 12 + 11]);
        c110 = _mm512_fnmadd_pd(a0, b, c110);
        c111 = _mm512_fnmadd_pd(a1, b, c111);
    }

    _mm512_storeu_pd(C + 0 * ldC + 0, c00);
    _mm512_storeu_pd(C + 0 * ldC + 8, c01);
    _mm512_storeu_pd(C + 1 * ldC + 0, c10);
    _mm512_storeu_pd(C + 1 * ldC + 8, c11);
    _mm512_storeu_pd(C + 2 * ldC + 0, c20);
    _mm512_storeu_pd(C + 2 * ldC + 8, c21);
    _mm512_storeu_pd(C + 3 * ldC + 0, c30);
    _mm512_storeu_pd(C + 3 * ldC + 8, c31);
    _mm512_storeu_pd(C + 4 * ldC + 0, c40);
    _mm512_storeu_pd(C + 4 * ldC + 8, c41);
    _mm512_storeu_pd(C + 5 * ldC + 0, c50);
    _mm512_storeu_pd(C + 5 * ldC + 8, c51);
    _mm512_storeu_pd(C + 6 * ldC + 0, c60);
    _mm512_storeu_pd(C + 6 * ldC + 8, c61);
    _mm512_storeu_pd(C + 7 * ldC + 0, c70);
    _mm512_storeu_pd(C + 7 * ldC + 8, c71);
    _mm512_storeu_pd(C + 8 * ldC + 0, c80);
    _mm512_storeu_pd(C + 8 * ldC + 8, c81);
    _mm512_storeu_pd(C + 9 * ldC + 0, c90);
    _mm512_storeu_pd(C + 9 * ldC + 8, c91);
    _mm512_storeu_pd(C + 10 * ldC + 0, c100);
    _mm512_storeu_pd(C + 10 * ldC + 8, c101);
    _mm512_storeu_pd(C + 11 * ldC + 0, c110);
    _mm512_storeu_pd(C + 11 * ldC + 8, c111);
}

static void __micro_kernel_nmul_packed_edge(
    const double* __restrict Ap, const double* __restrict Bp, double* __restrict C,
    size_t ldC, size_t k, int mr, int nr)
{
    const __mmask8 m0 = (mr >= 8) ? (__mmask8)0xFF
        : (__mmask8)((1u << mr) - 1u);
    const __mmask8 m1 = (mr >= 16) ? (__mmask8)0xFF
        : (mr > 8) ? (__mmask8)((1u << (mr - 8)) - 1u)
        : (__mmask8)0;

    __m512d c00 = _mm512_setzero_pd(), c01 = _mm512_setzero_pd();
    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        c00 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c01 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c10 = _mm512_setzero_pd(), c11 = _mm512_setzero_pd();
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        c10 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c11 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c20 = _mm512_setzero_pd(), c21 = _mm512_setzero_pd();
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        c20 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c21 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c30 = _mm512_setzero_pd(), c31 = _mm512_setzero_pd();
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        c30 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c31 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c40 = _mm512_setzero_pd(), c41 = _mm512_setzero_pd();
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        c40 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c41 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c50 = _mm512_setzero_pd(), c51 = _mm512_setzero_pd();
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        c50 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c51 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c60 = _mm512_setzero_pd(), c61 = _mm512_setzero_pd();
    if (6 < nr) {
        const size_t off = (size_t)(6) * ldC;
        c60 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c61 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c70 = _mm512_setzero_pd(), c71 = _mm512_setzero_pd();
    if (7 < nr) {
        const size_t off = (size_t)(7) * ldC;
        c70 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c71 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c80 = _mm512_setzero_pd(), c81 = _mm512_setzero_pd();
    if (8 < nr) {
        const size_t off = (size_t)(8) * ldC;
        c80 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c81 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c90 = _mm512_setzero_pd(), c91 = _mm512_setzero_pd();
    if (9 < nr) {
        const size_t off = (size_t)(9) * ldC;
        c90 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c91 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c100 = _mm512_setzero_pd(), c101 = _mm512_setzero_pd();
    if (10 < nr) {
        const size_t off = (size_t)(10) * ldC;
        c100 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c101 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }
    __m512d c110 = _mm512_setzero_pd(), c111 = _mm512_setzero_pd();
    if (11 < nr) {
        const size_t off = (size_t)(11) * ldC;
        c110 = _mm512_maskz_loadu_pd(m0, C + off + 0);
        c111 = _mm512_maskz_loadu_pd(m1, C + off + 8);
    }

    for (size_t p = 0; p < k; ++p) {
        const __m512d a0 = _mm512_load_pd(Ap + p * 16);
        const __m512d a1 = _mm512_load_pd(Ap + p * 16 + 8);

        __m512d b;
        b = _mm512_set1_pd(Bp[p * 12 + 0]);
        c00 = _mm512_fnmadd_pd(a0, b, c00);
        c01 = _mm512_fnmadd_pd(a1, b, c01);
        b = _mm512_set1_pd(Bp[p * 12 + 1]);
        c10 = _mm512_fnmadd_pd(a0, b, c10);
        c11 = _mm512_fnmadd_pd(a1, b, c11);
        b = _mm512_set1_pd(Bp[p * 12 + 2]);
        c20 = _mm512_fnmadd_pd(a0, b, c20);
        c21 = _mm512_fnmadd_pd(a1, b, c21);
        b = _mm512_set1_pd(Bp[p * 12 + 3]);
        c30 = _mm512_fnmadd_pd(a0, b, c30);
        c31 = _mm512_fnmadd_pd(a1, b, c31);
        b = _mm512_set1_pd(Bp[p * 12 + 4]);
        c40 = _mm512_fnmadd_pd(a0, b, c40);
        c41 = _mm512_fnmadd_pd(a1, b, c41);
        b = _mm512_set1_pd(Bp[p * 12 + 5]);
        c50 = _mm512_fnmadd_pd(a0, b, c50);
        c51 = _mm512_fnmadd_pd(a1, b, c51);
        b = _mm512_set1_pd(Bp[p * 12 + 6]);
        c60 = _mm512_fnmadd_pd(a0, b, c60);
        c61 = _mm512_fnmadd_pd(a1, b, c61);
        b = _mm512_set1_pd(Bp[p * 12 + 7]);
        c70 = _mm512_fnmadd_pd(a0, b, c70);
        c71 = _mm512_fnmadd_pd(a1, b, c71);
        b = _mm512_set1_pd(Bp[p * 12 + 8]);
        c80 = _mm512_fnmadd_pd(a0, b, c80);
        c81 = _mm512_fnmadd_pd(a1, b, c81);
        b = _mm512_set1_pd(Bp[p * 12 + 9]);
        c90 = _mm512_fnmadd_pd(a0, b, c90);
        c91 = _mm512_fnmadd_pd(a1, b, c91);
        b = _mm512_set1_pd(Bp[p * 12 + 10]);
        c100 = _mm512_fnmadd_pd(a0, b, c100);
        c101 = _mm512_fnmadd_pd(a1, b, c101);
        b = _mm512_set1_pd(Bp[p * 12 + 11]);
        c110 = _mm512_fnmadd_pd(a0, b, c110);
        c111 = _mm512_fnmadd_pd(a1, b, c111);
    }

    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c00);
        _mm512_mask_storeu_pd(C + off + 8, m1, c01);
    }
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c10);
        _mm512_mask_storeu_pd(C + off + 8, m1, c11);
    }
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c20);
        _mm512_mask_storeu_pd(C + off + 8, m1, c21);
    }
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c30);
        _mm512_mask_storeu_pd(C + off + 8, m1, c31);
    }
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c40);
        _mm512_mask_storeu_pd(C + off + 8, m1, c41);
    }
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c50);
        _mm512_mask_storeu_pd(C + off + 8, m1, c51);
    }
    if (6 < nr) {
        const size_t off = (size_t)(6) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c60);
        _mm512_mask_storeu_pd(C + off + 8, m1, c61);
    }
    if (7 < nr) {
        const size_t off = (size_t)(7) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c70);
        _mm512_mask_storeu_pd(C + off + 8, m1, c71);
    }
    if (8 < nr) {
        const size_t off = (size_t)(8) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c80);
        _mm512_mask_storeu_pd(C + off + 8, m1, c81);
    }
    if (9 < nr) {
        const size_t off = (size_t)(9) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c90);
        _mm512_mask_storeu_pd(C + off + 8, m1, c91);
    }
    if (10 < nr) {
        const size_t off = (size_t)(10) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c100);
        _mm512_mask_storeu_pd(C + off + 8, m1, c101);
    }
    if (11 < nr) {
        const size_t off = (size_t)(11) * ldC;
        _mm512_mask_storeu_pd(C + off + 0, m0, c110);
        _mm512_mask_storeu_pd(C + off + 8, m1, c111);
    }
}

#elif defined(__AVX2__)

template<>
constexpr int __micro_kernel_nr<double, double, double>() {
    return 6;
}

template<>
constexpr int __micro_kernel_mr<double, double, double>() {
    return 8;
}

template<>
inline void __pack_A_block<double, double>(
    const double* __restrict A,
    size_t ldA, size_t mc, size_t kc,
    double* __restrict Ap)
{
    constexpr int MR = 8;

    for (size_t ip = 0; ip* MR < mc; ++ip) {
        const int mm = std::min(MR, (int)(mc - ip * MR));
        double* dst = Ap + ip * kc * MR;

        if (mm == MR) {
            for (size_t p = 0; p < kc; ++p) {
                const double* src = A + p * ldA + ip * MR;
                _mm256_store_pd(dst, _mm256_loadu_pd(src));
                _mm256_store_pd(dst + 4, _mm256_loadu_pd(src + 4));
                dst += MR;
            }
        }
        else {
            const __m256i mask0 = _mm256_set_epi64x(
                mm > 3 ? -1LL : 0LL, mm > 2 ? -1LL : 0LL,
                mm > 1 ? -1LL : 0LL, mm > 0 ? -1LL : 0LL);
            const __m256i mask1 = _mm256_set_epi64x(
                mm > 7 ? -1LL : 0LL, mm > 6 ? -1LL : 0LL,
                mm > 5 ? -1LL : 0LL, mm > 4 ? -1LL : 0LL);
            for (size_t p = 0; p < kc; ++p) {
                const double* src = A + p * ldA + ip * MR;
                _mm256_store_pd(dst, _mm256_maskload_pd(src, mask0));
                _mm256_store_pd(dst + 4, _mm256_maskload_pd(src + 4, mask1));
                dst += MR;
            }
        }
    }
}
template<>
inline void __pack_AT_block<double, double>(
    const double* __restrict Vd,
    size_t ldV, size_t pc,
    size_t mc, size_t kc,
    double* __restrict Ap)
{
    constexpr size_t MR = 8;

    for (size_t ip = 0; ip * MR < mc; ++ip) {
        const size_t mm = std::min(MR, mc - ip * MR);
        double* dst = Ap + ip * kc * MR;
        const double* base = Vd + (ip * MR) * ldV + pc;

        size_t q = 0;
        if (mm == MR) {
            for (; q + 4 <= kc; q += 4) {
                __m256d v0 = _mm256_loadu_pd(base + 0 * ldV + q);
                __m256d v1 = _mm256_loadu_pd(base + 1 * ldV + q);
                __m256d v2 = _mm256_loadu_pd(base + 2 * ldV + q);
                __m256d v3 = _mm256_loadu_pd(base + 3 * ldV + q);
                __m256d v4 = _mm256_loadu_pd(base + 4 * ldV + q);
                __m256d v5 = _mm256_loadu_pd(base + 5 * ldV + q);
                __m256d v6 = _mm256_loadu_pd(base + 6 * ldV + q);
                __m256d v7 = _mm256_loadu_pd(base + 7 * ldV + q);

                __m256d r0, r1, r2, r3, r4, r5, r6, r7;
                __transpose4x4_pd(v0, v1, v2, v3, r0, r1, r2, r3);
                __transpose4x4_pd(v4, v5, v6, v7, r4, r5, r6, r7);

                _mm256_storeu_pd(dst + (q + 0) * MR + 0, r0);
                _mm256_storeu_pd(dst + (q + 0) * MR + 4, r4);

                _mm256_storeu_pd(dst + (q + 1) * MR + 0, r1);
                _mm256_storeu_pd(dst + (q + 1) * MR + 4, r5);

                _mm256_storeu_pd(dst + (q + 2) * MR + 0, r2);
                _mm256_storeu_pd(dst + (q + 2) * MR + 4, r6);

                _mm256_storeu_pd(dst + (q + 3) * MR + 0, r3);
                _mm256_storeu_pd(dst + (q + 3) * MR + 4, r7);
            }
        }
        for (; q < kc; ++q) {
            const double* src = base + q;
            for (size_t i = 0; i < mm; ++i) dst[q * MR + i] = src[i * ldV];
            for (size_t i = mm; i < MR; ++i) dst[q * MR + i] = 0.0;
        }
    }
}

template<>
inline void __pack_B_block<double, double>(
    const double* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    double* __restrict Bp)
{
    constexpr int NR = 6;

    for (size_t jp = 0; jp * NR < nc; ++jp) {
        const int nn = std::min(NR, (int)(nc - jp * NR));
        double* dst = Bp + jp * kc * NR;
        const double* Bcol = B + (jp * NR) * ldB;

        if (nn == NR) {
            size_t p = 0;
            for (; p + 4 <= kc; p += 4) {
                __m256d v0 = _mm256_loadu_pd(Bcol + 0 * ldB + p);
                __m256d v1 = _mm256_loadu_pd(Bcol + 1 * ldB + p);
                __m256d v2 = _mm256_loadu_pd(Bcol + 2 * ldB + p);
                __m256d v3 = _mm256_loadu_pd(Bcol + 3 * ldB + p);
                __m256d r0, r1, r2, r3;
                __transpose4x4_pd(v0, v1, v2, v3, r0, r1, r2, r3);

                __m256d v4 = _mm256_loadu_pd(Bcol + 4 * ldB + p);
                __m256d v5 = _mm256_loadu_pd(Bcol + 5 * ldB + p);
                __m256d t0 = _mm256_unpacklo_pd(v4, v5);
                __m256d t1 = _mm256_unpackhi_pd(v4, v5);

                _mm256_storeu_pd(dst + (p + 0) * NR + 0, r0);
                _mm_store_pd(dst + (p + 0) * NR + 4, _mm256_castpd256_pd128(t0));
                _mm256_storeu_pd(dst + (p + 1) * NR + 0, r1);
                _mm_store_pd(dst + (p + 1) * NR + 4, _mm256_castpd256_pd128(t1));
                _mm256_storeu_pd(dst + (p + 2) * NR + 0, r2);
                _mm_store_pd(dst + (p + 2) * NR + 4, _mm256_extractf128_pd(t0, 1));
                _mm256_storeu_pd(dst + (p + 3) * NR + 0, r3);
                _mm_store_pd(dst + (p + 3) * NR + 4, _mm256_extractf128_pd(t1, 1));
            }
            for (; p < kc; ++p)
                for (int j = 0; j < NR; ++j)
                    dst[p * NR + j] = Bcol[(size_t)j * ldB + p];
        }
        else {
            for (size_t j = 0; j < (size_t)nn; ++j) {
                const double* col = Bcol + j * ldB;
                for (size_t p = 0; p < kc; ++p)
                    dst[p * NR + j] = col[p];
            }
            for (size_t j = (size_t)nn; j < NR; ++j)
                for (size_t p = 0; p < kc; ++p)
                    dst[p * NR + j] = 0.0;
        }
    }
}

template<>
inline void __pack_BT_block<double, double>(
    const double* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    double* __restrict Bp)
{
    constexpr int NR = 6;

    for (size_t jp = 0; jp* NR < nc; ++jp) {
        const int nn = (int)((nc - jp * NR < (size_t)NR) ? (nc - jp * NR) : (size_t)NR);
        double* dst = Bp + jp * kc * NR;
        const double* Bcol = B + jp * NR;

        if (nn == NR) {
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                _mm256_storeu_pd(dst, _mm256_loadu_pd(src));
                _mm_storeu_pd(dst + 4, _mm_loadu_pd(src + 4));
                dst += NR;
            }
        }
        else {
            const __m256i m0 = _mm256_set_epi64x(
                nn > 3 ? -1LL : 0LL, nn > 2 ? -1LL : 0LL,
                nn > 1 ? -1LL : 0LL, nn > 0 ? -1LL : 0LL);
            const int rem = (nn > 4) ? (nn - 4) : 0;
            const __m128i m1 = _mm_set_epi64x(
                rem > 1 ? -1LL : 0LL, rem > 0 ? -1LL : 0LL);
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                _mm256_storeu_pd(dst, _mm256_maskload_pd(src, m0));
                _mm_storeu_pd(dst + 4, _mm_maskload_pd(src + 4, m1));
                dst += NR;
            }
        }
    }
}

template<>
inline void __pack_BT_block_d<double, double>(
    const double* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    const double* __restrict d,
    double* __restrict Bp)
{
    constexpr int NR = 6;

    for (size_t jp = 0; jp* NR < nc; ++jp) {
        const int nn = (int)((nc - jp * NR < (size_t)NR) ? (nc - jp * NR) : (size_t)NR);
        double* dst = Bp + jp * kc * NR;
        const double* Bcol = B + jp * NR;

        if (nn == NR) {
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                __m256d d256 = _mm256_set1_pd(d[p]);
                __m128d d128 = _mm_set1_pd(d[p]);
                _mm256_storeu_pd(dst, _mm256_mul_pd(_mm256_loadu_pd(src), d256));
                _mm_storeu_pd(dst + 4, _mm_mul_pd(_mm_loadu_pd(src + 4), d128));
                dst += NR;
            }
        }
        else {
            const __m256i m0 = _mm256_set_epi64x(
                nn > 3 ? -1LL : 0LL, nn > 2 ? -1LL : 0LL,
                nn > 1 ? -1LL : 0LL, nn > 0 ? -1LL : 0LL);
            const int rem = (nn > 4) ? (nn - 4) : 0;
            const __m128i m1 = _mm_set_epi64x(
                rem > 1 ? -1LL : 0LL, rem > 0 ? -1LL : 0LL);
            for (size_t p = 0; p < kc; ++p) {
                const double* src = Bcol + p * ldB;
                __m256d d256 = _mm256_set1_pd(d[p]);
                __m128d d128 = _mm_set1_pd(d[p]);
                _mm256_storeu_pd(dst,
                    _mm256_mul_pd(_mm256_maskload_pd(src, m0), d256));
                _mm_storeu_pd(dst + 4,
                    _mm_mul_pd(_mm_maskload_pd(src + 4, m1), d128));
                dst += NR;
            }
        }
    }
}

static void __micro_kernel_mul_packed_full(
    const double* __restrict Ap, const double* __restrict Bp, double* __restrict C,
    size_t ldC, size_t k)
{
    __m256d c00 = _mm256_loadu_pd(C + 0 * ldC + 0);
    __m256d c01 = _mm256_loadu_pd(C + 0 * ldC + 4);
    __m256d c10 = _mm256_loadu_pd(C + 1 * ldC + 0);
    __m256d c11 = _mm256_loadu_pd(C + 1 * ldC + 4);
    __m256d c20 = _mm256_loadu_pd(C + 2 * ldC + 0);
    __m256d c21 = _mm256_loadu_pd(C + 2 * ldC + 4);
    __m256d c30 = _mm256_loadu_pd(C + 3 * ldC + 0);
    __m256d c31 = _mm256_loadu_pd(C + 3 * ldC + 4);
    __m256d c40 = _mm256_loadu_pd(C + 4 * ldC + 0);
    __m256d c41 = _mm256_loadu_pd(C + 4 * ldC + 4);
    __m256d c50 = _mm256_loadu_pd(C + 5 * ldC + 0);
    __m256d c51 = _mm256_loadu_pd(C + 5 * ldC + 4);

    for (size_t p = 0; p < k; ++p) {
        const __m256d a0 = _mm256_load_pd(Ap + p * 8);
        const __m256d a1 = _mm256_load_pd(Ap + p * 8 + 4);

        __m256d b;
        b = _mm256_set1_pd(Bp[p * 6 + 0]);
        c00 = _mm256_fmadd_pd(a0, b, c00);
        c01 = _mm256_fmadd_pd(a1, b, c01);
        b = _mm256_set1_pd(Bp[p * 6 + 1]);
        c10 = _mm256_fmadd_pd(a0, b, c10);
        c11 = _mm256_fmadd_pd(a1, b, c11);
        b = _mm256_set1_pd(Bp[p * 6 + 2]);
        c20 = _mm256_fmadd_pd(a0, b, c20);
        c21 = _mm256_fmadd_pd(a1, b, c21);
        b = _mm256_set1_pd(Bp[p * 6 + 3]);
        c30 = _mm256_fmadd_pd(a0, b, c30);
        c31 = _mm256_fmadd_pd(a1, b, c31);
        b = _mm256_set1_pd(Bp[p * 6 + 4]);
        c40 = _mm256_fmadd_pd(a0, b, c40);
        c41 = _mm256_fmadd_pd(a1, b, c41);
        b = _mm256_set1_pd(Bp[p * 6 + 5]);
        c50 = _mm256_fmadd_pd(a0, b, c50);
        c51 = _mm256_fmadd_pd(a1, b, c51);
    }

    _mm256_storeu_pd(C + 0 * ldC + 0, c00);
    _mm256_storeu_pd(C + 0 * ldC + 4, c01);
    _mm256_storeu_pd(C + 1 * ldC + 0, c10);
    _mm256_storeu_pd(C + 1 * ldC + 4, c11);
    _mm256_storeu_pd(C + 2 * ldC + 0, c20);
    _mm256_storeu_pd(C + 2 * ldC + 4, c21);
    _mm256_storeu_pd(C + 3 * ldC + 0, c30);
    _mm256_storeu_pd(C + 3 * ldC + 4, c31);
    _mm256_storeu_pd(C + 4 * ldC + 0, c40);
    _mm256_storeu_pd(C + 4 * ldC + 4, c41);
    _mm256_storeu_pd(C + 5 * ldC + 0, c50);
    _mm256_storeu_pd(C + 5 * ldC + 4, c51);
}

static void __micro_kernel_mul_packed_edge(
    const double* __restrict Ap,
    const double* __restrict Bp,
    double* __restrict C, size_t ldC, size_t k,
    int mr, int nr)
{
    const __m256i m0 = _mm256_set_epi64x(
        mr > 3 ? -1LL : 0LL, mr > 2 ? -1LL : 0LL,
        mr > 1 ? -1LL : 0LL, mr > 0 ? -1LL : 0LL);
    const __m256i m1 = _mm256_set_epi64x(
        mr > 7 ? -1LL : 0LL, mr > 6 ? -1LL : 0LL,
        mr > 5 ? -1LL : 0LL, mr > 4 ? -1LL : 0LL);

    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        c00 = _mm256_maskload_pd(C + off + 0, m0);
        c01 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        c10 = _mm256_maskload_pd(C + off + 0, m0);
        c11 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        c20 = _mm256_maskload_pd(C + off + 0, m0);
        c21 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        c30 = _mm256_maskload_pd(C + off + 0, m0);
        c31 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        c40 = _mm256_maskload_pd(C + off + 0, m0);
        c41 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        c50 = _mm256_maskload_pd(C + off + 0, m0);
        c51 = _mm256_maskload_pd(C + off + 4, m1);
    }

    for (size_t p = 0; p < k; ++p) {
        const __m256d a0 = _mm256_load_pd(Ap + p * 8);
        const __m256d a1 = _mm256_load_pd(Ap + p * 8 + 4);

        __m256d b;
        b = _mm256_set1_pd(Bp[p * 6 + 0]);
        c00 = _mm256_fmadd_pd(a0, b, c00);
        c01 = _mm256_fmadd_pd(a1, b, c01);
        b = _mm256_set1_pd(Bp[p * 6 + 1]);
        c10 = _mm256_fmadd_pd(a0, b, c10);
        c11 = _mm256_fmadd_pd(a1, b, c11);
        b = _mm256_set1_pd(Bp[p * 6 + 2]);
        c20 = _mm256_fmadd_pd(a0, b, c20);
        c21 = _mm256_fmadd_pd(a1, b, c21);
        b = _mm256_set1_pd(Bp[p * 6 + 3]);
        c30 = _mm256_fmadd_pd(a0, b, c30);
        c31 = _mm256_fmadd_pd(a1, b, c31);
        b = _mm256_set1_pd(Bp[p * 6 + 4]);
        c40 = _mm256_fmadd_pd(a0, b, c40);
        c41 = _mm256_fmadd_pd(a1, b, c41);
        b = _mm256_set1_pd(Bp[p * 6 + 5]);
        c50 = _mm256_fmadd_pd(a0, b, c50);
        c51 = _mm256_fmadd_pd(a1, b, c51);
    }

    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c00);
        _mm256_maskstore_pd(C + off + 4, m1, c01);
    }
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c10);
        _mm256_maskstore_pd(C + off + 4, m1, c11);
    }
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c20);
        _mm256_maskstore_pd(C + off + 4, m1, c21);
    }
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c30);
        _mm256_maskstore_pd(C + off + 4, m1, c31);
    }
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c40);
        _mm256_maskstore_pd(C + off + 4, m1, c41);
    }
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c50);
        _mm256_maskstore_pd(C + off + 4, m1, c51);
    }
}


static void __micro_kernel_nmul_packed_full(
    const double* __restrict Ap, const double* __restrict Bp, double* __restrict C,
    size_t ldC, size_t k)
{
    __m256d c00 = _mm256_loadu_pd(C + 0 * ldC + 0);
    __m256d c01 = _mm256_loadu_pd(C + 0 * ldC + 4);
    __m256d c10 = _mm256_loadu_pd(C + 1 * ldC + 0);
    __m256d c11 = _mm256_loadu_pd(C + 1 * ldC + 4);
    __m256d c20 = _mm256_loadu_pd(C + 2 * ldC + 0);
    __m256d c21 = _mm256_loadu_pd(C + 2 * ldC + 4);
    __m256d c30 = _mm256_loadu_pd(C + 3 * ldC + 0);
    __m256d c31 = _mm256_loadu_pd(C + 3 * ldC + 4);
    __m256d c40 = _mm256_loadu_pd(C + 4 * ldC + 0);
    __m256d c41 = _mm256_loadu_pd(C + 4 * ldC + 4);
    __m256d c50 = _mm256_loadu_pd(C + 5 * ldC + 0);
    __m256d c51 = _mm256_loadu_pd(C + 5 * ldC + 4);

    for (size_t p = 0; p < k; ++p) {
        const __m256d a0 = _mm256_load_pd(Ap + p * 8);
        const __m256d a1 = _mm256_load_pd(Ap + p * 8 + 4);

        __m256d b;
        b = _mm256_set1_pd(Bp[p * 6 + 0]);
        c00 = _mm256_fnmadd_pd(a0, b, c00);
        c01 = _mm256_fnmadd_pd(a1, b, c01);
        b = _mm256_set1_pd(Bp[p * 6 + 1]);
        c10 = _mm256_fnmadd_pd(a0, b, c10);
        c11 = _mm256_fnmadd_pd(a1, b, c11);
        b = _mm256_set1_pd(Bp[p * 6 + 2]);
        c20 = _mm256_fnmadd_pd(a0, b, c20);
        c21 = _mm256_fnmadd_pd(a1, b, c21);
        b = _mm256_set1_pd(Bp[p * 6 + 3]);
        c30 = _mm256_fnmadd_pd(a0, b, c30);
        c31 = _mm256_fnmadd_pd(a1, b, c31);
        b = _mm256_set1_pd(Bp[p * 6 + 4]);
        c40 = _mm256_fnmadd_pd(a0, b, c40);
        c41 = _mm256_fnmadd_pd(a1, b, c41);
        b = _mm256_set1_pd(Bp[p * 6 + 5]);
        c50 = _mm256_fnmadd_pd(a0, b, c50);
        c51 = _mm256_fnmadd_pd(a1, b, c51);
    }

    _mm256_storeu_pd(C + 0 * ldC + 0, c00);
    _mm256_storeu_pd(C + 0 * ldC + 4, c01);
    _mm256_storeu_pd(C + 1 * ldC + 0, c10);
    _mm256_storeu_pd(C + 1 * ldC + 4, c11);
    _mm256_storeu_pd(C + 2 * ldC + 0, c20);
    _mm256_storeu_pd(C + 2 * ldC + 4, c21);
    _mm256_storeu_pd(C + 3 * ldC + 0, c30);
    _mm256_storeu_pd(C + 3 * ldC + 4, c31);
    _mm256_storeu_pd(C + 4 * ldC + 0, c40);
    _mm256_storeu_pd(C + 4 * ldC + 4, c41);
    _mm256_storeu_pd(C + 5 * ldC + 0, c50);
    _mm256_storeu_pd(C + 5 * ldC + 4, c51);
}

static void __micro_kernel_nmul_packed_edge(
    const double* __restrict Ap,
    const double* __restrict Bp,
    double* __restrict C, size_t ldC, size_t k,
    int mr, int nr)
{
    const __m256i m0 = _mm256_set_epi64x(
        mr > 3 ? -1LL : 0LL, mr > 2 ? -1LL : 0LL,
        mr > 1 ? -1LL : 0LL, mr > 0 ? -1LL : 0LL);
    const __m256i m1 = _mm256_set_epi64x(
        mr > 7 ? -1LL : 0LL, mr > 6 ? -1LL : 0LL,
        mr > 5 ? -1LL : 0LL, mr > 4 ? -1LL : 0LL);

    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        c00 = _mm256_maskload_pd(C + off + 0, m0);
        c01 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        c10 = _mm256_maskload_pd(C + off + 0, m0);
        c11 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        c20 = _mm256_maskload_pd(C + off + 0, m0);
        c21 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        c30 = _mm256_maskload_pd(C + off + 0, m0);
        c31 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        c40 = _mm256_maskload_pd(C + off + 0, m0);
        c41 = _mm256_maskload_pd(C + off + 4, m1);
    }
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        c50 = _mm256_maskload_pd(C + off + 0, m0);
        c51 = _mm256_maskload_pd(C + off + 4, m1);
    }

    for (size_t p = 0; p < k; ++p) {
        const __m256d a0 = _mm256_load_pd(Ap + p * 8);
        const __m256d a1 = _mm256_load_pd(Ap + p * 8 + 4);

        __m256d b;
        b = _mm256_set1_pd(Bp[p * 6 + 0]);
        c00 = _mm256_fnmadd_pd(a0, b, c00);
        c01 = _mm256_fnmadd_pd(a1, b, c01);
        b = _mm256_set1_pd(Bp[p * 6 + 1]);
        c10 = _mm256_fnmadd_pd(a0, b, c10);
        c11 = _mm256_fnmadd_pd(a1, b, c11);
        b = _mm256_set1_pd(Bp[p * 6 + 2]);
        c20 = _mm256_fnmadd_pd(a0, b, c20);
        c21 = _mm256_fnmadd_pd(a1, b, c21);
        b = _mm256_set1_pd(Bp[p * 6 + 3]);
        c30 = _mm256_fnmadd_pd(a0, b, c30);
        c31 = _mm256_fnmadd_pd(a1, b, c31);
        b = _mm256_set1_pd(Bp[p * 6 + 4]);
        c40 = _mm256_fnmadd_pd(a0, b, c40);
        c41 = _mm256_fnmadd_pd(a1, b, c41);
        b = _mm256_set1_pd(Bp[p * 6 + 5]);
        c50 = _mm256_fnmadd_pd(a0, b, c50);
        c51 = _mm256_fnmadd_pd(a1, b, c51);
    }

    if (0 < nr) {
        const size_t off = (size_t)(0) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c00);
        _mm256_maskstore_pd(C + off + 4, m1, c01);
    }
    if (1 < nr) {
        const size_t off = (size_t)(1) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c10);
        _mm256_maskstore_pd(C + off + 4, m1, c11);
    }
    if (2 < nr) {
        const size_t off = (size_t)(2) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c20);
        _mm256_maskstore_pd(C + off + 4, m1, c21);
    }
    if (3 < nr) {
        const size_t off = (size_t)(3) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c30);
        _mm256_maskstore_pd(C + off + 4, m1, c31);
    }
    if (4 < nr) {
        const size_t off = (size_t)(4) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c40);
        _mm256_maskstore_pd(C + off + 4, m1, c41);
    }
    if (5 < nr) {
        const size_t off = (size_t)(5) * ldC;
        _mm256_maskstore_pd(C + off + 0, m0, c50);
        _mm256_maskstore_pd(C + off + 4, m1, c51);
    }
}

#endif
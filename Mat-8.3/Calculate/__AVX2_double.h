
inline void __ApplyGivensOnTheRight(double* __restrict a, double* __restrict b, double c, double s, int len) {
    __m256d cc = _mm256_set1_pd(c);
    __m256d ss = _mm256_set1_pd(s);
    int n = len / 4;
    int i;
    for (i = 0; i < n; ++i) {
        __m256d ai = ((__m256d*)a)[i];
        __m256d bi = ((__m256d*)b)[i];
        ((__m256d*)a)[i] = _mm256_fmadd_pd(ai, cc, _mm256_mul_pd(bi, ss));
        ((__m256d*)b)[i] = _mm256_fnmadd_pd(ai, ss, _mm256_mul_pd(bi, cc));
    }
    n = len - n * 4;
    if (n != 0) {
        __m256d maskd = _mm256_castsi256_pd(_mm256_setr_epi64x(
            n > 0 ? -1LL : 0LL,
            n > 1 ? -1LL : 0LL,
            n > 2 ? -1LL : 0LL,
            n > 3 ? -1LL : 0LL
        ));
        __m256d ai = ((__m256d*)a)[i];
        __m256d bi = ((__m256d*)b)[i];
        ((__m256d*)a)[i] = _mm256_blendv_pd(ai, _mm256_fmadd_pd(ai, cc, _mm256_mul_pd(bi, ss)), maskd);
        ((__m256d*)b)[i] = _mm256_blendv_pd(bi, _mm256_fnmadd_pd(ai, ss, _mm256_mul_pd(bi, cc)), maskd);
    }
}

inline void __Schur_Householder_Right(
    double* __restrict a, double* __restrict b, double* __restrict c,
    const double* __restrict w, int len) {
    __m256d w0 = _mm256_set1_pd(w[0]);
    __m256d w1 = _mm256_set1_pd(w[1]);
    __m256d w2 = _mm256_set1_pd(w[2]);

    int n = len / 4;
    int i;
    for (i = 0; i < n; ++i) {
        __m256d sum =
            _mm256_fmadd_pd(((__m256d*)a)[i], w0,
                _mm256_fmadd_pd(((__m256d*)b)[i], w1,
                    _mm256_mul_pd(((__m256d*)c)[i], w2)));
        ((__m256d*)a)[i] = _mm256_fnmadd_pd(sum, w0, ((__m256d*)a)[i]);
        ((__m256d*)b)[i] = _mm256_fnmadd_pd(sum, w1, ((__m256d*)b)[i]);
        ((__m256d*)c)[i] = _mm256_fnmadd_pd(sum, w2, ((__m256d*)c)[i]);
    }
    n = len - n * 4;
    if (n != 0) {
        __m256d maskd = _mm256_castsi256_pd(_mm256_setr_epi64x(
            n > 0 ? -1LL : 0LL,
            n > 1 ? -1LL : 0LL,
            n > 2 ? -1LL : 0LL,
            n > 3 ? -1LL : 0LL
        ));
        __m256d ai = ((__m256d*)a)[i];
        __m256d bi = ((__m256d*)b)[i];
        __m256d ci = ((__m256d*)c)[i];
        __m256d sum = _mm256_blendv_pd(_mm256_setzero_pd(),
            _mm256_fmadd_pd(ai, w0,
                _mm256_fmadd_pd(bi, w1,
                    _mm256_mul_pd(ci, w2))), maskd);
        ((__m256d*)a)[i] = _mm256_fnmadd_pd(sum, w0, ai);
        ((__m256d*)b)[i] = _mm256_fnmadd_pd(sum, w1, bi);
        ((__m256d*)c)[i] = _mm256_fnmadd_pd(sum, w2, ci);
    }
}

inline void __times(double* __restrict a, double b, double* __restrict ans, int len) {
    __m256d bb = _mm256_set1_pd(b);
    int n = len / 4;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m256d*)ans)[i] = _mm256_mul_pd(((__m256d*)a)[i], bb);
    }
    n = len - n * 4;
    if (n != 0) {
        __m256d maskd = _mm256_castsi256_pd(_mm256_setr_epi64x(
            n > 0 ? -1LL : 0LL,
            n > 1 ? -1LL : 0LL,
            n > 2 ? -1LL : 0LL,
            n > 3 ? -1LL : 0LL
        ));
        ((__m256d*)ans)[i] = _mm256_blendv_pd(((__m256d*)ans)[i], _mm256_mul_pd(((__m256d*)a)[i], bb), maskd);
    }
}

inline void __add(double* __restrict a, double* __restrict b, double* __restrict ans, int len) {
    int n = len / 4;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m256d*)ans)[i] = _mm256_add_pd(((__m256d*)a)[i], ((__m256d*)b)[i]);
    }
    n = len - n * 4;
    if (n != 0) {
        __m256d maskd = _mm256_castsi256_pd(_mm256_setr_epi64x(
            n > 0 ? -1LL : 0LL,
            n > 1 ? -1LL : 0LL,
            n > 2 ? -1LL : 0LL,
            n > 3 ? -1LL : 0LL
        ));
        ((__m256d*)ans)[i] = _mm256_blendv_pd(((__m256d*)ans)[i], _mm256_add_pd(((__m256d*)a)[i], ((__m256d*)b)[i]), maskd);
    }
}

inline void __mixed(double* __restrict a, double* __restrict b, double q, double* __restrict ans, int len) {
    __m256d qq = _mm256_set1_pd(q);
    int n = len / 4;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m256d*)ans)[i] = _mm256_fmadd_pd(((__m256d*)b)[i], qq, ((__m256d*)a)[i]);
    }
    n = len - n * 4;
    if (n != 0) {
        __m256d maskd = _mm256_castsi256_pd(_mm256_setr_epi64x(
            n > 0 ? -1LL : 0LL,
            n > 1 ? -1LL : 0LL,
            n > 2 ? -1LL : 0LL,
            n > 3 ? -1LL : 0LL
        ));
        ((__m256d*)ans)[i] = _mm256_blendv_pd(((__m256d*)ans)[i], _mm256_fmadd_pd(((__m256d*)b)[i], qq, ((__m256d*)a)[i]), maskd);
    }
}

inline void __swap(double* __restrict a, double* __restrict b, int len) {
    int n = len / 4;
    int i;
    for (i = 0; i < n; ++i) {
        __m256d temp = ((__m256d*)a)[i];
        ((__m256d*)a)[i] = ((__m256d*)b)[i];
        ((__m256d*)b)[i] = temp;
    }
    n = len - n * 4;
    if (n != 0) {
        __m256d maskd = _mm256_castsi256_pd(_mm256_setr_epi64x(
            n > 0 ? -1LL : 0LL,
            n > 1 ? -1LL : 0LL,
            n > 2 ? -1LL : 0LL,
            n > 3 ? -1LL : 0LL
        ));
        __m256d temp = ((__m256d*)a)[i];
        ((__m256d*)a)[i] = _mm256_blendv_pd(temp, ((__m256d*)b)[i], maskd);
        ((__m256d*)b)[i] = _mm256_blendv_pd(((__m256d*)b)[i], temp, maskd);
    }
}

inline double __dot(double* __restrict a, double* __restrict b, int len) {
    __m256d ans0 = _mm256_setzero_pd();
    __m256d ans1 = _mm256_setzero_pd();
    __m256d ans2 = _mm256_setzero_pd();
    __m256d ans3 = _mm256_setzero_pd();
    int n = len / 4;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        ans0 = _mm256_fmadd_pd(((__m256d*)a)[i], ((__m256d*)b)[i], ans0);
        ans1 = _mm256_fmadd_pd(((__m256d*)a)[i + 1], ((__m256d*)b)[i + 1], ans1);
        ans2 = _mm256_fmadd_pd(((__m256d*)a)[i + 2], ((__m256d*)b)[i + 2], ans2);
        ans3 = _mm256_fmadd_pd(((__m256d*)a)[i + 3], ((__m256d*)b)[i + 3], ans3);
    }
    for (; i < n; ++i) {
        ans0 = _mm256_fmadd_pd(((__m256d*)a)[i], ((__m256d*)b)[i], ans0);
    }
    __m256d sum = _mm256_add_pd(_mm256_add_pd(ans0, ans1), _mm256_add_pd(ans2, ans3));

    n = len - n * 4;
    if (n != 0) {
        __m256d maskd = _mm256_castsi256_pd(_mm256_setr_epi64x(
            n > 0 ? -1LL : 0LL,
            n > 1 ? -1LL : 0LL,
            n > 2 ? -1LL : 0LL,
            n > 3 ? -1LL : 0LL
        ));
        sum = _mm256_blendv_pd(sum, _mm256_fmadd_pd(((__m256d*)a)[i], ((__m256d*)b)[i], sum), maskd);
    }

    __m128d low = _mm256_castpd256_pd128(sum);
    __m128d high = _mm256_extractf128_pd(sum, 1);
    __m128d pair_sum = _mm_add_pd(low, high);
    pair_sum = _mm_hadd_pd(pair_sum, pair_sum);
    return _mm_cvtsd_f64(pair_sum);
}

inline double __MaxAbs(double* a, int len) {
    __m256d sign_mask = _mm256_set1_pd(-0.0);
    __m256d max0 = _mm256_setzero_pd();
    __m256d max1 = _mm256_setzero_pd();
    __m256d max2 = _mm256_setzero_pd();
    __m256d max3 = _mm256_setzero_pd();
    int n = len / 4;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        max0 = _mm256_max_pd(max0, _mm256_andnot_pd(sign_mask, ((__m256d*)a)[i]));
        max1 = _mm256_max_pd(max1, _mm256_andnot_pd(sign_mask, ((__m256d*)a)[i + 1]));
        max2 = _mm256_max_pd(max2, _mm256_andnot_pd(sign_mask, ((__m256d*)a)[i + 2]));
        max3 = _mm256_max_pd(max3, _mm256_andnot_pd(sign_mask, ((__m256d*)a)[i + 3]));
    }
    for (; i < n; ++i) {
        max0 = _mm256_max_pd(max0, _mm256_andnot_pd(sign_mask, ((__m256d*)a)[i]));
    }
    __m256d maxv = _mm256_max_pd(_mm256_max_pd(max0, max1), _mm256_max_pd(max2, max3));

    n = len - n * 4;
    if (n != 0) {
        __m256d maskd = _mm256_castsi256_pd(_mm256_setr_epi64x(
            n > 0 ? -1LL : 0LL,
            n > 1 ? -1LL : 0LL,
            n > 2 ? -1LL : 0LL,
            n > 3 ? -1LL : 0LL
        ));
        maxv = _mm256_blendv_pd(maxv, _mm256_max_pd(maxv, _mm256_andnot_pd(sign_mask, ((__m256d*)a)[i])), maskd);
    }

    __m128d low = _mm256_castpd256_pd128(maxv);
    __m128d high = _mm256_extractf128_pd(maxv, 1);
    __m128d max128 = _mm_max_pd(low, high);
    __m128d swapped = _mm_permute_pd(max128, 0x1);
    max128 = _mm_max_pd(max128, swapped);
    return _mm_cvtsd_f64(max128);
}
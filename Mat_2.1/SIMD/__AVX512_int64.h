inline void __ApplyGivensOnTheRight(int64_t* __restrict a, int64_t* __restrict b, int64_t c, int64_t s, int len) {
    __m512i ai, bi;
    __m512i cc(_mm512_set1_epi64(c)), ss(_mm512_set1_epi64(s));
    int n = len / 8;
    int i;
    for (i = 0; i < n; ++i) {
        ai = ((__m512i*)a)[i];
        bi = ((__m512i*)b)[i];
        ((__m512i*)a)[i] = _mm512_add_epi64(_mm512_mullo_epi64(ai, cc), _mm512_mullo_epi64(bi, ss));
        ((__m512i*)b)[i] = _mm512_sub_epi64(_mm512_mullo_epi64(bi, cc), _mm512_mullo_epi64(ai, ss));
    }

    n = len - n * 8;
    if (n != 0) {
        __mmask8 mask = (1 << n) - 1;
        ai = ((__m512i*)a)[i];
        bi = ((__m512i*)b)[i];
        ((__m512i*)a)[i] = _mm512_mask_blend_epi64(mask, ai, _mm512_add_epi64(_mm512_mullo_epi64(ai, cc), _mm512_mullo_epi64(bi, ss)));
        ((__m512i*)b)[i] = _mm512_mask_blend_epi64(mask, bi, _mm512_sub_epi64(_mm512_mullo_epi64(bi, cc), _mm512_mullo_epi64(ai, ss)));
    }
}

inline void __Schur_Householder_Right(
    int64_t* __restrict a, int64_t* __restrict b, int64_t* __restrict c,
    const int64_t* __restrict w, int len) {
    __m512i sum;
    __m512i w0(_mm512_set1_epi64(w[0])), w1(_mm512_set1_epi64(w[1])), w2(_mm512_set1_epi64(w[2]));

    int n = len / 8;
    int i;
    for (i = 0; i < n; ++i) {
        sum =
            _mm512_add_epi64(
                _mm512_mullo_epi64(((__m512i*)a)[i], w0),
                _mm512_add_epi64(
                    _mm512_mullo_epi64(((__m512i*)b)[i], w1),
                    _mm512_mullo_epi64(((__m512i*)c)[i], w2)
                )
            );
        ((__m512i*)a)[i] = _mm512_sub_epi64(((__m512i*)a)[i], _mm512_mullo_epi64(sum, w0));
        ((__m512i*)b)[i] = _mm512_sub_epi64(((__m512i*)b)[i], _mm512_mullo_epi64(sum, w1));
        ((__m512i*)c)[i] = _mm512_sub_epi64(((__m512i*)c)[i], _mm512_mullo_epi64(sum, w2));
    }
    n = len - n * 8;
    if (n != 0) {
        sum = _mm512_mask_blend_epi64((1 << n) - 1, _mm512_setzero_si512(),
            _mm512_add_epi64(
                _mm512_mullo_epi64(((__m512i*)a)[i], w0),
                _mm512_add_epi64(
                    _mm512_mullo_epi64(((__m512i*)b)[i], w1),
                    _mm512_mullo_epi64(((__m512i*)c)[i], w2)
                )
            ));
        ((__m512i*)a)[i] = _mm512_sub_epi64(((__m512i*)a)[i], _mm512_mullo_epi64(sum, w0));
        ((__m512i*)b)[i] = _mm512_sub_epi64(((__m512i*)b)[i], _mm512_mullo_epi64(sum, w1));
        ((__m512i*)c)[i] = _mm512_sub_epi64(((__m512i*)c)[i], _mm512_mullo_epi64(sum, w2));
    }
}

inline void __times(int64_t* __restrict a, int64_t b, int64_t* __restrict ans, int len) {
    __m512i bb(_mm512_set1_epi64(b));

    int n = len / 8;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m512i*)ans)[i] = _mm512_mullo_epi64(((__m512i*)a)[i], bb);
    }
    n = len - n * 8;
    if (n != 0) {
        ((__m512i*)ans)[i] = _mm512_mask_blend_epi64((1 << n) - 1, ((__m512i*)ans)[i], _mm512_mullo_epi64(((__m512i*)a)[i], bb));
    }
}

inline void __add(int64_t* __restrict a, int64_t* __restrict b, int64_t* __restrict ans, int len) {
    int n = len / 8;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m512i*)ans)[i] = _mm512_add_epi64(((__m512i*)a)[i], ((__m512i*)b)[i]);
    }
    n = len - n * 8;
    if (n != 0) {
        ((__m512i*)ans)[i] = _mm512_mask_blend_epi64((1 << n) - 1, ((__m512i*)ans)[i], _mm512_add_epi64(((__m512i*)a)[i], ((__m512i*)b)[i]));
    }
}

//ans=a+b*p
inline void __mixed(int64_t* __restrict a, int64_t* __restrict b, int64_t q, int64_t* __restrict ans, int len) {
    __m512i qq(_mm512_set1_epi64(q));
    int n = len / 8;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m512i*)ans)[i] = _mm512_add_epi64(((__m512i*)a)[i], _mm512_mullo_epi64(((__m512i*)b)[i], qq));
    }
    n = len - n * 8;
    if (n != 0) {
        ((__m512i*)ans)[i] = _mm512_mask_blend_epi64((1 << n) - 1, ((__m512i*)ans)[i], _mm512_add_epi64(((__m512i*)a)[i], _mm512_mullo_epi64(((__m512i*)b)[i], qq)));
    }
}

inline void __swap(int64_t* __restrict a, int64_t* __restrict b, int len) {
    int n = len / 8;
    int i;
    for (i = 0; i < n; ++i) {
        __m512i temp = ((__m512i*)a)[i];
        ((__m512i*)a)[i] = ((__m512i*)b)[i];
        ((__m512i*)b)[i] = temp;
    }
    n = len - n * 8;
    if (n != 0) {
        __mmask8 mask = (1 << n) - 1;
        __m512i temp = ((__m512i*)a)[i];
        ((__m512i*)a)[i] = _mm512_mask_blend_epi64(mask, temp, ((__m512i*)b)[i]);
        ((__m512i*)b)[i] = _mm512_mask_blend_epi64(mask, ((__m512i*)b)[i], temp);
    }
}

inline int64_t __dot(int64_t* __restrict a, int64_t* __restrict b, int len) {
    __m512i ans0 = _mm512_setzero_si512();
    __m512i ans1 = _mm512_setzero_si512();
    __m512i ans2 = _mm512_setzero_si512();
    __m512i ans3 = _mm512_setzero_si512();
    int n = len / 8;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        ans0 = _mm512_add_epi64(_mm512_mullo_epi64(((__m512i*)a)[i], ((__m512i*)b)[i]), ans0);
        ans1 = _mm512_add_epi64(_mm512_mullo_epi64(((__m512i*)a)[i + 1], ((__m512i*)b)[i + 1]), ans1);
        ans2 = _mm512_add_epi64(_mm512_mullo_epi64(((__m512i*)a)[i + 2], ((__m512i*)b)[i + 2]), ans2);
        ans3 = _mm512_add_epi64(_mm512_mullo_epi64(((__m512i*)a)[i + 3], ((__m512i*)b)[i + 3]), ans3);
    }
    for (; i < n; ++i) {
        ans0 = _mm512_add_epi64(_mm512_mullo_epi64(((__m512i*)a)[i], ((__m512i*)b)[i]), ans0);
    }
    __m512i ans = _mm512_add_epi64(_mm512_add_epi64(ans0, ans1), _mm512_add_epi64(ans2, ans3));

    n = len - n * 8;
    if (n != 0) {
        ans = _mm512_mask_blend_epi64((1 << n) - 1, ans, _mm512_add_epi64(_mm512_mullo_epi64(((__m512i*)a)[i], ((__m512i*)b)[i]), ans));
    }
    return _mm512_reduce_add_epi64(ans);
}

int64_t __MaxAbs(int64_t* a, int len) {
    int n = len / 8;
    __m512i max0 = _mm512_setzero_si512();
    __m512i max1 = _mm512_setzero_si512();
    __m512i max2 = _mm512_setzero_si512();
    __m512i max3 = _mm512_setzero_si512();
    int i = 0;
    for (; i <= n - 4; i += 4) {
        max0 = _mm512_max_epi64(max0, _mm512_abs_epi64(((__m512i*)a)[i]));
        max1 = _mm512_max_epi64(max1, _mm512_abs_epi64(((__m512i*)a)[i + 1]));
        max2 = _mm512_max_epi64(max2, _mm512_abs_epi64(((__m512i*)a)[i + 2]));
        max3 = _mm512_max_epi64(max3, _mm512_abs_epi64(((__m512i*)a)[i + 3]));
    }
    for (; i < n; ++i) {
        max0 = _mm512_max_epi64(max0, _mm512_abs_epi64(((__m512i*)a)[i]));
    }
    __m512i max = _mm512_max_epi64(_mm512_max_epi64(max0, max1), _mm512_max_epi64(max2, max3));

    n = len - n * 8;
    if (n != 0) {
        max = _mm512_max_epi64(max,
            _mm512_mask_blend_epi64((1 << n) - 1, _mm512_setzero_si512(), _mm512_abs_epi64(((__m512i*)a)[i]))
        );
    }
    return _mm512_reduce_max_epi64(max);
}
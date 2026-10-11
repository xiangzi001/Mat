
inline void __times(int32_t* __restrict a, int32_t b, int32_t* __restrict ans, int len) {
    __m256i bb = _mm256_set1_epi32(b);
    int n = len / 8;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m256i*)ans)[i] = _mm256_mullo_epi32(((__m256i*)a)[i], bb);
    }
    n = len - n * 8;
    if (n != 0) {
        __m256i mask = _mm256_setr_epi32(
            n > 0 ? -1 : 0,
            n > 1 ? -1 : 0,
            n > 2 ? -1 : 0,
            n > 3 ? -1 : 0,
            n > 4 ? -1 : 0,
            n > 5 ? -1 : 0,
            n > 6 ? -1 : 0,
            n > 7 ? -1 : 0
        );
        __m256i result = _mm256_mullo_epi32(((__m256i*)a)[i], bb);
        ((__m256i*)ans)[i] = _mm256_blendv_epi8(((__m256i*)ans)[i], result, mask);
    }
}

inline void __add(int32_t* __restrict a, int32_t* __restrict b, int32_t* __restrict ans, int len) {
    int n = len / 8;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m256i*)ans)[i] = _mm256_add_epi32(((__m256i*)a)[i], ((__m256i*)b)[i]);
    }
    n = len - n * 8;
    if (n != 0) {
        __m256i mask = _mm256_setr_epi32(
            n > 0 ? -1 : 0,
            n > 1 ? -1 : 0,
            n > 2 ? -1 : 0,
            n > 3 ? -1 : 0,
            n > 4 ? -1 : 0,
            n > 5 ? -1 : 0,
            n > 6 ? -1 : 0,
            n > 7 ? -1 : 0
        );
        __m256i result = _mm256_add_epi32(((__m256i*)a)[i], ((__m256i*)b)[i]);
        ((__m256i*)ans)[i] = _mm256_blendv_epi8(((__m256i*)ans)[i], result, mask);
    }
}

inline void __mixed(int32_t* __restrict a, int32_t* __restrict b, int32_t q, int32_t* __restrict ans, int len) {
    __m256i qq = _mm256_set1_epi32(q);
    int n = len / 8;
    int i;
    for (i = 0; i < n; ++i) {
        __m256i prod = _mm256_mullo_epi32(((__m256i*)b)[i], qq);
        ((__m256i*)ans)[i] = _mm256_add_epi32(((__m256i*)a)[i], prod);
    }
    n = len - n * 8;
    if (n != 0) {
        __m256i mask = _mm256_setr_epi32(
            n > 0 ? -1 : 0,
            n > 1 ? -1 : 0,
            n > 2 ? -1 : 0,
            n > 3 ? -1 : 0,
            n > 4 ? -1 : 0,
            n > 5 ? -1 : 0,
            n > 6 ? -1 : 0,
            n > 7 ? -1 : 0
        );
        __m256i prod = _mm256_mullo_epi32(((__m256i*)b)[i], qq);
        __m256i result = _mm256_add_epi32(((__m256i*)a)[i], prod);
        ((__m256i*)ans)[i] = _mm256_blendv_epi8(((__m256i*)ans)[i], result, mask);
    }
}

inline void __swap(int32_t* __restrict a, int32_t* __restrict b, int len) {
    int n = len / 8;
    int i;
    for (i = 0; i < n; ++i) {
        __m256i temp = ((__m256i*)a)[i];
        ((__m256i*)a)[i] = ((__m256i*)b)[i];
        ((__m256i*)b)[i] = temp;
    }
    n = len - n * 8;
    if (n != 0) {
        __m256i mask = _mm256_setr_epi32(
            n > 0 ? -1 : 0,
            n > 1 ? -1 : 0,
            n > 2 ? -1 : 0,
            n > 3 ? -1 : 0,
            n > 4 ? -1 : 0,
            n > 5 ? -1 : 0,
            n > 6 ? -1 : 0,
            n > 7 ? -1 : 0
        );
        __m256i temp = ((__m256i*)a)[i];
        ((__m256i*)a)[i] = _mm256_blendv_epi8(temp, ((__m256i*)b)[i], mask);
        ((__m256i*)b)[i] = _mm256_blendv_epi8(((__m256i*)b)[i], temp, mask);
    }
}

inline int32_t __dot(int32_t* __restrict a, int32_t* __restrict b, int len) {
    __m256i ans0 = _mm256_setzero_si256();
    __m256i ans1 = _mm256_setzero_si256();
    __m256i ans2 = _mm256_setzero_si256();
    __m256i ans3 = _mm256_setzero_si256();
    int n = len / 8;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        ans0 = _mm256_add_epi32(ans0, _mm256_mullo_epi32(((__m256i*)a)[i], ((__m256i*)b)[i]));
        ans1 = _mm256_add_epi32(ans1, _mm256_mullo_epi32(((__m256i*)a)[i + 1], ((__m256i*)b)[i + 1]));
        ans2 = _mm256_add_epi32(ans2, _mm256_mullo_epi32(((__m256i*)a)[i + 2], ((__m256i*)b)[i + 2]));
        ans3 = _mm256_add_epi32(ans3, _mm256_mullo_epi32(((__m256i*)a)[i + 3], ((__m256i*)b)[i + 3]));
    }
    for (; i < n; ++i) {
        ans0 = _mm256_add_epi32(ans0, _mm256_mullo_epi32(((__m256i*)a)[i], ((__m256i*)b)[i]));
    }
    __m256i sum = _mm256_add_epi32(_mm256_add_epi32(ans0, ans1), _mm256_add_epi32(ans2, ans3));

    n = len - n * 8;
    if (n != 0) {
        __m256i mask = _mm256_setr_epi32(
            n > 0 ? -1 : 0,
            n > 1 ? -1 : 0,
            n > 2 ? -1 : 0,
            n > 3 ? -1 : 0,
            n > 4 ? -1 : 0,
            n > 5 ? -1 : 0,
            n > 6 ? -1 : 0,
            n > 7 ? -1 : 0
        );
        __m256i prod = _mm256_mullo_epi32(((__m256i*)a)[i], ((__m256i*)b)[i]);
        __m256i new_sum = _mm256_add_epi32(sum, prod);
        sum = _mm256_blendv_epi8(sum, new_sum, mask);
    }

    // 将 8 个 32 位整数水平相加
    __m128i low = _mm256_castsi256_si128(sum);
    __m128i high = _mm256_extracti128_si256(sum, 1);
    __m128i pair_sum = _mm_add_epi32(low, high);
    // 两次水平相加将 4 个元素合并为 1 个（低 32 位有效）
    pair_sum = _mm_hadd_epi32(pair_sum, pair_sum);
    pair_sum = _mm_hadd_epi32(pair_sum, pair_sum);
    return _mm_cvtsi128_si32(pair_sum);
}

inline int32_t __MaxAbs(int32_t* a, int len) {
    __m256i sign_mask = _mm256_set1_epi32(0x80000000);
    __m256i max0 = _mm256_setzero_si256();
    __m256i max1 = _mm256_setzero_si256();
    __m256i max2 = _mm256_setzero_si256();
    __m256i max3 = _mm256_setzero_si256();
    int n = len / 8;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        max0 = _mm256_max_epi32(max0, _mm256_andnot_si256(sign_mask, ((__m256i*)a)[i]));
        max1 = _mm256_max_epi32(max1, _mm256_andnot_si256(sign_mask, ((__m256i*)a)[i + 1]));
        max2 = _mm256_max_epi32(max2, _mm256_andnot_si256(sign_mask, ((__m256i*)a)[i + 2]));
        max3 = _mm256_max_epi32(max3, _mm256_andnot_si256(sign_mask, ((__m256i*)a)[i + 3]));
    }
    for (; i < n; ++i) {
        max0 = _mm256_max_epi32(max0, _mm256_andnot_si256(sign_mask, ((__m256i*)a)[i]));
    }
    __m256i maxv = _mm256_max_epi32(_mm256_max_epi32(max0, max1), _mm256_max_epi32(max2, max3));

    n = len - n * 8;
    if (n != 0) {
        __m256i mask = _mm256_setr_epi32(
            n > 0 ? -1 : 0,
            n > 1 ? -1 : 0,
            n > 2 ? -1 : 0,
            n > 3 ? -1 : 0,
            n > 4 ? -1 : 0,
            n > 5 ? -1 : 0,
            n > 6 ? -1 : 0,
            n > 7 ? -1 : 0
        );
        __m256i abs_val = _mm256_andnot_si256(sign_mask, ((__m256i*)a)[i]);
        __m256i new_max = _mm256_max_epi32(maxv, abs_val);
        maxv = _mm256_blendv_epi8(maxv, new_max, mask);
    }

    __m128i low = _mm256_castsi256_si128(maxv);
    __m128i high = _mm256_extracti128_si256(maxv, 1);
    __m128i max128 = _mm_max_epi32(low, high);
    // 对 4 个元素求最大值：水平交换两次
    __m128i swapped = _mm_shuffle_epi32(max128, _MM_SHUFFLE(2, 3, 0, 1));
    max128 = _mm_max_epi32(max128, swapped);
    swapped = _mm_shuffle_epi32(max128, _MM_SHUFFLE(1, 0, 3, 2));
    max128 = _mm_max_epi32(max128, swapped);
    return _mm_cvtsi128_si32(max128);
}
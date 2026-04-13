
#define XiangZiMatImport_SIMD

#include<immintrin.h>

inline void __ApplyGivensOnTheRight(double* __restrict a, double* __restrict b, double c, double s, int len) {
	__m512d ai, bi;
	__m512d cc(_mm512_set1_pd(c)), ss(_mm512_set1_pd(s));
	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		ai = ((__m512d*)a)[i];
		bi = ((__m512d*)b)[i];
		((__m512d*)a)[i] = _mm512_fmadd_pd(ai, cc, _mm512_mul_pd(bi, ss));
		((__m512d*)b)[i] = _mm512_fnmadd_pd(ai, ss, _mm512_mul_pd(bi, cc));
	}

	n = len - n * 8;
	if (n != 0) {
		__mmask8 mask = (1 << n) - 1;
		ai = ((__m512d*)a)[i];
		bi = ((__m512d*)b)[i];
		((__m512d*)a)[i] = _mm512_mask_blend_pd(mask, ai, _mm512_fmadd_pd(ai, cc, _mm512_mul_pd(bi, ss)));
		((__m512d*)b)[i] = _mm512_mask_blend_pd(mask, bi, _mm512_fnmadd_pd(ai, ss, _mm512_mul_pd(bi, cc)));
	}
}

inline void __Schur_HouseHolder_Right(
	double* __restrict a, double* __restrict b, double* __restrict c,
	const double* __restrict w, int len) {
	__m512d sum;
	__m512d w0(_mm512_set1_pd(w[0])), w1(_mm512_set1_pd(w[1])), w2(_mm512_set1_pd(w[2]));

	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		sum =
			_mm512_fmadd_pd(((__m512d*)a)[i], w0,
				_mm512_fmadd_pd(((__m512d*)b)[i], w1,
					_mm512_mul_pd(((__m512d*)c)[i], w2)));
		((__m512d*)a)[i] = _mm512_fnmadd_pd(sum, w0, ((__m512d*)a)[i]);
		((__m512d*)b)[i] = _mm512_fnmadd_pd(sum, w1, ((__m512d*)b)[i]);
		((__m512d*)c)[i] = _mm512_fnmadd_pd(sum, w2, ((__m512d*)c)[i]);
	}
	n = len - n * 8;
	if (n != 0) {
		sum = _mm512_mask_blend_pd((1 << n) - 1, _mm512_setzero_pd(),
			_mm512_fmadd_pd(((__m512d*)a)[i], w0,
				_mm512_fmadd_pd(((__m512d*)b)[i], w1,
					_mm512_mul_pd(((__m512d*)c)[i], w2))));
		((__m512d*)a)[i] = _mm512_fnmadd_pd(sum, w0, ((__m512d*)a)[i]);
		((__m512d*)b)[i] = _mm512_fnmadd_pd(sum, w1, ((__m512d*)b)[i]);
		((__m512d*)c)[i] = _mm512_fnmadd_pd(sum, w2, ((__m512d*)c)[i]);
	}
}

inline void __times(double* __restrict a, double b, double* __restrict ans, int len) {
	__m512d bb(_mm512_set1_pd(b));

	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		((__m512d*)ans)[i] = _mm512_mul_pd(((__m512d*)a)[i], bb);
	}
	n = len - n * 8;
	if (n != 0) {
		((__m512d*)ans)[i] = _mm512_mask_blend_pd((1 << n) - 1, ((__m512d*)ans)[i], _mm512_mul_pd(((__m512d*)a)[i], bb));
	}
}

inline void __add(double* __restrict a, double* __restrict b, double* __restrict ans, int len) {
	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		((__m512d*)ans)[i] = _mm512_add_pd(((__m512d*)a)[i], ((__m512d*)b)[i]);
	}
	n = len - n * 8;
	if (n != 0) {
		((__m512d*)ans)[i] = _mm512_mask_blend_pd((1 << n) - 1, ((__m512d*)ans)[i], _mm512_add_pd(((__m512d*)a)[i], ((__m512d*)b)[i]));
	}
}

//ans=a+b*p
inline void __mixed(double* __restrict a, double* __restrict b, double q, double* __restrict ans, int len) {
	__m512d qq(_mm512_set1_pd(q));
	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		((__m512d*)ans)[i] = _mm512_fmadd_pd(((__m512d*)b)[i], qq, ((__m512d*)a)[i]);
	}
	n = len - n * 8;
	if (n != 0) {
		((__m512d*)ans)[i] = _mm512_mask_blend_pd((1 << n) - 1, ((__m512d*)ans)[i], _mm512_fmadd_pd(((__m512d*)b)[i], qq, ((__m512d*)a)[i]));
	}
}

inline void __swap(double* __restrict a, double* __restrict b, int len) {
	__m512d temp;
	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		temp = ((__m512d*)a)[i];
		((__m512d*)a)[i] = ((__m512d*)b)[i];
		((__m512d*)b)[i] = temp;
	}
	n = len - n * 8;
	if (n != 0) {
		__mmask8 mask = (1 << n) - 1;
		temp = ((__m512d*)a)[i];
		((__m512d*)a)[i] = _mm512_mask_blend_pd(mask, temp, ((__m512d*)b)[i]);
		((__m512d*)b)[i] = _mm512_mask_blend_pd(mask, ((__m512d*)b)[i], temp);
	}
}

inline double __dot(double* __restrict a, double* __restrict b, int len) {
	__m512d ans(_mm512_setzero_pd());
	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		ans = _mm512_fmadd_pd(((__m512d*)a)[i], ((__m512d*)b)[i], ans);
	}
	n = len - n * 8;
	if (n != 0) {
		ans = _mm512_mask_blend_pd((1 << n) - 1, ans, _mm512_fmadd_pd(((__m512d*)a)[i], ((__m512d*)b)[i], ans));
	}

	return _mm512_reduce_add_pd(ans);
}

double __MaxAbs(double* a, int len) {
	int s = len / 8;
	__m512d max = _mm512_setzero_pd();
	for (int i = 0; i < s; ++i) {
		max = _mm512_max_pd(max, _mm512_abs_pd(((__m512d*)a)[i]));
	}
	len = len - s * 8;
	max = _mm512_max_pd(max,
		_mm512_mask_blend_pd((1 << len) - 1, _mm512_setzero_pd(), _mm512_abs_pd(((__m512d*)a)[s]))
	);
	return _mm512_reduce_max_pd(max);
}






void __micro_kernel_8x8_times_A_B(
	double* __restrict A, int ldA, double* __restrict B, int ldB,
	double* __restrict C, int ldC, int k)
{
	__m512d c0 = *(__m512d*)(C);
	__m512d c1 = *(__m512d*)(C + ldC);
	__m512d c2 = *(__m512d*)(C + 2 * ldC);
	__m512d c3 = *(__m512d*)(C + 3 * ldC);
	__m512d c4 = *(__m512d*)(C + 4 * ldC);
	__m512d c5 = *(__m512d*)(C + 5 * ldC);
	__m512d c6 = *(__m512d*)(C + 6 * ldC);
	__m512d c7 = *(__m512d*)(C + 7 * ldC);

	for (int p = 0; p < k; ++p) {
		__m512d a = *(__m512d*)(A + p * ldA);
		double b0 = B[0 * ldB + p];
		double b1 = B[1 * ldB + p];
		double b2 = B[2 * ldB + p];
		double b3 = B[3 * ldB + p];
		double b4 = B[4 * ldB + p];
		double b5 = B[5 * ldB + p];
		double b6 = B[6 * ldB + p];
		double b7 = B[7 * ldB + p];

		c0 = _mm512_fmadd_pd(a, _mm512_set1_pd(b0), c0);
		c1 = _mm512_fmadd_pd(a, _mm512_set1_pd(b1), c1);
		c2 = _mm512_fmadd_pd(a, _mm512_set1_pd(b2), c2);
		c3 = _mm512_fmadd_pd(a, _mm512_set1_pd(b3), c3);
		c4 = _mm512_fmadd_pd(a, _mm512_set1_pd(b4), c4);
		c5 = _mm512_fmadd_pd(a, _mm512_set1_pd(b5), c5);
		c6 = _mm512_fmadd_pd(a, _mm512_set1_pd(b6), c6);
		c7 = _mm512_fmadd_pd(a, _mm512_set1_pd(b7), c7);
	}

	*(__m512d*)(C) = c0;
	*(__m512d*)(C + ldC) = c1;
	*(__m512d*)(C + 2 * ldC) = c2;
	*(__m512d*)(C + 3 * ldC) = c3;
	*(__m512d*)(C + 4 * ldC) = c4;
	*(__m512d*)(C + 5 * ldC) = c5;
	*(__m512d*)(C + 6 * ldC) = c6;
	*(__m512d*)(C + 7 * ldC) = c7;
}

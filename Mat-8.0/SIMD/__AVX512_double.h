
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
void __micro_kernel_8x8_ntimes_A_B(
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

		c0 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b0), c0);
		c1 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b1), c1);
		c2 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b2), c2);
		c3 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b3), c3);
		c4 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b4), c4);
		c5 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b5), c5);
		c6 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b6), c6);
		c7 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b7), c7);
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
void __micro_kernel_8x8_ntimes_A_BT(
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
		double b0 = B[0];
		double b1 = B[1];
		double b2 = B[2];
		double b3 = B[3];
		double b4 = B[4];
		double b5 = B[5];
		double b6 = B[6];
		double b7 = B[7];

		c0 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b0), c0);
		c1 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b1), c1);
		c2 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b2), c2);
		c3 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b3), c3);
		c4 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b4), c4);
		c5 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b5), c5);
		c6 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b6), c6);
		c7 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b7), c7);

		B = B + ldB;
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
void __micro_kernel_8x8_ntimes_ADBT(
	double* __restrict A, int ldA, double* __restrict B, int ldB,
	double* __restrict C, int ldC, int k, double* d)
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

		double b0 = B[0];
		double b1 = B[1];
		double b2 = B[2];
		double b3 = B[3];
		double b4 = B[4];
		double b5 = B[5];
		double b6 = B[6];
		double b7 = B[7];

		double D = d[p];

		c0 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b0 * D), c0);
		c1 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b1 * D), c1);
		c2 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b2 * D), c2);
		c3 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b3 * D), c3);
		c4 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b4 * D), c4);
		c5 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b5 * D), c5);
		c6 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b6 * D), c6);
		c7 = _mm512_fnmadd_pd(a, _mm512_set1_pd(b7 * D), c7);

		B = B + ldB;
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
/*
void __micro_kernel_8x8_times_A_BT(
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
		double b0 = B[0];
		double b1 = B[1];
		double b2 = B[2];
		double b3 = B[3];
		double b4 = B[4];
		double b5 = B[5];
		double b6 = B[6];
		double b7 = B[7];

		c0 = _mm512_fmadd_pd(a, _mm512_set1_pd(b0), c0);
		c1 = _mm512_fmadd_pd(a, _mm512_set1_pd(b1), c1);
		c2 = _mm512_fmadd_pd(a, _mm512_set1_pd(b2), c2);
		c3 = _mm512_fmadd_pd(a, _mm512_set1_pd(b3), c3);
		c4 = _mm512_fmadd_pd(a, _mm512_set1_pd(b4), c4);
		c5 = _mm512_fmadd_pd(a, _mm512_set1_pd(b5), c5);
		c6 = _mm512_fmadd_pd(a, _mm512_set1_pd(b6), c6);
		c7 = _mm512_fmadd_pd(a, _mm512_set1_pd(b7), c7);

		B = B + ldB;
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

void __micro_kernel_4x4_times_AT_B(
	double* __restrict A, int ldA, double* __restrict B, int ldB,
	double* __restrict C, int ldC, int k)
{
	__m512d C00(_mm512_setzero_pd()), C01(_mm512_setzero_pd()), C02(_mm512_setzero_pd()), C03(_mm512_setzero_pd());
	__m512d C10(_mm512_setzero_pd()), C11(_mm512_setzero_pd()), C12(_mm512_setzero_pd()), C13(_mm512_setzero_pd());
	__m512d C20(_mm512_setzero_pd()), C21(_mm512_setzero_pd()), C22(_mm512_setzero_pd()), C23(_mm512_setzero_pd());
	__m512d C30(_mm512_setzero_pd()), C31(_mm512_setzero_pd()), C32(_mm512_setzero_pd()), C33(_mm512_setzero_pd());

	int n = k / 8;
	int p;
	for (p = 0; p < n; ++p) {
		__m512d a0 = ((__m512d*)(A + 0 * ldA))[p];
		__m512d a1 = ((__m512d*)(A + 1 * ldA))[p];
		__m512d a2 = ((__m512d*)(A + 2 * ldA))[p];
		__m512d a3 = ((__m512d*)(A + 3 * ldA))[p];

		__m512d b0 = ((__m512d*)(B + 0 * ldB))[p];
		__m512d b1 = ((__m512d*)(B + 1 * ldB))[p];
		__m512d b2 = ((__m512d*)(B + 2 * ldB))[p];
		__m512d b3 = ((__m512d*)(B + 3 * ldB))[p];

		C00 = _mm512_fmadd_pd(a0, b0, C00); C01 = _mm512_fmadd_pd(a1, b0, C01); C02 = _mm512_fmadd_pd(a2, b0, C02); C03 = _mm512_fmadd_pd(a3, b0, C03);
		C10 = _mm512_fmadd_pd(a0, b1, C10); C11 = _mm512_fmadd_pd(a1, b1, C11); C12 = _mm512_fmadd_pd(a2, b1, C12); C13 = _mm512_fmadd_pd(a3, b1, C13);
		C20 = _mm512_fmadd_pd(a0, b2, C20); C21 = _mm512_fmadd_pd(a1, b2, C21); C22 = _mm512_fmadd_pd(a2, b2, C22); C23 = _mm512_fmadd_pd(a3, b2, C23);
		C30 = _mm512_fmadd_pd(a0, b3, C30); C31 = _mm512_fmadd_pd(a1, b3, C31); C32 = _mm512_fmadd_pd(a2, b3, C32); C33 = _mm512_fmadd_pd(a3, b3, C33);
	}
	n = k - 8 * n;
	if (n != 0) {
		__m512d a0 = _mm512_maskz_loadu_pd((1 << n) - 1, ((__m512d*)(A + 0 * ldA)) + p);
		__m512d a1 = _mm512_maskz_loadu_pd((1 << n) - 1, ((__m512d*)(A + 1 * ldA)) + p);
		__m512d a2 = _mm512_maskz_loadu_pd((1 << n) - 1, ((__m512d*)(A + 2 * ldA)) + p);
		__m512d a3 = _mm512_maskz_loadu_pd((1 << n) - 1, ((__m512d*)(A + 3 * ldA)) + p);

		__m512d b0 = ((__m512d*)(B + 0 * ldB))[p];
		__m512d b1 = ((__m512d*)(B + 1 * ldB))[p];
		__m512d b2 = ((__m512d*)(B + 2 * ldB))[p];
		__m512d b3 = ((__m512d*)(B + 3 * ldB))[p];

		C00 = _mm512_fmadd_pd(a0, b0, C00); C01 = _mm512_fmadd_pd(a1, b0, C01); C02 = _mm512_fmadd_pd(a2, b0, C02); C03 = _mm512_fmadd_pd(a3, b0, C03);
		C10 = _mm512_fmadd_pd(a0, b1, C10); C11 = _mm512_fmadd_pd(a1, b1, C11); C12 = _mm512_fmadd_pd(a2, b1, C12); C13 = _mm512_fmadd_pd(a3, b1, C13);
		C20 = _mm512_fmadd_pd(a0, b2, C20); C21 = _mm512_fmadd_pd(a1, b2, C21); C22 = _mm512_fmadd_pd(a2, b2, C22); C23 = _mm512_fmadd_pd(a3, b2, C23);
		C30 = _mm512_fmadd_pd(a0, b3, C30); C31 = _mm512_fmadd_pd(a1, b3, C31); C32 = _mm512_fmadd_pd(a2, b3, C32); C33 = _mm512_fmadd_pd(a3, b3, C33);
	}

	double c00 = C[0 * ldC + 0], c01 = C[0 * ldC + 1], c02 = C[0 * ldC + 2], c03 = C[0 * ldC + 3];
	double c10 = C[1 * ldC + 0], c11 = C[1 * ldC + 1], c12 = C[1 * ldC + 2], c13 = C[1 * ldC + 3];
	double c20 = C[2 * ldC + 0], c21 = C[2 * ldC + 1], c22 = C[2 * ldC + 2], c23 = C[2 * ldC + 3];
	double c30 = C[3 * ldC + 0], c31 = C[3 * ldC + 1], c32 = C[3 * ldC + 2], c33 = C[3 * ldC + 3];

	c00 += _mm512_reduce_add_pd(C00); c01 += _mm512_reduce_add_pd(C01); c02 += _mm512_reduce_add_pd(C02); c03 += _mm512_reduce_add_pd(C03);
	c10 += _mm512_reduce_add_pd(C10); c11 += _mm512_reduce_add_pd(C11); c12 += _mm512_reduce_add_pd(C12); c13 += _mm512_reduce_add_pd(C13);
	c20 += _mm512_reduce_add_pd(C20); c21 += _mm512_reduce_add_pd(C21); c22 += _mm512_reduce_add_pd(C22); c23 += _mm512_reduce_add_pd(C23);
	c30 += _mm512_reduce_add_pd(C30); c31 += _mm512_reduce_add_pd(C31); c32 += _mm512_reduce_add_pd(C32); c33 += _mm512_reduce_add_pd(C33);

	C[0 * ldC + 0] = c00; C[0 * ldC + 1] = c01; C[0 * ldC + 2] = c02; C[0 * ldC + 3] = c03;
	C[1 * ldC + 0] = c10; C[1 * ldC + 1] = c11; C[1 * ldC + 2] = c12; C[1 * ldC + 3] = c13;
	C[2 * ldC + 0] = c20; C[2 * ldC + 1] = c21; C[2 * ldC + 2] = c22; C[2 * ldC + 3] = c23;
	C[3 * ldC + 0] = c30; C[3 * ldC + 1] = c31; C[3 * ldC + 2] = c32; C[3 * ldC + 3] = c33;
}
*/
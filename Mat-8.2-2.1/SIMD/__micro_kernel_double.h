
#ifdef __AVX512__

template<>
constexpr int __micro_kernel_size<double, double, double>() {
	return 8;
}

void __micro_kernel_times_A_B(
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
void __micro_kernel_ntimes_A_B(
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
void __micro_kernel_ntimes_A_BT(
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
void __micro_kernel_ntimes_ADBT(
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

#endif
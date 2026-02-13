

#include"AVX512.h"

void times_A_B(MAT<double>& A, MAT<double>& B, MAT<double>& C, double* w) {
	int i, j, k;
	int n = A.cols / 8;
	int m = A.cols - n * 8;

	__m512d __add;
	__m512d* tempA;
	__m512d* tempB = (__m512d*)(w + m);

	__mmask8 mask = (1 << m) - 1;

	for (j = 0; j < C.cols; ++j) {
		for (k = 0; k < B.rows; ++k) {
			w[k] = B[k][j];
		}
		for (i = 0; i < C.rows; ++i) {
			__add = _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(*(__m512d*)(A[i]), *(__m512d*)(w)));
			tempA = (__m512d*)(A[i] + m);
			for (k = 0; k < n; ++k) {
				__add = _mm512_add_pd(__add, _mm512_mul_pd(tempA[k], tempB[k]));
			}
			C[i][j] = _mm512_reduce_add_pd(__add);
		}
	}
}
void times_A_B(MAT<float>& A, MAT<float>& B, MAT<float>& C, float* w) {
	int i, j, k;
	int n = A.cols / 16;
	int m = A.cols - n * 16;

	__m512 __add;
	__m512* tempA;
	__m512* tempB = (__m512*)(w + m);

	__mmask16 mask = (1 << m) - 1;

	for (j = 0; j < C.cols; ++j) {
		for (k = 0; k < B.rows; ++k) {
			w[k] = B[k][j];
		}
		for (i = 0; i < C.rows; ++i) {
			__add = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(*(__m512*)(A[i]), *(__m512*)(w)));
			tempA = (__m512*)(A[i] + m);
			for (k = 0; k < n; ++k) {
				__add = _mm512_add_ps(__add, _mm512_mul_ps(tempA[k], tempB[k]));
			}
			C[i][j] = _mm512_reduce_add_ps(__add);
		}
	}
}


void times_A_B(const MAT<double>& A, const MAT<double>& B, const MAT<double>& C) {
	int i, j, k;
	ZeroMemory(C.data, sizeof(double) * C.rows * C.cols);
	int n = C.cols / 8;
	int m = C.cols - n * 8;
	__m512d mul;
	__m512d* tempB;
	__m512d* tempC;

	__mmask8 mask = (1 << m) - 1;

	for (i = 0; i < C.rows; ++i) {
		tempC = (__m512d*)(C.data + C.cols * i + m);
		for (k = 0; k < A.cols; ++k) {
			mul = _mm512_set1_pd((A.data + A.cols * i)[k]);
			*(__m512d*)(C.data + C.cols * i) = _mm512_add_pd(*(__m512d*)(C.data + C.cols * i), _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(mul, *(__m512d*)(B.data + B.cols * k))));
			tempB = (__m512d*)(B.data + B.cols * k + m);
			for (j = 0; j < n; ++j)
				tempC[j] = _mm512_add_pd(tempC[j], _mm512_mul_pd(mul, tempB[j]));
		}
	}
}
void times_A_B(const MAT<float>& A, const MAT<float>& B, const MAT<float>& C) {
	int i, j, k;
	ZeroMemory(C.data, sizeof(float) * C.rows * C.cols);
	int n = C.cols / 16;
	int m = C.cols - n * 16;
	__m512 mul;
	__m512* tempB;
	__m512* tempC;

	__mmask16 mask = (1 << m) - 1;

	for (i = 0; i < C.rows; ++i) {
		tempC = (__m512*)(C.data + C.cols * i + m);
		for (k = 0; k < A.cols; ++k) {
			mul = _mm512_set1_ps((A.data + A.cols * i)[k]);
			*(__m512*)(C.data + C.cols * i) = _mm512_add_ps(*(__m512*)(C.data + C.cols * i), _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(mul, *(__m512*)(B.data + B.cols * k))));
			tempB = (__m512*)(B.data + B.cols * k + m);
			for (j = 0; j < n; ++j)
				tempC[j] = _mm512_add_ps(tempC[j], _mm512_mul_ps(mul, tempB[j]));
		}
	}
}


void times_A_BT(MAT<double>& A, MAT<double>& B, MAT<double>& C) {
	int i, j, k;
	int n = A.cols / 8;
	int m = A.cols - n * 8;
	__m512d __add;
	__m512d* tempA;
	__m512d* tempB;

	__mmask8 mask = (1 << m) - 1;

	for (i = 0; i < C.rows; ++i) {
		for (j = 0; j < C.rows; ++j) {
			__add = _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(*(__m512d*)(A[i]), *(__m512d*)(B[j])));
			tempA = (__m512d*)(A[i] + m);
			tempB = (__m512d*)(B[j] + m);
			for (k = 0; k < n; ++k) {
				__add = _mm512_add_pd(__add, _mm512_mul_pd(tempA[k], tempB[k]));
			}
			C[i][j] = _mm512_reduce_add_pd(__add);
		}
	}
}
void times_A_BT(MAT<float>& A, MAT<float>& B, MAT<float>& C) {
	int i, j, k;
	int n = A.cols / 16;
	int m = A.cols - n * 16;
	__m512 __add;
	__m512* tempA;
	__m512* tempB;

	__mmask16 mask = (1 << m) - 1;

	for (i = 0; i < C.rows; ++i) {
		for (j = 0; j < C.rows; ++j) {
			__add = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(*(__m512*)(A[i]), *(__m512*)(B[j])));
			tempA = (__m512*)(A[i] + m);
			tempB = (__m512*)(B[j] + m);
			for (k = 0; k < n; ++k) {
				__add = _mm512_add_ps(__add, _mm512_mul_ps(tempA[k], tempB[k]));
			}
			C[i][j] = _mm512_reduce_add_ps(__add);
		}
	}
}


void times_AT_B(MAT<double>& A, MAT<double>& B, MAT<double>& C) {
	int i, j, k;
	ZeroMemory(C.data, sizeof(double) * C.rows * C.cols);
	int n = C.cols / 8;
	int m = C.cols - n * 8;
	__m512d mul;
	__m512d* tempB;
	__m512d* tempC;

	__mmask8 mask = (1 << m) - 1;

	for (k = 0; k < A.rows; ++k) {
		tempB = (__m512d*)(B[k] + m);
		for (i = 0; i < C.rows; ++i) {
			tempC = (__m512d*)(C[i] + m);
			mul = _mm512_set1_pd(A[k][i]);
			*(__m512d*)(C[i]) = _mm512_add_pd(*(__m512d*)(C[i]), _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(mul, *(__m512d*)(B[k]))));
			for (j = 0; j < n; ++j) {
				tempC[j] = _mm512_add_pd(tempC[j], _mm512_mul_pd(mul, tempB[j]));
			}
		}
	}
}
void times_AT_B(MAT<float>& A, MAT<float>& B, MAT<float>& C) {
	int i, j, k;
	ZeroMemory(C.data, sizeof(float) * C.rows * C.cols);
	int n = B.cols / 16;
	int m = B.cols - n * 16;
	__m512 mul;
	__m512* tempB;
	__m512* tempC;

	__mmask16 mask = (1 << m) - 1;

	for (k = 0; k < A.rows; ++k) {
		tempB = (__m512*)(B[k] + m);
		for (i = 0; i < C.rows; ++i) {
			tempC = (__m512*)(C[i] + m);
			mul = _mm512_set1_ps(A[k][i]);
			*(__m512*)(C[i]) = _mm512_add_ps(*(__m512*)(C[i]), _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(mul, *(__m512*)(B[k]))));
			for (j = 0; j < n; ++j) {
				tempC[j] = _mm512_add_ps(tempC[j], _mm512_mul_ps(mul, tempB[j]));
			}
		}
	}
}

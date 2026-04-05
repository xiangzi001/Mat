
template<typename T>
inline void __times(T* __restrict a, T b, T* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] * b;
	}
}

template<typename T>
inline void __add(T* __restrict a, T b, T* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] + b;
	}
}
template<typename T>
inline void __add(T* __restrict a, T* __restrict b, T* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] + b[i];
	}
}
template<typename t>
inline void __add(complex<t>* a, complex<t>* b, complex<t>* ans, int len) {
	__add((t*)a, (t*)b, (t*)ans, len * 2);
}

template<typename T>
inline void __sub(T* __restrict a, T* __restrict b, T* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] - b[i];
	}
}
template<typename t>
inline void __sub(complex<t>* a, complex<t>* b, complex<t>* ans, int len) {
	__sub((t*)a, (t*)b, (t*)ans, len * 2);
}

//ans=a+b*p
template<typename T>
inline void __mixed(T* __restrict a, T* __restrict b, T q, T* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] + b[i] * q;
	}
}
//ans=a+b*p
template<typename t1, typename t2, typename t3, typename t4>
inline void __mixed(t1* __restrict a, t2* __restrict b, t3 q, t4* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] + b[i] * q;
	}
}
template<typename T>
inline void __swap(T* __restrict a, T* __restrict b, int len) {
	T temp;
	for (int i = 0; i < len; ++i) {
		temp = a[i];
		a[i] = b[i];
		b[i] = temp;
	}
}
/*
template<typename T>
inline T __dot(T* __restrict a, T* __restrict b, int len) {
	T ans(0);
	for (int i = 0; i < len; ++i) {
		ans += a[i] * b[i];
	}
	return ans;
}
*/
template<typename T>
inline T __dot(T* __restrict a, T* __restrict b, int len) {
	T temp[4] = {};
	int n = 4 * ((int)(len / 4));
	for (int i = 0; i < n; i += 4) {
		temp[0] += a[i] * b[i];
		temp[1] += a[i + 1] * b[i + 1];
		temp[2] += a[i + 2] * b[i + 2];
		temp[3] += a[i + 3] * b[i + 3];
	}
	T ans = temp[0] + temp[1] + temp[2] + temp[3];
	for (int i = n; i < len; ++i) {
		ans += a[i] * b[i];
	}
	return ans;
}

template<typename t>
inline complex<t> __dot(complex<t>* __restrict a, complex<t>* __restrict b, int len) {
	complex<t> ans(0);
	for (int i = 0; i < len; ++i) {
		ans += a[i].conj() * b[i];
	}
	return ans;
}

template<class t>
MAT<t> operator * (t a, const MAT<t>& A) {
	MAT<t> n(A.rows, A.cols);

	__times(A.data, a, n.data, A.rows * A.cols);

	return n;
}
template<class t>
MAT<t> operator * (const MAT<t>& A, t a) {
	MAT<t> n(A.rows, A.cols);

	__times(A.data, a, n.data, A.rows * A.cols);

	return n;
}
template<class t>
MAT<t> operator / (const MAT<t>& A, t a) {
	MAT<t> n(A.rows, A.cols);

	__times(A.data, t(1) / a, n.data, A.rows * A.cols);

	return n;
}

template<typename t1, typename t2>
auto operator + (const MAT<t1>& A, const MAT<t2>& B) {
	MAT<decltype(std::declval<t1>() + std::declval<t2>())> n(A.rows, A.cols);
	__add(A.data, B.data, n.data, A.rows * A.cols);
	return n;
}
template<typename t1, typename t2>
auto operator - (const MAT<t1>& A, const MAT<t2>& B) {
	MAT<decltype(std::declval<t1>() - std::declval<t2>())> n(A.rows, A.cols);
	__sub(A.data, B.data, n.data, A.rows * A.cols);
	return n;
}

template<typename T>
void __micro_kernel_4x4_times_A_B(
	T* __restrict A, int ldA, T* __restrict B, int ldB,
	T* __restrict C, int ldC, int k)
{
	T c00 = C[0 * ldC + 0], c01 = C[0 * ldC + 1], c02 = C[0 * ldC + 2], c03 = C[0 * ldC + 3];
	T c10 = C[1 * ldC + 0], c11 = C[1 * ldC + 1], c12 = C[1 * ldC + 2], c13 = C[1 * ldC + 3];
	T c20 = C[2 * ldC + 0], c21 = C[2 * ldC + 1], c22 = C[2 * ldC + 2], c23 = C[2 * ldC + 3];
	T c30 = C[3 * ldC + 0], c31 = C[3 * ldC + 1], c32 = C[3 * ldC + 2], c33 = C[3 * ldC + 3];

	for (int p = 0; p < k; ++p) {
		T a0 = A[0];
		T a1 = A[1];
		T a2 = A[2];
		T a3 = A[3];

		T b0 = B[0 * ldB + p];
		T b1 = B[1 * ldB + p];
		T b2 = B[2 * ldB + p];
		T b3 = B[3 * ldB + p];

		c00 += a0 * b0; c01 += a1 * b0; c02 += a2 * b0; c03 += a3 * b0;
		c10 += a0 * b1; c11 += a1 * b1; c12 += a2 * b1; c13 += a3 * b1;
		c20 += a0 * b2; c21 += a1 * b2; c22 += a2 * b2; c23 += a3 * b2;
		c30 += a0 * b3; c31 += a1 * b3; c32 += a2 * b3; c33 += a3 * b3;

		A = A + ldA;
	}
	C[0 * ldC + 0] = c00; C[0 * ldC + 1] = c01; C[0 * ldC + 2] = c02; C[0 * ldC + 3] = c03;
	C[1 * ldC + 0] = c10; C[1 * ldC + 1] = c11; C[1 * ldC + 2] = c12; C[1 * ldC + 3] = c13;
	C[2 * ldC + 0] = c20; C[2 * ldC + 1] = c21; C[2 * ldC + 2] = c22; C[2 * ldC + 3] = c23;
	C[3 * ldC + 0] = c30; C[3 * ldC + 1] = c31; C[3 * ldC + 2] = c32; C[3 * ldC + 3] = c33;
}
template<typename T>
void __micro_kernel_4x4_ntimes_A_B(
	T* __restrict A, int ldA, T* __restrict B, int ldB,
	T* __restrict C, int ldC, int k)
{
	T c00 = C[0 * ldC + 0], c01 = C[0 * ldC + 1], c02 = C[0 * ldC + 2], c03 = C[0 * ldC + 3];
	T c10 = C[1 * ldC + 0], c11 = C[1 * ldC + 1], c12 = C[1 * ldC + 2], c13 = C[1 * ldC + 3];
	T c20 = C[2 * ldC + 0], c21 = C[2 * ldC + 1], c22 = C[2 * ldC + 2], c23 = C[2 * ldC + 3];
	T c30 = C[3 * ldC + 0], c31 = C[3 * ldC + 1], c32 = C[3 * ldC + 2], c33 = C[3 * ldC + 3];

	for (int p = 0; p < k; ++p) {
		T a0 = A[p * ldA + 0];
		T a1 = A[p * ldA + 1];
		T a2 = A[p * ldA + 2];
		T a3 = A[p * ldA + 3];

		T b0 = B[0 * ldB + p];
		T b1 = B[1 * ldB + p];
		T b2 = B[2 * ldB + p];
		T b3 = B[3 * ldB + p];

		c00 -= a0 * b0; c01 -= a1 * b0; c02 -= a2 * b0; c03 -= a3 * b0;
		c10 -= a0 * b1; c11 -= a1 * b1; c12 -= a2 * b1; c13 -= a3 * b1;
		c20 -= a0 * b2; c21 -= a1 * b2; c22 -= a2 * b2; c23 -= a3 * b2;
		c30 -= a0 * b3; c31 -= a1 * b3; c32 -= a2 * b3; c33 -= a3 * b3;
	}
	C[0 * ldC + 0] = c00; C[0 * ldC + 1] = c01; C[0 * ldC + 2] = c02; C[0 * ldC + 3] = c03;
	C[1 * ldC + 0] = c10; C[1 * ldC + 1] = c11; C[1 * ldC + 2] = c12; C[1 * ldC + 3] = c13;
	C[2 * ldC + 0] = c20; C[2 * ldC + 1] = c21; C[2 * ldC + 2] = c22; C[2 * ldC + 3] = c23;
	C[3 * ldC + 0] = c30; C[3 * ldC + 1] = c31; C[3 * ldC + 2] = c32; C[3 * ldC + 3] = c33;
}
template<typename T>
void __micro_kernel_4x4_ntimes_A_BT(
	T* __restrict A, int ldA, T* __restrict B, int ldB,
	T* __restrict C, int ldC, int k)
{
	T c00 = C[0 * ldC + 0], c01 = C[0 * ldC + 1], c02 = C[0 * ldC + 2], c03 = C[0 * ldC + 3];
	T c10 = C[1 * ldC + 0], c11 = C[1 * ldC + 1], c12 = C[1 * ldC + 2], c13 = C[1 * ldC + 3];
	T c20 = C[2 * ldC + 0], c21 = C[2 * ldC + 1], c22 = C[2 * ldC + 2], c23 = C[2 * ldC + 3];
	T c30 = C[3 * ldC + 0], c31 = C[3 * ldC + 1], c32 = C[3 * ldC + 2], c33 = C[3 * ldC + 3];

	for (int p = 0; p < k; ++p) {
		T a0 = A[p * ldA + 0];
		T a1 = A[p * ldA + 1];
		T a2 = A[p * ldA + 2];
		T a3 = A[p * ldA + 3];

		T b0 = B[p * ldB + 0];
		T b1 = B[p * ldB + 1];
		T b2 = B[p * ldB + 2];
		T b3 = B[p * ldB + 3];

		c00 -= a0 * b0; c01 -= a1 * b0; c02 -= a2 * b0; c03 -= a3 * b0;
		c10 -= a0 * b1; c11 -= a1 * b1; c12 -= a2 * b1; c13 -= a3 * b1;
		c20 -= a0 * b2; c21 -= a1 * b2; c22 -= a2 * b2; c23 -= a3 * b2;
		c30 -= a0 * b3; c31 -= a1 * b3; c32 -= a2 * b3; c33 -= a3 * b3;
	}
	C[0 * ldC + 0] = c00; C[0 * ldC + 1] = c01; C[0 * ldC + 2] = c02; C[0 * ldC + 3] = c03;
	C[1 * ldC + 0] = c10; C[1 * ldC + 1] = c11; C[1 * ldC + 2] = c12; C[1 * ldC + 3] = c13;
	C[2 * ldC + 0] = c20; C[2 * ldC + 1] = c21; C[2 * ldC + 2] = c22; C[2 * ldC + 3] = c23;
	C[3 * ldC + 0] = c30; C[3 * ldC + 1] = c31; C[3 * ldC + 2] = c32; C[3 * ldC + 3] = c33;
}
template<typename T>
void __micro_kernel_4x4_ntimes_ADBT(
	T* __restrict A, int ldA, T* __restrict B, int ldB,
	T* __restrict C, int ldC, int k, T* d)
{
	T c00 = C[0 * ldC + 0], c01 = C[0 * ldC + 1], c02 = C[0 * ldC + 2], c03 = C[0 * ldC + 3];
	T c10 = C[1 * ldC + 0], c11 = C[1 * ldC + 1], c12 = C[1 * ldC + 2], c13 = C[1 * ldC + 3];
	T c20 = C[2 * ldC + 0], c21 = C[2 * ldC + 1], c22 = C[2 * ldC + 2], c23 = C[2 * ldC + 3];
	T c30 = C[3 * ldC + 0], c31 = C[3 * ldC + 1], c32 = C[3 * ldC + 2], c33 = C[3 * ldC + 3];

	for (int p = 0; p < k; ++p) {
		T a0 = A[p * ldA + 0];
		T a1 = A[p * ldA + 1];
		T a2 = A[p * ldA + 2];
		T a3 = A[p * ldA + 3];

		T b0 = B[p * ldB + 0];
		T b1 = B[p * ldB + 1];
		T b2 = B[p * ldB + 2];
		T b3 = B[p * ldB + 3];

		T D = d[p];
		b0 *= D; b1 *= D; b2 *= D; b3 *= D;

		c00 -= a0 * b0; c01 -= a1 * b0; c02 -= a2 * b0; c03 -= a3 * b0;
		c10 -= a0 * b1; c11 -= a1 * b1; c12 -= a2 * b1; c13 -= a3 * b1;
		c20 -= a0 * b2; c21 -= a1 * b2; c22 -= a2 * b2; c23 -= a3 * b2;
		c30 -= a0 * b3; c31 -= a1 * b3; c32 -= a2 * b3; c33 -= a3 * b3;
	}
	C[0 * ldC + 0] = c00; C[0 * ldC + 1] = c01; C[0 * ldC + 2] = c02; C[0 * ldC + 3] = c03;
	C[1 * ldC + 0] = c10; C[1 * ldC + 1] = c11; C[1 * ldC + 2] = c12; C[1 * ldC + 3] = c13;
	C[2 * ldC + 0] = c20; C[2 * ldC + 1] = c21; C[2 * ldC + 2] = c22; C[2 * ldC + 3] = c23;
	C[3 * ldC + 0] = c30; C[3 * ldC + 1] = c31; C[3 * ldC + 2] = c32; C[3 * ldC + 3] = c33;
}
/*
template<typename T>
void __micro_kernel_4x4_times_A_BT(
	T* __restrict A, int ldA, T* __restrict B, int ldB,
	T* __restrict C, int ldC, int k)
{
	T c00 = C[0 * ldC + 0], c01 = C[0 * ldC + 1], c02 = C[0 * ldC + 2], c03 = C[0 * ldC + 3];
	T c10 = C[1 * ldC + 0], c11 = C[1 * ldC + 1], c12 = C[1 * ldC + 2], c13 = C[1 * ldC + 3];
	T c20 = C[2 * ldC + 0], c21 = C[2 * ldC + 1], c22 = C[2 * ldC + 2], c23 = C[2 * ldC + 3];
	T c30 = C[3 * ldC + 0], c31 = C[3 * ldC + 1], c32 = C[3 * ldC + 2], c33 = C[3 * ldC + 3];

	for (int p = 0; p < k; ++p) {
		T a0 = A[p * ldA + 0];
		T a1 = A[p * ldA + 1];
		T a2 = A[p * ldA + 2];
		T a3 = A[p * ldA + 3];

		T b0 = B[p * ldB + 0];
		T b1 = B[p * ldB + 1];
		T b2 = B[p * ldB + 2];
		T b3 = B[p * ldB + 3];

		c00 += a0 * b0; c01 += a1 * b0; c02 += a2 * b0; c03 += a3 * b0;
		c10 += a0 * b1; c11 += a1 * b1; c12 += a2 * b1; c13 += a3 * b1;
		c20 += a0 * b2; c21 += a1 * b2; c22 += a2 * b2; c23 += a3 * b2;
		c30 += a0 * b3; c31 += a1 * b3; c32 += a2 * b3; c33 += a3 * b3;
	}
	C[0 * ldC + 0] = c00; C[0 * ldC + 1] = c01; C[0 * ldC + 2] = c02; C[0 * ldC + 3] = c03;
	C[1 * ldC + 0] = c10; C[1 * ldC + 1] = c11; C[1 * ldC + 2] = c12; C[1 * ldC + 3] = c13;
	C[2 * ldC + 0] = c20; C[2 * ldC + 1] = c21; C[2 * ldC + 2] = c22; C[2 * ldC + 3] = c23;
	C[3 * ldC + 0] = c30; C[3 * ldC + 1] = c31; C[3 * ldC + 2] = c32; C[3 * ldC + 3] = c33;
}
template<typename T>
void __micro_kernel_4x4_times_AT_B(
	T* __restrict A, int ldA, T* __restrict B, int ldB,
	T* __restrict C, int ldC, int k)
{
	T c00 = C[0 * ldC + 0], c01 = C[0 * ldC + 1], c02 = C[0 * ldC + 2], c03 = C[0 * ldC + 3];
	T c10 = C[1 * ldC + 0], c11 = C[1 * ldC + 1], c12 = C[1 * ldC + 2], c13 = C[1 * ldC + 3];
	T c20 = C[2 * ldC + 0], c21 = C[2 * ldC + 1], c22 = C[2 * ldC + 2], c23 = C[2 * ldC + 3];
	T c30 = C[3 * ldC + 0], c31 = C[3 * ldC + 1], c32 = C[3 * ldC + 2], c33 = C[3 * ldC + 3];

	T* A_col = (T*)alloca(sizeof(T) * k * 4);
	for (int p = 0; p < k; ++p) {
		A_col[4 * p + 0] = A[0 * ldA + p];
		A_col[4 * p + 1] = A[1 * ldA + p];
		A_col[4 * p + 2] = A[2 * ldA + p];
		A_col[4 * p + 3] = A[3 * ldA + p];
	}

	for (int p = 0; p < k; ++p) {
		T a0 = A_col[p * 4 + 0];
		T a1 = A_col[p * 4 + 1];
		T a2 = A_col[p * 4 + 2];
		T a3 = A_col[p * 4 + 3];

		T b0 = B[0 * ldB + p];
		T b1 = B[1 * ldB + p];
		T b2 = B[2 * ldB + p];
		T b3 = B[3 * ldB + p];

		c00 += a0 * b0; c01 += a1 * b0; c02 += a2 * b0; c03 += a3 * b0;
		c10 += a0 * b1; c11 += a1 * b1; c12 += a2 * b1; c13 += a3 * b1;
		c20 += a0 * b2; c21 += a1 * b2; c22 += a2 * b2; c23 += a3 * b2;
		c30 += a0 * b3; c31 += a1 * b3; c32 += a2 * b3; c33 += a3 * b3;
	}
	C[0 * ldC + 0] = c00; C[0 * ldC + 1] = c01; C[0 * ldC + 2] = c02; C[0 * ldC + 3] = c03;
	C[1 * ldC + 0] = c10; C[1 * ldC + 1] = c11; C[1 * ldC + 2] = c12; C[1 * ldC + 3] = c13;
	C[2 * ldC + 0] = c20; C[2 * ldC + 1] = c21; C[2 * ldC + 2] = c22; C[2 * ldC + 3] = c23;
	C[3 * ldC + 0] = c30; C[3 * ldC + 1] = c31; C[3 * ldC + 2] = c32; C[3 * ldC + 3] = c33;
}

template<typename T>
void __micro_kernel_8x8_times_A_B(
	T* __restrict A, int ldA, T* __restrict B, int ldB,
	T* __restrict C, int ldC, int k)
{
	T* C0 = C + 0 * ldC, * C1 = C + 1 * ldC, * C2 = C + 2 * ldC, * C3 = C + 3 * ldC,
		* C4 = C + 4 * ldC, * C5 = C + 5 * ldC, * C6 = C + 6 * ldC, * C7 = C + 7 * ldC;

	T c00 = C0[0], c01 = C0[1], c02 = C0[2], c03 = C0[3];
	T c04 = C0[4], c05 = C0[5], c06 = C0[6], c07 = C0[7];

	T c10 = C1[0], c11 = C1[1], c12 = C1[2], c13 = C1[3];
	T c14 = C1[4], c15 = C1[5], c16 = C1[6], c17 = C1[7];

	T c20 = C2[0], c21 = C2[1], c22 = C2[2], c23 = C2[3];
	T c24 = C2[4], c25 = C2[5], c26 = C2[6], c27 = C2[7];

	T c30 = C3[0], c31 = C3[1], c32 = C3[2], c33 = C3[3];
	T c34 = C3[4], c35 = C3[5], c36 = C3[6], c37 = C3[7];

	T c40 = C4[0], c41 = C4[1], c42 = C4[2], c43 = C4[3];
	T c44 = C4[4], c45 = C4[5], c46 = C4[6], c47 = C4[7];

	T c50 = C5[0], c51 = C5[1], c52 = C5[2], c53 = C5[3];
	T c54 = C5[4], c55 = C5[5], c56 = C5[6], c57 = C5[7];

	T c60 = C6[0], c61 = C6[1], c62 = C6[2], c63 = C6[3];
	T c64 = C6[4], c65 = C6[5], c66 = C6[6], c67 = C6[7];

	T c70 = C7[0], c71 = C7[1], c72 = C7[2], c73 = C7[3];
	T c74 = C7[4], c75 = C7[5], c76 = C7[6], c77 = C7[7];

	T* B0 = B + 0 * ldB, * B1 = B + 1 * ldB, * B2 = B + 2 * ldB, * B3 = B + 3 * ldB,
		* B4 = B + 4 * ldB, * B5 = B + 5 * ldB, * B6 = B + 6 * ldB, * B7 = B + 7 * ldB;
	for (int p = 0; p < k; ++p) {
		T a0 = A[0];
		T a1 = A[1];
		T a2 = A[2];
		T a3 = A[3];
		T a4 = A[4];
		T a5 = A[5];
		T a6 = A[6];
		T a7 = A[7];

		T b0 = B0[p];
		T b1 = B1[p];
		T b2 = B2[p];
		T b3 = B3[p];
		T b4 = B4[p];
		T b5 = B5[p];
		T b6 = B6[p];
		T b7 = B7[p];

		c00 += a0 * b0; c01 += a1 * b0; c02 += a2 * b0; c03 += a3 * b0;
		c04 += a4 * b0; c05 += a5 * b0; c06 += a6 * b0; c07 += a7 * b0;

		c10 += a0 * b1; c11 += a1 * b1; c12 += a2 * b1; c13 += a3 * b1;
		c14 += a4 * b1; c15 += a5 * b1; c16 += a6 * b1; c17 += a7 * b1;

		c20 += a0 * b2; c21 += a1 * b2; c22 += a2 * b2; c23 += a3 * b2;
		c24 += a4 * b2; c25 += a5 * b2; c26 += a6 * b2; c27 += a7 * b2;

		c30 += a0 * b3; c31 += a1 * b3; c32 += a2 * b3; c33 += a3 * b3;
		c34 += a4 * b3; c35 += a5 * b3; c36 += a6 * b3; c37 += a7 * b3;

		c40 += a0 * b4; c41 += a1 * b4; c42 += a2 * b4; c43 += a3 * b4;
		c44 += a4 * b4; c45 += a5 * b4; c46 += a6 * b4; c47 += a7 * b4;

		c50 += a0 * b5; c51 += a1 * b5; c52 += a2 * b5; c53 += a3 * b5;
		c54 += a4 * b5; c55 += a5 * b5; c56 += a6 * b5; c57 += a7 * b5;

		c60 += a0 * b6; c61 += a1 * b6; c62 += a2 * b6; c63 += a3 * b6;
		c64 += a4 * b6; c65 += a5 * b6; c66 += a6 * b6; c67 += a7 * b6;

		c70 += a0 * b7; c71 += a1 * b7; c72 += a2 * b7; c73 += a3 * b7;
		c74 += a4 * b7; c75 += a5 * b7; c76 += a6 * b7; c77 += a7 * b7;

		A = A + ldA;
	}

	C0[0] = c00, C0[1] = c01, C0[2] = c02, C0[3] = c03;
	C0[4] = c04, C0[5] = c05, C0[6] = c06, C0[7] = c07;

	C1[0] = c10, C1[1] = c11, C1[2] = c12, C1[3] = c13;
	C1[4] = c14, C1[5] = c15, C1[6] = c16, C1[7] = c17;

	C2[0] = c20, C2[1] = c21, C2[2] = c22, C2[3] = c23;
	C2[4] = c24, C2[5] = c25, C2[6] = c26, C2[7] = c27;

	C3[0] = c30, C3[1] = c31, C3[2] = c32, C3[3] = c33;
	C3[4] = c34, C3[5] = c35, C3[6] = c36, C3[7] = c37;

	C4[0] = c40, C4[1] = c41, C4[2] = c42, C4[3] = c43;
	C4[4] = c44, C4[5] = c45, C4[6] = c46, C4[7] = c47;

	C5[0] = c50, C5[1] = c51, C5[2] = c52, C5[3] = c53;
	C5[4] = c54, C5[5] = c55, C5[6] = c56, C5[7] = c57;

	C6[0] = c60, C6[1] = c61, C6[2] = c62, C6[3] = c63;
	C6[4] = c64, C6[5] = c65, C6[6] = c66, C6[7] = c67;

	C7[0] = c70, C7[1] = c71, C7[2] = c72, C7[3] = c73;
	C7[4] = c74, C7[5] = c75, C7[6] = c76, C7[7] = c77;
}
*/
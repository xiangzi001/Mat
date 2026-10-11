
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

template<typename t>
auto __MaxAbs(t* a, int len) {
	auto max = fabs(a[0]);
	t temp;
	for (int i = 1; i < len; ++i) {
		temp = fabs(a[i]);
		if (temp > max) max = temp;
	}
	return max;
}




template<typename t1, typename t2, typename T>
void __micro_kernel_4x4_times_A_B(
	t1* __restrict A, int ldA, t2* __restrict B, int ldB,
	T* __restrict C, int ldC, int k)
{
	T c00 = C[0 * ldC + 0], c01 = C[0 * ldC + 1], c02 = C[0 * ldC + 2], c03 = C[0 * ldC + 3];
	T c10 = C[1 * ldC + 0], c11 = C[1 * ldC + 1], c12 = C[1 * ldC + 2], c13 = C[1 * ldC + 3];
	T c20 = C[2 * ldC + 0], c21 = C[2 * ldC + 1], c22 = C[2 * ldC + 2], c23 = C[2 * ldC + 3];
	T c30 = C[3 * ldC + 0], c31 = C[3 * ldC + 1], c32 = C[3 * ldC + 2], c33 = C[3 * ldC + 3];

	for (int p = 0; p < k; ++p) {
		t1 a0 = A[0];
		t1 a1 = A[1];
		t1 a2 = A[2];
		t1 a3 = A[3];

		t2 b0 = B[0 * ldB + p];
		t2 b1 = B[1 * ldB + p];
		t2 b2 = B[2 * ldB + p];
		t2 b3 = B[3 * ldB + p];

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

#ifdef XiangZiMatImport_SIMD
void __times_A_B(const MAT<double>& A, const MAT<double>& B, const MAT<double>& C) {
	int i, j;

	__set0(C.data, C.rows * C.cols);
	for (j = 0; j < C.cols; j = j + 8) {
		if (j + 8 > C.cols) {
			int k;
			for (; j < C.cols; ++j) {
				double* b = B.data + j * B.rows;
				for (k = 0; k < A.cols; ++k) {
					__mixed(C.data + j * C.rows, A.data + k * A.rows, b[k], C.data + j * C.rows, C.rows);
				}
			}
			break;
		}
		else {
			for (i = 0; i < C.rows; i = i + 8) {
				if (i + 8 > C.rows) {
					int y, k;
					for (y = j; y < j + 8; ++y) {
						double* b = B.data + j * B.rows;
						for (k = 0; k < A.cols; ++k) {
							__mixed(C.data + j * C.rows, A.data + k * A.rows, b[k], C.data + j * C.rows, C.rows);
						}
					}
				}
				else __micro_kernel_8x8_times_A_B(
					A.data + i, A.rows, B.data + j * B.rows, B.rows, C.data + j * C.rows + i, C.rows, A.cols
				);
			}
		}
	}
}
#endif // XiangZiMatImport_SIMD

template<typename t1, typename t2, typename t3>
void __times_A_B(const MAT<t1>& A, const MAT<t2>& B, const MAT<t3>& C) {
	int i, j;

	__set0(C.data, C.rows * C.cols);
	for (j = 0; j < C.cols; j = j + 4) {
		if (j + 4 > C.cols) {
			int k;
			for (; j < C.cols; ++j) {
				t3* b = B.data + j * B.rows;
				for (k = 0; k < A.cols; ++k) {
					__mixed(C.data + j * C.rows, A.data + k * A.rows, b[k], C.data + j * C.rows, C.rows);
				}
			}
			break;
		}
		else {
			for (i = 0; i < C.rows; i = i + 4) {
				if (i + 4 > C.rows) {
					int y, k;
					for (y = j; y < j + 4; ++y) {
						t3* b = B.data + j * B.rows;
						for (k = 0; k < A.cols; ++k) {
							__mixed(C.data + j * C.rows, A.data + k * A.rows, b[k], C.data + j * C.rows, C.rows);
						}
					}
				}
				else __micro_kernel_4x4_times_A_B(
					A.data + i, A.rows, B.data + j * B.rows, B.rows, C.data + j * C.rows + i, C.rows, A.cols
				);
			}
		}
	}
}
template<typename t1, typename t2>
auto operator * (const MAT<t1>& A, const MAT<t2>& B) {
	MAT<decltype(std::declval<t1>()* std::declval<t2>())> n(A.rows, B.cols);
	__times_A_B(A, B, n);
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

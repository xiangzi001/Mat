
#include"Norm.h"
#include"LU.h"
#include"LLT.h"
#include"Solve_direct.h"
#include"Solve_iterative.h"

template<typename t>
MatData<t> _T(MAT<t>& A) {
	MatData<t> n(A.cols, A.rows);
	for (int i = 0; i < A.cols; ++i)
		for (int j = 0; j < A.rows; ++j) {
			n[i][j] = A[j][i];
		}

	return n;
}
template<typename t>
MatData<t> _T(MatData<t> A) {
	MAT<t> a(A);
	return _T(a);
}

template<typename t>
MatData<complex<t>> _H(MAT<complex<t>>& A) {
	MatData<complex<t>> n(A.cols, A.rows);
	for (int i = 0; i < A.cols; ++i)
		for (int j = 0; j < A.rows; ++j) {
			n[i][j] = A[j][i].conj();
		}

	return n;
}
template<typename t>
MatData<complex<t>> _H(MatData<complex<t>> A) {
	MAT<complex<t>> a(A);
	return _H(a);
}


template<typename T>
void times(MAT<T>& A, MAT<T>& B, MAT<T>& C) {
	int i, j, k = C.rows * C.cols;
	for (i = 0; i < k; ++i) {
		C.data[i] = 0;
	}
	T v;
	for (i = 0; i < C.rows; ++i) {
		for (k = 0; k < A.cols; ++k) {
			v = A[i][k];
			for (j = 0; j < C.cols; ++j)
				C[i][j] += v * B[k][j];
		}
	}
}

template<typename T>
void times_A_BT(MAT<T>& A, MAT<T>& B, MAT<T>& C) {
	int i, j, k;
	for (i = 0; i < C.rows; ++i) {
		for (j = 0; j < C.rows; ++j) {
			C[i][j] = 0;
			for (k = 0; k < A.cols; ++k)
				C[i][j] += A[i][k] * B[j][k];
		}
	}
}

template<typename T>
void times_AT_B(MAT<T>& A, MAT<T>& B, MAT<T>& C) {
	int i, j, k = C.rows * C.cols;
	for (i = 0; i < k; ++i)
		C.data[i] = 0;
	for (k = 0; k < A.rows; ++k) {
		for (i = 0; i < C.rows; ++i) {
			for (j = 0; j < C.cols; ++j) {
				C[i][j] += A[k][i] * B[k][j];
			}
		}
	}
}

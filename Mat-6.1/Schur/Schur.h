
#include"__Schur_R_Hessenberg.h"
#include"__Schur_R_QR.h"
#include"__Schur_I.h"

template<typename t>
void Schur_R(MAT<t>& A, MAT<t>& T, MAT<t>& U, t eps) {
	int i;
	int s(T.rows * T.cols);
	for (int i = 0; i < s; ++i) {
		T.data[i] = A.data[i];
	}
	s = U.rows * U.cols;
	for (i = 0; i < s; ++i) {
		U.data[i] = 0;
	}
	for (i = 0; i < U.rows; ++i) {
		U[i][i] = 1;
	}

	t* w = new t[A.rows + 7];
	t* v = new t[A.rows + 64 / sizeof(t) - 1];

	__Schur_HouseHolder_R(U, T, w, v);

	delete[] w;
	delete[] v;

	__Schur_QR_R(U, T, 0, T.rows, eps);
}

template<typename t1, typename t2>
void Schur_I(MAT<t1>& A, MAT<complex<t2>>& T, MAT<complex<t2>>& U, t2 eps) {
	int i;
	int s(T.rows * T.cols);
	for (int i = 0; i < s; ++i) {
		T.data[i] = A.data[i];
	}
	s = U.rows * U.cols;
	for (i = 0; i < s; ++i) {
		U.data[i] = 0;
	}
	for (i = 0; i < U.rows; ++i) {
		U[i][i] = 1;
	}

	complex<t2>* w = new complex<t2>[A.rows];

	__Schur_HouseHolder_I(U, T, w);

	delete[] w;

	__Schur_Givens_I(U, T, 0, T.rows, eps);
}

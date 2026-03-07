
#include"__Schur_R.h"
#include"__Schur_I.h"

template<typename t1, typename t2>
void Schur_R(MAT<t1>& A, MAT<t2>& T, MAT<t2>& U, double eps) {

	if (A.data != T.data)
		__copy(T.data, A.data, T.rows * T.cols);
	__set0(U.data, U.rows * U.cols);
	for (int i = 0; i < U.rows; ++i) {
		U[i][i] = 1;
	}

	t2* w = new t2[A.rows];
	t2* v = new t2[A.rows];

	__Schur_HouseHolder_R(U, T, w, v);

	delete[] w;
	delete[] v;

	__Schur_QR_R(U, T, 0, T.rows, eps);
}

template<typename t1, typename t2>
void Schur_I(MAT<t1>& A, MAT<complex<t2>>& T, MAT<complex<t2>>& U, double eps) {

	if (A.data != T.data)
		__copy(T.data, A.data, T.rows * T.cols);
	__set0(U.data, U.rows * U.cols);
	for (int i = 0; i < U.rows; ++i) {
		U[i][i] = 1;
	}

	complex<t2>* w = new complex<t2>[A.rows];
	complex<t2>* v = new complex<t2>[A.rows];

	__Schur_HouseHolder_I(U, T, w, v);

	delete[] w;
	delete[] v;

	__Schur_Givens_I(U, T, 0, T.rows, eps);
}

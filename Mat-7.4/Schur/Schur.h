
#include"__Schur_R.h"
#include"__Schur_I.h"

template<typename t1, typename t2>
void Schur_R(MAT<t1>& A, MAT<t2>& T, MAT<t2>& U, double eps) {

	if (A.data != T.data)
		__copy(T.data, A.data, T.rows * T.cols);

	t2* v = new t2[A.rows];
	__Schur_HouseHolder_R(U, T, v);
	delete[] v;

	__Schur_QR_R(U, T, 0, T.rows, eps);
}

template<typename t1, typename t2>
void Schur_I(MAT<t1>& A, MAT<complex<t2>>& T, MAT<complex<t2>>& U, double eps) {

	if (A.data != T.data)
		__copy(T.data, A.data, T.rows * T.cols);

	complex<t2>* v = new complex<t2>[A.rows];

	__Schur_HouseHolder_I(U, T, v);

	delete[] v;

	__Schur_Givens_I(U, T, 0, T.rows, eps);
}


#include"__QR.h"
#include"__Function.h"

template<typename t1, typename t2>
void EV(MAT<t1>& A, MAT<complex<t2>>& E, double eps) {
	int i, j(E.rows * E.cols);
	for (i = 0; i < j; ++i) {
		E.data[i] = A.data[i];
	}

	complex<t2>* w = new complex<t2>[E.cols];
	__EV_HouseHolder(E, w);
	delete[] w;
	__EV_Givens(E, 0, E.rows, eps);

	for (i = 0; i < E.rows; ++i) {
		for (j = i + 1; j < E.cols; ++j) {
			E[i][j] = 0;
		}
	}
}

template<typename t, typename _t>
void Roots(Func<_t>& func, complex<t>* ans, double eps) {
	int n (func.times);
	_t* data (func.data);
	complex<t> temp(-1 / data[n]);
	MAT<complex<t>> A(n, n);

	--n;
	int i;
	for (i = 0; i < A.rows; ++i)
		A[0][i] = data[n - i] * temp;

	n = A.rows * A.rows;
	for (i = A.rows; i < n; ++i)
		A.data[i] = 0;
	for (i = 1; i < A.rows; ++i) 
		A[i][i - 1] = 1;
	
	__EV_Givens(A, 0, A.rows, eps);

	for (i = 0; i < A.rows; ++i) {
		ans[i] = A[i][i];
	}
}

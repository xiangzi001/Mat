
#include"__QR_I.h"
#include"__QR_R.h"
#include"__QR_Q.h"

template<typename t>
void EV(MAT<t>& A, MAT<complex<t>>& E, double eps) {
	int i;
	t a, b;

	MAT<t> B;
	B.rows = E.rows;
	B.cols = E.cols;
	B.data = (t*)E.data;

	memcpy(B.data, A.data, sizeof(t) * B.rows * B.cols);

	t* w = new t[A.rows * 2];
	complex<t>* v = (complex<t>*)w;

	/*
	bool flag = true;
	while (flag) {
		flag = __EV_R_BalanceMatrix(B, w, w + A.rows + 64 / sizeof(a), 1);
	}
	__EV_HouseHolder_R(B, w, w + A.rows + 64 / sizeof(a));
	__EV_QR_R(B, 0, B.rows, eps);
	*/

	i = 0;
	while (i < B.cols - 1) {
		if (fabs(B(i + 1, i)) < eps) {
			v[i] = B(i, i);
			++i;
		}
		else {
			a = B(i, i) + B(i + 1, i + 1);
			b = sqrt((complex<t>)(a * a - (B(i, i) * B(i + 1, i + 1) - B(i, i + 1) * B(i + 1, i)) * 4));
			v[i] = ((complex<t>)a + b) * 0.5;
			v[i + 1] = ((complex<t>)a - b) * 0.5;
			i += 2;
		}
	}
	if (i < B.cols) {
		v[i] = B(i, i);
	}

	ZeroMemory(E.data, sizeof(complex<a>) * E.rows * E.cols);
	for (i = 0; i < E.rows; ++i) {
		E[i][i] = v[i];
	}

	B.data = 0;
	delete[] w;
}

template<typename t>
void EV(MAT<complex<t>>& A, MAT<complex<t>>& E, double eps) {
	int i, j;

	memcpy(E.data, A.data, sizeof(complex<t>) * E.rows * E.cols);

	complex<t>* w = new complex<t>[E.cols];
	complex<t>* v = new complex<t>[E.cols];
	__EV_HouseHolder_I(E, w, v);
	delete[] w;
	delete[] v;
	__EV_Givens(E, 0, E.rows, eps);

	for (i = 0; i < E.rows; ++i) {
		for (j = i + 1; j < E.cols; ++j) {
			E[i][j] = 0;
		}
	}
}

#ifdef XiangZiFuncImport

template<typename t>
void Roots(Func<complex<t>>& func, complex<t>* ans, double eps) {
	int n (func.times);
	complex<t>* data (func.data);
	complex<t> temp(-1 / data[n]);
	MAT<complex<t>> A(n, n);

	--n;
	int i;
	for (i = 0; i < A.rows; ++i)
		A(0, i) = data[n - i] * temp;

	n = A.rows * A.rows;
	for (i = A.rows; i < n; ++i)
		A.data[i] = 0;
	for (i = 1; i < A.rows; ++i) 
		A(i, i - 1) = 1;
	
	__EV_Givens(A, 0, A.rows, eps);

	for (i = 0; i < A.rows; ++i) {
		ans[i] = A(i, i);
	}
}
template<typename t>
void Roots(Func<t>& func, complex<t>* ans, double eps) {
	int n(func.times);
	t* data(func.data);
	t temp(-1 / data[n]);
	complex<t> a, b;
	MAT<t> A(n, n);

	int i;
	ZeroMemory(A.data + n, sizeof(t) * n * (n - 1));
	--n;
	for (i = 0; i < A.rows; ++i)
		A(0, i) = data[n - i] * temp;
	for (i = 1; i < A.rows; ++i)
		A(i, i - 1) = 1;

	__EV_QR_R(A, 0, A.rows, eps);

	n = 0;
	while (n < A.cols - 1) {
		if (fabs(A(n + 1, n)) < eps) {
			ans[n] = A(n, n);
			++n;
		}
		else {
			a = A(n, n) + A(n + 1, n + 1);
			b = sqrt((complex<t>)(a * a - (A(n, n) * A(n + 1, n + 1) - A(n, n + 1) * A(n + 1, n)) * 4));
			ans[n] = ((complex<t>)a + b) * 0.5;
			ans[n + 1] = ((complex<t>)a - b) * 0.5;
			n += 2;
		}
	}
	if (n < A.cols) {
		ans[n] = A(n, n);
	}
}

#endif // XiangZiFuncImport

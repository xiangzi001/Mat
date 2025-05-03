
#include"__QR_I.h"
#include"__QR_R.h"

template<typename t>
void EV(MAT<t>& A, MAT<complex<t>>& E, double eps) {
	int i;

	MAT<t> B;
	B.rows = E.rows;
	B.cols = E.cols;
	B.data = (t*)E.data;

	memcpy(B.data, A.data, sizeof(t) * B.rows * B.cols);

	t* w = new t[A.rows * 2 + 128 / sizeof(t)];

	bool flag = true;
	while (flag) {
		flag = __EV_R_BalanceMatrix(B, w, w + A.rows + 64 / sizeof(t), 1);
	}
	__EV_HouseHolder_R(B, w, w + A.rows + 64 / sizeof(t));
	__EV_QR_R(B, 0, B.rows, eps);

	complex<t> a, b;
	complex<t>* v = (complex<t>*)w;
	int n = 0;
	while (n < B.cols - 1) {
		if (fabs(B[n + 1][n]) < eps) {
			v[n] = B[n][n];
			++n;
		}
		else {
			a = B[n][n] + B[n + 1][n + 1];
			b = sqrt(a * a - (B[n][n] * B[n + 1][n + 1] - B[n][n + 1] * B[n + 1][n]) * 4);
			v[n] = (a + b) * 0.5;
			v[n + 1] = (a - b) * 0.5;
			n += 2;
		}
	}
	if (n < B.cols) {
		v[n] = B[n][n];
	}

	ZeroMemory(E.data, sizeof(complex<t>) * E.rows * E.cols);
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
	__EV_HouseHolder(E, w);
	delete[] w;
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
template<typename t>
void Roots(Func<t>& func, complex<t>* ans, double eps) {
	int n(func.times);
	t* data(func.data);
	t temp(-1 / data[n]);
	complex<t> a, b;
	MAT<t> A(n, n);

	--n;
	int i;
	for (i = 0; i < A.rows; ++i)
		A[0][i] = data[n - i] * temp;

	n = A.rows * A.rows;
	for (i = A.rows; i < n; ++i)
		A.data[i] = 0;
	for (i = 1; i < A.rows; ++i)
		A[i][i - 1] = 1;

	__EV_QR_R(A, 0, A.rows, eps);

	n = 0;
	while (n < A.cols - 1) {
		if (fabs(A[n + 1][n]) < eps) {
			ans[n] = A[n][n];
			++n;
		}
		else {
			a = A[n][n] + A[n + 1][n + 1];
			b = sqrt(a * a - (A[n][n] * A[n + 1][n + 1] - A[n][n + 1] * A[n + 1][n]) * 4);
			ans[n] = (a + b) * 0.5;
			ans[n + 1] = (a - b) * 0.5;
			n += 2;
		}
	}
	if (n < A.cols) {
		ans[n] = A[n][n];
	}
}

#endif // XiangZiFuncImport

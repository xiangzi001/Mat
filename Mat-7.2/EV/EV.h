
#include"__QR_I.h"
#include"__QR_R.h"
#include"__QR_Q.h"

template<typename t>
void __EV(MAT<t>& A, complex<t>* v, double eps) {
	int i;
	t a, b;
	t* w = (t*) v;

	/*
	bool flag = true;
	while (flag) {
		flag = __EV_R_BalanceMatrix(A, w, w + A.rows + 64 / sizeof(a), 1);
	}
	__EV_HouseHolder_R(A, w, w + A.rows + 64 / sizeof(a));
	__EV_QR_R(A, 0, A.rows, eps);
	*/

	i = 0;
	while (i < A.cols - 1) {
		if (fabs(A(i + 1, i)) < eps) {
			v[i] = A(i, i);
			++i;
		}
		else {
			a = A(i, i) + A(i + 1, i + 1);
			b = sqrt((complex<t>)(a * a - (A(i, i) * A(i + 1, i + 1) - A(i, i + 1) * A(i + 1, i)) * 4));
			v[i] = ((complex<t>)a + b) * 0.5;
			v[i + 1] = ((complex<t>)a - b) * 0.5;
			i += 2;
		}
	}
	if (i < A.cols) {
		v[i] = A(i, i);
	}
}

template<typename t1, typename t2>
void EV(MAT<t1>& A, complex<t2>* v, double eps) {
	MAT<t2>B = A;
	__EV(B, v, eps);
}

template<typename t>
void __EV(MAT<complex<t>>& A, complex<t>* v, double eps) {
	int i;
	complex<t>* w = new complex<t>[A.cols];
	__EV_HouseHolder_I(A, w, v);
	delete[] w;
	__EV_Givens(A, 0, A.rows, eps);

	for (i = 0; i < A.rows; ++i) {
		v[i] = A(i, i);
	}
}
template<typename t1, typename t2>
void EV(MAT<complex<t1>>& A, complex<t2>* v, double eps) {
	MAT<complex<t2>>B = A;
	__EV(B, v, eps);
}

#ifdef XiangZiFuncImport

template<typename t1, typename t2>
void Roots(Func<complex<t1>>& func, complex<t2>* ans, double eps) {
	int n (func.times);
	complex<t1>* data (func.data);
	complex<t2> temp(-complex<t2>(1) / data[n]);
	MAT<complex<t2>> A(n, n);

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

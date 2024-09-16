
#include"__Base.h"
#include"__QR_1.h"
#include"__QR_2.h"
#include"__Jacobi.h"

template<typename T>
void EVD_Q(MAT<T>& A, MAT<T>& E, MAT<T>& Q, double eps, EVD_type type) {
	switch (type)
	{
	case EVD_QR: {
		for (int i = 0; i < E.rows * E.cols; ++i) {
			E.data[i] = A.data[i];
		}
		__EVD_Q_QR_1(E, Q, eps);
		break;
	}
	case EVD_QR_2: {
		for (int i = 0; i < E.rows * E.cols; ++i) {
			E.data[i] = A.data[i];
		}
		__EVD_Q_QR_2(E, Q, eps);
		break;
	}
	case EVD_Jacobi: {
		for (int i = 0; i < E.rows * E.cols; ++i) {
			E.data[i] = A.data[i];
		}
		__EVD_Q_Jacobi(E, Q, eps);
		break;
	}
	default:
		break;
	}
}

template<typename T>
void EVD_UP(MAT<T>& A, MAT<T>& X, T* w) {
	int i, j, k;
	double len;
	T sum;
	T* temp;
	for (i = A.cols - 1; i > 0; --i) {
		for (j = i - 1; j >= 0; --j) {
			temp = A[j];
			sum = 0;
			for (k = j + 1; k < i; ++k) {
				sum += temp[k] * w[k];
			}
			sum += temp[i];
			w[j] = sum / (A[i][i] - temp[j]);
		}

		len = 0;
		for (j = 0; j < i; ++j)
			len += len2(w[j]);
		len = 1 / sqrt(len + 1);
		for (j = 0; j < i; ++j)
			w[j] *= len;
		w[i] = len;

		for (j = 0; j < X.rows; ++j) {
			temp = X[j];
			sum = 0;
			for (k = 0; k <= i; ++k) {
				sum += temp[k] * w[k];
			}
			temp[i] = sum;
		}
	}
	for (i = 0; i < A.rows; ++i) {
		temp = A[i];
		for (j = i + 1; j < A.cols; ++j) {
			temp[j] = 0;
		}
	}
}

template<typename t1, typename t2>
void EVD_X(MAT<t1>& A, MAT<complex<t2>>& E, MAT<complex<t2>>& X, double eps) {
	int i;
	int s(E.rows * E.cols);
	for (int i = 0; i < s; ++i) {
		E.data[i] = A.data[i];
	}
	s = X.rows * X.cols;
	for (i = 0; i < s; ++i) {
		X.data[i] = 0;
	}
	for (i = 0; i < X.rows; ++i) {
		X[i][i] = 1;
	}

	complex<t2>* w = new complex<t2>[A.rows];

	__Schur_HouseHolder_I(X, E, w);

	__Schur_Givens_I(X, E, 0, E.rows, eps);

	EVD_UP(E, X, w);

	delete[] w;
}

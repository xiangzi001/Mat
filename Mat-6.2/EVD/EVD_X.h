
template<typename T, typename t>
void EVD_init(MAT<T>& A, MAT<t>& E, MAT<t>& X) {
	E.init(A.rows, A.rows);
	X.init(A.rows, A.rows);
}

template<typename T>
void EVD_UP(MAT<T>& A, MAT<T>& X, T* w) {
	int i, j, k;
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
		w[i] = 1;

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

template<typename t>
void EVD_X(MAT<complex<t>>& A, MAT<complex<t>>& E, MAT<complex<t>>& X, double eps) {
	int i;

	memcpy(E.data, A.data, sizeof(complex<t>) * E.rows * E.cols);
	ZeroMemory(X.data, sizeof(complex<t>) * X.rows * X.cols);
	for (i = 0; i < X.rows; ++i) {
		X[i][i] = 1;
	}

	complex<t>* w = new complex<t>[A.rows];

	__Schur_HouseHolder_I(X, E, w);

	__Schur_Givens_I(X, E, 0, E.rows, eps);

	EVD_UP(E, X, w);

	delete[] w;
}

template<typename T>
void EVD_UP2_Vec(MAT<T>& A, MAT<T>& Q, MAT<complex<T>>& X, complex<T>* w, complex<T>* x, int i, int n) {
	complex<T> s, t;
	int j;
	while (i >= 0) {
		if (w[i].imag == 0) {
			s = 0;
			for (j = i + 1; j <= n; ++j) {
				s += A[i][j] * x[j];
			}
			x[i] = s / (w[n] - w[i]);
			--i;
		}
		else {
			s = 0;
			for (j = i + 1; j <= n; ++j) {
				s += A[i - 1][j] * x[j];
			}
			t = 0;
			for (j = i + 1; j <= n; ++j) {
				t += A[i][j] * x[j];
			}
			x[i] = 1 / ((-w[n] + A[i - 1][i - 1]) * (-w[n] + A[i][i]) - A[i - 1][i] * A[i][i - 1]);

			x[i - 1] = (t * A[i - 1][i] - s * (-w[n] + A[i][i])) * x[i];
			x[i] = (-t * (-w[n] + A[i - 1][i - 1]) + s * A[i][i - 1]) * x[i];
			i -= 2;
		}
	}

	for (i = 0; i < A.rows; ++i) {
		s = 0;
		for (j = 0; j <= n; ++j) {
			s += Q[i][j] * x[j];
		}
		X[i][n] = s;
	}
}

template<typename T>
void EVD_UP2(MAT<T>& A, MAT<T>& Q, MAT<complex<T>>& E, MAT<complex<T>>& X, complex<T>* w, complex<T>* x, double eps) {
	int i,n = 0;

	complex<T> s, t;

	while (n < A.cols - 1) {
		if (fabs(A[n + 1][n]) < eps) {
			w[n] = A[n][n];
			++n;
		}
		else {
			t = A[n][n] + A[n + 1][n + 1];
			s = sqrt(t * t - (A[n][n] * A[n + 1][n + 1] - A[n][n + 1] * A[n + 1][n]) * 4);
			w[n] = (t + s) * 0.5;
			w[n + 1] = (t - s) * 0.5;
			n += 2;
		}
	}
	if (n < A.cols) {
		w[n] = A[n][n];
	}

	n = A.cols - 1;
	while(n >= 0) {
		if (w[n].imag == 0) {
			x[n] = 1;
			EVD_UP2_Vec(A, Q, X, w, x, n - 1, n);
			--n;
		}
		else {
			x[n] = A[n][n - 1];
			x[n - 1] = w[n] - A[n][n];
			EVD_UP2_Vec(A, Q, X, w, x, n - 2, n);

			for (i = 0; i < X.rows; ++i) {
				X[i][n - 1].real = X[i][n].real;
				X[i][n - 1].imag = -X[i][n].imag;
			}
			n -= 2;
		}
	}

	n = E.rows * E.cols;
	for (i = 0; i < n; ++i) {
		E.data[i] = 0;
	}
	for (i = 0; i < E.rows; ++i) {
		E[i][i] = w[i];
	}

	for (i = 0; i < X.cols; ++i) {
		w[i] = 0;
	}
	T* temp = (T*)w;
	int j;
	for (i = 0; i < X.rows; ++i) {
		for (j = 0; j < X.cols; ++j) {
			temp[j] += X[i][j].len2();
		}
	}
	for (i = 0; i < X.cols; ++i) {
		temp[i] = 1.0 / sqrt(temp[i]);
	}
	for (i = 0; i < X.rows; ++i) {
		for (j = 0; j < X.cols; ++j) {
			X[i][j] *= temp[j];
		}
	}
}

template<typename t>
void EVD_X(MAT<t>& A, MAT<complex<t>>& E, MAT<complex<t>>& X, double eps) {
	int i;
	int s(E.rows * E.cols);

	MAT<t> U, T;
	U.rows = E.rows; U.cols = E.cols; U.data = (t*)(E.data);
	T.rows = E.rows; T.cols = E.cols; T.data = (t*)(E.data) + E.rows * E.cols;
	for (i = 0; i < U.rows; ++i) {
		U.data[i] = 1;
	}
	ZeroMemory(U.data + U.rows, sizeof(t) * U.rows * (U.cols - 1));
	memcpy(T.data, A.data, sizeof(t) * T.rows* T.cols);

	t* w = new t[A.rows * 2 + 128 / sizeof(t)];

	bool flag = true;
	while (flag) {
		flag = __EVD_X_BalanceMatrix(U, T, w, w + A.rows + 64 / sizeof(t), 1);
	}
	for (i = 1; i < U.rows; ++i) {
		U[i][i] = U.data[i];
	}
	ZeroMemory(U.data + 1, sizeof(t) * (U.cols - 1));

	__Schur_HouseHolder_R(U, T, w, w + A.rows + 64 / sizeof(t));
	__Schur_QR_R(U, T, 0, T.rows, eps);

	complex<t>* v = new complex<t>[A.rows];
	EVD_UP2(T, U, E, X, (complex<t>*)w, v, eps);

	delete[] v;
	delete[] w;

	U.data = 0;
	T.data = 0;
}

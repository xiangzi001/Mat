
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

template<typename t>
void EVD_X(MAT<complex<t>>& A, MAT<complex<t>>& E, MAT<complex<t>>& X, double eps) {
	int i;
	int s(E.rows * E.cols);
	for (i = 0; i < s; ++i) {
		E.data[i] = A.data[i];
	}
	s = X.rows * X.cols;
	for (i = 0; i < s; ++i) {
		X.data[i] = 0;
	}
	for (i = 0; i < X.rows; ++i) {
		X[i][i] = 1;
	}

	complex<t>* w = new complex<t>[A.rows];

	__Schur_HouseHolder_I(X, E, w);

	__Schur_Givens_I(X, E, 0, E.rows, eps);

	EVD_UP(E, X, w);

	delete[] w;
}
/*
template<typename t>
void EVD_UP3(MAT<complex<t>>& A, MAT<complex<t>>& X, t eps) {
	t c;
	complex<t> t, s, _s, temp;
	int k, n = A.cols;
	while (n > 0) {

		if (fabs(A[n - 1][n - 2]) < eps) {
			--n;
			continue;
		}

		t = A[n - 2][n - 2] + A[n - 1][n - 1];
		s = sqrt(t * t - (A[n - 2][n - 2] * A[n - 1][n - 1] - A[n - 2][n - 1] * A[n - 1][n - 2]) * 4);

		temp = (t - s) * 0.5;

		t = A[n - 1][n - 2] / (A[n - 2][n - 2] - temp);
		c = 1 / sqrt(t.real * t.real + t.imag * t.imag + 1);
		s = t * c;
		_s = s.conj();

		for (k = n - 2; k < A.rows; ++k) {
			temp = A[n - 2][k] * c + A[n - 1][k] * _s;
			A[n - 1][k] = A[n - 1][k] * c - A[n - 2][k] * s;
			A[n - 2][k] = temp;
		}
		for (k = 0; k < n; ++k) {
			temp = A[k][n - 2] * c + A[k][n - 1] * s;
			A[k][n - 1] = A[k][n - 1] * c - A[k][n - 2] * _s;
			A[k][n - 2] = temp;
		}
		for (k = 0; k < X.rows; ++k) {
			temp = X[k][n - 2] * c + X[k][n - 1] * s;
			X[k][n - 1] = X[k][n - 1] * c - X[k][n - 2] * _s;
			X[k][n - 2] = temp;
		}
		n -= 2;
	}
}
*/
template<typename T>
void EVD_UP2_Vec(MAT<T>& A, MAT<T>& Q, MAT<complex<T>>& X, complex<T>* w, complex<T>* x, int i, int n) {
	complex<T> s, t;
	double len;
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
	len = 0;
	for (i = 0; i <= n; ++i)
		len += x[i].len2();
	len = 1 / sqrt(len);
	for (i = 0; i <= n; ++i)
		x[i] *= len;
	for (i = 0; i < A.rows; ++i) {
		s = 0;
		for (j = 0; j <= n; ++j) {
			s += Q[i][j] * x[j];
		}
		X[i][n] = s;
	}
}

template<typename T>
void EVD_UP2(MAT<T>& A, MAT<T>& Q, MAT<complex<T>>& E, MAT<complex<T>>& X, complex<T>* w, T eps) {
	int i,n = 0;

	complex<T>* x = w + A.cols;
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
}

template<typename t>
void EVD_X(MAT<t>& A, MAT<complex<t>>& E, MAT<complex<t>>& X, double eps) {
	int i, j;
	int s(E.rows * E.cols);

	MAT<t> U, T;
	U.rows = E.rows; U.cols = E.cols; U.data = (t*)(E.data);
	T.rows = E.rows; T.cols = E.cols; T.data = (t*)(E.data) + E.rows * E.cols;
	for (i = 0; i < s; ++i) {
		U.data[i] = 0;
	}
	for (i = 0; i < U.rows; ++i) {
		U[i][i] = 1;
	}
	for (i = 0; i < s; ++i) {
		T.data[i] = A.data[i];
	}

	MAT<t> Temp(3, 3);
	complex<t>* w = new complex<t>[A.rows * 2];

	__Schur_HouseHolder_R(U, T, (t*)w);
	__Schur_QR_R(U, T, Temp, 0, T.rows, eps);

	EVD_UP2(T, U, E, X, w, eps);

	U.data = 0;
	T.data = 0;
	delete[] w;
}

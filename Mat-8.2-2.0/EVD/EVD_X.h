
template<typename T>
void EVD_UP1_Vec(MAT<T>& A, MAT<T>& Q, MAT<T>& X, T* x, int n) {
	T w = A(n, n);
	T temp;
	T* Temp;
	T* Temp2;
	int i, j;

	__copy(x, A[n], n);
	x[n] = 1;

	i = n - 1;
	while (i >= 0) {
		Temp = A[i];
		x[i] /= (w - Temp[i]);
		temp = x[i];

		for (j = 0; j < i; ++j) {
			x[j] += Temp[j] * temp;
		}
		--i;
	}

	Temp2 = X[n];
	__set0(Temp2, X.rows);
	for (j = 0; j <= n; ++j) {
		Temp = Q[j];
		temp = x[j];
		for (i = 0; i < A.rows; ++i) {
			Temp2[i] += Temp[i] * temp;
		}
	}
	x[n] = w;
}
template<typename T>
void EVD_UP1(MAT<complex<T>>& E, MAT<complex<T>>& X, complex<T>* x) {
	int i, n = 0;

	T t;
	complex<T> s;

	n = E.cols - 1;
	while (n >= 0) {
		EVD_UP1_Vec(X, E, X, x, n);
		--n;
	}

	__set0(E.data, E.rows * E.cols);
	for (i = 0; i < E.rows; ++i) {
		E(i, i) = x[i];
	}
	
	T temp;
	int j;
	for (i = 0; i < X.cols; ++i) {
		t = 0;
		x = X[i];
		for (j = 0; j < X.rows; ++j) {
			temp = x[j].len2();
			if (t < temp) {
				n = j;
				t = temp;
			}
		}
		s = x[n];
		s = s.conj() / sqrt(__dot((T*)(X[i]), (T*)(X[i]), X.rows * 2) * (s.real * s.real + s.imag * s.imag));
		__times(X[i], s, X[i], X.rows);
	}
}

template<typename t1, typename t2>
void EVD_X(MAT<complex<t1>>& A, MAT<complex<t2>>& E, MAT<complex<t2>>& X, t2 eps) {

	if (A.data != X.data)
		__copy(X.data, A.data, X.rows * X.cols);
	__set0(E.data, A.rows * A.cols);
	for (int i = 0; i < X.rows; ++i) {
		E(i, i) = 1;
	}

	complex<t2>* v = new complex<t2>[A.rows];

	__Schur_Householder_I(E, X, v);
	__Schur_Givens_I(E, X, 0, E.rows, eps);

	EVD_UP1(E, X, v);

	delete[] v;
}

/*
template<typename T>
bool __EVD_X_BalanceMatrix(MAT<T>& Q, MAT<T>& R, T* w, T* v, T eps) {
	int i = 0, j;
	double temp;
	bool ret = false;

	for (i = 0; i < R.rows; ++i) {
		v[i] = -R[i][i] * R[i][i];
	}
	for (i = 0; i < R.rows; ++i) {
		temp = 0;
		for (j = 0; j < R.rows; ++j) {
			temp += R[i][j] * R[i][j];
		}
		w[i] = temp + v[i];
	}
	for (i = 0; i < R.rows; ++i) {
		for (j = 0; j < R.rows; ++j) {
			v[j] += R[i][j] * R[i][j];
		}
	}

	for (i = 0; i < R.rows; ++i) {
		w[i] = pow(w[i] / v[i], 0.25);
		ret = ret || (fabs(w[i] - 1) > eps);
	}

	for (i = 0; i < Q.rows; ++i) {
		Q.data[i] *= w[i];
	}
	for (i = 0; i < R.rows; ++i) {
		for (j = 0; j < R.rows; ++j) {
			R[i][j] *= w[j];
		}
	}
	for (i = 0; i < R.rows; ++i) {
		temp = (T)1 / w[i];
		for (j = 0; j < R.rows; ++j) {
			R[i][j] *= temp;
		}
	}
	return ret;
}
*/

template<typename T>
void EVD_UP2_Vec(MAT<T>& A, MAT<T>& Q, MAT<complex<T>>& X, complex<T>* w, complex<T>* x, int i, int n) {
	complex<T> s, t;
	T* Temp;
	int j;
	while (i >= 0) {
		if (w[i].imag == 0) {
			s = 0;
			Temp = A[i];
			for (j = i + 1; j <= n; ++j) {
				s += Temp[j] * x[j];
			}
			x[i] = s / (w[n] - w[i]);
			--i;
		}
		else {
			s = 0;
			Temp = A[i - 1];
			for (j = i + 1; j <= n; ++j) {
				s += Temp[j] * x[j];
			}
			t = 0;
			Temp = A[i];
			for (j = i + 1; j <= n; ++j) {
				t += Temp[j] * x[j];
			}
			x[i] = 1 / ((-w[n] + A[i - 1][i - 1]) * (-w[n] + A[i][i]) - A[i - 1][i] * A[i][i - 1]);

			x[i - 1] = (t * A[i - 1][i] - s * (-w[n] + A[i][i])) * x[i];
			x[i] = (-t * (-w[n] + A[i - 1][i - 1]) + s * A[i][i - 1]) * x[i];
			i -= 2;
		}
	}

	complex<T>* Temp2 = X[n];
	for (j = 0; j <= n; ++j) {
		Temp = Q[j];
		s = x[j];
		for (i = 0; i < A.rows; ++i) {
			Temp2[i] += Temp[i] * s;
		}
	}
}

template<typename T>
void EVD_UP2(MAT<T>& A, MAT<T>& Q, MAT<complex<T>>& E, MAT<complex<T>>& X, complex<T>* w, complex<T>* x, T eps) {
	int i,n = 0;

	T t;
	complex<T> s;

	while (n < A.cols - 1) {
		if (fabs(A(n, n + 1)) < eps) {
			w[n] = A(n, n);
			++n;
		}
		else {
			t = A(n, n) + A(n + 1, n + 1);
			s = sqrt((complex<T>)(t * t - (A(n, n) * A(n + 1, n + 1) - A(n, n + 1) * A(n + 1, n)) * 4));
			w[n] = ((complex<T>)t + s) * 0.5;
			w[n + 1] = ((complex<T>)t - s) * 0.5;
			n += 2;
		}
	}
	if (n < A.cols) {
		w[n] = A(n, n);
	}

	__set0(X.data, X.rows * X.cols);
	n = A.cols - 1;
	while(n >= 0) {
		if (w[n].imag == 0) {
			x[n] = 1;
			EVD_UP2_Vec(A, Q, X, w, x, n - 1, n);
			--n;
		}
		else {
			x[n] = A(n - 1, n);
			x[n - 1] = w[n] - A(n, n);
			EVD_UP2_Vec(A, Q, X, w, x, n - 2, n);

			complex<T>* a = X[n - 1];
			complex<T>* b = X[n];
			for (i = 0; i < X.rows; ++i) {
				a[i].real = b[i].real;
				a[i].imag = -b[i].imag;
			}
			n -= 2;
		}
	}

	__set0(E.data, E.rows * E.cols);
	for (i = 0; i < E.rows; ++i) {
		E[i][i] = w[i];
	}

	for (i = 0; i < X.cols; ++i) {
		t = (T)1 / sqrt(__dot((T*)(X[i]), (T*)(X[i]), X.rows * 2));
		__times((T*)(X[i]), t, (T*)(X[i]), X.rows * 2);
	}
}

template<typename t1, typename t2>
void EVD_X(MAT<t1>& A, MAT<complex<t2>>& E, MAT<complex<t2>>& X, t2 eps) {
	
	MAT<t2> U, T;
	U.rows = E.rows; U.cols = E.cols; U.data = (t2*)(E.data);
	T.rows = E.rows; T.cols = E.cols; T.data = (t2*)(E.data) + E.rows * E.cols;
	__set0(U.data, U.rows * U.cols);
	for (int i = 0; i < U.rows; ++i) {
		U[i][i] = 1;
	}
	if (A.data != T.data)
	__copy(T.data, A.data, T.rows * T.cols);

	t2* w = new t2[A.rows * 2];

	/*
	for (i = 0; i < U.rows; ++i) {
		U.data[i] = 1;
	}
	bool flag = true;
	while (flag) {
		flag = __EVD_X_BalanceMatrix(U, T, w, w + A.rows, 1);
	}
	for (i = 1; i < U.rows; ++i) {
		U[i][i] = U.data[i];
	}
	__set0(U.data + 1, (U.cols - 1));
	*/
	
	__Schur_Householder_R(U, T, w);

	int i, j;
	for (i = 1; i < T.rows; ++i) {
		w[i] = T(i, i - 1);
	}
	for (i = 0; i < T.rows; ++i) {
		for (j = 0; j < i; ++j) {
			T(i, j) = T(j, i);
		}
	}
	for (i = 2; i < T.rows; ++i) {
		__set0(T[i], i - 1);
	}
	for (i = 1; i < T.rows; ++i) {
		T(i - 1, i) = w[i];
	}

	__Schur_QR_R(U, T, 0, T.rows, eps);

	complex<t2>* v = new complex<t2>[A.rows];
	EVD_UP2(T, U, E, X, (complex<t2>*)w, v, eps);

	delete[] v;
	delete[] w;

	U.data = 0;
	T.data = 0;
}

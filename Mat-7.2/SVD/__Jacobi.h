
template<typename T>
void __LQT_HouseHolder(MAT<T>& Q, MAT<T>& R, T* w, T* v) {
	int n = 0;
	if (R.rows > R.cols) {
		n = R.cols;
	}
	else {
		n = R.rows;
	};
	int r;

	T cr, hr;

	for (r = 0; r < n; ++r) {
		for (int i = r; i < R.cols; ++i) {
			w[i] = R(r, i);
		}
		cr = sqrt(__dot(w + r, w + r, R.cols - r));

		if (w[r] < 0.0)cr = -cr;
		w[r] += cr;
		hr = 1.0 / sqrt(cr * w[r]);

		__times(w + r, hr, w + r, R.cols - r);
		__ApplyHouseHolderOnTheRight(Q, w, v, r);
		__ApplyHouseHolderOnTheRight(R, w, v, r);
	}
}

template <typename T>
inline void __SVD_Jacobi_Rot(MAT<T>& U, MAT<T>& S, MAT<T>& V, int i, int j, double eps) {
	T c, s;
	double Angle;
	T* Temp;
	T temp;

	c = S(i, i) * S(i, j) + S(j, j) * S(j, i);
	s = S(i, j) * S(i, j) + S(j, j) * S(j, j) - S(i, i) * S(i, i) - S(j, i) * S(j, i);

	Angle = -0.5 * atan2(2 * c, s);
	c = cos(Angle);
	s = sin(Angle);

	__ApplyGivensOnTheRight(V[i], V[j], c, s, V.rows);
	__ApplyGivensOnTheRight(S[i], S[j], c, s, S.cols);

	c = S(i, i) * S(j, i) + S(j, j) * S(i, j);
	s = S(j, i) * S(j, i) + S(j, j) * S(j, j) - S(i, i) * S(i, i) - S(i, j) * S(i, j);

	Angle = -0.5 * atan2(2 * c, s);
	c = cos(Angle);
	s = sin(Angle);

	__ApplyGivensOnTheRight(U[i], U[j], c, s, V.rows);

	for (int k = 0; k < S.cols; ++k) {
		Temp = S[k];
		temp = Temp[i];
		Temp[i] = Temp[j] * s + temp * c;
		Temp[j] = Temp[j] * c - temp * s;
	}
}

template <typename T>
void __SVD_Jacobi_2(MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int m, n;
	bool flag = true;
	while (flag) {
		flag = false;
		for (m = 1; m < S.cols; ++m) {
			for (n = 0; n < m; ++n) {
				if (fabs(S(n, m)) > eps || fabs(S(m, n)) > eps) {
					flag = true;
					__SVD_Jacobi_Rot(U, S, V, n, m, eps);
				}
			}
		}
	}
}

template <typename T>
void __SVD_Jacobi(MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i;
	ZeroMemory(V.data, sizeof(T) * V.rows * V.cols);
	for (i = 0; i < V.rows; ++i) {
		V[i][i] = 1;
	}
	ZeroMemory(U.data, sizeof(T) * U.rows * U.cols);
	for (i = 0; i < U.rows; ++i) {
		U[i][i] = 1;
	}

	if (S.rows >= S.cols) {

		T* w = new T[S.rows];
		T* v = new T[S.rows];
		__QR_HouseHolder(U, S, w, v);
		delete[] w;
		delete[] v;

		__SVD_Jacobi_2(U, S, V, eps);
	}
	else {

		T* w = new T[S.cols];
		T* v = new T[S.cols];
		__LQT_HouseHolder(V, S, w, v);
		delete[] w;
		delete[] v;

		i = S.cols;
		S.cols = S.rows;
		__SVD_Jacobi_2(U, S, V, eps);
		S.cols = i;
	}
}

/*
template <typename T>
bool __SVD_Jacobi_Rot(MAT<T>& U, MAT<T>& V, int i, int j, double eps) {
	int k = 0;
	T c, s;

	c = __dot(U[i], U[j], U.rows);
	s = __dot(U[j], U[j], U.rows) - __dot(U[i], U[i], U.rows);

	double Angle = - 0.5 * atan2(2 * c, s);
	if (fabs(Angle) < eps) {
		return true;
	}

	c = cos(Angle);
	s = sin(Angle);

	__ApplyGivensOnTheRight(V[i], V[j], c, s, V.rows);
	__ApplyGivensOnTheRight(U[i], U[j], c, s, U.rows);

	return false;
}
template <typename T>
void __SVD_Jacobi(MAT<T>& U, MAT<T>& V, double eps) {
	int i, j, m(0), n(U.cols - 1);
	bool flag = false;
	while (m < n) {
		flag = true;
		for (j = m + 1; j <= n; ++j) {
			flag = flag && __SVD_Jacobi_Rot(U, V, m, j, eps);
		}
		if (flag) {
			++m;
			continue;
		}

		flag = true;
		for (j = m; j < n; ++j) {
			flag = flag && __SVD_Jacobi_Rot(U, V, j, n, eps);
		}
		if (flag) {
			--n;
			continue;
		}

		for (i = m + 1; i < n; ++i) {
			for (j = i + 1; j <= n; ++j) {
				__SVD_Jacobi_Rot(U, V, i, j, eps);
			}
		}
	}
}


template <typename T>
void __SVD_Jacobi(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i, j;
	T* Temp;

	if (A.rows >= A.cols) {
		T* w = new T[A.rows];
		T* v = new T[A.rows];
		int m = 0, n = S.cols - 1;

		ZeroMemory(U.data, sizeof(T) * U.rows * U.cols);
		for (i = 0; i < U.rows; ++i) {
			U[i][i] = 1;
		}
		memcpy(S.data, A.data, sizeof(T) * A.rows * A.cols);

		__QR_HouseHolder(U, S, w, v);

		for (i = 0; i < S.cols; ++i) {
			Temp = V[i];
			for (j = 0; j < i; ++j) {
				Temp[j] = 0;
			}
			for (j = i; j < S.cols; ++j) {
				Temp[j] = S[j][i];
			}
		}
		ZeroMemory(S.data, sizeof(T) * S.rows * S.cols);

		__SVD_Jacobi(V, U, eps);

		for (i = 0; i < V.cols; ++i) {
			w[i] = sqrt(__dot(V[i], V[i], V.rows));
			__times(V[i], 1.0 / w[i], V[i], V.rows);
		}
		for (i = 0; i < S.cols; ++i) {
			S(i, i) = w[i];
		}
		delete[] w;
		delete[] v;
	}
	else {
		int m = 0, n = S.rows - 1;
		T* w = new T[A.cols];
		T* v = new T[A.cols];

		ZeroMemory(V.data, sizeof(T) * V.rows * V.cols);
		for (i = 0; i < V.rows; ++i) {
			V[i][i] = 1;
		}
		memcpy(S.data, A.data, sizeof(T) * A.rows * A.cols);

		__LQT_HouseHolder(V, S, w, v);

		for (i = 0; i < S.rows; ++i) {
			Temp = U[i];
			for (j = 0; j < i; ++j) {
				Temp[j] = 0;
			}
			for (j = i; j < S.rows; ++j) {
				Temp[j] = S[i][j];
			}
		}
		ZeroMemory(S.data, sizeof(T) * S.rows * S.cols);

		__SVD_Jacobi(U, V, eps);

		for (i = 0; i < U.cols; ++i) {
			w[i] = sqrt(__dot(U[i], U[i], U.rows));
			__times(U[i], 1.0 / w[i], U[i], U.rows);
		}
		for (i = 0; i < S.rows; ++i) {
			S(i, i) = w[i];
		}

		delete[] w;
		delete[] v;
	}
}
*/
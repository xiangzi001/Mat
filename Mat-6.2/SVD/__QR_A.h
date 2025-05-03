
template<typename T>
void __SVD_Givens_UP_A(T* a, T* b, T* value, MAT<T>& U, MAT<T>& V, int m, int n, double eps) {
	int i, k;
	T temp, s, c, B;
	double angle;

	while (n > m) {
		if (fabs(b[m]) < eps) {
			++m;
			continue;
		}
		if (fabs(b[n - 1]) < eps) {
			--n;
			continue;
		}
		for (i = m + 1; i < n - 1; ++i) {
			if (fabs(b[i]) < eps) {
				__SVD_Givens_UP_A(a, b, value, U, V, m, i, eps);
				m = i + 1;
			}
		}

		temp = a[m] * a[m] - value[n] * value[n];

		angle = atan2(b[m] * a[m], temp);
		c = cos(angle);
		s = sin(angle);

		temp = c * a[m] + s * b[m];
		b[m] = c * b[m] - s * a[m];
		a[m] = temp;
		B = s * a[m + 1];
		a[m + 1] = c * a[m + 1];

		for (k = 0; k < V.rows; k++) {
			temp = c * V[k][m] + s * V[k][m + 1];
			V[k][m + 1] = c * V[k][m + 1] - s * V[k][m];
			V[k][m] = temp;
		}

		angle = atan2(B, a[m]);
		c = cos(angle);
		s = sin(angle);

		a[m] = c * a[m] + s * B;
		temp = c * b[m] + s * a[m + 1];
		a[m + 1] = c * a[m + 1] - s * b[m];
		b[m] = temp;
		B = s * b[m + 1];
		b[m + 1] = c * b[m + 1];

		for (k = 0; k < U.rows; k++) {
			temp = c * U[k][m] + s * U[k][m + 1];
			U[k][m + 1] = c * U[k][m + 1] - s * U[k][m];
			U[k][m] = temp;
		}

		for (i = m + 1; i < n; ++i) {
			angle = atan2(B, b[i - 1]);
			c = cos(angle);
			s = sin(angle);

			b[i - 1] = c * b[i - 1] + s * B;
			temp = c * a[i] + s * b[i];
			b[i] = c * b[i] - s * a[i];
			a[i] = temp;
			B = s * a[i + 1];
			a[i + 1] = c * a[i + 1];

			for (k = 0; k < V.rows; k++) {
				temp = c * V[k][i] + s * V[k][i + 1];
				V[k][i + 1] = c * V[k][i + 1] - s * V[k][i];
				V[k][i] = temp;
			}

			angle = atan2(B, a[i]);
			c = cos(angle);
			s = sin(angle);

			a[i] = c * a[i] + s * B;
			temp = c * b[i] + s * a[i + 1];
			a[i + 1] = c * a[i + 1] - s * b[i];
			b[i] = temp;
			B = s * b[i + 1];
			b[i + 1] = c * b[i + 1];

			for (k = 0; k < U.rows; k++) {
				temp = c * U[k][i] + s * U[k][i + 1];
				U[k][i + 1] = c * U[k][i + 1] - s * U[k][i];
				U[k][i] = temp;
			}
		}
		if (fabs(b[n - 1]) < eps) {
			--n;
		}
		else {
			value[n] = a[n];
		}
	}
}

template<typename T>
void __SVD_QR_A(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i = 0;
	int s;
	int r;

	ZeroMemory(U.data, sizeof(T) * U.cols * U.rows);
	ZeroMemory(V.data, sizeof(T) * V.cols * V.rows);
	memcpy(S.data, A.data, sizeof(T) * S.cols * S.rows);
	for (i = 0; i < U.rows; ++i) {
		U[i][i] = 1;
	}
	for (i = 0; i < V.rows; ++i) {
		V[i][i] = 1;
	}

	T* w, * v;
	if (S.rows > S.cols) {
		//T* w = (T*)alloca(sizeof(T) * S.rows);
		s = sizeof(T) * (S.rows - 1) / 64 + 1;
		w = new T[s * 64 / sizeof(T)];
		v = new T[s * 64 / sizeof(T)];
		s = S.cols - 1;

		for (r = 0; r < s; ++r) {
			__SVD_HouseHolder_left(S, U, w, v, r, r);
			__SVD_HouseHolder_right(S, V, w, r, r + 1);
		}
		__SVD_HouseHolder_left(S, U, w, v, s, s);

		T* a(S[S.rows - 1]);
		T* b(S[S.rows - 2]);
		for (r = 0; r < S.cols; ++r) {
			a[r] = S[r][r];
		}
		for (r = 1; r < S.cols; ++r) {
			b[r - 1] = S[r - 1][r];
		}
		for (i = 0; i < S.rows; ++i) {
			w[i] = a[i];
		}
		for (i = 0; i < S.rows; ++i) {
			v[i] = b[i];
		}
		__SVD_Givens_UP_2(w, v, 0, S.rows - 1, eps);
		__SVD_Givens_UP_A(a, b, w, U, V, 0, S.rows - 1, eps);

		for (r = 0; r < S.cols; ++r) {
			S[r][r] = a[r];
		}
		for (r = 1; r < S.cols; ++r) {
			S[r - 1][r] = 0;
		}
		for (r = 0; r < S.cols; ++r) {
			a[r] = 0;
		}
	}
	else {
		//T* w = (T*)alloca(sizeof(T) * S.cols);
		s = sizeof(T) * (S.cols - 1) / 64 + 1;
		w = new T[s * 64 / sizeof(T)];
		v = new T[s * 64 / sizeof(T)];
		s = S.rows - 1;

		for (r = 0; r < s; ++r) {
			__SVD_HouseHolder_right(S, V, w, r, r);
			__SVD_HouseHolder_left(S, U, w, v, r, r + 1);
		}
		__SVD_HouseHolder_right(S, V, w, s, s);

		T* a(S.data);
		T* b(a + S.rows);
		for (r = 0; r < S.rows; ++r) {
			a[r] = S[r][r];
		}
		for (r = 1; r < S.rows; ++r) {
			b[r - 1] = S[r][r - 1];
		}
		for (i = 0; i < S.rows; ++i) {
			w[i] = a[i];
		}
		for (i = 0; i < S.rows; ++i) {
			v[i] = b[i];
		}
		__SVD_Givens_UP_2(w, v, 0, S.rows - 1, eps);
		__SVD_Givens_UP_A(a, b, w, V, U, 0, S.rows - 1, eps);

		for (r = 0; r < S.rows; ++r) {
			S[r][r] = a[r];
		}
		for (r = 1; r < S.rows; ++r) {
			S[r][r - 1] = 0;
		}
		for (r = 1; r < S.rows; ++r) {
			a[r] = 0;
		}
	}
	delete[] w;
	delete[] v;
}


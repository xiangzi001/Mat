
template<typename T>
void __SVD_Givens_UP_2(T*& a, T*& b, MAT<T>& V, int m, int n, T eps) {
	int i, k;
	T temp, t, s, c, B;

	while (n > m) {
		temp = a[m] * a[m] - a[n] * a[n] - b[n - 1] * b[n - 1];
		t = b[m] * a[m] / temp;
		//t = b[m] / a[m];
		if (fabs(t) < eps) {
			++m;
			continue;
		}
		c = 1 / sqrt(t * t + 1);
		s = t * c;

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

		t = B / a[m];
		c = 1 / sqrt(t * t + 1);
		s = t * c;

		a[m] = c * a[m] + s * B;
		temp = c * b[m] + s * a[m + 1];
		a[m + 1] = c * a[m + 1] - s * b[m];
		b[m] = temp;
		B = s * b[m + 1];
		b[m + 1] = c * b[m + 1];

		for (i = m + 1; i < n; ++i) {

			t = B / b[i - 1];
			if (fabs(t) < eps) {
				if (n > i + 1) {
					__SVD_Givens_UP_2(a, b, V, m, i, eps);
					m = i + 1;
					break;
				}
				else {
					--n;
					break;
				}
			}
			c = 1 / sqrt(t * t + 1);
			s = t * c;

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

			t = B / a[i];
			c = 1 / sqrt(t * t + 1);
			s = t * c;

			a[i] = c * a[i] + s * B;
			temp = c * b[i] + s * a[i + 1];
			a[i + 1] = c * a[i + 1] - s * b[i];
			b[i] = temp;
			B = s * b[i + 1];
			b[i + 1] = c * b[i + 1];

		}
	}
}

template<typename T>
void __SVD_QR_2(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i, j;
	int s = U.rows * U.cols;

	for (i = 0; i < s; ++i) {
		U.data[i] = 0;
	}
	for (i = 0; i < U.rows; ++i) {
		U[i][i] = 1;
	}
	s = V.rows * V.cols;
	for (i = 0; i < s; ++i) {
		V.data[i] = 0;
	}
	for (i = 0; i < V.rows; ++i) {
		V[i][i] = 1;
	}
	s = A.rows * A.cols;
	for (i = 0; i < s; ++i) {
		S.data[i] = A.data[i];
	}

	if (S.rows > S.cols) {
		//T* w = (T*)alloca(sizeof(T) * S.rows);
		T* w = new T[S.rows];
		s = S.cols - 1;

		for (i = 0; i < s; i++) {
			__SV_HouseHolder_left(S, w, i, i);
			__SVD_HouseHolder_right(S, V, w, i, i + 1);
		}
		__SV_HouseHolder_left(S, w, s, s);

		T* a(S[S.rows - 1]);
		T* b(S[S.rows - 2]);
		for (i = 0; i < S.cols; i++) {
			a[i] = S[i][i];
		}
		for (i = 1; i < S.cols; i++) {
			b[i - 1] = S[i - 1][i];
		}
		__SVD_Givens_UP_2(a, b, V, 0, S.cols - 1, eps);

		print(S);
		for (i = 0; i < U.cols; ++i)
			w[i] = 1 / a[i];
		//U*S = A*V
		times_A_B(A, V, U, w);
		for (i = 0; i < U.rows; ++i) {
			for (j = 0; j < U.cols; ++j) {
				U[i][j] = U[i][j] * w[j];
			}
		}
		delete[] w;

		for (i = 0; i < S.cols; i++) {
			S[i][i] = a[i];
		}
		for (i = 1; i < S.cols; i++) {
			S[i - 1][i] = 0;
		}
		for (i = 0; i < S.cols; i++) {
			a[i] = 0;
		}
	}
	else {
		//T* w = (T*)alloca(sizeof(T) * S.cols);
		T* w = new T[S.cols];
		s = S.rows - 1;

		for (i = 0; i < s; i++) {
			__SV_HouseHolder_right(S, w, i, i);
			__SVD_HouseHolder_left(S, U, w, i, i + 1);
		}
		__SV_HouseHolder_right(S, w, s, s);

		T* a(S.data);
		T* b(a + S.rows);
		for (i = 0; i < S.rows; i++) {
			a[i] = S[i][i];
		}
		for (i = 1; i < S.rows; i++) {
			b[i - 1] = S[i][i - 1];
		}
		__SVD_Givens_UP_2(a, b, U, 0, S.rows - 1, eps);

		for (i = 0; i < V.cols; ++i)
			w[i] = 1 / a[i];
		//V*S = _T(A)*U
		times_AT_B(A, U, V);
		for (i = 0; i < V.rows; ++i) {
			for (j = 0; j < V.cols; ++j) {
				V[i][j] = V[i][j] * w[j];
			}
		}
		delete[] w;

		for (i = 0; i < S.rows; i++) {
			S[i][i] = a[i];
		}
		for (i = 1; i < S.rows; i++) {
			S[i][i - 1] = 0;
		}
		for (i = 1; i < S.rows; i++) {
			a[i] = 0;
		}
	}
}

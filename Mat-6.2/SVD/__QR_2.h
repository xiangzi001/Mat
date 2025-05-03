
template<typename T>
void __SVD_Givens_UP_2(T* a, T* b, int m, int n, double eps) {
	int i, k;
	T temp, t, s, c, B;

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
				__SVD_Givens_UP_2(a, b, m, i, eps);
				m = i + 1;
			}
		}

		temp = a[m] * a[m] - a[n] * a[n] - b[n - 1] * b[n - 1];
		t = b[m] * a[m] / temp;
		c = 1 / sqrt(t * t + 1);
		s = t * c;

		temp = c * a[m] + s * b[m];
		b[m] = c * b[m] - s * a[m];
		a[m] = temp;
		B = s * a[m + 1];
		a[m + 1] = c * a[m + 1];

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
			c = 1 / sqrt(t * t + 1);
			s = t * c;

			b[i - 1] = c * b[i - 1] + s * B;
			temp = c * a[i] + s * b[i];
			b[i] = c * b[i] - s * a[i];
			a[i] = temp;
			B = s * a[i + 1];
			a[i + 1] = c * a[i + 1];

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
void __SVD_Givens_UP_2_UV(T* a, T* b, T* value, MAT<T>& V, int n, double eps) {
	int i, k;
	T temp, t, s, c, B;

	for (; n > 0; --n) {
		temp = a[0] * a[0] - value[n] * value[n];
		t = b[0] * a[0] / temp;
		c = 1 / sqrt(t * t + 1);
		s = t * c;

		temp = c * a[0] + s * b[0];
		b[0] = c * b[0] - s * a[0];
		a[0] = temp;
		B = s * a[1];
		a[1] = c * a[1];

		for (k = 0; k < V.rows; k++) {
			temp = c * V[k][0] + s * V[k][1];
			V[k][1] = c * V[k][1] - s * V[k][0];
			V[k][0] = temp;
		}

		t = B / a[0];
		c = 1 / sqrt(t * t + 1);
		s = t * c;

		a[0] = c * a[0] + s * B;
		temp = c * b[0] + s * a[1];
		a[1] = c * a[1] - s * b[0];
		b[0] = temp;
		B = s * b[1];
		b[1] = c * b[1];

		for (i = 1; i < n; ++i) {
			t = B / b[i - 1];
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
	int s;

	memcpy(S.data, A.data, sizeof(T) * S.cols * S.rows);

	T *w, *v;
	if (S.rows > S.cols) {
		ZeroMemory(V.data, sizeof(T) * V.cols * V.rows);
		for (i = 0; i < V.rows; ++i) {
			V[i][i] = 1;
		}
		//T* w = (T*)alloca(sizeof(T) * S.rows);
		s = sizeof(T) * (S.rows - 1) / 64 + 1;
		w = new T[s * 64 / sizeof(T)];
		v = new T[s * 64 / sizeof(T)];
		s = S.cols - 1;

		for (i = 0; i < s; ++i) {
			__SV_HouseHolder_left(S, w, v, i, i);
			__SVD_HouseHolder_right(S, V, w, i, i + 1);
		}
		__SV_HouseHolder_left(S, w, v, s, s);

		T* a(S[S.rows - 1]);
		T* b(S[S.rows - 2]);
		for (i = 0; i < S.cols; ++i) {
			a[i] = S[i][i];
		}
		for (i = 1; i < S.cols; ++i) {
			b[i - 1] = S[i - 1][i];
		}
		for (i = 0; i < S.rows; ++i) {
			w[i] = a[i];
		}
		for (i = 0; i < S.rows; ++i) {
			v[i] = b[i];
		}
		__SVD_Givens_UP_2(a, b, 0, S.cols - 1, eps);
		__SVD_Givens_UP_2_UV(w, v, a, V, S.cols - 1, eps);

		//U*S = A*V
		times_A_B(A, V, U, w);
		__SVD_Q_times(U, w, a);

		for (i = 0; i < S.cols; ++i) {
			S[i][i] = a[i];
		}
		for (i = 1; i < S.cols; ++i) {
			S[i - 1][i] = 0;
		}
		for (i = 0; i < S.cols; ++i) {
			a[i] = 0;
		}
	}
	else {
		ZeroMemory(U.data, sizeof(T) * U.cols * U.rows);
		for (i = 0; i < U.rows; ++i) {
			U[i][i] = 1;
		}
		//T* w = (T*)alloca(sizeof(T) * S.cols);
		s = sizeof(T) * (S.cols - 1) / 64 + 1;
		w = new T[s * 64 / sizeof(T)];
		v = new T[s * 64 / sizeof(T)];
		s = S.rows - 1;

		for (i = 0; i < s; ++i) {
			__SV_HouseHolder_right(S, w, i, i);
			__SVD_HouseHolder_left(S, U, w, v, i, i + 1);
		}
		__SV_HouseHolder_right(S, w, s, s);

		T* a(S.data);
		T* b(a + S.rows);
		for (i = 0; i < S.rows; ++i) {
			a[i] = S[i][i];
		}
		for (i = 1; i < S.rows; ++i) {
			b[i - 1] = S[i][i - 1];
		}
		for (i = 0; i < S.rows; ++i) {
			w[i] = a[i];
		}
		for (i = 0; i < S.rows; ++i) {
			v[i] = b[i];
		}
		__SVD_Givens_UP_2(a, b, 0, S.rows - 1, eps);
		__SVD_Givens_UP_2_UV(w, v, a, U, S.rows - 1, eps);

		//V*S = _T(A)*U
		times_AT_B(A, U, V);
		__SVD_Q_times(V, w, a);

		for (i = 0; i < S.rows; ++i) {
			S[i][i] = a[i];
		}
		for (i = 1; i < S.rows; ++i) {
			S[i][i - 1] = 0;
		}
		for (i = 1; i < S.rows; ++i) {
			a[i] = 0;
		}
	}
	delete[] w;
	delete[] v;
}

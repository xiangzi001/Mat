
template<typename T>
void __SVD_HouseHolder_Up(MAT<T>& A, MAT<T>& U, MAT<T>& V, T* v) {
	int m, n, i;

	T cr, hr;
	T* w;

	for (m = 0; m < A.cols - 1; ++m) {
		n = m + 1;

		w = U[m];
		__copy(w + m, A[m] + m, A.rows - m);
		cr = sqrt(__dot(w + m, w + m, A.rows - m));
		if (w[m] < 0.0)cr = -cr;
		w[m] += cr;
		hr = 1.0 / sqrt(cr * w[m]);
		__times(w + m, hr, w + m, A.rows - m);
		//__ApplyHouseHolderOnTheRight(U, w, v, m);
		__ApplyHouseHolderOnTheLeft2(A, w, m, m);

		w = V[m];
		for (i = n; i < A.cols; ++i) {
			w[i] = A(m, i);
		}
		cr = sqrt(__dot(w + n, w + n, A.cols - n));
		if (w[n] < 0.0)cr = -cr;
		w[n] += cr;
		hr = 1.0 / sqrt(cr * w[n]);
		__times(w + n, hr, w + n, A.cols - n);
		//__ApplyHouseHolderOnTheRight(V, w, v, n);
		__ApplyHouseHolderOnTheRight3(A, w, v, m, n);
	}

	w = U[m];
	__copy(w + m, A[m] + m, A.rows - m);
	cr = sqrt(__dot(w + m, w + m, A.rows - m));
	if (w[m] < 0.0)cr = -cr;
	w[m] += cr;
	hr = 1.0 / sqrt(cr * w[m]);
	__times(w + m, hr, w + m, A.rows - m);
	//__ApplyHouseHolderOnTheRight(U, w, v, m);
	__ApplyHouseHolderOnTheLeft2(A, w, m, m);

	__HouseHolder_Q1(U, U.rows);
	__HouseHolder_Q3(V);
}

template<typename T>
void __SVD_HouseHolder_Down(MAT<T>& A, MAT<T>& U, MAT<T>& V, T* v) {
	int m, n, i;

	T cr, hr;
	T* w;

	for (m = 0; m < A.rows - 1; ++m) {
		n = m + 1;

		w = V[m];
		for (i = m; i < A.cols; ++i) {
			w[i] = A(m, i);
		}
		cr = sqrt(__dot(w + m, w + m, A.cols - m));
		if (w[m] < 0.0)cr = -cr;
		w[m] += cr;
		hr = 1.0 / sqrt(cr * w[m]);
		__times(w + m, hr, w + m, A.cols - m);
		//__ApplyHouseHolderOnTheRight(V, w, v, m);
		__ApplyHouseHolderOnTheRight3(A, w, v, m, m);

		w = U[m];
		__copy(w + n, A[m] + n, A.rows - n);
		cr = sqrt(__dot(w + n, w + n, A.rows - n));
		if (w[n] < 0.0)cr = -cr;
		w[n] += cr;
		hr = 1.0 / sqrt(cr * w[n]);
		__times(w + n, hr, w + n, A.rows - n);
		//__ApplyHouseHolderOnTheRight(U, w, v, n);
		__ApplyHouseHolderOnTheLeft2(A, w, m, n);
	}

	w = V[m];
	for (i = m; i < A.cols; ++i) {
		w[i] = A(m, i);
	}
	cr = sqrt(__dot(w + m, w + m, A.cols - m));
	if (w[m] < 0.0)cr = -cr;
	w[m] += cr;
	hr = 1.0 / sqrt(cr * w[m]);
	__times(w + m, hr, w + m, A.cols - m);
	//__ApplyHouseHolderOnTheRight(V, w, v, m);
	__ApplyHouseHolderOnTheRight3(A, w, v, m, m);

	__HouseHolder_Q3(U);
	__HouseHolder_Q1(V, V.rows);
}

template<typename T>
void __SVD_Givens_UP(T* a, T* b, MAT<T>& U, MAT<T>& V, int m, int n, double eps) {
	int i;
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
				__SVD_Givens_UP(a, b, U, V, m, i, eps);
				m = i + 1;
			}
		}

		temp = a[m] * a[m] - a[n] * a[n] - b[n - 1] * b[n - 1];
		if (fabs(b[m] * a[m]) > fabs(temp)) {
			t = temp / (b[m] * a[m]);
			s = 1 / sqrt(t * t + 1);
			c = t * s;
		}
		else {
			t = b[m] * a[m] / temp;
			c = 1 / sqrt(t * t + 1);
			s = t * c;
		}

		temp = c * a[m] + s * b[m];
		b[m] = c * b[m] - s * a[m];
		a[m] = temp;
		B = s * a[m + 1];
		a[m + 1] = c * a[m + 1];

		__ApplyGivensOnTheRight(V[m], V[m + 1], c, s, V.rows);

		if (fabs(B) > fabs(a[m])) {
			t = a[m] / B;
			s = 1 / sqrt(t * t + 1);
			c = t * s;
		}
		else {
			t = B / a[m];
			c = 1 / sqrt(t * t + 1);
			s = t * c;
		}

		a[m] = c * a[m] + s * B;
		temp = c * b[m] + s * a[m + 1];
		a[m + 1] = c * a[m + 1] - s * b[m];
		b[m] = temp;
		B = s * b[m + 1];
		b[m + 1] = c * b[m + 1];

		__ApplyGivensOnTheRight(U[m], U[m + 1], c, s, U.rows);

		for (i = m + 1; i < n; ++i) {

			if (fabs(B) > fabs(b[i - 1])) {
				t = b[i - 1] / B;
				s = 1 / sqrt(t * t + 1);
				c = t * s;
			}
			else {
				t = B / b[i - 1];
				c = 1 / sqrt(t * t + 1);
				s = t * c;
			}

			b[i - 1] = c * b[i - 1] + s * B;
			temp = c * a[i] + s * b[i];
			b[i] = c * b[i] - s * a[i];
			a[i] = temp;
			B = s * a[i + 1];
			a[i + 1] = c * a[i + 1];

			__ApplyGivensOnTheRight(V[i], V[i + 1], c, s, V.rows);

			if (fabs(B) > fabs(a[i])) {
				t = a[i] / B;
				s = 1 / sqrt(t * t + 1);
				c = t * s;
			}
			else {
				t = B / a[i];
				c = 1 / sqrt(t * t + 1);
				s = t * c;
			}

			a[i] = c * a[i] + s * B;
			temp = c * b[i] + s * a[i + 1];
			a[i + 1] = c * a[i + 1] - s * b[i];
			b[i] = temp;
			B = s * b[i + 1];
			b[i + 1] = c * b[i + 1];

			__ApplyGivensOnTheRight(U[i], U[i + 1], c, s, U.rows);

		}
	}
}

template<typename T>
void __SVD_QR(MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i = 0;
	int r;

	T* v;
	if (S.rows > S.cols) {
		v = new T[S.rows];
		__SVD_HouseHolder_Up(S, U, V, v);
		delete[] v;

		T* a(S[0]);
		T* b(S[1]);
		for (r = 1; r < S.cols; ++r) {
			a[r] = S(r, r);
		}
		for (r = 1; r < S.cols; ++r) {
			b[r - 1] = S(r - 1, r);
		}
		__SVD_Givens_UP(a, b, U, V, 0, S.cols - 1, eps);
		for (r = 0; r < S.cols; ++r) {
			S[r][r] = a[r];
		}
		for (r = 1; r < S.cols; ++r) {
			S(r - 1, r) = 0;
		}
		for (r = 1; r < S.cols; ++r) {
			a[r] = 0;
		}
	}
	else {
		v = new T[S.cols];
		__SVD_HouseHolder_Down(S, U, V, v);
		delete[] v;

		T* a(S[S.rows - 1]);
		T* b(S[S.rows - 2]);
		for (r = 0; r < S.rows; ++r) {
			a[r] = S(r, r);
		}
		for (r = 1; r < S.rows; ++r) {
			b[r - 1] = S(r, r - 1);
		}
		__SVD_Givens_UP(a, b, V, U, 0, S.rows - 1, eps);

		for (r = 0; r < S.rows; ++r) {
			S(r, r) = a[r];
		}
		for (r = 1; r < S.rows; ++r) {
			S(r, r - 1) = 0;
		}
		for (r = 0; r < S.rows - 1; ++r) {
			a[r] = 0;
		}
	}
}

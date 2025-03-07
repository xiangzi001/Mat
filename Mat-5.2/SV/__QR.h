
template<typename T>
void __SV_HouseHolder_left(MAT<T>& A, T*& w, int r, int len) {
	int i, j, k;
	T temp, sum, cr, hr;
	temp = 0;
	for (k = r; k < A.rows; ++k)
		temp += fabs(A[k][r]);
	if (temp > 0)
	{
		sum = 0;
		for (k = len; k < A.rows; ++k)
			sum += A[k][r] * A[k][r];
		cr = sqrt(sum);
		if (A[len][r] > 0.0)cr = -cr;
		hr = cr * (cr - A[len][r]);

		//定义w
		w[len] = A[len][r] - cr;
		for (i = len + 1; i < A.rows; ++i)
		{
			w[i] = A[i][r];
		}

		//定义A
		for (j = r; j < A.cols; ++j)
		{
			sum = 0;
			for (i = len; i < A.rows; ++i)
				sum += A[i][j] * w[i];
			sum = sum / hr;
			for (i = len; i < A.rows; ++i)
				A[i][j] = A[i][j] - w[i] * sum;
		}
	}
}
template<typename T>
void __SV_HouseHolder_right(MAT<T>& A, T*& w, int r, int len) {
	int i, j, k;
	T temp, sum, cr, hr;
	temp = 0;
	for (k = r; k < A.cols; ++k)
		temp += fabs(A[r][k]);
	if (temp > 0)
	{
		sum = 0;
		for (k = len; k < A.cols; ++k)
			sum += A[r][k] * A[r][k];
		cr = sqrt(sum);
		if (A[r][len] > 0.0)cr = -cr;
		hr = cr * (cr - A[r][len]);

		//定义w
		w[len] = A[r][len] - cr;
		for (i = len + 1; i < A.cols; ++i)
		{
			w[i] = A[r][i];
		}

		//定义A
		for (j = r; j < A.rows; ++j)
		{
			sum = 0;
			for (i = len; i < A.cols; ++i)
				sum += A[j][i] * w[i];
			sum = sum / hr;
			for (i = len; i < A.cols; ++i)
				A[j][i] = A[j][i] - w[i] * sum;
		}

	}
}

template<typename T>
void __SV_Givens_UP(T*& a, T*& b, int m, int n, T eps) {
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
					__SV_Givens_UP(a, b, m, i, eps);
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
void __SV_QR(MAT<T>& A, MAT<T>& S, double eps) {
	int i = 0;
	int s = A.rows * A.cols;
	int r;
	for (i = 0; i < s; ++i) {
		S.data[i] = A.data[i];
	}

	if (S.rows > S.cols) {
		//T* w = (T*)alloca(sizeof(T) * S.rows);
		T* w = new T[S.rows];
		s = S.cols - 1;

		for (r = 0; r < s; ++r) {
			__SV_HouseHolder_left(S, w, r, r);
			__SV_HouseHolder_right(S, w, r, r + 1);
		}
		__SV_HouseHolder_left(S, w, s, s);
		delete[] w;

		T* a(S[S.rows - 2]);
		T* b(S[S.rows - 1]);
		for (r = 0; r < S.cols; ++r) {
			a[r] = S[r][r];
		}
		for (r = 1; r < S.cols; ++r) {
			b[r - 1] = S[r - 1][r];
		}
		__SV_Givens_UP(a, b, 0, S.cols - 1, eps);
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
		T* w = new T[S.cols];
		s = S.rows - 1;

		for (r = 0; r < s; ++r) {
			__SV_HouseHolder_right(S, w, r, r);
			__SV_HouseHolder_left(S, w, r, r + 1);
		}
		__SV_HouseHolder_right(S, w, s, s);
		delete[] w;

		T* a(S.data);
		T* b(a + S.rows);
		for (r = 0; r < S.rows; ++r) {
			a[r] = S[r][r];
		}
		for (r = 1; r < S.rows; ++r) {
			b[r - 1] = S[r][r - 1];
		}
		__SV_Givens_UP(a, b, 0, S.rows - 1, eps);
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
}

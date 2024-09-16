
template<typename T>
void SVD_HouseHolder_left(MAT<T>& A, MAT<T>& U, T*& w, int& r, int len) {
	int i, j, k;
	T sum(0), cr, hr;
	for (k = len; k < A.rows; ++k)
		sum += A[k][r] * A[k][r];
	if (sum == 0) {
		return;
	}
	cr = sqrt(sum);
	if (A[len][r] < 0.0)cr = -cr;

	//定义w
	w[len] = A[len][r] + cr;
	for (i = len + 1; i < A.rows; ++i)
	{
		w[i] = A[i][r];
	}

	hr = 1 / (cr * (cr + A[len][r]));
	//定义A
	for (j = r; j < A.cols; ++j)
	{
		sum = 0;
		for (i = len; i < A.rows; ++i)
			sum += A[i][j] * w[i];
		sum = sum * hr;
		for (i = len; i < A.rows; ++i)
			A[i][j] = A[i][j] - w[i] * sum;
	}

	//定义U
	for (i = 0; i < U.rows; ++i)
	{
		sum = 0;
		for (j = len; j < U.rows; ++j)
			sum += U[i][j] * w[j];
		sum = sum * hr;
		for (j = len; j < U.cols; ++j)
		{
			U[i][j] = U[i][j] - w[j] * sum;
		}
	}
}
template<typename T>
void SVD_HouseHolder_right(MAT<T>& A, MAT<T>& V, T*& w, int& r, int len) {
	int i, j, k;
	T sum(0), cr, hr;
	for (k = len; k < A.cols; ++k)
		sum += A[r][k] * A[r][k];
	if (sum == 0) {
		return;
	}
	cr = sqrt(sum);
	if (A[r][len] < 0)cr = -cr;

	//定义w
	w[len] = A[r][len] + cr;
	for (i = len + 1; i < A.cols; ++i)
	{
		w[i] = A[r][i];
	}

	hr = 1 / (cr * (cr + A[r][len]));
	//定义A
	for (j = r; j < A.rows; ++j)
	{
		sum = 0;
		for (i = len; i < A.cols; ++i)
			sum += A[j][i] * w[i];
		sum = sum * hr;
		for (i = len; i < A.cols; ++i)
			A[j][i] = A[j][i] - w[i] * sum;
	}
	//定义V
	for (i = 0; i < V.rows; ++i)
	{
		sum = 0;
		for (j = len; j < V.cols; ++j)
			sum += V[i][j] * w[j];
		sum = sum * hr;
		for (j = len; j < V.cols; ++j)
		{
			V[i][j] = V[i][j] - w[j] * sum;
		}
	}
}
/*
template<typename T>
void SVD_Givens_left(MAT<T>& A, MAT<T>& U, int i, int j) {
	int k;
	T temp;
	double t( A[i][j] / A[i - 1][j]);
	double c ( 1 / sqrt(t * t + 1) );
	double s ( t * c );

	//double theta(atan2(A[i][j], A[i - 1][j]));
	//double c(cos(theta));
	//double s(sin(theta));

	int len (j + 3);
	if (A.cols < len){
		len = A.cols;
	}

	for (k = j; k < len; k++) {
		temp = c * A[i - 1][k] + s * A[i][k];
		A[i][k] = -s * A[i - 1][k] + c * A[i][k];
		A[i - 1][k] = temp;
	}
	for (k = 0; k < U.rows; k++) {
		temp = c * U[k][i - 1] + s * U[k][i];
		U[k][i] = -s * U[k][i - 1] + c * U[k][i];
		U[k][i - 1] = temp;
	}
}

template<typename T>
void SVD_Givens_right(MAT<T>& A, MAT<T>& V, int i, int j) {
	int k;
	T temp;
	double t( A[j][i] / A[j][i - 1]);
	double c(1 / sqrt(t * t + 1));
	double s(t * c);

	//double theta(atan2(A[j][i], A[j][i - 1]));
	//double c(cos(theta));
	//double s(sin(theta));

	int len (j + 3);
	if (A.rows < len) {
		len = A.rows;
	}

	for (k = j; k < len; k++) {
		temp = c * A[k][i - 1] + s * A[k][i];
		A[k][i] = -s * A[k][i - 1] + c * A[k][i];
		A[k][i - 1] = temp;
	}
	for (k = 0; k < V.rows; k++) {
		temp = c * V[k][i - 1] + s * V[k][i];
		V[k][i] = -s * V[k][i - 1] + c * V[k][i];
		V[k][i - 1] = temp;
	}
}
*/

template<typename T>
void SVD_Givens_UP_1(T*& a, T*& b, MAT<T>& U, MAT<T>& V, int m, int n, T eps) {
	int i, k;
	T temp,t,s,c,B;

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

		for (k = 0; k < U.rows; k++) {
			temp = c * U[k][m] + s * U[k][m + 1];
			U[k][m + 1] = c * U[k][m + 1] - s * U[k][m];
			U[k][m] = temp;
		}

		for (i = m + 1; i < n; ++i) {

			t = B / b[i - 1];
			if (fabs(t) < eps) {
				if (n > i + 1) {
					SVD_Givens_UP_1(a, b, U, V, m, i, eps);
					m = i+1;
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

			for (k = 0; k < U.rows; k++) {
				temp = c * U[k][i] + s * U[k][i + 1];
				U[k][i + 1] = c * U[k][i + 1] - s * U[k][i];
				U[k][i] = temp;
			}

		}
	}
}

template<typename T>
void SVD_QR_1(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i = 0;
	int s = U.rows * U.cols;
	int r;
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

		for (r = 0; r < s; r++) {
			SVD_HouseHolder_left(S, U, w, r, r);
			SVD_HouseHolder_right(S, V, w, r, r+1);
		}
		SVD_HouseHolder_left(S, U, w, s, s);
		delete[] w;

		T* a(S[S.rows - 1]);
		T* b(S[S.rows - 2]);
		for (r = 0; r < S.cols; r++) {
			a[r] = S[r][r];
		}
		for (r = 1; r < S.cols; r++) {
			b[r - 1] = S[r - 1][r];
		}
		SVD_Givens_UP_1(a, b, U, V, 0, S.cols-1, eps);
		for (r = 0; r < S.cols; r++) {
			S[r][r] = a[r];
		}
		for (r = 1; r < S.cols; r++) {
			S[r - 1][r] = 0;
		}
		for (r = 0; r < S.cols; r++) {
			a[r] = 0;
		}
	}
	else {
		//T* w = (T*)alloca(sizeof(T) * S.cols);
		T* w = new T[S.cols];
		s = S.rows - 1;

		for (r = 0; r < s; r++) {
			SVD_HouseHolder_right(S, V, w, r, r);
			SVD_HouseHolder_left(S, U, w, r, r+1);
		}
		SVD_HouseHolder_right(S, V, w, s, s);
		delete[] w;

		T* a(S.data);
		T* b(a+S.rows);
		for (r = 0; r < S.rows; r++) {
			a[r] = S[r][r];
		}
		for (r = 1; r < S.rows; r++) {
			b[r - 1] = S[r][r - 1];
		}
		SVD_Givens_UP_1(a, b, V, U, 0, S.rows - 1, eps);
		for (r = 0; r < S.rows; r++) {
			S[r][r] = a[r];
		}
		for (r = 1; r < S.rows; r++) {
			S[r][r - 1] = 0;
		}
		for (r = 1; r < S.rows; r++) {
			a[r] = 0;
		}
	}
}


template<typename T>
void SVD_Givens_UP_2(T*& a, T*& b, MAT<T>& V, int m, int n, T eps) {
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
					SVD_Givens_UP_2(a, b, V, m, i, eps);
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
void SVD_QR_2(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
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
			SingularValue_HouseHolder_left(S, w, i, i);
			SVD_HouseHolder_right(S, V, w, i, i + 1);
		}
		SingularValue_HouseHolder_left(S, w, s, s);

		T* a(S[S.rows - 1]);
		T* b(S[S.rows - 2]);
		for (i = 0; i < S.cols; i++) {
			a[i] = S[i][i];
		}
		for (i = 1; i < S.cols; i++) {
			b[i - 1] = S[i - 1][i];
		}
		SVD_Givens_UP_2(a, b, V, 0, S.cols - 1, eps);

		print(S);
		for (i = 0; i < U.cols; ++i)
			w[i] = 1 / a[i];
		//U*S = A*V
		times(A, V, U);
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
			SingularValue_HouseHolder_right(S, w, i, i);
			SVD_HouseHolder_left(S, U, w, i, i + 1);
		}
		SingularValue_HouseHolder_right(S, w, s, s);

		T* a(S.data);
		T* b(a + S.rows);
		for (i = 0; i < S.rows; i++) {
			a[i] = S[i][i];
		}
		for (i = 1; i < S.rows; i++) {
			b[i - 1] = S[i][i - 1];
		}
		SVD_Givens_UP_2(a, b, U, 0, S.rows - 1, eps);

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


template <typename T>
bool Jacobi_SVD_U(MAT<T>& A, MAT<T>& U, int& i, int& j, T& eps) {
	int k = 0;
	T b = 0;
	T a = 0;

	for (k = 0; k < A.cols; ++k) {
		b = b - A[i][k] * A[i][k];
	}
	for (k = 0; k < A.cols; ++k) {
		a = a + A[i][k] * A[j][k];
	}
	for (k = 0; k < A.cols; ++k) {
		b = b + A[j][k] * A[j][k];
	}

	//T t = (-b + sqrt(b * b + 4 * a * a)) / (2 * a);
	//T c = 1 / sqrt(t * t + 1);
	//T s = t * c;

	double Angle = 0.5 * atan2(2 * a, b);
	T c = cos(Angle);
	T s = sin(Angle);

	for (k = 0; k < A.cols; ++k) {
		a = A[i][k] * c - A[j][k] * s;
		b = A[i][k] * s + A[j][k] * c;
		A[i][k] = a;
		A[j][k] = b;
	}
	for (k = 0; k < U.rows; ++k) {
		a = U[k][i] * c - U[k][j] * s;
		b = U[k][i] * s + U[k][j] * c;
		U[k][i] = a;
		U[k][j] = b;
	}

	if (s < eps) {
		return false;
	}
	else {
		return true;
	}
}

template <typename T>
bool Jacobi_SVD_V(MAT<T>& A, MAT<T>& V, int& i, int& j, T& eps) {
	int k = 0;
	T b = 0;
	T a = 0;

	for (k = 0; k < A.rows; ++k) {
		b = b - A[k][i] * A[k][i];
		a = a + A[k][i] * A[k][j];
		b = b + A[k][j] * A[k][j];
	}

	//T t = (-b + sqrt(b * b + 4 * a * a)) / (2 * a);
	//T c = 1 / sqrt(t * t + 1);
	//T s = t * c;

	double Angle = 0.5 * atan2(2 * a, b);
	T c = cos(Angle);
	T s = sin(Angle);

	for (k = 0; k < A.rows; ++k) {
		a = A[k][i] * c - A[k][j] * s;
		b = A[k][i] * s + A[k][j] * c;
		A[k][i] = a;
		A[k][j] = b;
	}
	for (k = 0; k < V.rows; ++k) {
		a = V[k][i] * c - V[k][j] * s;
		b = V[k][i] * s + V[k][j] * c;
		V[k][i] = a;
		V[k][j] = b;
	}

	if (s < eps) {
		return false;
	}
	else {
		return true;
	}
}

template <typename T>
void SVD_Jacobi(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i;
	int j;
	int s = S.rows * S.cols;
	for (i = 0; i < s; ++i) {
		S.data[i] = A.data[i];
	}

	if (A.rows > A.cols) {
		s = U.rows * U.cols;
		for (i = 0; i < s; ++i) {
			U.data[i] = 0;
		}
		for (i = 0; i < U.rows; ++i) {
			U[i][i] = 1;
		}

		bool flag = true;
		while (flag) {
			flag = false;
			for (i = 0; i < S.cols; ++i) {
				for (j = i + 1; j < S.rows; ++j) {
					if (Jacobi_SVD_U(S, U, j, i, eps)) {
						flag = true;
					}
				}
			}
		}

		T temp;
		for (i = 0; i < S.cols; ++i) {
			temp = 0;
			for (j = 0; j < S.cols; ++j) {
				temp += S[i][j] * S[i][j];
			}
			temp = 1.0 / sqrt(temp);

			for (j = 0; j < V.rows; ++j) {
				V[j][i] = S[i][j] * temp;
			}
			for (j = 0; j < S.cols; ++j) {
				S[i][j] = 0;
			}
			S[i][i] = 1.0 / temp;
		}
	}
	else {
		s = V.rows * V.cols;
		for (i = 0; i < s; ++i) {
			V.data[i] = 0;
		}
		for (i = 0; i < V.rows; ++i) {
			V[i][i] = 1;
		}

		bool flag = true;
		while (flag) {
			flag = false;
			for (i = 0; i < S.rows; ++i) {
				for (j = i + 1; j < S.cols; ++j) {
					if (Jacobi_SVD_V(S, V, j, i, eps)) {
						flag = true;
					}
				}
			}
		}

		T temp;
		for (j = 0; j < S.rows; ++j) {
			temp = 0;
			for (i = 0; i < S.rows; ++i) {
				temp += S[i][j] * S[i][j];
			}
			temp = 1.0 / sqrt(temp);

			for (i = 0; i < U.rows; ++i) {
				U[i][j] = S[i][j] * temp;
			}
			for (i = 0; i < S.rows; ++i) {
				S[i][j] = 0;
			}
			S[j][j] = 1.0 / temp;
		}
	}
}
/*
template <typename T>
void SVD_EVD_QR_full(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i, j;
	if (A.rows > A.cols) {
		MAT<T> E(A.rows,A.rows);

		T* w = new T[A.rows];
		//A*_T(A) = U*E*_T(U)

		times_A_BT(A, A, E);

		EVD_Q_QR_test(E, U, E.data, w, eps);

		for (i = 0; i < S.cols; ++i) {
			S[i][i] = sqrt(E.data[i]);
			w[i] = 1.0 / S[i][i];
		}

		//V*E = _T(A)*U
		times_AT_B(A, U, V);
		for (i = 0; i < V.rows; ++i) {
			for (j = 0; j < V.cols; ++j) {
				V[i][j] = V[i][j] * w[j];
			}
		}

		delete[] w;
	}
	else {
		MAT<T> E(A.cols, A.cols);
		T* w = new T[A.cols];
		//_T(A)*A = V*E*_T(V)

		times_AT_B(A, A, E);

		EVD_Q_QR_test(E, V, E.data, w, eps);

		for (int i = 0; i < A.rows; ++i) {
			S[i][i] = sqrt(E.data[i]);
			w[i] = 1.0 / S[i][i];
		}

		//U*E = A*V
		times(A, V, U);
		for (int i = 0; i < U.rows; ++i) {
			for (int j = 0; j < U.cols; ++j) {
				U[i][j] = U[i][j] * w[j];
			}
		}
		delete[] w;
	}
}
*/

template <typename T>
void SVD_EVD_QR_1(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i,j;
	if (A.rows > A.cols) {
		T* w = new T[A.cols];
		//_T(A)*A = V*S*_T(V)

		times_AT_B(A, A, S);

		EVD_Q_QR_test_1(S, V, S.data, w, eps);

		for (i = 0; i < S.rows; ++i) {
			S[i][i] = sqrt(S.data[i]);
			w[i] = 1.0 / S[i][i];
		}
		for (i = 1; i < S.rows; ++i) {
			S[i - 1][i] = 0;
			S[i][i - 1] = 0;
		}
		for (i = 2; i < S.rows; ++i) {
			S.data[i] = 0;
		}

		//U*S = A*V
		times(A, V, U);
		for (i = 0; i < U.rows; ++i) {
			for (j = 0; j < U.cols; ++j) {
				U[i][j] = U[i][j] * w[j];
			}
		}
		delete[] w;
	}
	else {
		T* w = new T[A.rows];
		//A*_T(A) = U*S*_T(U)

		times_A_BT(A, A, S);

		EVD_Q_QR_test_1(S, U, S.data, w, eps);

		for (i = 0; i < S.rows; ++i) {
			S[i][i] = sqrt(S.data[i]);
			w[i] = 1.0 / S[i][i];
		}
		for (i = 1; i < S.rows; ++i) {
			S[i - 1][i] = 0;
			S[i][i - 1] = 0;
		}
		for (i = 2; i < S.rows; ++i) {
			S.data[i] = 0;
		}

		//V*S = _T(A)*U
		times_AT_B(A, U, V);
		for (i = 0; i < V.rows; ++i) {
			for (j = 0; j < V.cols; ++j) {
				V[i][j] = V[i][j] * w[j];
			}
		}

		delete[] w;
	}
}

template <typename T>
void SVD_EVD_QR_2(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i, j;
	if (A.rows > A.cols) {
		T* w = new T[A.cols];
		//_T(A)*A = V*S*_T(V)

		times_AT_B(A, A, S);

		EVD_Q_QR_test_2(S, V, w, eps);

		for (i = 0; i < S.rows; ++i) {
			S[i][i] = sqrt(w[i]);
			w[i] = 1.0 / S[i][i];
		}

		//U*S = A*V
		times(A, V, U);
		for (i = 0; i < U.rows; ++i) {
			for (j = 0; j < U.cols; ++j) {
				U[i][j] = U[i][j] * w[j];
			}
		}
		delete[] w;
	}
	else {
		T* w = new T[A.rows];
		//A*_T(A) = U*S*_T(U)

		times_A_BT(A, A, S);

		EVD_Q_QR_test_2(S, U, w, eps);

		for (i = 0; i < S.rows; ++i) {
			S[i][i] = sqrt(w[i]);
			w[i] = 1.0 / S[i][i];
		}

		//V*S = _T(A)*U
		times_AT_B(A, U, V);
		for (i = 0; i < V.rows; ++i) {
			for (j = 0; j < V.cols; ++j) {
				V[i][j] = V[i][j] * w[j];
			}
		}

		delete[] w;
	}
}

template <typename T>
void SVD_EVD_Jacobi(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	if (A.rows > A.cols) {
		//_T(A)*A = V*S*_T(V)
		times_AT_B(A, A, S);

		EVD_Q_Jacobi(S, V, eps);

		for (int i = 0; i < A.cols; ++i) {
			S[i][i] = sqrt(S[i][i]);
		}

		//U*S = A*V
		times(A, V, U);
		for (int i = 0; i < A.rows; ++i) {
			for (int j = 0; j < A.cols; ++j) {
				U[i][j] = U[i][j] / S[j][j];
			}
		}
	}
	else {
		//A*_T(A) = U*S*_T(U)
		MAT<T> B(A.cols, A.rows);
		B.rows = A.rows;

		times_A_BT(A, A, B);

		EVD_Q_Jacobi(B, U, eps);

		for (int i = 0; i < A.rows; ++i) {
			S[i][i] = sqrt(B[i][i]);
		}

		//V*S = _T(A)*U
		B.rows = A.cols;
		times_AT_B(A, U, B);
		for (int i = 0; i < A.cols; ++i) {
			for (int j = 0; j < A.rows; ++j) {
				V[i][j] = B[i][j] / S[j][j];
			}
		}
	}
}

//THIN A(r, c) U(r, s) S(s, s) V(c, s)
//FULL  A(r, c) U(r, r) S(r, c) V(c, c)
template<typename T>
void SVD(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps, SVD_type type) {
	switch (type) {
	case _SVD_QR_1: {
		SVD_QR_1(A, U, S, V, eps);
		break;
	}
	case _SVD_QR_2: {
		SVD_QR_2(A, U, S, V, eps);
		break;
	}
	case _SVD_Jacobi: {
		SVD_Jacobi(A, U, S, V, eps);
		break;
	}
	//case _SVD_Eigen_QR_full: {
	//	SVD_EVD_QR_full(A, U, S, V, eps);
	//	break;
	//}
	case _SVD_Eigen_QR_1: {
		SVD_EVD_QR_1(A, U, S, V, eps);
		break;
	}
	case _SVD_Eigen_QR_2: {
		SVD_EVD_QR_2(A, U, S, V, eps);
		break;
	}
	case _SVD_Eigen_Jacobi: {
		SVD_EVD_Jacobi(A, U, S, V, eps);
		break;
	}
	default:
		break;
	}
}

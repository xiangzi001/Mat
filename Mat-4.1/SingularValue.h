
template<typename T>
void HouseHolder_SingularValue_left(MAT<T>& A, T*& w, int r, int len) {
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
void HouseHolder_SingularValue_right(MAT<T>& A, T*& w, int r, int len) {
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
void Givens_SingularValue_left(MAT<T>& A, int& i, int& j, int& k, double& theta, T& c, T& s, T& temp) {
	theta = atan2(A[i][j], A[i - 1][j]);
	c = cos(theta);
	s = sin(theta);
	for (k = j; k < A.cols; ++k) {
		temp = c * A[i - 1][k] + s * A[i][k];
		A[i][k] = -s * A[i - 1][k] + c * A[i][k];
		A[i - 1][k] = temp;
	}
}

template<typename T>
void Givens_SingularValue_right(MAT<T>& A, int& i, int& j, int& k, double& theta, T& c, T& s, T& temp) {
	theta = atan2(A[j][i], A[j][i - 1]);
	c = cos(theta);
	s = sin(theta);
	for (k = j; k < A.rows; ++k) {
		temp = c * A[k][i - 1] + s * A[k][i];
		A[k][i] = -s * A[k][i - 1] + c * A[k][i];
		A[k][i - 1] = temp;
	}
}

template<typename T>
void Givens_SingularValue_UP(MAT<T>& A, int m, int n, T& eps) {
	int x, y, k;
	double theta;
	T c, s, temp;
	while (n - m - 1) {

		if (fabs(A[m][m + 1]) < eps) {
			m++;
			continue;
		}
		x = m + 1;
		y = m;

		Givens_SingularValue_right(A, x, y, k, theta, c, s, temp);
		Givens_SingularValue_left(A, x, y, k, theta, c, s, temp);

		for (x = m + 2; x < n; ++x) {
			if (fabs(A[x - 1][x]) < eps) {
				if (n - x - 1) {
					Givens_SingularValue_UP(A, m, x, eps);
					m = x;
					break;
				}
				else {
					n--;
					break;
				}
			}
			y = x - 2;
			Givens_SingularValue_right(A, x, y, k, theta, c, s, temp);
			++y;
			Givens_SingularValue_left(A, x, y, k, theta, c, s, temp);
		}
	}
}

template<typename T>
void Givens_SingularValue_DOWN(MAT<T>& A, int m, int n, T& eps) {
	int x, y, k;
	double theta;
	T c, s, temp;
	while (n - m - 1) {

		if (fabs(A[m + 1][m]) < eps) {
			m++;
			continue;
		}
		//print(A);
		x = m + 1;
		y = m;

		Givens_SingularValue_left(A, x, y, k, theta, c, s, temp);
		Givens_SingularValue_right(A, x, y, k, theta, c, s, temp);

		for (x = m + 2; x < n; ++x) {
			if (fabs(A[x][x - 1]) < eps) {
				if (n - x - 1) {
					Givens_SingularValue_DOWN(A, m, x, eps);
					m = x;
					break;
				}
				else {
					n--;
					break;
				}
			}
			y = x - 2;
			Givens_SingularValue_left(A, x, y, k, theta, c, s, temp);
			++y;
			Givens_SingularValue_right(A, x, y, k, theta, c, s, temp);
		}
	}
}

template<typename T>
void SingularValue_QR(MAT<T>& A, MAT<T>& S, double eps) {
	int i = 0;
	int s = A.rows * A.cols;
	for (i = 0; i < s; ++i) {
		S.data[i] = A.data[i];
	}

	if (S.rows > S.cols) {
		T* w = new T[S.rows];
		s = S.cols - 1;
		for (int r = 0; r < s; r++) {
			HouseHolder_SingularValue_left(S, w, r, r);
			HouseHolder_SingularValue_right(S, w, r, r + 1);
		}
		HouseHolder_SingularValue_left(S, w, s, s);

		Givens_SingularValue_UP(S, 0, S.cols, eps);

		delete[] w;
	}
	else {
		T* w = new T[S.cols];
		int r;
		s = S.rows - 1;
		for (r = 0; r < s; r++) {
			HouseHolder_SingularValue_right(S, w, r, r);
			HouseHolder_SingularValue_left(S, w, r, r + 1);
		}
		HouseHolder_SingularValue_right(S, w, s, s);

		Givens_SingularValue_DOWN(S, 0, S.rows, eps);

		delete[] w;
	}
}

template <typename T>
bool Jacobi_SingularValue(MAT<T>& A, int& i, int& j, T& eps) {
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

	if (s < eps) {
		return false;
	}
	else {
		return true;
	}
}

template <typename T>
void SingularValue_Jacobi(MAT<T>& A, MAT<T>& S, double eps) {
	int i;
	int j;
	int n, m;
	int s = S.rows * S.cols;
	for (i = 0; i < s; ++i) {
		S.data[i] = A.data[i];
	}

	if (A.rows > A.cols) {
		n = A.rows;
		m = A.cols;
	}
	else {
		n = A.cols;
		m = A.rows;
	}

	bool flag = true;
	s = 0;
	while (flag) {
		flag = false;
		for (i = s; i < m; ++i) {
			for (j = i + 1; j < n; ++j) {
				if (Jacobi_SingularValue(S, j, i, eps)) {
					flag = true;
				}
			}
			if (!flag) {
				s++;
			}
		}
	}

	T temp;
	for (j = 0; j < S.rows; ++j) {
		temp = 0;
		for (i = 0; i < S.rows; ++i) {
			temp += S[i][j] * S[i][j];
		}
		temp = sqrt(temp);

		for (i = 0; i < S.rows; ++i) {
			S[i][j] = 0;
		}
		S[j][j] = temp;
	}

}

template <typename T>
void SingularValue_EVD_QR(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	if (A.rows > A.cols) {
		T* w = new T[A.cols];
		//_T(A)*A = V*S*_T(V)

		times_AT_B(A, A, S);

		EVD_Q_QR_test(S, V, w, eps);

		for (int i = 0; i < A.cols; ++i) {
			S[i][i] = sqrt(S[i][i]);
			w[i] = 1.0 / S[i][i];
		}

		//U*S = A*V
		times(A, V, U);
		for (int i = 0; i < A.rows; ++i) {
			for (int j = 0; j < A.cols; ++j) {
				U[i][j] = U[i][j] * w[j];
			}
		}
		delete[] w;
	}
	else {
		T* w = new T[A.rows];
		//A*_T(A) = U*S*_T(U)

		times_A_BT(A, A, S);

		EVD_Q_QR_test(S, U, w, eps);

		for (int i = 0; i < A.rows; ++i) {
			S[i][i] = sqrt(S[i][i]);
			w[i] = 1.0 / S[i][i];
		}

		//V*S = _T(A)*U
		times_AT_B(A, U, V);
		for (int i = 0; i < A.cols; ++i) {
			for (int j = 0; j < A.rows; ++j) {
				V[i][j] = V[i][j] * w[j];
			}
		}

		delete[] w;
	}
}

template <typename T>
void SingularValue_EVD_Jacobi(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
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
void SingularValue(MAT<T>& A, MAT<T>& S, double eps, SVD_type type) {
	switch (type) {
	case _SVD_QR: {
		SingularValue_QR(A, S, eps);
		break;
	}
	case _SVD_Jacobi: {
		SingularValue_Jacobi(A, S, eps);
		break;
	}
	case _SVD_Eigen_QR_1: {
		break;
	}
	case _SVD_Eigen_Jacobi: {
		break;
	}
	default:
		break;
	}
}

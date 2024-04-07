
template<typename T>
void SVD_init_full(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V){
	U.init(A.rows, A.rows);
	S.init(A.rows, A.cols);
	V.init(A.cols, A.cols);
}

template<typename T>
void SVD_init_thin(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V) {
	int s;
	if (A.rows > A.cols) {
		s = A.cols;
	}
	else {
		s = A.rows;
	}
	U.init(A.rows, s);
	S.init(s, s);
	V.init(A.cols, s);
}

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

template<typename T>
void SVD_QR_UP(MAT<T>& A, MAT<T>& U, MAT<T>& V, int m, int n, T& eps) {
	int x,y;
	while (n - m - 1) {

		if (fabs(A[m][m + 1]) < eps) {
			m++;
			continue;
		}
		x = m + 1;
		y = m;

		SVD_Givens_right(A, V, x, y );
		SVD_Givens_left(A, U, x, y );

		for (x = m + 2; x < n; ++x) {
			if (fabs(A[x - 1][x]) < eps) {
				if (n > x + 1) {
					SVD_QR_UP(A, U, V, m, x, eps);
					m = x;
					break;
				}
				else {
					--n;
					break;
				}
			}
			y = x - 2;
			SVD_Givens_right(A, V, x, y);
			++y;
			SVD_Givens_left(A, U, x, y);
		}
	}
}

template<typename T>
void SVD_QR_DOWN(MAT<T>& A, MAT<T>& U, MAT<T>& V, int m, int n, T& eps) {
	int x, y;
	while (n - m - 1) {

		if (fabs(A[m+1][m]) < eps) {
			m++;
			continue;
		}
		//print(A);
		x = m + 1;
		y = m;

		SVD_Givens_left(A, U, x, y );
		SVD_Givens_right(A, V, x, y );

		for (x = m + 2; x < n; ++x) {
			if (fabs(A[x][x - 1]) < eps) {
				if (n > x + 1) {
					SVD_QR_DOWN(A, U, V, m, x, eps);
					m = x;
					break;
				}
				else {
					--n;
					break;
				}
			}
			y = x - 2;
			SVD_Givens_left(A, U, x, y );
			++y;
			SVD_Givens_right(A, V, x, y );
		}
	}
}

template<typename T>
void SVD_QR(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i = 0;
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
		T* w = (T*)alloca(sizeof(T) * S.rows);
		//T* w = new T[S.rows];
		s = S.cols - 1;
		for (int r = 0; r < s; r++) {
			SVD_HouseHolder_left(S, U, w, r, r);
			SVD_HouseHolder_right(S, V, w, r, r+1);
		}
		SVD_HouseHolder_left(S, U, w, s, s);
		//delete[] w;

		SVD_QR_UP(S, U, V, 0, S.cols, eps);

	}
	else {
		T* w = (T*)alloca(sizeof(T) * S.cols);
		//T* w = new T[S.cols];
		s = S.rows - 1;
		for (int r = 0; r < s; r++) {
			SVD_HouseHolder_right(S, V, w, r, r);
			SVD_HouseHolder_left(S, U, w, r, r+1);
		}
		SVD_HouseHolder_right(S, V, w, s, s);
		//delete[] w;

		SVD_QR_DOWN(S, U, V, 0, S.rows, eps);

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

template <typename T>
void SVD_EVD_QR_full(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	if (A.rows > A.cols) {
		MAT<T> E(A.rows,A.rows);

		T* w = new T[A.rows];
		//A*_T(A) = U*E*_T(U)

		times_A_BT(A, A, E);

		EVD_Q_QR_test(E, U, w, eps);

		for (int i = 0; i < A.cols; ++i) {
			S[i][i] = sqrt(E[i][i]);
			w[i] = 1.0 / S[i][i];
		}

		//V*E = _T(A)*U
		times_AT_B(A, U, V);
		for (int i = 0; i < A.cols; ++i) {
			for (int j = 0; j < A.cols; ++j) {
				V[i][j] = V[i][j] * w[j];
			}
		}

		delete[] w;
	}
	else {
		MAT<T> E(A.cols, A.cols);
		T* w = (T*)alloca(sizeof(T) * S.cols);
		//T* w = new T[A.cols];
		//_T(A)*A = V*E*_T(V)

		times_AT_B(A, A, E);

		EVD_Q_QR_test(E, V, w, eps);

		for (int i = 0; i < A.rows; ++i) {
			S[i][i] = sqrt(E[i][i]);
			w[i] = 1.0 / S[i][i];
		}

		//U*E = A*V
		times(A, V, U);
		for (int i = 0; i < A.rows; ++i) {
			for (int j = 0; j < A.rows; ++j) {
				U[i][j] = U[i][j] * w[j];
			}
		}
		//delete[] w;
	}
}

template <typename T>
void SVD_EVD_QR_thin(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
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
	case _SVD_QR: {
		SVD_QR(A, U, S, V, eps);
		break;
	}
	case _SVD_Jacobi: {
		SVD_Jacobi(A, U, S, V, eps);
		break;
	}
	case _SVD_Eigen_QR_full: {
		SVD_EVD_QR_full(A, U, S, V, eps);
		break;
	}
	case _SVD_Eigen_QR_thin: {
		SVD_EVD_QR_thin(A, U, S, V, eps);
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

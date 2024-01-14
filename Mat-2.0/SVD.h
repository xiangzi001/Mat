
enum SVD_type {
	_SVD_QR,
	_EVD_QR,
	_EVD_Jacobi
};

template<typename T>
void HouseHolder_SVD_left(MAT<T>& A, MAT<T>& U, T* ur, int r) {
	int i, j, k;
	T temp, sum, cr, hr;
	temp = 0;
	for (k = r; k < A.rows; k++)
		temp += fabs(A[k][r]);
	if (temp > 0)
	{
		sum = 0;
		for (k = r; k < A.rows; k++)
			sum += A[k][r] * A[k][r];
		cr = sqrt(sum);
		if (A[r][r] > 0.0)cr = -cr;
		hr = cr * (cr - A[r][r]);

		//定义ur
		ur[r] = A[r][r] - cr;
		for (i = r + 1; i < A.rows; i++)
		{
			ur[i] = A[i][r];
		}

		//定义U
		for (i = 0; i < U.rows; i++)
		{
			sum = 0;
			for (j = r; j < U.rows; j++)
				sum += U[i][j] * ur[j];
			sum = sum / hr;
			for (j = r; j < U.cols; j++)
			{
				U[i][j] = U[i][j] - ur[j] * sum;
			}
		}

		//定义A
		for (j = r; j < A.cols; j++)
		{
			sum = 0;
			for (i = r; i < A.rows; i++)
				sum += A[i][j] * ur[i];
			sum = sum / hr;
			for (i = r; i < A.rows; i++)
				A[i][j] = A[i][j] - ur[i] * sum;
		}
	}
}
template<typename T>
void HouseHolder_SVD_right(MAT<T>& A, MAT<T>& V, T* &ur, int &r) {
	int i, j, k;
	T temp, sum, cr, hr;
	temp = 0;
	for (k = r; k < A.cols; k++)
		temp += fabs(A[r][k]);
	if (temp > 0)
	{
		sum = 0;
		for (k = r + 1; k < A.cols; k++)
			sum += A[r][k] * A[r][k];
		cr = sqrt(sum);
		if (A[r][r + 1] > 0.0)cr = -cr;
		hr = cr * (cr - A[r][r + 1]);

		//定义ur
		ur[r + 1] = A[r][r + 1] - cr;
		for (i = r + 2; i < A.cols; i++)
		{
			ur[i] = A[r][i];
		}

		//定义V
		for (i = 0; i < V.rows; i++)
		{
			sum = 0;
			for (j = r + 1; j < V.cols; j++)
				sum += V[i][j] * ur[j];
			sum = sum / hr;
			for (j = r + 1; j < V.cols; j++)
			{
				V[i][j] = V[i][j] - ur[j] * sum;
			}
		}

		//定义A
		for (j = r; j < A.rows; j++)
		{
			sum = 0;
			for (i = r + 1; i < A.cols; i++)
				sum += A[j][i] * ur[i];
			sum = sum / hr;
			for (i = r + 1; i < A.cols; i++)
				A[j][i] = A[j][i] - ur[i] * sum;
		}
	}
}

template<typename T>
void Givens_SVD_left(MAT<T>& A, MAT<T>& U, int &i, int &j) {
	double theta = atan2(A[i][j], A[i - 1][j]);
	T c = cos(theta);
	T s = sin(theta);
	for (int k = 0; k < A.cols; k++) {
		T temp = c * A[i - 1][k] + s * A[i][k];
		A[i][k] = -s * A[i - 1][k] + c * A[i][k];
		A[i - 1][k] = temp;
	}
	for (int k = 0; k < U.rows; k++) {
		T temp = c * U[k][i - 1] + s * U[k][i];
		U[k][i] = -s * U[k][i - 1] + c * U[k][i];
		U[k][i - 1] = temp;
	}
}

template<typename T>
void Givens_SVD_UP(MAT<T>& A, MAT<T>& U, MAT<T>& V, int m, int n) {
	int x = m+1;
	int y = m;
	int k;

	double theta = atan2(A[y][x], A[y][x - 1]);
	T c = cos(theta);
	T s = sin(theta);
	T temp;
	for (k = y; k < y+2; k++) {
		temp = c * A[k][x - 1] + s * A[k][x];
		A[k][x] = -s * A[k][x - 1] + c * A[k][x];
		A[k][x - 1] = temp;
	}
	for (k = 0; k < V.rows; k++) {
		temp = c * V[k][x - 1] + s * V[k][x];
		V[k][x] = -s * V[k][x - 1] + c * V[k][x];
		V[k][x - 1] = temp;
	}
	theta = atan2(A[x][y], A[x - 1][y]);
	c = cos(theta);
	s = sin(theta);
	for (k = y; k < y+3; k++) {
		temp = c * A[x - 1][k] + s * A[x][k];
		A[x][k] = -s * A[x - 1][k] + c * A[x][k];
		A[x - 1][k] = temp;
	}
	for (k = 0; k < U.rows; k++) {
		temp = c * U[k][x - 1] + s * U[k][x];
		U[k][x] = -s * U[k][x - 1] + c * U[k][x];
		U[k][x - 1] = temp;
	}

	for (x = m+2; x < n; x++) {
		y = x - 2;
		theta = atan2(A[y][x], A[y][x - 1]);
		c = cos(theta);
		s = sin(theta);
		for (k = y; k < y+3; k++) {
			temp = c * A[k][x - 1] + s * A[k][x];
			A[k][x] = -s * A[k][x - 1] + c * A[k][x];
			A[k][x - 1] = temp;
		}
		for (k = 0; k < V.rows; k++) {
			temp = c * V[k][x - 1] + s * V[k][x];
			V[k][x] = -s * V[k][x - 1] + c * V[k][x];
			V[k][x - 1] = temp;
		}
		y++;
		theta = atan2(A[x][y], A[x - 1][y]);
		c = cos(theta);
		s = sin(theta);
		for (k = y; k < y+3; k++) {
			temp = c * A[x - 1][k] + s * A[x][k];
			A[x][k] = -s * A[x - 1][k] + c * A[x][k];
			A[x - 1][k] = temp;
		}
		for (k = 0; k < U.rows; k++) {
			temp = c * U[k][x - 1] + s * U[k][x];
			U[k][x] = -s * U[k][x - 1] + c * U[k][x];
			U[k][x - 1] = temp;
		}
	}
}

template<typename T>
void SVD_QR(MAT<T>& A, MAT<T>& U, MAT<T>& E, MAT<T>& V, T eps) {
	int i = 0;
	int s = U.rows * U.cols;
	for (i = 0; i < s; i++) {
		U.data[i] = 0;
	}
	for (i = 0; i < U.rows; i++) {
		U[i][i] = 1;
	}
	s = V.rows * V.cols;
	for (i = 0; i < s; i++) {
		V.data[i] = 0;
	}
	for (i = 0; i < V.rows; i++) {
		V[i][i] = 1;
	}
	s = A.rows * A.cols;
	for (i = 0; i < s; i++) {
		E.data[i] = A.data[i];
	}

	if (E.rows > E.cols) {
		T* ur = new T[E.rows]; 
		for (int r = 0; r < E.cols; r++) {
			HouseHolder_SVD_left(E, U, ur, r);
			HouseHolder_SVD_right(E, V, ur, r);
		}
		int m = 0;
		int n = E.cols;
		while (n - m) {
			Givens_SVD_UP(E, U, V, m, n);

			for (; m < n; m++) {
				if (fabs(E[m][m+1]) > eps) {
					break;
				}
			}
			for (; n > m; n--) {
				if (fabs(E[n - 2][n - 1]) > eps) {
					break;
				}
			}
		}
		delete[] ur;
	}
	else {
		T* ur = new T[E.cols];
		int r;
		for (r = 0; r < E.rows; r++) {
			HouseHolder_SVD_left(E, U, ur, r);
			HouseHolder_SVD_right(E, V, ur, r);
		}
		delete[] ur;
	}
}

template <typename T>
void SVD_EVD_QR(MAT<T>& A, MAT<T>& U, MAT<T>& E, MAT<T>& V, T eps) {
	int n, m;
	if (A.rows > A.cols) {
		n = A.cols;
		m = A.rows;
	}
	else {
		n = A.rows;
		m = A.cols;
	}

	//A*T(A) = U*S*T(U)

	MAT<T> B(m, A.rows);
	B.rows = A.rows;

	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.rows; j++) {
			B[i][j] = 0;
			for (int k = 0; k < A.cols; k++)
				B[i][j] = B[i][j] + A[i][k] * A[j][k];
		}
	}

	EVD_Q_QR(B, U, eps);

	for (int i = 0; i < n; i++) {
		E[i][i] = sqrt(B[i][i]);
	}

	//V = T(A)*U*E*inv(S)
	B.rows = A.cols;

	times_AT_B(A, U, B);

	for (int i = 0; i < A.cols; i++) {
		for (int j = 0; j < n; j++) {
			V[i][j] = B[i][j] / E[j][j];
		}
	}
}

template <typename T>
void SVD_EVD_Jacobi(MAT<T>& A, MAT<T>& U, MAT<T>& E, MAT<T>& V, T eps) {
	int n, m;
	if (A.rows > A.cols) {
		n = A.cols;
		m = A.rows;
	}
	else {
		n = A.rows;
		m = A.cols;
	}

	//A*T(A) = U*S*T(U)

	MAT<T> B(m, A.rows);
	B.rows = A.rows;

	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.rows; j++) {
			B[i][j] = 0;
			for (int k = 0; k < A.cols; k++)
				B[i][j] = B[i][j] + A[i][k] * A[j][k];
		}
	}

	EVD_Q_Jacobi(B, U, eps);

	for (int i = 0; i < n; i++) {
		E[i][i] = sqrt(B[i][i]);
	}

	//V = T(A)*U*E*inv(S)
	B.rows = A.cols;

	times_AT_B(A, U, B);

	for (int i = 0; i < A.cols; i++) {
		for (int j = 0; j < n; j++) {
			V[i][j] = B[i][j] / E[j][j];
		}
	}
}

// A(r, c) U(r, r) E(r, c) V(c, c)
template<typename T>
void SVD(MAT<T>& A, MAT<T>& U, MAT<T>& E, MAT<T>& V, T eps, SVD_type type) {
	switch (type) {
	case _SVD_QR:{
		SVD_QR(A, U, E, V, eps);
	}
	case _EVD_QR: {
		SVD_EVD_QR(A, U, E, V, eps);
		break;
	}
	case _EVD_Jacobi: {
		SVD_EVD_Jacobi(A, U, E, V, eps);
		break;
	}
	default:
		break;
	}
}

void bidiagonalize(double** matrix, int m, int n, double** U, double** V) {
	// 双对角化过程
	for (int k = 0; k < n; k++) {
		double x = 0.0;
		for (int i = k; i < m; i++) {
			x += matrix[i][k] * matrix[i][k];
		}
		double sigma = sqrt(x);
		if (matrix[k][k] > 0) {
			sigma = -sigma;
		}
		double alpha = sqrt(2 * x - 2 * matrix[k][k] * sigma);
		double beta = 1 / (sqrt(2 * x));
		matrix[k][k] -= sigma;
		for (int i = k + 1; i < m; i++) {
			matrix[i][k] = beta * matrix[i][k];
		}
		for (int j = k + 1; j < n; j++) {
			double sum = 0.0;
			for (int i = k; i < m; i++) {
				sum += matrix[i][k] * matrix[i][j];
			}
			for (int i = k; i < m; i++) {
				matrix[i][j] -= 2 * matrix[i][k] * sum;
			}
		}
		if (k < n - 2) {
			x = 0.0;
			for (int j = k + 1; j < n; j++) {
				x += matrix[k][j] * matrix[k][j];
			}
			sigma = sqrt(x);
			if (matrix[k][k + 1] > 0) {
				sigma = -sigma;
			}
			alpha = sqrt(2 * x - 2 * matrix[k][k + 1] * sigma);
			beta = 1 / (sqrt(2 * x));
			matrix[k][k + 1] -= sigma;
			for (int j = k + 2; j < n; j++) {
				matrix[k][j] = beta * matrix[k][j];
			}
			for (int i = k; i < m; i++) {
				double sum = 0.0;
				for (int j = k + 1; j < n; j++) {
					sum += matrix[i][j] * matrix[k][j];
				}
				for (int j = k + 1; j < n; j++) {
					matrix[i][j] -= 2 * matrix[k][j] * sum;
				}
			}
		}
	}

	// 构造 U 和 V 矩阵
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			if (i == j) {
				U[i][j] = 1.0;
			}
			else {
				U[i][j] = 0.0;
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j) {
				V[i][j] = 1.0;
			}
			else {
				V[i][j] = 0.0;
			}
		}
	}
}
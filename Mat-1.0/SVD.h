
enum SVD_type {
	_EVD_HouseHolder,
	_EVD_Givens,
	_EVD_Jacobi
};

template<typename T>
void SVD_EVD_HouseHolder(MAT<T>& A, MAT<T>& U, MAT<T>& E, MAT<T>& V, T* ur, T eps) {
	bool flag = true;
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

	MAT<T> B(A.rows, A.rows);
	MAT<T> S(A.rows, A.rows);

	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.rows; j++) {
			B[i][j] = 0;
			for (int k = 0; k < A.cols; k++)
				B[i][j] = B[i][j] + A[i][k] * A[j][k];
		}
	}

	MAT<T> Temp;
	Temp.data = new T[m * m];
	Temp.rows = A.rows;
	Temp.cols = Temp.rows;

	EVD_Q_QR_HouseHolder(B, S, U, Temp, ur, eps);

	for (int i = 0; i < n; i++) {
		E[i][i] = sqrt(S[i][i]);
	}

	//V = T(A)*U*E*inv(S)
	Temp.rows = A.cols;
	Temp.cols = A.rows;

	int size = m * m;
	for (int i = 0; i < size; i++)
		Temp.data[i] = 0;
	for (int k = 0; k < A.rows; k++) {
		for (int i = 0; i < A.cols; i++) {
			for (int j = 0; j < n; j++) {
				Temp[i][j] = Temp[i][j] + A[k][i] * U[k][j];
			}
		}
	}
	for (int i = 0; i < A.cols; i++) {
		for (int j = 0; j < n; j++) {
			V[i][j] = Temp[i][j] / E[j][j];
		}
	}
}

template<typename T>
void SVD_EVD_Givens(MAT<T>& A, MAT<T>& U, MAT<T>& E, MAT<T>& V, T eps) {
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

	MAT<T> B(A.rows, A.rows);
	MAT<T> S(A.rows, A.rows);

	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.rows; j++) {
			B[i][j] = 0;
			for (int k = 0; k < A.cols; k++)
				B[i][j] = B[i][j] + A[i][k] * A[j][k];
		}
	}

	MAT<T> Temp;
	Temp.data = new T[m * m];
	Temp.rows = A.rows;
	Temp.cols = Temp.rows;

	EVD_Q_QR_Givens(B, S, U, Temp, eps);

	for (int i = 0; i < n; i++) {
		E[i][i] = sqrt(S[i][i]);
	}

	//V = T(A)*U*E*inv(S)
	Temp.rows = A.cols;
	Temp.cols = A.rows;

	times_AT_B(A, U, Temp);

	for (int i = 0; i < A.cols; i++) {
		for (int j = 0; j < n; j++) {
			V[i][j] = Temp[i][j] / E[j][j];
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
	case _EVD_HouseHolder: {
		T* ur = new T[A.rows];
		SVD_EVD_HouseHolder(A, U, E, V, ur, eps);
		delete[] ur;
		break;
	}
	case _EVD_Givens: {
		SVD_EVD_Givens(A, U, E, V, eps);
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
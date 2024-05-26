
template<typename T>
void EVD_UP(MAT<T>& A, MAT<T>& X, T* w) {
	int i, j, k;
	T sum;
	for (i = A.cols - 1; i > 0; --i) {
		for (j = i - 1; j >= 0; --j) {
			sum = 0;
			for (k = j + 1; k < i; ++k) {
				sum += A[j][k] * w[k];
			}
			sum += A[j][i];
			w[j] = sum / (A[i][i] - A[j][j]);
		}
		for (j = 0; j < X.rows; ++j) {
			sum = 0;
			for (k = 0; k < i; ++k) {
				sum += X[j][k] * w[k];
			}
			X[j][i] += sum;
		}
	}
	for (i = 0; i < A.rows; ++i) {
		for (j = i + 1; j < A.cols; ++j) {
			A[i][j] = 0;
		}
	}
}

template<typename t>
void EVD_X(MAT<complex<t>>& A, MAT<complex<t>>& E, MAT<complex<t>>& X, double eps) {
	int i;
	int s(E.rows * E.cols);
	for (int i = 0; i < s; ++i) {
		E.data[i] = A.data[i];
	}
	s = X.rows * X.cols;
	for (i = 0; i < s; ++i) {
		X.data[i] = 0;
	}
	for (i = 0; i < X.rows; ++i) {
		X[i][i] = 1;
	}

	complex<t>* w = new complex<t>[A.rows];

	Schur_HouseHolder(X, E, w);

	Schur_Givens(X, E, 0, E.rows, eps);

	EVD_UP(E, X, w);

	delete[] w;
}


template<typename T>
void EVD_UP(MAT<T>& A, MAT<T>& X, T* w) {
	int i, j, k;
	double len;
	T sum;
	T* temp;
	for (i = A.cols - 1; i > 0; --i) {
		for (j = i - 1; j >= 0; --j) {
			temp = A[j];
			sum = 0;
			for (k = j + 1; k < i; ++k) {
				sum += temp[k] * w[k];
			}
			sum += temp[i];
			w[j] = sum / (A[i][i] - temp[j]);
		}

		len = 0;
		for (j = 0; j < i; ++j)
			len += len2(w[j]);
		len = 1 / sqrt(len + 1);
		for (j = 0; j < i; ++j)
			w[j] *= len;
		w[i] = len;

		for (j = 0; j < X.rows; ++j) {
			temp = X[j];
			sum = 0;
			for (k = 0; k <= i; ++k) {
				sum += temp[k] * w[k];
			}
			temp[i] = sum;
		}
	}
	for (i = 0; i < A.rows; ++i) {
		temp = A[i];
		for (j = i + 1; j < A.cols; ++j) {
			temp[j] = 0;
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

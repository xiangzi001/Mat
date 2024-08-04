
template <typename T>
void random(MAT<T>& A, int r, int c, int Big, int Small) {
	A.rows = r;
	A.cols = c;
	int s = r * c;
	delete[] A.data;
	A.data = new T[s];
	int temp = r;
	for (int i = 0; i < s; ++i) {
		random(temp,i);
		temp = fabs(temp);
		A.data[i] = (temp % (Big - Small)) + Small;
	}
	//for (int i = 0; i < s; ++i) {
	//	srand((int)(time(0)) + i);
	//	A.data[i] = (rand() % (Big - Small)) + Small;
	//}
}

template <typename T>
void random(MAT<complex<T>>& A, int r, int c, int Big, int Small) {
	A.rows = r;
	A.cols = c;
	int s = r * c;
	delete[] A.data;
	A.data = new complex<T>[s];
	int temp = r;
	for (int i = 0; i < s; ++i) {
		random(temp, i);
		temp = fabs(temp);
		A.data[i].real = (temp % (Big - Small)) + Small;

		random(temp, A.data[i].real);
		temp = fabs(temp);
		A.data[i].imag = (temp % (Big - Small)) + Small;
	}
	//for (int i = 0; i < s; ++i) {
	//	srand((int)(time(0)) + i);
	//	A.data[i] = (rand() % (Big - Small)) + Small;
	//}
}

template<typename t>
MatData<t> _T(MAT<t>& A) {
	MatData<t> n(A.cols, A.rows);
	for (int i = 0; i < A.cols; ++i)
		for (int j = 0; j < A.rows; ++j) {
			n[i][j] = A[j][i];
		}

	return n;
}
template<typename t>
MatData<t> _T(MatData<t> A) {
	MAT<t> a(A);
	return _T(a);
}

template<typename t>
MatData<complex<t>> _H(MAT<complex<t>>& A) {
	MatData<complex<t>> n(A.cols, A.rows);
	for (int i = 0; i < A.cols; ++i)
		for (int j = 0; j < A.rows; ++j) {
			n[i][j] = A[j][i].conj();
		}

	return n;
}
template<typename t>
MatData<complex<t>> _H(MatData<complex<t>> A) {
	MAT<complex<t>> a(A);
	return _H(a);
}


template<typename T>
void times(MAT<T>& A, MAT<T>& B, MAT<T>& C) {
	int pos = C.rows * C.cols;
	for (int i = 0; i < pos; ++i) {
		C.data[i] = 0;
	}
	T v;
	for (int i = 0; i < C.rows; ++i) {
		for (int k = 0; k < A.cols; ++k) {
			v = A[i][k];
			for (int j = 0; j < C.cols; ++j)
				C[i][j] += v * B[k][j];
		}
	}
}

template<typename T>
void times_A_BT(MAT<T>& A, MAT<T>& B, MAT<T>& C) {
	for (int i = 0; i < C.rows; ++i) {
		for (int j = 0; j < C.rows; ++j) {
			C[i][j] = 0;
			for (int k = 0; k < A.cols; ++k)
				C[i][j] += A[i][k] * B[j][k];
		}
	}
}

template<typename T>
void times_AT_B(MAT<T>& A, MAT<T>& B, MAT<T>& C) {
	int size = C.rows * C.cols;
	for (int i = 0; i < size; ++i)
		C.data[i] = 0;
	for (int k = 0; k < A.rows; ++k) {
		for (int i = 0; i < C.rows; ++i) {
			for (int j = 0; j < C.cols; ++j) {
				C[i][j] += A[k][i] * B[k][j];
			}
		}
	}
}

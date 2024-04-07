

template<typename T>
T det(MAT<T> n) {
	int x, y, i, pos;
	double big, now;
	T r = 1, temp;
	for (y = 0; y < n.cols; ++y) {
		pos = y;
		big = fabs(n[y][y]);
		for (i = y + 1; i < n.rows; ++i) {
			now = fabs(n[i][y]);
			if (now > big) {
				pos = i;
				big = now;
			}
		}
		if (pos != y) {
			for (i = y; i < n.cols; ++i) {
				temp = n[pos][i];
				n[pos][i] = n[y][i];
				n[y][i] = temp;
			}
			r = -r;
		}
		for (x = y + 1; x < n.rows; ++x) {
			temp = n[x][y] / n[y][y];
			for (i = y; i < n.rows; ++i)
				n[x][i] -= n[y][i] * temp;
		}
	}
	for (x = 0; x < n.rows; ++x)
		r = r * n[x][x];

	n.data = 0;
	return r;
}

template<typename T>
T det_c(MAT<T> n) {
	MAT<T> m(n.rows, n.cols, n.data);
	n.data = 0;
	return det(m);
}

template<typename T>
void solve(MAT<T>& A, MAT<T>& b) {
	int x, y, i, j, k;
	T temp;
	for (y = 0; y < A.cols; ++y) {
		if (A[y][y] == 0)
			for (k = y + 1; k < A.rows; ++k)
				if (A[k][y] != 0) {
					for (i = y; i < A.cols; ++i) {
						temp = A[k][i];
						A[k][i] = A[y][i];
						A[y][i] = temp;
					}
					for (i = y; i < b.cols; ++i) {
						temp = b[k][i];
						b[k][i] = b[y][i];
						b[y][i] = temp;
					}
					break;
				}
		for (x = y + 1; x < A.rows; ++x) {
			temp = -A[x][y] / A[y][y];
			for (k = y; k < A.rows; ++k)
				A[x][k] += A[y][k] * temp;
			for (k = 0; k < b.cols; ++k)
				b[x][k] += b[y][k] * temp;
		}
	}
	for (y = A.cols - 1; y >= 0; y--) {
		temp = 1 / A[y][y];
		for (x = 0; x < y; ++x) {
			for (k = 0; k < b.cols; ++k) {
				b[x][k] -= b[y][k] * A[x][y] * temp;
			}
		}
		for (k = 0; k < b.cols; ++k) {
			b[y][k] *= temp;
		}
	}
	/*
	for (int i = 0; i < A.rows; ++i)
		if (A[i][i] != 0) {
			T temp = 1 / A[i][i];
			for (int n = i; n < A.cols; ++n)
				A[i][n] = A[i][n] * temp;
			for (int n = 0; n < b.cols; ++n)
				b[i][n] = b[i][n] * temp;
		}
		else return;

	for (int y = A.cols - 1; y > 0; y--)
		for (int x = 0; x < y; ++x) {
			for (int n = 0; n < b.cols; ++n)
				b[x][n] = b[x][n] - A[x][y] * b[y][n];
		}
	*/
}

template<typename T>
MAT<T> solve_c(MAT<T>& A, MAT<T>& b) {
	MAT<T> B(A.rows, A.cols, A.data);
	MAT<T> c(b.rows, b.cols, b.data);
	solve(B, c);
	return c;
}

template<typename T>
void solve_0(MAT<T>& A, T* b, int n) {

	for (int j = 0; j < n; ++j) {
		T temp = 1 / A[j][j];
		for (int k = j; k < A.cols; ++k)
			A[j][k] = A[j][k] * temp;
	}

	for (int j = n - 1; j > 0; j--) {
		for (int x = 0; x < j; ++x) {
			A[x][n] = A[x][n] - A[x][j] * A[j][n];
		}
	}

	T len = 1;
	for (int j = 0; j < n; ++j) {
		len = len + A[j][n] * A[j][n];
	}
	len = 1.0 / sqrt(len);

	for (int j = 0; j < n; ++j) {
		b[j] = A[j][n] * len;
	}
	b[n] = -len;
}

template<typename T>
void solve_0(MAT<T>& A, T* b) {
	//int i = 0;
	int n = A.cols;

	for (int y = 0; y < A.cols; ++y) {
		if (A[y][y] == 0) {
			n = y;
			break;
		}
		for (int x = y + 1; x < A.rows; ++x) {
			T temp(-A[x][y] / A[y][y]);
			for (int k = y; k < A.rows; ++k)
				A[x][k] = A[x][k] + A[y][k] * temp;
		}
	}

	solve_0(A, b, n);
}

template<typename T>
MAT<T> Inv(MAT<T>& A) {
	MAT<T> I;
	I.iden(A.rows);
	solve(A, I);
	return I;
}

template<typename T>
MAT<T> Inv_c(MAT<T>& A) {
	MAT<T> I;
	I.iden(A.rows);
	MAT<T> B(A.rows, A.cols, A.data);
	solve(B, I);
	return I;
}

template<typename T>
void Inv(MAT<T>& A, MAT<T>& B) {
	int i, s = B.rows * B.cols;
	for (i = 0; i < s; ++i) {
		B.data[i] = 0;
	}
	for (i = 0; i < B.rows; ++i) {
		B[i][i] = 1;
	}
	solve(A, B);
}

template<typename T>
void Inv_c(MAT<T>& A, MAT<T>& B) {
	int i, s = B.rows * B.cols;
	for (i = 0; i < s; ++i) {
		B.data[i] = 0;
	}
	for (i = 0; i < B.rows; ++i) {
		B[i][i] = 1;
	}
	MAT<T> n(A.rows, A.cols, A.data);
	solve(n, B);
}

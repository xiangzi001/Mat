

template<typename T>
T det(MAT<T>& n) {
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

	return r;
}
template<typename T>
T det(MatData<T> n) {
	MAT<T> A(n);
	return det(A);
}

template<typename T>
T det_c(MAT<T>& n) {
	MAT<T> m(n.rows, n.cols, n.data);
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
		temp = 1 / A[y][y];
		A[y][y] = 1;
		for (k = y+1; k < A.cols; ++k)
			A[y][k] *= temp;
		for (k = 0; k < b.cols; ++k)
			b[y][k] *= temp;

		for (x = y + 1; x < A.rows; ++x) {
			temp = -A[x][y];
			for (k = y; k < A.rows; ++k)
				A[x][k] += A[y][k] * temp;
			for (k = 0; k < b.cols; ++k)
				b[x][k] += b[y][k] * temp;
		}
	}
	for (y = A.cols - 1; y >= 0; --y) {
		for (x = 0; x < y; ++x) {
			for (k = 0; k < b.cols; ++k) {
				b[x][k] -= b[y][k] * A[x][y];
			}
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
void solve(MatData<T> A, MatData<T> B) {
	MAT<T> a(A);
	MAT<T> b(B);
	solve(a, b);
}
template<typename T>
void solve(MatData<T> A, MAT<T>& B) {
	MAT<T> a(A);
	solve(a, B);
}

template<typename T>
MatData<T> solve_c(MAT<T>& A, MAT<T>& b) {
	MAT<T> B(A.rows, A.cols, A.data);
	MatData<T> c(b.rows, b.cols, b.data);
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
MatData<T> inv(MAT<T>& A) {
	MatData<T> ret;
	ret.iden(A.rows);

	MAT<T> b(ret);
	solve(A, b);

	b.data = 0;
	return ret;
}
template<typename T>
MatData<T> inv(MatData<T> A) {
	MatData<T> ret;
	ret.iden(A.rows);

	MAT<T> a(A);
	MAT<T> b(ret);
	solve(a, b);

	b.data = 0;
	return ret;
}

template<typename T>
MatData<T> inv_c(MAT<T>& A) {
	MatData<T> ret;
	ret.iden(A.rows);

	MAT<T> a(A.rows, A.cols, A.data);
	MAT<T> b(ret);
	solve(a, b);

	b.data = 0;
	return ret;
}

template<typename T>
void inv(MAT<T>& A, MAT<T>& B) {
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
int Rank(MAT<T>& A, double eps = 0) {
	int r, c;
	for (r = 0, c = 0; c < A.cols; c++) {

		// 1. 找到最大值
		int t = r;
		for (int i = r; i < A.cols; ++i) {
			if (fabs(A[i][c]) > fabs(A[t][c]))
				t = i;
		}

		if (fabs(A[t][c]) < eps) continue;

		// 2. 交换到首行
		for (int i = c; i < A.rows; ++i) swap(A[r][i], A[t][i]);

		// 3. 本行首列置为1
		for (int i = A.rows - 1; i >= c; --i) A[r][i] /= A[r][c];

		// 4. 下面的行首列置为0
		for (int i = r + 1; i < A.cols; ++i) {
			if (fabs(A[i][c]) < eps) continue;
			for (int j = A.rows - 1; j >= c; --j) {
				A[i][j] -= A[i][c] * A[r][j]; // - 首航本列 * i, r首列商
			}
		}

		r++;
	}
	return r;
}

template<typename T>
int Rank_c(MAT<T>& A, double eps = 0) {
	MAT<T> B(A.rows, A.cols, A.data);
	return Rank(B);
}

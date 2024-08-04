

template<typename T>
T Det(MatData<T> n) {
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
T Det(MAT<T>& n) {
	MatData<T> m(n.rows, n.cols, n.data);
	return Det(m);
}

template<typename T>
MatData<T> Solve_direct(MatData<T> A, MatData<T> B) {
	int x, y, i, j, pos;
	double now, big;
	T temp;

	for (y = 0; y < A.cols; ++y) {
		///*
		j = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.rows; ++i) {
			now = fabs(A[i][y]);
			if (now > big) {
				j = i;
				big = now;
			}
		}
		if (j != y) {
			for (i = y; i < A.cols; ++i) {
				temp = A[j][i];
				A[j][i] = A[y][i];
				A[y][i] = temp;
			}
			for (i = 0; i < B.cols; ++i) {
				temp = B[j][i];
				B[j][i] = B[y][i];
				B[y][i] = temp;
			}
		}
		//*/
		/*
		if (A[y][y] == 0)
			for (j = y + 1; j < A.rows; ++j)
				if (A[j][y] != 0) {
					for (i = y; i < A.cols; ++i) {
						temp = A[j][i];
						A[j][i] = A[y][i];
						A[y][i] = temp;
					}
					for (i = 0; i < B.cols; ++i) {
						temp = B[j][i];
						B[j][i] = B[y][i];
						B[y][i] = temp;
					}
					break;
				}
		//*/
		temp = 1 / A[y][y];
		A[y][y] = 1;
		for (j = y+1; j < A.cols; ++j)
			A[y][j] *= temp;
		for (j = 0; j < B.cols; ++j)
			B[y][j] *= temp;

		for (x = y + 1; x < A.rows; ++x) {
			temp = -A[x][y];
			for (j = y; j < A.rows; ++j)
				A[x][j] += A[y][j] * temp;
			for (j = 0; j < B.cols; ++j)
				B[x][j] += B[y][j] * temp;
		}
	}
	for (y = A.cols - 1; y >= 0; --y) {
		for (x = 0; x < y; ++x) {
			for (j = 0; j < B.cols; ++j) {
				B[x][j] -= B[y][j] * A[x][y];
			}
		}
	}
	return B;
	/*
	for (int i = 0; i < A.rows; ++i)
		if (A[i][i] != 0) {
			T temp = 1 / A[i][i];
			for (int n = i; n < A.cols; ++n)
				A[i][n] = A[i][n] * temp;
			for (int n = 0; n < B.cols; ++n)
				B[i][n] = B[i][n] * temp;
		}
		else return;

	for (int y = A.cols - 1; y > 0; y--)
		for (int x = 0; x < y; ++x) {
			for (int n = 0; n < B.cols; ++n)
				B[x][n] = B[x][n] - A[x][y] * B[y][n];
		}
	*/
}

template<typename T>
MatData<T> Solve_direct(MAT<T>& A, MatData<T> B) {
	MatData<T> a(A.rows, A.cols, A.data);
	return Solve_direct(a, B);
}
template<typename T>
MatData<T> Solve_direct(MatData<T> A, MAT<T>& B) {
	MatData<T> b(B.rows, B.cols, B.data);
	return Solve_direct(A, b);
}

template<typename T>
MatData<T> Solve_direct(MAT<T>& A, MAT<T>& B) {
	MatData<T> a(A.rows, A.cols, A.data);
	MatData<T> b(B.rows, B.cols, B.data);
	return Solve_direct(a, b);
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
MatData<T> Inv(MatData<T> A) {
	MatData<T> ret;
	ret.iden(A.rows);
	Solve_direct(A,  ret);
	return ret;
}
template<typename T>
MatData<T> Inv(MAT<T>& A) {
	MatData<T> ret;
	ret.iden(A.rows);
	MatData<T> a(A.cols, A.rows, A.data);
	Solve_direct(a, ret);
	return ret;
}

template<typename T>
int Rank(MatData<T> A, double eps = 0) {
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
int Rank(MAT<T>& A, double eps = 0) {
	MatData<T> a(A.rows, A.cols, A.data);
	return Rank(a, eps);
}

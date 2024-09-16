

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

		++r;
	}
	return r;
}

template<typename T>
int Rank(MAT<T>& A, double eps = 0) {
	MatData<T> a(A.rows, A.cols, A.data);
	return Rank(a, eps);
}

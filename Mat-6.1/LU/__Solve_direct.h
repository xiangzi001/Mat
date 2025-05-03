template<typename T>
T Det(MAT<T> n) {
	return __Det(n);
}

template<typename T>
MAT<T> Solve_direct(MAT<T> A, MAT<T> B) {
	__Solve_direct(A, B);
	return B;
}

template<typename T>
MAT<T> Inv(MAT<T> A) {
	MAT<T> B;
	B.iden(A.rows);

	__Solve_direct(A, B);

	return B;
}

template<typename T>
int Rank(MAT<T> A, double eps = 0) {
	int r, c;
	for (r = 0, c = 0; c < A.cols; c++) {

		// 1. 找到最大值
		int t = r;
		for (int i = r; i < A.cols; ++i) {
			if (fabs(A[i][c]) > fabs(A[t][c]))
				t = i;
		}

		if (fabs(A[t][c]) <= eps) continue;

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

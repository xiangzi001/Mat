
template <typename T>
void __EVD_Q_Jacobi(MAT<T>& A, MAT<T>& Q, T dbEps)
{
	int i, j;
	int m, n;
	int k(Q.rows * Q.cols);
	for (i = 0; i < k; ++i) {
		Q.data[i] = 0;
	}
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	T Max, temp;
	T dbApp, dbApq, dbAqq;
	double angle;
	T s, c, s2, c2;

	// 第二步：对矩阵A.data即矩阵A，求出最大索引m、n即p、q

	while (1)
	{
		// 第二步：对矩阵A.data即矩阵A，求出最大索引m、n即p、q
		Max = A.data[1];
		m = 0;
		n = 1; // 因为求的是非对角线上最大值，所有从1开始

		for (i = 1; i < A.rows; ++i)			//行
		{
			for (j = 0; j < i; ++j)		//列
			{
				temp = fabs(A[i][j]);

				if (temp > Max)
				{
					Max = temp;
					m = i;
					n = j;
				}
			}
		}

		if (Max < dbEps)
			break;

		dbApp = A[m][m];
		dbApq = A[m][n];
		dbAqq = A[n][n];

		angle = 0.5 * atan2(-2 * dbApq, dbAqq - dbApp);

		s = sin(angle);
		c = cos(angle);
		s2 = sin(2 * angle);
		c2 = cos(2 * angle);

		A[m][m] = dbApp * c * c +
			dbAqq * s * s + 2 * dbApq * c * s;
		A[n][n] = dbApp * s * s +
			dbAqq * c * c - 2 * dbApq * c * s;
		A[m][n] = 0.5 * (dbAqq - dbApp) * s2 + dbApq * c2;
		A[n][m] = A[m][n];

		for (i = 0; i < A.rows; ++i)
		{
			if ((i != n) && (i != m))
			{
				temp = A[i][m];
				A[i][m] = A[i][n] * s + temp * c;
				A[i][n] = A[i][n] * c - temp * s;
			}
		}

		for (j = 0; j < A.rows; ++j)
		{
			if ((j != n) && (j != m))
			{
				temp = A[m][j];
				A[m][j] = A[n][j] * s + temp * c;
				A[n][j] = A[n][j] * c - temp * s;
			}
		}

		for (i = 0; i < Q.rows; ++i)
		{
			temp = Q[i][m];
			Q[i][m] = Q[i][n] * s + temp * c;
			Q[i][n] = Q[i][n] * c - temp * s;
		}

	}
	/*
	for (i = 0; i < A.rows; ++i) {
		T m = 0;
		int c = i;
		T value;
		for (j = i; j < A.rows; ++j) {
			if (A[j][j] > m) {
				m = A[j][j];
				c = j;
			}
		}
		if (c != i) {
			value = A[c][c];
			A[c][c] = A[i][i];
			A[i][i] = value;
			for (j = 0; j < A.rows; ++j) {
				value = Q[c][j];
				Q[c][j] = Q[i][j];
				Q[i][j] = value;
			}
		}
	}
	*/
}

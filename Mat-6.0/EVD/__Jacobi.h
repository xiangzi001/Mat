
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

	T Max = 1, temp;
	double angle;
	T s, c;

	while (Max * Max > dbEps){
		Max = A.data[1];
		m = 1;
		n = 0;
		for (i = 1; i < A.rows; ++i){
			for (j = 0; j < i; ++j){
				temp = fabs(A[i][j]);
				if (temp > Max){
					Max = temp;
					m = i;
					n = j;
				}
			}
		}

		angle = 0.5 * atan2(-2 * A[m][n], A[n][n] - A[m][m]);
		s = sin(angle);
		c = cos(angle);

		for (i = 0; i < A.rows; ++i) {
			temp = A[i][m];
			A[i][m] = A[i][n] * s + temp * c;
			A[i][n] = A[i][n] * c - temp * s;
		}

		for (j = 0; j < A.rows; ++j) {
			temp = A[m][j];
			A[m][j] = A[n][j] * s + temp * c;
			A[n][j] = A[n][j] * c - temp * s;
		}
		for (i = 0; i < Q.rows; ++i){
			temp = Q[i][m];
			Q[i][m] = Q[i][n] * s + temp * c;
			Q[i][n] = Q[i][n] * c - temp * s;
		}

	}
}

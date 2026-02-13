
template <typename T>
void __EVD_Q_Jacobi(MAT<T>& A, MAT<T>& Q, double eps)
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
	T* Temp;
	double angle;
	T s, c;

	while (Max * Max > eps){
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

		for (j = 0; j < A.rows; ++j) {
			Temp = A[j];
			temp = Temp[m];
			Temp[m] = Temp[n] * s + temp * c;
			Temp[n] = Temp[n] * c - temp * s;
		}
		__ApplyGivensOnTheRight(A[m], A[n], c, s, Q.rows);
		__ApplyGivensOnTheRight(Q[m], Q[n], c, s, Q.rows);
	}
}

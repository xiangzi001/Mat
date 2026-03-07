
template <typename T>
void __EVD_Q_Jacobi(MAT<T>& A, MAT<T>& Q, double eps)
{
	int m, n, i;
	__set0(Q.data, Q.rows * Q.cols);
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	T temp;
	T* Temp;
	double angle;
	T s, c;
	bool flag = true;

	while (flag){
		flag = false;
		for (m = 1; m < A.rows; ++m) {
			for (n = 0; n < m; ++n) {
				flag = flag || fabs(A(n, m)) > eps;
				angle = 0.5 * atan2(-2 * A[m][n], A[n][n] - A[m][m]);
				s = sin(angle);
				c = cos(angle);

				for (i = 0; i < A.rows; ++i) {
					Temp = A[i];
					temp = Temp[m];
					Temp[m] = Temp[n] * s + temp * c;
					Temp[n] = Temp[n] * c - temp * s;
				}
				__ApplyGivensOnTheRight(A[m], A[n], c, s, Q.rows);
				__ApplyGivensOnTheRight(Q[m], Q[n], c, s, Q.rows);
			}
		}
	}
}


template <typename T>
void LLT(MAT<T>& A, MAT<T>& L) {
	for (int j = 0; j < A.cols; ++j)
		for (int i = j; i < A.rows; ++i) {
			if (i == j) {
				T sum = 0;
				for (int k = 0; k < j; ++k) {
					sum = sum + L[j][k] * L[j][k];
				}
				L[i][j] = sqrt(A[i][j] - sum);
				//L[i][j] = pow(A[i][j] - sum, 0.5);
			}
			else {
				T sum = 0;
				for (int k = 0; k < j; ++k) {
					sum = sum + L[i][k] * L[j][k];
				}
				L[i][j] = (A[i][j] - sum) / L[j][j];
			}
		}
}

template <typename T>
void LDLT(MAT<T>& A, MAT<T>& L, MAT<T>& D) {
	for (int i = 0; i < A.rows; ++i)
	{
		T temp = 0;
		for (int j = 0; j < i; ++j) {
			temp = temp + L[i][j] * L[i][j] * D[j][j];
		}
		D[i][i] = A[i][i] - temp;

		L[i][i] = 1;
		for (int j = i + 1; j < L.rows; ++j) {
			temp = 0;
			for (int k = 0; k < i; ++k) {
				temp = temp + L[i][k] * L[j][k] * D[k][k];
			}
			L[j][i] = (A[j][i] - temp) / D[i][i];
		}
	}
}

template <typename T>
void LDLT_2(MAT<T>& A, MAT<T>& L, MAT<T>& D) {
	for (int i = 0; i < L.rows; ++i)
	{
		//更新L矩阵  下三角矩阵
		for (int j = 0; j <= i; ++j)
		{
			if (j == i) L[i][i] = 1;
			else
			{
				L[i][j] = A[i][j];
				for (int k = 0; k < j; ++k)
				{
					L[i][j] -= L[i][k] * D[k][j];
				}
				L[i][j] /= D[j][j];
			}
		}

		//更新D矩阵  上三角矩阵
		for (int j = i; j < L.rows; ++j)
		{
			D[i][j] = A[i][j];
			for (int k = 0; k < i; ++k)
			{
				D[i][j] -= L[i][k] * D[k][j];
			}
		}
	}
	for (int i = 0; i < L.rows; ++i)
	{
		D[i][i] = D[i][i] / L[i][i];
		for (int j = i+1; j < L.rows; ++j)
		{
			D[i][j] = 0;
		}
	}
}
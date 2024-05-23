
template<typename T>
void HouseHolder_EigenValue(MAT<T>& A, T* w) {
	int i, j, k, r;
	T temp, sum, cr, hr;
	for (r = 0; r < A.rows - 1; r++) {
		temp = 0;
		for (k = r; k < A.rows; ++k)
			temp += fabs(A[k][r]);
		if (temp > 0)
		{
			sum = 0;
			for (k = r + 1; k < A.rows; ++k)
				sum += A[k][r] * A[k][r];
			cr = sqrt(sum);
			if (A[r + 1][r] > 0.0)cr = -cr;
			hr = cr * (cr - A[r + 1][r]);

			//定义w
			w[r + 1] = A[r + 1][r] - cr;
			for (i = r + 2; i < A.rows; ++i)
			{
				w[i] = A[i][r];
			}

			//定义A
			for (i = r; i < A.rows; ++i)
			{
				sum = 0;
				for (j = r + 1; j < A.rows; ++j)
					sum += A[i][j] * w[j];
				sum = sum / hr;
				for (j = r + 1; j < A.cols; ++j)
				{
					A[i][j] = A[i][j] - w[j] * sum;
				}
			}
			for (j = r; j < A.cols; ++j)
			{
				sum = 0;
				for (i = r + 1; i < A.rows; ++i)
					sum += A[i][j] * w[i];
				sum = sum / hr;
				for (i = r + 1; i < A.rows; ++i)
					A[i][j] = A[i][j] - w[i] * sum;
			}
		}
	}
}

template<typename T>
void Givens_EigenValue(MAT<T>& A, int m, int n, T eps, T* Temp) {

	while (n - m > 2) {
		//print(A);

		if (fabs(A[m + 1][m]) < eps) {
			m++;
			continue;
		}

		double theta = atan2(A[m + 1][m], A[m][m]);
		T c = cos(theta);
		T s = sin(theta);
		for (int k = m; k < A.rows; ++k) {
			T temp = c * A[m][k] + s * A[m + 1][k];
			A[m + 1][k] = -s * A[m][k] + c * A[m + 1][k];
			A[m][k] = temp;
		}
		for (int k = m; k < A.rows; ++k) {
			T temp = c * A[k][m] + s * A[k][m + 1];
			A[k][m + 1] = -s * A[k][m] + c * A[k][m + 1];
			A[k][m] = temp;
		}
		
		for (int i = m + 2; i < n; ++i) {
			if ((fabs(A[i][i - 1]) < eps)&&(fabs(A[i][i - 2]) < eps)){
				switch (n - i) {
				case 1: {
					n--;
					break;
				}
				case 2: {
					n = n-2;
					break;
				}
				default: {
					Givens_EigenValue(A, m, i, eps, Temp);
					m = i;
					break;
				}
				}
				break;
			}
			else {
				theta = atan2(A[i][i - 2], A[i - 1][i - 2]);
				c = cos(theta);
				s = sin(theta);

				for (int k = m; k < A.rows; ++k) {
					T temp = c * A[i - 1][k] + s * A[i][k];
					A[i][k] = -s * A[i - 1][k] + c * A[i][k];
					A[i - 1][k] = temp;
				}
				for (int k = m; k < A.rows; ++k) {
					T temp = c * A[k][i - 1] + s * A[k][i];
					A[k][i] = -s * A[k][i - 1] + c * A[k][i];
					A[k][i - 1] = temp;
				}
			}
		}
		for (int i = m; i < n; ++i) {
			//Temp[i] = Temp[i] + A[n - 1][n - 1];
			//A[i][i] = A[i][i] - A[n - 1][n - 1];
		}
	}
}

template<typename T>
void EigenValue_QR(MAT<T>& A, T eps) {

	T* w = new T[A.rows];
	HouseHolder_EigenValue(A, w);

	for (int i = 0; i < A.rows; ++i) {
		w[i] = 0;
	}
	Givens_EigenValue(A, 0, A.rows, eps, w);

	for (int i = 0; i < A.rows; ++i) {
		A[i][i] = A[i][i] + w[i];
	}

	delete[] w;
}


template<typename T>
void EigenValue_QR_test(MAT<T>& A, T* w, T eps) {

	HouseHolder_EigenValue(A, w);

	for (int i = 0; i < A.rows; ++i) {
		w[i] = 0;
	}
	Givens_EigenValue(A, 0, A.rows, eps, w);

	for (int i = 0; i < A.rows; ++i) {
		A[i][i] = A[i][i] + w[i];
	}

	for (int i = 0; i < A.rows; ++i) {
		T max = 0;
		int c = i;
		T value;
		for (int j = i; j < A.rows; ++j) {
			if (A[j][j] > max) {
				max = A[j][j];
				c = j;
			}
		}
		if (c != i) {
			value = A[c][c];
			A[c][c] = A[i][i];
			A[i][i] = value;
		}
	}
}

template <typename T>
void EigenValue_Jacobi(MAT<T>& A, T dbEps)
{
	while (1)
	{
		T dbMax = A.data[1];
		int nRow = 0;
		int nCol = 1; 
		for (int i = 1; i < A.rows; ++i)			//行
		{
			for (int j = 0; j < i; ++j)		//列
			{
				T d = fabs(A[i][j]);

				if (d > dbMax)
				{
					dbMax = d;
					nRow = i;
					nCol = j;
				}
			}
		}

		if (dbMax < dbEps)
			break;

		T dbApp = A[nRow][nRow];
		T dbApq = A[nRow][nCol];
		T dbAqq = A[nCol][nCol];

		T dbAngle = 0.5 * atan2(-2 * dbApq, dbAqq - dbApp);

		T dbSinTheta = sin(dbAngle);
		T dbCosTheta = cos(dbAngle);
		T dbSin2Theta = sin(2 * dbAngle);
		T dbCos2Theta = cos(2 * dbAngle);

		A[nRow][nRow] = dbApp * dbCosTheta * dbCosTheta +
			dbAqq * dbSinTheta * dbSinTheta + 2 * dbApq * dbCosTheta * dbSinTheta;
		A[nCol][nCol] = dbApp * dbSinTheta * dbSinTheta +
			dbAqq * dbCosTheta * dbCosTheta - 2 * dbApq * dbCosTheta * dbSinTheta;
		A[nRow][nCol] = 0.5 * (dbAqq - dbApp) * dbSin2Theta + dbApq * dbCos2Theta;
		A[nCol][nRow] = A[nRow][nCol];

		for (int i = 0; i < A.rows; ++i)
		{
			if ((i != nCol) && (i != nRow))
			{
				int u = i * A.rows + nRow;
				int w = i * A.rows + nCol;
				dbMax = A.data[u];
				A.data[u] = A.data[w] * dbSinTheta + dbMax * dbCosTheta;
				A.data[w] = A.data[w] * dbCosTheta - dbMax * dbSinTheta;
			}
		}

		for (int j = 0; j < A.rows; ++j)
		{
			if ((j != nCol) && (j != nRow))
			{
				int u = nRow * A.rows + j;
				int w = nCol * A.rows + j;
				dbMax = A.data[u];
				A.data[u] = A.data[w] * dbSinTheta + dbMax * dbCosTheta;
				A.data[w] = A.data[w] * dbCosTheta - dbMax * dbSinTheta;
			}
		}
	}

	for (int i = 0; i < A.rows; ++i) {
		T m = 0;
		int c = i;
		T value;
		for (int j = i; j < A.rows; ++j) {
			if (A[j][j] > m) {
				m = A[j][j];
				c = j;
			}
		}
		if (c != i) {
			value = A[c][c];
			A[c][c] = A[i][i];
			A[i][i] = value;
		}
	}
}


template<typename T>
void EigenValue(MAT<T>& A, MAT<T>& E, T eps, EVD_type type) {
	switch (type)
	{
	case _QR: {
		for (int i = 0; i < E.rows * E.cols; ++i) {
			E.data[i] = A.data[i];
		}
		EigenValue_QR(E, eps);
		break;
	}
	case _Jacobi: {
		for (int i = 0; i < E.rows * E.cols; ++i) {
			E.data[i] = A.data[i];
		}
		EigenValue_Jacobi(E, eps);
		break;
	}
	default:
		break;
	}
}

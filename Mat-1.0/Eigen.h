
enum EVD_type {
	_QR_HouseHolder,
	_QR_Givens,
	_Jacobi,
	_LLT
};

template<typename T>
void EigenValue(MAT<T>& A, MAT<T>& Q, T eps) {
	bool flag = true;

	T* ur = new T[A.rows];
	MAT<T> C(A.rows, A.rows);
	T* Temp;

	while (flag) {

		QR_HouseHolder(Q, A, ur);
		times(A, Q, C);

		Temp = A.data;
		A.data = C.data;
		C.data = Temp;
		flag = false;
		for (int i = 2; i < A.rows; i++) {
			for (int j = 0; j < i - 1; j++)
				if (fabs(A[i][j]) > eps) {
					flag = true;
					break;
				}
		}
	}
	delete[] ur;
}

template<typename T>
void EVD_X_QR_HouseHolder(MAT<T>& A, MAT<T>& E, MAT<T>& X, MAT<T>& Temp, T* ur, T eps) {
	bool flag = true;
	T* _data;

	QR_HouseHolder(X, A, ur);
	times(A, X, E);

	while (flag) {
		// 计算X和A
		QR_HouseHolder(A, E, ur);

		times(E, A, Temp);
		_data = E.data;
		E.data = Temp.data;
		Temp.data = _data;

		times(X, A, Temp);
		_data = X.data;
		X.data = Temp.data;
		Temp.data = _data;

		flag = false;
		for (int i = 2; i < A.rows; i++) {
			for (int j = 0; j < i - 1; j++)
				if (fabs(E[i][j]) > eps) {
					flag = true;
					break;
				}
		}
	}
}

template<typename T>
void EVD_X_QR_Givens(MAT<T>& A, MAT<T>& E, MAT<T>& X, MAT<T>& Temp, T eps) {
	bool flag = true;
	T* _data;

	QR_Givens(X, A);
	times(A, X, E);

	while (flag) {
		// 计算X和A
		QR_Givens(A, E);

		times(E, A, Temp);
		_data = E.data;
		E.data = Temp.data;
		Temp.data = _data;

		times(X, A, Temp);
		_data = X.data;
		X.data = Temp.data;
		Temp.data = _data;

		flag = false;
		for (int i = 2; i < A.rows; i++) {
			for (int j = 0; j < i - 1; j++)
				if (fabs(E[i][j]) > eps) {
					flag = true;
					break;
				}
		}
	}
}


template <typename T>
void EVD_X_Jacobi(MAT<T>& A, MAT<T>& X, T dbEps)
{
	// 第一步：初始化特征向量矩阵pbdEigenValues即V
	for (int i = 0; i < X.rows; i++)
	{
		// 对角线为1，其他为0
		X[i][i] = 1.0f;
		for (int j = 0; j < X.rows; j++)
		{
			if (i != j)
				X[i][j] = 0.0f;
		}
	}

	while (1)
	{
		// 第二步：对矩阵A.data即矩阵A，求出最大索引nRow、nCol即p、q
		T dbMax = A.data[1];
		int nRow = 0;
		int nCol = 1; // 因为求的是非对角线上最大值，所有从1开始
		for (int i = 2; i < A.rows; i++)			//行
		{
			for (int j = 0; j < i-1; j++)		//列
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

		for (int i = 0; i < A.rows; i++)
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

		for (int j = 0; j < A.rows; j++)
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

		for (int i = 0; i < X.rows; i++)
		{
			int u = i * X.rows + nRow;
			int w = i * X.rows + nCol;

			dbMax = X.data[u];
			X.data[u] = X.data[w] * dbSinTheta + dbMax * dbCosTheta;
			X.data[w] = X.data[w] * dbCosTheta - dbMax * dbSinTheta;
		}

	}

	for (int i = 0; i < A.rows; i++) {
		T m = 0;
		int c = i;
		T value;
		for (int j = i; j < A.rows; j++) {
			if (A[j][j] > m) {
				m = A[j][j];
				c = j;
			}
		}
		if (c != i) {
			value = A[c][c];
			A[c][c] = A[i][i];
			A[i][i] = value;
			for (int j = 0; j < A.rows; j++) {
				value = X[c][j];
				X[c][j] = X[i][j];
				X[i][j] = value;
			}
		}
	}
}

template<typename T>
void EVD_X(MAT<T>& A, MAT<T>& E, MAT<T>& X, T eps, EVD_type type) {
	switch (type)
	{
	case _QR_HouseHolder: {
		T* ur = new T[A.rows];
		MAT<T> Temp(A.rows, A.rows);
		EVD_X_QR_HouseHolder(A, E, X, Temp, ur, eps);
		delete[] ur;
		break;
	}
	case _QR_Givens: {
		MAT<T> Temp(A.rows, A.rows);
		EVD_X_QR_Givens(A, E, X, Temp, eps);
		break;
	}
	case _Jacobi: {
		for (int i = 0; i < E.rows * E.cols; i++) {
			E.data[i] = A.data[i];
		}
		EVD_X_Jacobi(E, X, eps);
		break;
	}
	default:
		break;
	}
}





template<typename T>
void EVD_Q_QR_HouseHolder(MAT<T>& A, MAT<T>& E, MAT<T>& X, MAT<T>& Temp, T* ur, T eps) {
	bool flag = true;
	T* _data;

	QR_HouseHolder(X, A, ur);
	times(A, X, E);

	while (flag) {
		// 计算X和A
		QR_HouseHolder(A, E, ur);

		times(E, A, Temp);
		_data = E.data;
		E.data = Temp.data;
		Temp.data = _data;

		times(X, A, Temp);
		_data = X.data;
		X.data = Temp.data;
		Temp.data = _data;

		flag = false;
		for (int i = 1; i < A.rows; i++) {
			for (int j = 0; j < i; j++)
				if (fabs(E[i][j]) > eps) {
					flag = true;
					break;
				}
		}
	}
}

template<typename T>
void EVD_Q_QR_Givens(MAT<T>& A, MAT<T>& E, MAT<T>& X, MAT<T>& Temp, T eps) {
	bool flag = true;
	T* _data;
	int i = 1;
	int j = 0;

	QR_Givens(X, A, eps);
	times(A, X, E);

	while (flag) {
		// 计算X和A
		QR_Givens(A, E, eps);

		times(E, A, Temp);
		_data = E.data;
		E.data = Temp.data;
		Temp.data = _data;

		times(X, A, Temp);
		_data = X.data;
		X.data = Temp.data;
		Temp.data = _data;

		flag = false;
		for (int i = 1; i < A.rows; i++) {
			for (int j = 0; j < i; j++)
				if (fabs(E[i][j]) > eps) {
					flag = true;
					i = A.rows;
					break;
				}
		}
	}
}

template<typename T>
void EVD_Q_LLT(MAT<T>& A, MAT<T>& E, MAT<T>& Q, T eps) {
	bool flag = true;

	Cholesky(A, Q);
	times_AT_B(Q, Q, E);

	while (flag) {
		// 计算Q和A
		Cholesky(E, A);
		times_AT_B(A, A, E);

		flag = false;
		for (int i = 1; i < A.rows; i++) {
			for (int j = 0; j < i; j++)
				if (fabs(E[i][j]) > eps) {
					flag = true;
					break;
				}
		}
	}
}

template <typename T>
void EVD_Q_Jacobi(MAT<T>& A, MAT<T>& Q, T dbEps)
{
	// 第一步：初始化特征向量矩阵pbdEigenValues即V
	for (int i = 0; i < Q.rows; i++)
	{
		// 对角线为1，其他为0
		Q[i][i] = 1.0f;
		for (int j = 0; j < Q.rows; j++)
		{
			if (i != j)
				Q[i][j] = 0.0f;
		}
	}

	while (1)
	{
		// 第二步：对矩阵A.data即矩阵A，求出最大索引nRow、nCol即p、q
		T dbMax = A.data[1];
		int nRow = 0;
		int nCol = 1; // 因为求的是非对角线上最大值，所有从1开始
		for (int i = 1; i < A.rows; i++)			//行
		{
			for (int j = 0; j < i; j++)		//列
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

		for (int i = 0; i < A.rows; i++)
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

		for (int j = 0; j < A.rows; j++)
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

		for (int i = 0; i < Q.rows; i++)
		{
			int u = i * Q.rows + nRow;
			int w = i * Q.rows + nCol;

			dbMax = Q.data[u];
			Q.data[u] = Q.data[w] * dbSinTheta + dbMax * dbCosTheta;
			Q.data[w] = Q.data[w] * dbCosTheta - dbMax * dbSinTheta;
		}

	}

	for (int i = 0; i < A.rows; i++) {
		T m = 0;
		int c = i;
		T value;
		for (int j = i; j < A.rows; j++) {
			if (A[j][j] > m) {
				m = A[j][j];
				c = j;
			}
		}
		if (c != i) {
			value = A[c][c];
			A[c][c] = A[i][i];
			A[i][i] = value;
			for (int j = 0; j < A.rows; j++) {
				value = Q[c][j];
				Q[c][j] = Q[i][j];
				Q[i][j] = value;
			}
		}
	}
}


template<typename T>
void EVD_Q(MAT<T>& A, MAT<T>& E, MAT<T>& Q, T eps, EVD_type type) {
	switch (type)
	{
	case _QR_HouseHolder: {
		T* ur = new T[A.rows];
		MAT<T> Temp(A.rows, A.rows);
		EVD_Q_QR_HouseHolder(A, E, Q, Temp, ur, eps);
		delete[] ur;
		break;
	}
	case _QR_Givens: {
		MAT<T> Temp(A.rows, A.rows);
		EVD_Q_QR_Givens(A, E, Q, Temp, eps);
		break;
	}
	case _Jacobi: {
		for (int i = 0; i < E.rows * E.cols; i++) {
			E.data[i] = A.data[i];
		}
		EVD_Q_Jacobi(E, Q, eps);
		break;
	}
	case _LLT: {
		EVD_Q_LLT(A, E, Q, eps);
		break;
	}
	default:
		break;
	}
}
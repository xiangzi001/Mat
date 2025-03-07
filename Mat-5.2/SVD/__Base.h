
/*
template<typename T>
void SVD_Givens_left(MAT<T>& A, MAT<T>& U, int i, int j) {
	int k;
	T temp;
	double t( A[i][j] / A[i - 1][j]);
	double c ( 1 / sqrt(t * t + 1) );
	double s ( t * c );

	//double theta(atan2(A[i][j], A[i - 1][j]));
	//double c(cos(theta));
	//double s(sin(theta));

	int len (j + 3);
	if (A.cols < len){
		len = A.cols;
	}

	for (k = j; k < len; k++) {
		temp = c * A[i - 1][k] + s * A[i][k];
		A[i][k] = -s * A[i - 1][k] + c * A[i][k];
		A[i - 1][k] = temp;
	}
	for (k = 0; k < U.rows; k++) {
		temp = c * U[k][i - 1] + s * U[k][i];
		U[k][i] = -s * U[k][i - 1] + c * U[k][i];
		U[k][i - 1] = temp;
	}
}

template<typename T>
void SVD_Givens_right(MAT<T>& A, MAT<T>& V, int i, int j) {
	int k;
	T temp;
	double t( A[j][i] / A[j][i - 1]);
	double c(1 / sqrt(t * t + 1));
	double s(t * c);

	//double theta(atan2(A[j][i], A[j][i - 1]));
	//double c(cos(theta));
	//double s(sin(theta));

	int len (j + 3);
	if (A.rows < len) {
		len = A.rows;
	}

	for (k = j; k < len; k++) {
		temp = c * A[k][i - 1] + s * A[k][i];
		A[k][i] = -s * A[k][i - 1] + c * A[k][i];
		A[k][i - 1] = temp;
	}
	for (k = 0; k < V.rows; k++) {
		temp = c * V[k][i - 1] + s * V[k][i];
		V[k][i] = -s * V[k][i - 1] + c * V[k][i];
		V[k][i - 1] = temp;
	}
}
*/



/*
template <typename T>
void SVD_EVD_QR_full(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i, j;
	if (A.rows > A.cols) {
		MAT<T> E(A.rows,A.rows);

		T* w = new T[A.rows];
		//A*_T(A) = U*E*_T(U)

		times_A_BT(A, A, E);

		EVD_Q_QR_test(E, U, E.data, w, eps);

		for (i = 0; i < S.cols; ++i) {
			S[i][i] = sqrt(E.data[i]);
			w[i] = 1.0 / S[i][i];
		}

		//V*E = _T(A)*U
		times_AT_B(A, U, V);
		for (i = 0; i < V.rows; ++i) {
			for (j = 0; j < V.cols; ++j) {
				V[i][j] = V[i][j] * w[j];
			}
		}

		delete[] w;
	}
	else {
		MAT<T> E(A.cols, A.cols);
		T* w = new T[A.cols];
		//_T(A)*A = V*E*_T(V)

		times_AT_B(A, A, E);

		EVD_Q_QR_test(E, V, E.data, w, eps);

		for (int i = 0; i < A.rows; ++i) {
			S[i][i] = sqrt(E.data[i]);
			w[i] = 1.0 / S[i][i];
		}

		//U*E = A*V
		times(A, V, U);
		for (int i = 0; i < U.rows; ++i) {
			for (int j = 0; j < U.cols; ++j) {
				U[i][j] = U[i][j] * w[j];
			}
		}
		delete[] w;
	}
}
*/
template<typename T>
void __SVD_HouseHolder_left(MAT<T>& A, MAT<T>& U, T*& w, int& r, int len) {
	int i, j, k;
	T sum(0), cr, hr;
	for (k = len; k < A.rows; ++k)
		sum += A[k][r] * A[k][r];
	if (sum == 0) {
		return;
	}
	cr = sqrt(sum);
	if (A[len][r] < 0.0)cr = -cr;

	//定义w
	w[len] = A[len][r] + cr;
	for (i = len + 1; i < A.rows; ++i)
	{
		w[i] = A[i][r];
	}

	hr = 1 / (cr * (cr + A[len][r]));
	//定义A
	for (j = r; j < A.cols; ++j)
	{
		sum = 0;
		for (i = len; i < A.rows; ++i)
			sum += A[i][j] * w[i];
		sum = sum * hr;
		for (i = len; i < A.rows; ++i)
			A[i][j] = A[i][j] - w[i] * sum;
	}

	//定义U
	for (i = 0; i < U.rows; ++i)
	{
		sum = 0;
		for (j = len; j < U.rows; ++j)
			sum += U[i][j] * w[j];
		sum = sum * hr;
		for (j = len; j < U.cols; ++j)
		{
			U[i][j] = U[i][j] - w[j] * sum;
		}
	}
}
template<typename T>
void __SVD_HouseHolder_right(MAT<T>& A, MAT<T>& V, T*& w, int& r, int len) {
	int i, j, k;
	T sum(0), cr, hr;
	for (k = len; k < A.cols; ++k)
		sum += A[r][k] * A[r][k];
	if (sum == 0) {
		return;
	}
	cr = sqrt(sum);
	if (A[r][len] < 0)cr = -cr;

	//定义w
	w[len] = A[r][len] + cr;
	for (i = len + 1; i < A.cols; ++i)
	{
		w[i] = A[r][i];
	}

	hr = 1 / (cr * (cr + A[r][len]));
	//定义A
	for (j = r; j < A.rows; ++j)
	{
		sum = 0;
		for (i = len; i < A.cols; ++i)
			sum += A[j][i] * w[i];
		sum = sum * hr;
		for (i = len; i < A.cols; ++i)
			A[j][i] = A[j][i] - w[i] * sum;
	}
	//定义V
	for (i = 0; i < V.rows; ++i)
	{
		sum = 0;
		for (j = len; j < V.cols; ++j)
			sum += V[i][j] * w[j];
		sum = sum * hr;
		for (j = len; j < V.cols; ++j)
		{
			V[i][j] = V[i][j] - w[j] * sum;
		}
	}
}

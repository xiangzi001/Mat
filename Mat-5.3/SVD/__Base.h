
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

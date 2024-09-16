
template<typename T>
void __EVD_Q_HouseHolder(MAT<T>& Q, MAT<T>& R, T* w) {
	int i, j, k, r;
	T sum, cr, hr;
	for (r = 0; r < R.rows - 2; ++r) {
		sum = 0;
		for (k = r + 1; k < R.rows; ++k)
			sum += R[k][r] * R[k][r];
		if (sum == 0) {
			continue;
		}
		cr = sqrt(sum);
		if (R[r + 1][r] > 0.0)cr = -cr;
		hr = 1.0 / (cr * (cr - R[r + 1][r]));

		//定义w
		w[r + 1] = R[r + 1][r] - cr;
		for (i = r + 2; i < R.rows; ++i)
		{
			w[i] = R[i][r];
		}

		//定义Q
		for (i = 0; i < R.rows; ++i)
		{
			sum = 0;
			for (j = r + 1; j < R.rows; ++j)
				sum += Q[i][j] * w[j];
			sum *= hr;
			for (j = r + 1; j < R.cols; ++j)
			{
				Q[i][j] -= w[j] * sum;
			}
		}

		//定义R
		for (i = r; i < R.rows; ++i)
		{
			sum = 0;
			for (j = r + 1; j < R.rows; ++j)
				sum += R[i][j] * w[j];
			sum *= hr;
			for (j = r + 1; j < R.cols; ++j)
			{
				R[i][j] -= w[j] * sum;
			}
		}
		for (j = r; j < R.cols; ++j)
		{
			sum = 0;
			for (i = r + 1; i < R.rows; ++i)
				sum += R[i][j] * w[i];
			sum *= hr;
			for (i = r + 1; i < R.rows; ++i)
				R[i][j] -= w[i] * sum;
		}

	}
}

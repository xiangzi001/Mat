
template <typename T>
double Norm_one(MAT<T>& A) {
	double ret(0);
	double temp;
	int i;
	for (int j = 0; j < A.cols; ++j) {
		temp = 0;
		for (i = 0; i < A.rows; ++i) {
			temp += fabs(A[i][j]);
		}
		if (temp > ret)
			ret = temp;
	}

	return ret;
}

template <typename T>
double Norm_inf(MAT<T>& A) {
	double ret(0);
	double temp;
	int j;
	for (int i = 0; i < A.rows; ++i) {
		temp = 0;
		for (j = 0; j < A.cols; ++j) {
			temp += fabs(A[i][j]);
		}
		if (temp > ret)
			ret = temp;
	}

	return ret;
}

template <typename T>
T Norm_F(MAT<T>& A) {
	T ret = 0;
	for (int i = 0; i < A.rows*A.cols; ++i) {
		ret += A.data[i] * A.data[i];
	}

	return sqrt(ret);
}

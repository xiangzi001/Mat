
template <typename T>
double Norm_one(const MAT<T>& A) {
	double ret(0);
	double temp;
	int i;
	T* a;
	for (int j = 0; j < A.cols; ++j) {
		temp = 0;
		a = A.data + A.rows * j;
		for (i = 0; i < A.rows; ++i) {
			temp += fabs(a[i]);
		}
		if (temp > ret)
			ret = temp;
	}

	return ret;
}

template <typename T>
double Norm_inf(const MAT<T>& A) {
	double ret(0);
	double temp;
	int j;
	for (int i = 0; i < A.rows; ++i) {
		temp = 0;
		for (j = 0; j < A.cols; ++j) {
			temp += fabs(A.data[j * A.rows + i]);
		}
		if (temp > ret)
			ret = temp;
	}

	return ret;
}

template <typename T>
double Norm_F(const MAT<T>& A) {
	double ret = 0;
	for (int i = 0; i < A.rows * A.cols; ++i) {
		ret += A.data[i] * A.data[i];
	}

	return sqrt(ret);
}

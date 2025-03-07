
template <typename T>
bool Jacobi___SV(MAT<T>& A, int& i, int& j, T& eps) {
	int k = 0;
	T b = 0;
	T a = 0;

	for (k = 0; k < A.rows; ++k) {
		b = b - A[k][i] * A[k][i];
		a = a + A[k][i] * A[k][j];
		b = b + A[k][j] * A[k][j];
	}

	//T t = (-b + sqrt(b * b + 4 * a * a)) / (2 * a);
	//T c = 1 / sqrt(t * t + 1);
	//T s = t * c;

	double Angle = 0.5 * atan2(2 * a, b);
	T c = cos(Angle);
	T s = sin(Angle);

	for (k = 0; k < A.rows; ++k) {
		a = A[k][i] * c - A[k][j] * s;
		b = A[k][i] * s + A[k][j] * c;
		A[k][i] = a;
		A[k][j] = b;
	}

	if (s < eps) {
		return false;
	}
	else {
		return true;
	}
}

template <typename T>
void __SV_Jacobi(MAT<T>& A, MAT<T>& S, double eps) {
	int i;
	int j;
	int n, m;
	int s = S.rows * S.cols;
	for (i = 0; i < s; ++i) {
		S.data[i] = A.data[i];
	}

	if (A.rows > A.cols) {
		n = A.rows;
		m = A.cols;
	}
	else {
		n = A.cols;
		m = A.rows;
	}

	bool flag = true;
	s = 0;
	while (flag) {
		flag = false;
		for (i = s; i < m; ++i) {
			for (j = i + 1; j < n; ++j) {
				if (Jacobi___SV(S, j, i, eps)) {
					flag = true;
				}
			}
			if (!flag) {
				s++;
			}
		}
	}

	T temp;
	for (j = 0; j < S.rows; ++j) {
		temp = 0;
		for (i = 0; i < S.rows; ++i) {
			temp += S[i][j] * S[i][j];
		}
		temp = sqrt(temp);

		for (i = 0; i < S.rows; ++i) {
			S[i][j] = 0;
		}
		S[j][j] = temp;
	}

}

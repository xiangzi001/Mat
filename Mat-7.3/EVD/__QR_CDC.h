
template<typename T>
void __GramSchmidt(MAT<T>& Q) {
	int i, j;
	T len;

	for (j = 0; j < Q.cols; ++j) {
		for (i = 0; i < j; ++i) {
			len = __dot(Q[i], Q[j], Q.rows);
			__mixed2(Q[j], Q[i], -len, Q[j], Q.rows);
		}
		len = sqrt(__dot(Q[j], Q[j], Q.rows));
		__times(Q[j], 1.0 / len, Q[j], Q.rows);
	}
}

template <class T>
void quickSort(T* arr, int left, int right, int* pos) {
	if (left >= right) {
		return;
	}

	int i = left;
	int j = right;
	T Temp = arr[left];
	int temp = pos[left];
	while (i < j) {
		while (i < j && arr[j] >= Temp) {
			j--;
		}
		arr[i] = arr[j];
		pos[i] = pos[j];

		while (i < j && arr[i] <= Temp) {
			i++;
		}
		arr[j] = arr[i];
		pos[j] = pos[i];
	}
	arr[i] = Temp;
	pos[i] = temp;

	quickSort(arr, left, i - 1, pos);
	quickSort(arr, i + 1, right, pos);
}

template<typename T>
inline T __EVD_Q_QR_CDC_Solve_EigenValue(int n, T* z, T* d, T x) {
	T a = 0;
	for (int i = 0; i < n; ++i) {
		a += z[i] * z[i] / (d[i] - x);
	}
	a = a + 1;
	T b = 0;
	for (int i = 0; i < n; ++i) {
		b += z[i] * z[i] / ((d[i] - x) * (d[i] - x));
	}
	b = b;
	return a / b;
}
template<typename T>
inline T __EVD_Q_QR_CDC_Solve_EigenValue_Start(int n, T* z, T* d, int i) {
	if (i == n - 1) {
		T temp = 0;
		for (int k = 0; k < i; ++k) {
			temp += z[k] * z[k] / d[k];
		}
		return (1 + temp) / (z[i] * z[i]);
	}
	else {
		T a = z[i] * z[i];
		T temp = 0;
		for (int k = 0; k < i; ++k) {
			temp += z[k] * z[k] / d[k];
		}
		for (int k = i + 1; k < n; ++k) {
			temp += z[k] * z[k] / d[k];
		}
		T b = 1 + z[i] * z[i] / d[i + 1] + temp;
		T c = (1 + temp) / d[i + 1] - z[i + 1] * z[i + 1] / (d[i + 1] * d[i + 1]);
		return (b + sqrt(b * b - 4 * a * c)) / (2 * a);
	}
}

template<typename T>
void __EVD_Q_QR_CDC_Solve(T* w, T* v, MAT<T>& Q, MAT<T>& Temp, int* pos, double eps) {
	if (Q.rows <= 32) {
		__set0(Q.data, Q.rows * Q.cols);
		for (int i = 0; i < Q.rows; ++i) {
			Q[i][i] = 1;
		}

		__EVD_Q_Givens(w, v, Q, 0, Q.rows - 1, eps);
	}
	else {
		int s = Q.rows / 2;
		int t = Q.rows - s;

		T* u = Temp.data + t * t + s * s;
		T* z = u + Q.rows;
		T* d = z + Q.rows;

		MAT<T> Q1, Q2;
		Q1.rows = s;
		Q1.cols = s;
		Q1.data = Temp.data;
		Q2.rows = t;
		Q2.cols = t;
		Q2.data = Temp.data + s * s;

		MAT<T> Temp1, Temp2;
		Temp1.rows = s;
		Temp1.cols = s;
		Temp1.data = Q.data;
		Temp2.rows = t;
		Temp2.cols = t;
		Temp2.data = Q.data + s * s;

		T p = v[s - 1];
		v[s - 1] = 0;
		w[s - 1] -= p;
		w[s] -= p;

		__EVD_Q_QR_CDC_Solve(w, v, Q1, Temp1, pos, eps);
		__EVD_Q_QR_CDC_Solve(w + s, v + s, Q2, Temp2, pos, eps);
		
		int i, j, m = 0;
		T x, temp, shift = 0;

		__copy(d, w, Q.rows);
		__copy(v, w, Q.rows);
		__times(v, 1.0 / p, v, Q.rows);
		for (i = 0; i < s; ++i) {
			u[i] = Q1(s - 1, i);
		}
		for (i = 0; i < t; ++i) {
			u[i + s] = Q2(0, i);
		}

		for (i = 0; i < Q.rows; ++i) {
			pos[i] = i;
		}
		for (i = 0; i < Q.rows; ++i) {
			if (u[i] * u[i] < eps) {
				j = pos[i];
				pos[i] = pos[m];
				pos[m] = j;

				temp = v[i];
				v[i] = v[m];
				v[m] = temp;

				++m;
			}
		}
		quickSort(v, m, Q.rows - 1, pos);
		for (i = 0; i < Q.rows; ++i) {
			z[i] = u[pos[i]];
		}

		// EigenValues
		for (i = 0; i < m; ++i) {
			w[i] = (v[i] + shift) * p;
		}
		for (i = m; i < Q.rows; ++i) {
			shift += v[i];
			__add(v, -v[i], v, Q.rows);

			x = __EVD_Q_QR_CDC_Solve_EigenValue_Start(Q.rows, z, v, i);
			temp = __EVD_Q_QR_CDC_Solve_EigenValue(Q.rows, z, v, 1.0 / x);

			while (fabs(temp) > eps) {
				x += temp * x * x;
				temp = __EVD_Q_QR_CDC_Solve_EigenValue(Q.rows, z, v, 1.0 / x);
			}
			w[i] = (1.0 / x + shift) * p;
		}
		/*
		for (j = 0; j < Q.rows; ++j) {
			cout << w[j] << ", ";
		}
		cout << endl;
		//*/

		// EigenVectors
		__set0(Q.data, Q.rows * Q.cols);
		for (i = 0; i < m; ++i) {
			j = pos[i];
			if (j < s)
				__copy(Q[i], Q1[j], s);
			else
				__copy(Q[i] + s, Q2[j - s], t);
		}
		for (i = m; i < Q.rows; ++i) {
			for (j = 0; j < Q.rows; ++j) {
				v[j] = u[j] / (d[j] - w[i]);
			}
			p = sqrt(__dot(v, v, Q.rows));
			__times(v, 1.0 / p, v, Q.rows);
			for (j = 0; j < s; ++j) {
				__mixed2(Q[i], Q1[j], v[j], Q[i], s);
			}
			for (j = 0; j < t; ++j) {
				__mixed2(Q[i] + s, Q2[j], v[s + j], Q[i] + s, t);
			}
		}

		Q1.data = 0;
		Q2.data = 0;
		Temp1.data = 0;
		Temp2.data = 0;
	}
}


template<typename T>
void __EVD_Q_QR_CDC(MAT<T>& A, MAT<T>& Q, double eps) {
	int i, s;

	MAT<T> Temp(Q.rows, Q.cols);
	__set0(Temp.data, Temp.rows * Temp.cols);
	for (i = 0; i < Temp.rows; ++i) {
		Temp[i][i] = 1;
	}

	T* w = new T[A.rows];
	T* v = new T[A.cols];
	__EVD_Q_HouseHolder(Temp, A, w, v);

	s = A.rows - 1;
	for (i = 0; i < A.rows; ++i) {
		w[i] = A[i][i];
	}
	for (i = 0; i < s; ++i) {
		v[i] = A[i + 1][i];
	}
	v[A.rows - 1] = 0;

	int* pos = new int[A.cols];

	__EVD_Q_QR_CDC_Solve(w, v, A, Q, pos, eps);

	times_A_B(Temp, A, Q);
	__set0(A.data, A.rows * A.cols);
	for (i = 0; i < A.rows; ++i) {
		A[i][i] = w[i];
	}

	delete[] w;
	delete[] v;
	delete[] pos;
}

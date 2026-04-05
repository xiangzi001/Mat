
template<typename T>
void __GramSchmidt(MAT<T>& Q) {
	int i, j;
	T len;

	for (j = 0; j < Q.cols; ++j) {
		for (i = 0; i < j; ++i) {
			len = __dot(Q[i], Q[j], Q.rows);
			__mixed(Q[j], Q[i], -len, Q[j], Q.rows);
		}
		len = sqrt(__dot(Q[j], Q[j], Q.rows));
		__times(Q[j], (T)1 / len, Q[j], Q.rows);
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
inline T __EVD_Q_QR_CDC_Solve_EigenValue(int n, T* z, T* d, T x, int k) {
	T temp = 0;
	T a = 0;
	for (int i = 0; i < k; ++i) {
		a += z[i] / (d[i] - x);
	}
	for (int i = n - 1; i >= k; --i) {
		temp += z[i] / (d[i] - x);
	}
	a = a + temp + 1;

	temp = 0;
	T b = 0;
	for (int i = 0; i < k; ++i) {
		b += z[i] / ((d[i] - x) * (d[i] - x));
	}
	for (int i = n - 1; i >= k; --i) {
		temp += z[i] / ((d[i] - x) * (d[i] - x));
	}
	b = b + temp;
	return a / b;
}
template<typename T>
inline T __EVD_Q_QR_CDC_Solve_EigenValue_Start(int n, T* z, T* d, int i, T eps) {
	if (i == n - 1) {
		return 1;
	}
	else {
		T a = 0;
		T temp = 0;
		for (int k = 0; k < i; ++k) {
			temp += z[k] / d[k];
		}
		for (int k = n - 1; k > i;--k) {
			a += z[k] / d[k];
		}
		temp = temp + a;
		a = z[i];
		T b = 1 + z[i] / d[i + 1] + temp;

		if (a < eps) {
			if (b > 0) {
				return b / a;
			}
			else {
				T c = (1 + temp) / d[i + 1] - z[i + 1] / (d[i + 1] * d[i + 1]);
				return c / b;
			}
		}
		else {
			T c = (1 + temp) / d[i + 1] - z[i + 1] / (d[i + 1] * d[i + 1]);
			return (b + sqrt(b * b - 4 * a * c)) / (2 * a);
		}
	}
}

template<typename T>
inline void __EVD_Q_QR_CDC_u(int n, T* w, T* d, T* u, T p, T eps) {
	T temp;
	int i, j;
	for (i = 0; i < n; ++i) {
		temp = p / (w[i] - d[i]);

		for (j = 0; j < i; ++j)
			temp *= (d[j] - d[i]) / (w[j] - d[i]);
		for (j = i + 1; j < n; ++j)
			temp *= (d[j] - d[i]) / (w[j] - d[i]);

		if (u[i] < 0)
			u[i] = -(T)1 / sqrt(temp);
		else
			u[i] = (T)1 / sqrt(temp);
	}
}

template<typename T>
void __EVD_Q_QR_CDC_Solve(T* w, T* v, MAT<T>& Q, MAT<T>& Temp, int* pos, T eps) {
	if (Q.rows <= 500) {
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
		
		int i, j;
		T x, temp, shift = 0;

		__copy(d, w, Q.rows);
		__copy(v, w, Q.rows);
		for (i = 0; i < s; ++i) {
			u[i] = Q1(s - 1, i);
		}
		for (i = 0; i < t; ++i) {
			u[i + s] = Q2(0, i);
		}

		for (i = 0; i < Q.rows; ++i) {
			pos[i] = i;
		}

		__times(v, (T)1 / p, v, Q.rows);
		quickSort(v, 0, Q.rows - 1, pos);
		for (i = 0; i < Q.rows; ++i) {
			z[i] = u[pos[i]];
		}
		for (i = 0; i < Q.rows; ++i) {
			z[i] = z[i] * z[i];
		}

		// EigenValues
		for (i = 0; i < Q.rows; ++i) {
			shift += v[i];
			__add(v, -v[i], v, Q.rows);

			x = __EVD_Q_QR_CDC_Solve_EigenValue_Start(Q.rows, z, v, i, eps);
			temp = __EVD_Q_QR_CDC_Solve_EigenValue(Q.rows, z, v, (T)1 / x, i);

			while (fabs(temp) > eps) {
				x += temp * x * x;
				temp = __EVD_Q_QR_CDC_Solve_EigenValue(Q.rows, z, v, (T)1 / x, i);
			}
			w[i] = ((T)1 / x + shift) * p;
		}

		/*
		for (i = 0; i < Q.rows; ++i) {
			std::cout << w[i] << std::endl;
		}
		std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
		std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
		std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
		std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
		std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
		//*/
		// EigenVectors
		__set0(Q.data, Q.rows * Q.cols);
		__EVD_Q_QR_CDC_u(Q.rows, w, d, u, p, eps);
		/*
		std::cout << p << std::endl;
		std::cout << m << std::endl;
		std::cout << "u1 : ";
		for (i = 0; i < Q.rows; ++i) {
			std::cout << u[i] << ", ";
		}
		std::cout << std::endl;
		std::cout << std::endl;
		__EVD_Q_QR_CDC_u(Q.rows, w, d, u, p, eps);
		std::cout << "u2 : ";
		for (i = 0; i < Q.rows; ++i) {
			std::cout << u[i] << ", ";
		}
		std::cout << std::endl;
		std::cout << std::endl;
		//*/

		for (i = 0; i < Q.rows; ++i) {
			for (j = 0; j < Q.rows; ++j) {
				v[j] = u[j] / (d[j] - w[i]);
			}

			p = (T)1 / eps;
			for (j = 0; j < Q.rows; ++j) {
				if (v[j] * v[j] > p) {
					break;
				}
			}
			/*
			for (i = 0; i < Q.rows; ++i) {
				std::cout << v[i] << std::endl;
			}
			std::cout << std::endl << std::endl << std::endl << std::endl << std::endl;
			std::cout << std::endl << std::endl << std::endl << std::endl << std::endl;
			std::cout << std::endl << std::endl << std::endl << std::endl << std::endl;
			std::cout << std::endl << std::endl << std::endl << std::endl << std::endl;
			std::cout << std::endl << std::endl << std::endl << std::endl << std::endl;
			*/
			if (j == Q.rows) {
				p = sqrt(__dot(v, v, Q.rows));
				__times(v, (T)1 / p, v, Q.rows);
				for (j = 0; j < s; ++j) {
					__mixed(Q[i], Q1[j], v[j], Q[i], s);
				}
				for (j = 0; j < t; ++j) {
					__mixed(Q[i] + s, Q2[j], v[s + j], Q[i] + s, t);
				}
			}
			else {
				if (j < s)
					__copy(Q[i], Q1[j], s);
				else
					__copy(Q[i] + s, Q2[j - s], t);
			}
		}

		Q1.data = 0;
		Q2.data = 0;
		Temp1.data = 0;
		Temp2.data = 0;
	}
}


template<typename T>
void __EVD_Q_QR_CDC(MAT<T>& A, MAT<T>& Q, T eps) {
	int i, s;

	MAT<T> Temp(Q.rows, Q.cols);
	__set0(Temp.data, Temp.rows * Temp.cols);
	for (i = 0; i < Temp.rows; ++i) {
		Temp[i][i] = 1;
	}

	T* w = new T[A.rows];
	T* v = new T[A.cols];
	__EVD_Q_HouseHolder(Temp, A, w);

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

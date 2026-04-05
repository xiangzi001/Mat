
template <class T>
void __QuickSortBigToSmall(T* arr, int left, int right) {
	if (left >= right) {
		return;
	}
	int i = left;
	int j = right;
	T pivot = arr[left];
	while (i < j) {
		while (i < j && arr[j] <= pivot) {
			j--;
		}
		arr[i] = arr[j];
		while (i < j && arr[i] >= pivot) {
			i++;
		}
		arr[j] = arr[i];
	}
	arr[i] = pivot;
	__QuickSortBigToSmall(arr, left, i - 1);
	__QuickSortBigToSmall(arr, i + 1, right);
}

template<typename T>
T __EigenSolve_DC_Function1(int n, T* u, T* d, T x) {
	T ret = x;
	for (int i = 1; i < n; ++i) {
		ret += u[i] * u[i] / (d[i] - x);
	}
	return ret;
}

template<typename T>
T __EigenSolve_DC(int n, T ai, T* u, T* d, T x, T y, T eps) {

	eps = n * 0.5 * fabs(x + y) * eps;

	while (x - y > eps) {
		if (__EigenSolve_DC_Function1(n, u, d, (x + y) / 2) - ai > 0) {
			x = (x + y) / 2;
		}
		else {
			y = (x + y) / 2;
		}
	}

	return (x + y) / 2;
}

template<typename T>
bool __EVD_Q_Divide(T* w, T* v, MAT<T>& Q, MAT<T>& Temp, T* u, T* x, int* pos, int n, T eps) {
	if (n <= 8) {
		int i;
		__set0(Q.data, Q.rows * Q.cols);
		for (i = 0; i < Q.rows; ++i) {
			Q[i][i] = 1;
		}

		__EVD_Q_Givens(w, v, Q, 0, n - 1, eps);

		return false;
	}
	else {
		int s = (n - 1) / 2;
		int t = n - (s + 1);

		MAT<T> Q1, Q2;
		Q1.rows = s;
		Q1.cols = s;
		Q1.data = Q.data;
		Q2.rows = t;
		Q2.cols = t;
		Q2.data = Q.data + s * s;

		MAT<T> Temp1, Temp2;
		Temp1.rows = s;
		Temp1.cols = s;
		Temp1.data = Temp.data;
		Temp2.rows = t;
		Temp2.cols = t;
		Temp2.data = Temp.data + s * s;

		T a = w[s];
		T b1 = v[s - 1];
		T b2 = v[s];
		v[s - 1] = 0;

		bool flag1, flag2;
		flag1 = __EVD_Q_Divide(w, v, Q1, Temp1, u, x, pos, s, eps);
		flag2 = __EVD_Q_Divide(w + s + 1, v + s + 1, Q2, Temp2, u, x, pos, t, eps);
		v[0] = 0;

		int i, j, k;
		// sort
		///*
		if (flag1 && flag2) {
			i = 0;
			j = 0;
			pos = pos + 1;
			v = v + 1;
			while (i < s && j < t) {
				if (w[i] > w[s + 1 + j]) {
					pos[i] = i + j + 1;
					v[i + j] = w[i];
					++i;
				}
				else {
					pos[s + j] = i + j + 1;
					v[i + j] = w[s + 1 + j];
					++j;
				}
			}
			if (i == s) {
				v = v - 1;
				j = s + 1 + j;
				for (; j < n; ++j) {
					pos[j - 1] = j;
					v[j] = w[j];
				}
			}
			else {
				for (; i < s; ++i) {
					pos[i + t] = i + t + 1;
					v[i + t] = w[i];
				}
				v = v - 1;
			}
		}
		else {
			for (i = s - 1; i >= 0; --i) {
				w[i + 1] = w[i];
			}
			for (i = 1; i < n; ++i) {
				v[i] = w[i];
			}
			__QuickSortBigToSmall(v, 1, n - 1);
			for (i = 1; i < n; ++i) {
				j = 1;
				while (1) {
					if (w[i] == v[j]) {
						pos[i] = j;
						break;
					}
					else {
						++j;
					}
				}
			}
			pos = pos + 1;
		}
		//*/

		// Q0
		if (flag1) {
			__set0(Temp1.data, s * s);
		}
		if (flag2) {
			__set0(Temp2.data, t * t);
		}
		//__set0(Temp.data, n * n);
		for (i = 0; i < s; ++i) {
			for (j = 0; j < s; ++j) {
				Temp[i][pos[j]] = Q1[i][j];
			}
		}
		Temp[s][0] = 1;
		for (i = 0; i < t; ++i) {
			for (j = 0; j < t; ++j) {
				Temp[i + s + 1][pos[j + s]] = Q2[i][j];
			}
		}

		// u
		T* q = Q1[s - 1];
		for (i = 0; i < s; ++i) {
			u[pos[i]] = b1 * q[i];
		}
		q = Q2[0];
		for (i = 0; i < t; ++i) {
			u[pos[i + s]] = b2 * q[i];
		}

		// EigenValues
		b1 = 0;
		for (i = 1; i < n; ++i) {
			b1 += u[i] * u[i];
		}
		b1 = sqrt(b1);
		for (k = 0; k < n; ++k) {
			T max, min;
			if (k == 0) {
				b2 = v[1];
				max = fabs(b2 - a) + b1;
				min = 0;
			}
			else if (k == n - 1) {
				b2 = v[n - 1];
				max = 0;
				min = -(fabs(b2 - a) + b1);
			}
			else {
				/*
				b2 = 0;
				max = v[k];
				min = v[k + 1];
				*/
				if (__EigenSolve_DC_Function1(n, u, v, (v[k] + v[k + 1]) / 2) - a > 0) {
					b2 = v[k + 1];
					max = (v[k] - v[k + 1]) / 2;
					min = 0;
				}
				else {
					b2 = v[k];
					max = 0;
					min = (-v[k] + v[k + 1]) / 2;
				}
			}

			for (i = 1; i < n; ++i) {
				v[i] -= b2;
			}
			w[k] = __EigenSolve_DC(n, a - b2, u, v, max, min, eps) + b2;
			for (i = 1; i < n; ++i) {
				v[i] += b2;
			}
		}

		for (i = 1; i < n; ++i) {
			b1 = 1;
			for (k = 0; k < n; ++k) {
				b1 *= (v[i] - w[k]);
			}
			b2 = 1;
			for (k = 1; k < i; ++k) {
				b2 *= (v[i] - v[k]);
			}
			for (k = i + 1; k < n; ++k) {
				b2 *= (v[i] - v[k]);
			}
			if (u[i] < 0) {
				u[i] = -sqrt(-b1 / b2);
			}
			else {
				u[i] = sqrt(-b1 / b2);
			}
		}

		// EigenVec
		for (k = 0; k < n; ++k) {
			x[0] = -1;
			for (i = 1; i < n; ++i) {
				x[i] = u[i] / (v[i] - w[k]);
			}
			a = 1;
			for (i = 1; i < n; ++i) {
				a += x[i] * x[i];
			}
			a = 1 / sqrt(a);
			for (i = 0; i < n; ++i) {
				x[i] *= a;
			}

			// Q
			for (i = 0; i < n; ++i) {
				a = 0;
				for (j = 0; j < n; ++j) {
					a += Temp[i][j] * x[j];
				}
				Q[i][k] = a;
			}
		}

		Q1.data = 0;
		Q2.data = 0;
		Temp1.data = 0;
		Temp2.data = 0;
		return true;
	}
}

template<typename T>
void __EVD_Q_QR_DC(MAT<T>& A, MAT<T>& Q, T eps) {
	int i, s;

	MAT<T> Temp(Q.rows, Q.cols);
	__set0(Temp.data, Temp.rows * Temp.cols);
	for (i = 0; i < Temp.rows; ++i) {
		Temp[i][i] = 1;
	}

	T* w = new T[A.rows + 64 / sizeof(T) - 1];
	T* v = new T[A.cols + 64 / sizeof(T) - 1];
	__EVD_Q_HouseHolder(Temp, A, w, v);

	s = A.rows - 1;
	for (i = 0; i < A.rows; ++i) {
		w[i] = A[i][i];
	}
	for (i = 0; i < s; ++i) {
		v[i] = A[i + 1][i];
	}
	v[A.rows - 1] = 0;

	s = (A.cols - 1) / (64 / sizeof(T)) + 1;
	T* u = new T[A.cols];
	T* x = new T[s * (64 / sizeof(T))];
	int* pos = new int[A.cols];

	__set0(Q.data, Q.rows * Q.cols);

	__EVD_Q_Divide(w, v, A, Q, u, x, pos, A.cols, eps);

	//times_A_B(Temp, A, Q);
	__set0(A.data, A.rows * A.cols);
	for (i = 0; i < A.rows; ++i) {
		A[i][i] = w[i];
	}

	delete[] w;
	delete[] v;
	delete[] u;
	delete[] x;
	delete[] pos;
}

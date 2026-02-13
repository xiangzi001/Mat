

template<typename T>
void Solve_Jacobi(MAT<T>& A, T* b, T*& x, double eps) {
	int len(A.cols);
	int i, j;
	T* _x = new T[len];
	T* temp;
	T sum;

	bool run = true;
	while (run) {
		run = false;
		for (i = 0; i < len; ++i) {
			sum = 0;
			for (j = 0; j < i; ++j) {
				sum += A[i][j] * x[j];
			}
			for (j = i+1; j < len; ++j) {
				sum += A[i][j] * x[j];
			}
			_x[i] = (b[i] - sum) / A[i][i];
			if (fabs(x[i] - _x[i]) > eps){
				run = true;
			}
		}
		temp = x;
		x = _x;
		_x = temp;
	}

	delete[] _x;
}

template<typename T>
void Solve_GS(MAT<T>& A, T* b, T* x, double eps) {
	int len(A.cols);
	int i, j;
	T sum;

	for (j = 0; j < len; ++j) {
		x[j] = 0;
	}
	bool run = true;
	while (run) {
		run = false;
		for (i = 0; i < len; ++i) {
			sum = 0;
			for (j = 0; j < i; ++j) {
				sum += A[i][j] * x[j];
			}
			for (j = i + 1; j < len; ++j) {
				sum += A[i][j] * x[j];
			}
			sum = (b[i] - sum) / A[i][i];
			if (fabs(x[i] - sum) > eps) {
				run = true;
			}
			x[i] = sum;
		}
	}
}

template<typename T>
void Solve_SOR(MAT<T>& A, T* b, T* x, double eps, T w = 0.1) {
	int len(A.cols);
	int i, j, k;
	T sum;

	for (j = 0; j < len; ++j) {
		x[j] = 0;
	}

	T new_eps = 1000;
	while (new_eps > eps) {
		new_eps = 0;
		for (i = 0; i < len; ++i) {
			sum = 0;
			for (j = 0; j < i; ++j) {
				sum += A[i][j] * x[j];
			}
			for (j = i + 1; j < len; ++j) {
				sum += A[i][j] * x[j];
			}
			sum = (1 - w) * x[i] + (b[i] - sum) * w / A[i][i];
			new_eps += fabs(x[i] - sum);
			x[i] = sum;
		}
	}
}

template<typename T>
void Solve_Krylov_SD(MAT<T>& A, T* b, T* x, double eps) {
	int i, j;
	//int k = 0;
	T sum, t;
	T d = eps + 1;

	T* r = new T[A.rows];

	for (i = 0; i < A.rows; ++i) {
		x[i] = 0;
	}
	for (i = 0; i < A.rows; ++i) {
		r[i] = b[i];
	}

	while (d > eps) {
		//++k;

		t = 0;
		for (i = 0; i < A.rows; ++i) {
			sum = 0;
			for (j = 0; j < A.cols; ++j) {
				sum += A[i][j] * r[j];
			}
			t += sum * r[i];
		}
		sum = 0;
		for (i = 0; i < A.rows; ++i) {
			sum += r[i] * r[i];
		}
		t = sum / t;

		for (i = 0; i < A.rows; ++i) {
			x[i] += t * r[i];
		}

		d = 0;
		for (i = 0; i < A.rows; ++i) {
			sum = b[i];
			for (j = 0; j < A.cols; ++j) {
				sum -= A[i][j] * x[j];
			}
			r[i] = sum;
			sum = fabs(sum);
			if (sum > d) {
				d = sum;
			}
		}
	}
	delete[] r;
}

template<typename T>
void Solve_Krylov_CG(MAT<T>& A, T* b, T* x, double eps) {
	int i, j, k;
	T sum, t, r0;
	T d = 1;

	T* r = new T[A.rows];
	T* p = new T[A.rows];

	for (i = 0; i < A.rows; ++i) {
		x[i] = 0;
	}
	for (i = 0; i < A.rows; ++i) {
		r[i] = b[i];
	}

	//int n = 0;
	while (d > eps) {

		for (i = 0; i < A.rows; ++i) {
			p[i] = r[i];
		}
		r0 = 0;
		for (i = 0; i < A.rows; ++i) {
			r0 += r[i] * r[i];
		}
		//++n;

		for (k = 0; k < A.cols; ++k) {
			//while( r0 > eps ){
			t = 0;
			for (i = 0; i < A.rows; ++i) {
				sum = 0;
				for (j = 0; j < A.cols; ++j) {
					sum += A[i][j] * p[j];
				}
				t += sum * p[i];
			}
			t = r0 / t;

			for (i = 0; i < A.rows; ++i) {
				x[i] += t * r[i];
			}

			for (i = 0; i < A.rows; ++i) {
				sum = 0;
				for (j = 0; j < A.cols; ++j) {
					sum += A[i][j] * p[j];
				}
				r[i] -= t * sum;
			}

			t = r0;

			r0 = 0;
			for (i = 0; i < A.rows; ++i) {
				r0 += r[i] * r[i];
			}
			t = r0 / t;

			for (i = 0; i < A.rows; ++i) {
				p[i] = r[i] + t * p[i];
			}
		}

		d = 0;
		for (i = 0; i < A.rows; ++i) {
			sum = b[i];
			for (j = 0; j < A.cols; ++j) {
				sum -= A[i][j] * x[j];
			}
			r[i] = sum;
			sum = fabs(sum);
			if (sum > d) {
				d = sum;
			}
		}
	}
	delete[] r;
	delete[] p;
}
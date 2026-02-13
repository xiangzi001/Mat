
///*
template<typename T>
void __Solve_direct2(MAT<T>& A, MAT<T>& B) {
	int x, y, i, j;
	double now, big;
	T temp;

	for (y = 0; y < A.rows; ++y) {
		j = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.cols; ++i) {
			now = fabs(A[i][y]);
			if (now > big) {
				j = i;
				big = now;
			}
		}
		if (j != y) {
			__swap(A[j] + y, A[y] + y, A.rows - y);
			__swap(B[j], B[y], B.rows);
		}

		temp = 1 / A[y][y];
		__times(A[y] + y, temp, A[y] + y, A.rows - y);
		__times(B[y], temp, B[y], B.rows);

		for (x = y + 1; x < A.cols; ++x) {
			temp = -A[x][y];
			__mixed3(A[x] + y, A[y] + y, temp, A[x] + y, A.rows - y);
			__mixed3(B[x], B[y], temp, B[x], B.rows);
		}
	}
	for (y = A.rows - 1; y >= 0; --y) {
		for (x = 0; x < y; ++x) {
			__mixed3(B[x], B[y], -A[x][y], B[x], B.rows);
		}
	}
}

template<typename T>
void __Solve_direct(MAT<T>& A, MAT<T>& B) {
	int r, i, j;
	double now, big;
	T temp, temp2;

	for (r = 0; r < A.rows; ++r) {
		j = r;
		big = fabs(A(r, r));
		for (i = r + 1; i < A.cols; ++i) {
			now = fabs(A(i, r));
			if (now > big) {
				j = i;
				big = now;
			}
		}

		if (j == r) {
			temp = 1.0 / A(r, r);
			for (i = r; i < A.cols; ++i) {
				A(r, i) *= temp;
			}
			for (i = 0; i < B.cols; ++i) {
				B(r, i) *= temp;
			}
		}
		else {
			temp = 1.0 / A(j, r);
			for (i = r; i < A.cols; ++i) {
				temp2 = A(r, i);
				A(r, i) = A(j, i) * temp;
				A(j, i) = temp2;
			}
			for (i = 0; i < B.cols; ++i) {
				temp2 = B(r, i);
				B(r, i) = B(j, i) * temp;
				B(j, i) = temp2;
			}
		}
		for (i = r + 1; i < A.cols; ++i) {
			__mixed3(A[i] + r + 1, A[r] + r + 1, -A(r, i), A[i] + r + 1, A.rows - r - 1);
		}
		for (i = 0; i < B.cols; ++i) {
			__mixed3(B[i] + r + 1, A[r] + r + 1, -B(r, i), B[i] + r + 1, B.rows - r - 1);
		}
	}
	for (r = 0; r < B.cols; ++r) {
		for (i = B.rows - 1; i > 0; --i) {
			__mixed3(B[r], A[i], -B(i, r), B[r], i);
		}
	}
}

template<typename T>
T __Det(MAT<T>& A) {
	int x, y, i, pos;
	T big, now;
	T r = 1, temp;
	for (y = 0; y < A.rows; ++y) {
		pos = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.cols; ++i) {
			now = fabs(A[i][y]);
			if (now > big) {
				pos = i;
				big = now;
			}
		}
		if (pos != y) {
			__swap(A[pos] + y, A[y] + y, A.rows - y);
			r = -r;
		}
		for (x = y + 1; x < A.cols; ++x) {
			temp = -A[x][y] / A[y][y];
			__mixed3(A[x] + y, A[y] + y, temp, A[x] + y, A.rows - y);
		}
	}
	for (x = 0; x < A.cols; ++x)
		r = r * A[x][x];

	return r;
}
//*/

template<typename T>
T Det(MAT<T> n) {
	return __Det(n);
}

//X*A=B
template<typename T>
MAT<T> Solve_direct(MAT<T> A, MAT<T> B) {
	__Solve_direct(A, B);
	return B;
}

template<typename T>
MAT<T> Inv(MAT<T> A) {
	MAT<T> B;
	B.iden(A.cols);

	__Solve_direct2(A, B);

	return B;
}

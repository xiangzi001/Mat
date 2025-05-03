

template<typename T>
void __Solve_direct(MAT<T>& A, MAT<T>& B) {
	int x, y, i, j;
	double now, big;
	T temp;

	for (y = 0; y < A.cols; ++y) {
		j = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.rows; ++i) {
			now = fabs(A[i][y]);
			if (now > big) {
				j = i;
				big = now;
			}
		}
		if (j != y) {
			for (i = y; i < A.cols; ++i) {
				temp = A[j][i];
				A[j][i] = A[y][i];
				A[y][i] = temp;
			}
			for (i = 0; i < B.cols; ++i) {
				temp = B[j][i];
				B[j][i] = B[y][i];
				B[y][i] = temp;
			}
		}
		temp = 1 / A[y][y];
		A[y][y] = 1;
		for (j = y + 1; j < A.cols; ++j)
			A[y][j] *= temp;
		for (j = 0; j < B.cols; ++j)
			B[y][j] *= temp;

		for (x = y + 1; x < A.rows; ++x) {
			temp = -A[x][y];
			for (j = y; j < A.rows; ++j)
				A[x][j] += A[y][j] * temp;
			for (j = 0; j < B.cols; ++j)
				B[x][j] += B[y][j] * temp;
		}
	}
	for (y = A.cols - 1; y >= 0; --y) {
		for (x = 0; x < y; ++x) {
			for (j = 0; j < B.cols; ++j) {
				B[x][j] -= B[y][j] * A[x][y];
			}
		}
	}
}

void __Solve_direct(MAT<double>& A, MAT<double>& B) {
	int x, y, i, j;
	int s = A.cols % 8;
	double t, big;
	int n1 = A.cols / 8;
	int m1 = s;
	const __mmask8 mask1 = (1 << m1) - 1;
	const int n2 = B.cols / 8;
	const int m2 = B.cols - n2 * 8;
	const __mmask8 mask2 = (1 << m2) - 1;

	__m512d copy;
	__m512d temp;
	__m512d* temp1;
	__m512d* temp2;

	for (y = 0; y < s; ++y) {
		j = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.rows; ++i) {
			t = fabs(A[i][y]);
			if (t > big) {
				j = i;
				big = t;
			}
		}
		if (j != y) {
			temp = _mm512_set1_pd(1.0 / A[j][y]);

			copy = _mm512_mul_pd(temp, *(__m512d*)(A[j]));
			*(__m512d*)(A[j]) = _mm512_mask_blend_pd(mask1, *(__m512d*)(A[j]), *(__m512d*)(A[y]));
			*(__m512d*)(A[y]) = _mm512_mask_blend_pd(mask1, *(__m512d*)(A[y]), copy);

			temp1 = (__m512d*)(A[j] + m1);
			temp2 = (__m512d*)(A[y] + m1);
			for (i = 0; i < n1; ++i) {
				copy = _mm512_mul_pd(temp, temp1[i]);
				temp1[i] = temp2[i];
				temp2[i] = copy;
			}

			copy = _mm512_mul_pd(temp, *(__m512d*)(B[j]));
			*(__m512d*)(B[j]) = _mm512_mask_blend_pd(mask2, *(__m512d*)(B[j]), *(__m512d*)(B[y]));
			*(__m512d*)(B[y]) = _mm512_mask_blend_pd(mask2, *(__m512d*)(B[y]), copy);

			temp1 = (__m512d*)(B[j] + m2);
			temp2 = (__m512d*)(B[y] + m2);
			for (i = 0; i < n2; ++i) {
				copy = _mm512_mul_pd(temp, temp1[i]);
				temp1[i] = temp2[i];
				temp2[i] = copy;
			}
		}
		else {
			temp = _mm512_set1_pd(1.0 / A[y][y]);
			*(__m512d*)(B[y]) = _mm512_mul_pd(_mm512_mask_blend_pd(mask2, _mm512_set1_pd(1.0), temp), *(__m512d*)(B[y]));
			temp2 = (__m512d*)(B[y] + m2);
			for (j = 0; j < n2; ++j)
				temp2[j] = _mm512_mul_pd(temp, temp2[j]);
			*(__m512d*)(A[y]) = _mm512_mul_pd(_mm512_mask_blend_pd(mask1, _mm512_set1_pd(1.0), temp), *(__m512d*)(A[y]));
			temp1 = (__m512d*)(A[y] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_mul_pd(temp, temp1[j]);
		}

		temp2 = (__m512d*)(B[y] + m2);
		for (x = y + 1; x < A.rows; ++x) {
			temp = _mm512_set1_pd(-A[x][y]);
			*(__m512d*)(B[x]) = _mm512_add_pd(*(__m512d*)(B[x]), _mm512_mul_pd(_mm512_mask_blend_pd(mask2, _mm512_setzero_pd(), temp), *(__m512d*)(B[y])));
			temp1 = (__m512d*)(B[x] + m2);
			for (j = 0; j < n2; ++j)
				temp1[j] = _mm512_add_pd(temp1[j], _mm512_mul_pd(temp, temp2[j]));
		}
		temp2 = (__m512d*)(A[y] + m1);
		for (x = y + 1; x < A.rows; ++x) {
			temp = _mm512_set1_pd(-A[x][y]);
			*(__m512d*)(A[x]) = _mm512_add_pd(*(__m512d*)(A[x]), _mm512_mul_pd(_mm512_mask_blend_pd(mask1, _mm512_setzero_pd(), temp), *(__m512d*)(A[y])));
			temp1 = (__m512d*)(A[x] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_add_pd(temp1[j], _mm512_mul_pd(temp, temp2[j]));
		}
	}
	for (y = s; y < A.cols - 1; ++y) {
		n1 = (A.cols - y - 1) / 8 + 1;
		m1 = A.cols - 8 * n1;
		j = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.rows; ++i) {
			t = fabs(A[i][y]);
			if (t > big) {
				j = i;
				big = t;
			}
		}
		if (j != y) {
			temp = _mm512_set1_pd(1.0 / A[j][y]);

			temp1 = (__m512d*)(A[j] + m1);
			temp2 = (__m512d*)(A[y] + m1);
			for (i = 0; i < n1; ++i) {
				copy = _mm512_mul_pd(temp, temp1[i]);
				temp1[i] = temp2[i];
				temp2[i] = copy;
			}

			copy = _mm512_mul_pd(temp, *(__m512d*)(B[j]));
			*(__m512d*)(B[j]) = _mm512_mask_blend_pd(mask2, *(__m512d*)(B[j]), *(__m512d*)(B[y]));
			*(__m512d*)(B[y]) = _mm512_mask_blend_pd(mask2, *(__m512d*)(B[y]), copy);

			temp1 = (__m512d*)(B[j] + m2);
			temp2 = (__m512d*)(B[y] + m2);
			for (i = 0; i < n2; ++i) {
				copy = _mm512_mul_pd(temp, temp1[i]);
				temp1[i] = temp2[i];
				temp2[i] = copy;
			}
		}
		else {
			temp = _mm512_set1_pd(1.0 / A[y][y]);

			*(__m512d*)(B[y]) = _mm512_mul_pd(_mm512_mask_blend_pd(mask2, _mm512_set1_pd(1.0), temp), *(__m512d*)(B[y]));
			temp2 = (__m512d*)(B[y] + m2);
			for (j = 0; j < n2; ++j)
				temp2[j] = _mm512_mul_pd(temp, temp2[j]);
			temp1 = (__m512d*)(A[y] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_mul_pd(temp, temp1[j]);
		}

		temp2 = (__m512d*)(B[y] + m2);
		for (x = y + 1; x < A.rows; ++x) {
			temp = _mm512_set1_pd(-A[x][y]);
			*(__m512d*)(B[x]) = _mm512_add_pd(*(__m512d*)(B[x]), _mm512_mul_pd(_mm512_mask_blend_pd(mask2, _mm512_setzero_pd(), temp), *(__m512d*)(B[y])));
			temp1 = (__m512d*)(B[x] + m2);
			for (j = 0; j < n2; ++j)
				temp1[j] = _mm512_add_pd(temp1[j], _mm512_mul_pd(temp, temp2[j]));
		}
		temp2 = (__m512d*)(A[y] + m1);
		for (x = y + 1; x < A.rows; ++x) {
			temp = _mm512_set1_pd(-A[x][y]);
			temp1 = (__m512d*)(A[x] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_add_pd(temp1[j], _mm512_mul_pd(temp, temp2[j]));
		}
	}

	y = A.cols - 1;
	temp = _mm512_set1_pd(1.0 / A[y][y]);
	*(__m512d*)(B[y]) = _mm512_mul_pd(_mm512_mask_blend_pd(mask2, _mm512_set1_pd(1.0), temp), *(__m512d*)(B[y]));
	temp2 = (__m512d*)(B[y] + m2);
	for (j = 0; j < n2; ++j)
		temp2[j] = _mm512_mul_pd(temp, temp2[j]);

	for (y = A.cols - 1; y >= 0; --y) {
		temp2 = (__m512d*)(B[y] + m2);
		for (x = 0; x < y; ++x) {
			temp = _mm512_set1_pd(-A[x][y]);
			*(__m512d*)(B[x]) = _mm512_add_pd(*(__m512d*)(B[x]), _mm512_mul_pd(_mm512_mask_blend_pd(mask2, _mm512_setzero_pd(), temp), *(__m512d*)(B[y])));
			temp1 = (__m512d*)(B[x] + m2);
			for (j = 0; j < n2; ++j)
				temp1[j] = _mm512_add_pd(temp1[j], _mm512_mul_pd(temp, temp2[j]));
		}
	}
}

void __Solve_direct(MAT<float>& A, MAT<float>& B) {
	int x, y, i, j;
	int s = A.cols % 16;
	float t, big;
	int n1 = A.cols / 16;
	int m1 = s;
	const __mmask16 mask1 = (1 << m1) - 1;
	const int n2 = B.cols / 16;
	const int m2 = B.cols - n2 * 16;
	const __mmask16 mask2 = (1 << m2) - 1;

	__m512 copy;
	__m512 temp;
	__m512* temp1;
	__m512* temp2;

	for (y = 0; y < s; ++y) {
		j = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.rows; ++i) {
			t = fabs(A[i][y]);
			if (t > big) {
				j = i;
				big = t;
			}
		}
		if (j != y) {
			temp = _mm512_set1_ps(1.0 / A[j][y]);

			copy = _mm512_mul_ps(temp, *(__m512*)(A[j]));
			*(__m512*)(A[j]) = _mm512_mask_blend_ps(mask1, *(__m512*)(A[j]), *(__m512*)(A[y]));
			*(__m512*)(A[y]) = _mm512_mask_blend_ps(mask1, *(__m512*)(A[y]), copy);

			temp1 = (__m512*)(A[j] + m1);
			temp2 = (__m512*)(A[y] + m1);
			for (i = 0; i < n1; ++i) {
				copy = _mm512_mul_ps(temp, temp1[i]);
				temp1[i] = temp2[i];
				temp2[i] = copy;
			}

			copy = _mm512_mul_ps(temp, *(__m512*)(B[j]));
			*(__m512*)(B[j]) = _mm512_mask_blend_ps(mask2, *(__m512*)(B[j]), *(__m512*)(B[y]));
			*(__m512*)(B[y]) = _mm512_mask_blend_ps(mask2, *(__m512*)(B[y]), copy);

			temp1 = (__m512*)(B[j] + m2);
			temp2 = (__m512*)(B[y] + m2);
			for (i = 0; i < n2; ++i) {
				copy = _mm512_mul_ps(temp, temp1[i]);
				temp1[i] = temp2[i];
				temp2[i] = copy;
			}
		}
		else {
			temp = _mm512_set1_ps(1.0 / A[y][y]);
			*(__m512*)(B[y]) = _mm512_mul_ps(_mm512_mask_blend_ps(mask2, _mm512_set1_ps(1.0), temp), *(__m512*)(B[y]));
			temp2 = (__m512*)(B[y] + m2);
			for (j = 0; j < n2; ++j)
				temp2[j] = _mm512_mul_ps(temp, temp2[j]);
			*(__m512*)(A[y]) = _mm512_mul_ps(_mm512_mask_blend_ps(mask1, _mm512_set1_ps(1.0), temp), *(__m512*)(A[y]));
			temp1 = (__m512*)(A[y] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_mul_ps(temp, temp1[j]);
		}

		temp2 = (__m512*)(B[y] + m2);
		for (x = y + 1; x < A.rows; ++x) {
			temp = _mm512_set1_ps(-A[x][y]);
			*(__m512*)(B[x]) = _mm512_add_ps(*(__m512*)(B[x]), _mm512_mul_ps(_mm512_mask_blend_ps(mask2, _mm512_setzero_ps(), temp), *(__m512*)(B[y])));
			temp1 = (__m512*)(B[x] + m2);
			for (j = 0; j < n2; ++j)
				temp1[j] = _mm512_add_ps(temp1[j], _mm512_mul_ps(temp, temp2[j]));
		}
		temp2 = (__m512*)(A[y] + m1);
		for (x = y + 1; x < A.rows; ++x) {
			temp = _mm512_set1_ps(-A[x][y]);
			*(__m512*)(A[x]) = _mm512_add_ps(*(__m512*)(A[x]), _mm512_mul_ps(_mm512_mask_blend_ps(mask1, _mm512_setzero_ps(), temp), *(__m512*)(A[y])));
			temp1 = (__m512*)(A[x] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_add_ps(temp1[j], _mm512_mul_ps(temp, temp2[j]));
		}
	}
	for (y = s; y < A.cols - 1; ++y) {
		n1 = (A.cols - y - 1) / 16 + 1;
		m1 = A.cols - 16 * n1;
		j = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.rows; ++i) {
			t = fabs(A[i][y]);
			if (t > big) {
				j = i;
				big = t;
			}
		}
		if (j != y) {
			temp = _mm512_set1_ps(1.0 / A[j][y]);

			temp1 = (__m512*)(A[j] + m1);
			temp2 = (__m512*)(A[y] + m1);
			for (i = 0; i < n1; ++i) {
				copy = _mm512_mul_ps(temp, temp1[i]);
				temp1[i] = temp2[i];
				temp2[i] = copy;
			}

			copy = _mm512_mul_ps(temp, *(__m512*)(B[j]));
			*(__m512*)(B[j]) = _mm512_mask_blend_ps(mask2, *(__m512*)(B[j]), *(__m512*)(B[y]));
			*(__m512*)(B[y]) = _mm512_mask_blend_ps(mask2, *(__m512*)(B[y]), copy);

			temp1 = (__m512*)(B[j] + m2);
			temp2 = (__m512*)(B[y] + m2);
			for (i = 0; i < n2; ++i) {
				copy = _mm512_mul_ps(temp, temp1[i]);
				temp1[i] = temp2[i];
				temp2[i] = copy;
			}
		}
		else {
			temp = _mm512_set1_ps(1.0 / A[y][y]);

			*(__m512*)(B[y]) = _mm512_mul_ps(_mm512_mask_blend_ps(mask2, _mm512_set1_ps(1.0), temp), *(__m512*)(B[y]));
			temp2 = (__m512*)(B[y] + m2);
			for (j = 0; j < n2; ++j)
				temp2[j] = _mm512_mul_ps(temp, temp2[j]);
			temp1 = (__m512*)(A[y] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_mul_ps(temp, temp1[j]);
		}

		temp2 = (__m512*)(B[y] + m2);
		for (x = y + 1; x < A.rows; ++x) {
			temp = _mm512_set1_ps(-A[x][y]);
			*(__m512*)(B[x]) = _mm512_add_ps(*(__m512*)(B[x]), _mm512_mul_ps(_mm512_mask_blend_ps(mask2, _mm512_setzero_ps(), temp), *(__m512*)(B[y])));
			temp1 = (__m512*)(B[x] + m2);
			for (j = 0; j < n2; ++j)
				temp1[j] = _mm512_add_ps(temp1[j], _mm512_mul_ps(temp, temp2[j]));
		}
		temp2 = (__m512*)(A[y] + m1);
		for (x = y + 1; x < A.rows; ++x) {
			temp = _mm512_set1_ps(-A[x][y]);
			temp1 = (__m512*)(A[x] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_add_ps(temp1[j], _mm512_mul_ps(temp, temp2[j]));
		}
	}

	y = A.cols - 1;
	temp = _mm512_set1_ps(1.0 / A[y][y]);
	*(__m512*)(B[y]) = _mm512_mul_ps(_mm512_mask_blend_ps(mask2, _mm512_set1_ps(1.0), temp), *(__m512*)(B[y]));
	temp2 = (__m512*)(B[y] + m2);
	for (j = 0; j < n2; ++j)
		temp2[j] = _mm512_mul_ps(temp, temp2[j]);

	for (y = A.cols - 1; y >= 0; --y) {
		temp2 = (__m512*)(B[y] + m2);
		for (x = 0; x < y; ++x) {
			temp = _mm512_set1_ps(-A[x][y]);
			*(__m512*)(B[x]) = _mm512_add_ps(*(__m512*)(B[x]), _mm512_mul_ps(_mm512_mask_blend_ps(mask2, _mm512_setzero_ps(), temp), *(__m512*)(B[y])));
			temp1 = (__m512*)(B[x] + m2);
			for (j = 0; j < n2; ++j)
				temp1[j] = _mm512_add_ps(temp1[j], _mm512_mul_ps(temp, temp2[j]));
		}
	}
}


template<typename T>
T __Det(MAT<T>& n) {
	int x, y, i, pos;
	T big, now;
	T r = 1, temp;
	for (y = 0; y < n.cols; ++y) {
		pos = y;
		big = fabs(n[y][y]);
		for (i = y + 1; i < n.rows; ++i) {
			now = fabs(n[i][y]);
			if (now > big) {
				pos = i;
				big = now;
			}
		}
		if (pos != y) {
			for (i = y; i < n.cols; ++i) {
				temp = n[pos][i];
				n[pos][i] = n[y][i];
				n[y][i] = temp;
			}
			r = -r;
		}
		for (x = y + 1; x < n.rows; ++x) {
			temp = n[x][y] / n[y][y];
			for (i = y; i < n.rows; ++i)
				n[x][i] -= n[y][i] * temp;
		}
	}
	for (x = 0; x < n.rows; ++x)
		r = r * n[x][x];

	return r;
}

double __Det(MAT<double>& A) {
	int x, y, i, j;
	int s = A.cols % 8;
	double t, big;
	double ret = 1;
	int n1 = A.cols / 8;
	int m1 = s;
	const __mmask8 mask1 = (1 << m1) - 1;

	__m512d copy;
	__m512d temp;
	__m512d* temp1;
	__m512d* temp2;

	for (y = 0; y < s; ++y) {
		j = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.rows; ++i) {
			t = fabs(A[i][y]);
			if (t > big) {
				j = i;
				big = t;
			}
		}
		if (j != y) {
			ret *= -A[j][y];
			temp = _mm512_set1_pd(1.0 / A[j][y]);

			copy = _mm512_mul_pd(temp, *(__m512d*)(A[j]));
			*(__m512d*)(A[j]) = _mm512_mask_blend_pd(mask1, *(__m512d*)(A[j]), *(__m512d*)(A[y]));
			*(__m512d*)(A[y]) = _mm512_mask_blend_pd(mask1, *(__m512d*)(A[y]), copy);

			temp1 = (__m512d*)(A[j] + m1);
			temp2 = (__m512d*)(A[y] + m1);
			for (i = 0; i < n1; ++i) {
				copy = _mm512_mul_pd(temp, temp1[i]);
				temp1[i] = temp2[i];
				temp2[i] = copy;
			}
		}
		else {
			ret *= A[y][y];
			temp = _mm512_set1_pd(1.0 / A[y][y]);
			*(__m512d*)(A[y]) = _mm512_mul_pd(_mm512_mask_blend_pd(mask1, _mm512_set1_pd(1.0), temp), *(__m512d*)(A[y]));
			temp1 = (__m512d*)(A[y] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_mul_pd(temp, temp1[j]);
		}
		temp2 = (__m512d*)(A[y] + m1);
		for (x = y + 1; x < A.rows; ++x) {
			temp = _mm512_set1_pd(-A[x][y]);
			*(__m512d*)(A[x]) = _mm512_add_pd(*(__m512d*)(A[x]), _mm512_mul_pd(_mm512_mask_blend_pd(mask1, _mm512_setzero_pd(), temp), *(__m512d*)(A[y])));
			temp1 = (__m512d*)(A[x] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_add_pd(temp1[j], _mm512_mul_pd(temp, temp2[j]));
		}
	}
	for (y = s; y < A.cols - 1; ++y) {
		n1 = (A.cols - y - 1) / 8 + 1;
		m1 = A.cols - 8 * n1;
		j = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.rows; ++i) {
			t = fabs(A[i][y]);
			if (t > big) {
				j = i;
				big = t;
			}
		}
		if (j != y) {
			ret *= -A[j][y];
			temp = _mm512_set1_pd(1.0 / A[j][y]);

			temp1 = (__m512d*)(A[j] + m1);
			temp2 = (__m512d*)(A[y] + m1);
			for (i = 0; i < n1; ++i) {
				copy = _mm512_mul_pd(temp, temp1[i]);
				temp1[i] = temp2[i];
				temp2[i] = copy;
			}
		}
		else {
			ret *= A[y][y];
			temp = _mm512_set1_pd(1.0 / A[y][y]);
			temp1 = (__m512d*)(A[y] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_mul_pd(temp, temp1[j]);
		}
		temp2 = (__m512d*)(A[y] + m1);
		for (x = y + 1; x < A.rows; ++x) {
			temp = _mm512_set1_pd(-A[x][y]);
			temp1 = (__m512d*)(A[x] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_add_pd(temp1[j], _mm512_mul_pd(temp, temp2[j]));
		}
	}
	y = A.cols - 1;

	return ret * A[y][y];
}

float __Det(MAT<float>& A) {
	int x, y, i, j;
	int s = A.cols % 16;
	float t, big;
	float ret = 1;
	int n1 = A.cols / 16;
	int m1 = s;
	const __mmask16 mask1 = (1 << m1) - 1;

	__m512 copy;
	__m512 temp;
	__m512* temp1;
	__m512* temp2;

	for (y = 0; y < s; ++y) {
		j = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.rows; ++i) {
			t = fabs(A[i][y]);
			if (t > big) {
				j = i;
				big = t;
			}
		}
		if (j != y) {
			ret *= -A[j][y];
			temp = _mm512_set1_ps(1.0 / A[j][y]);

			copy = _mm512_mul_ps(temp, *(__m512*)(A[j]));
			*(__m512*)(A[j]) = _mm512_mask_blend_ps(mask1, *(__m512*)(A[j]), *(__m512*)(A[y]));
			*(__m512*)(A[y]) = _mm512_mask_blend_ps(mask1, *(__m512*)(A[y]), copy);

			temp1 = (__m512*)(A[j] + m1);
			temp2 = (__m512*)(A[y] + m1);
			for (i = 0; i < n1; ++i) {
				copy = _mm512_mul_ps(temp, temp1[i]);
				temp1[i] = temp2[i];
				temp2[i] = copy;
			}
		}
		else {
			ret *= A[y][y];
			temp = _mm512_set1_ps(1.0 / A[y][y]);
			*(__m512*)(A[y]) = _mm512_mul_ps(_mm512_mask_blend_ps(mask1, _mm512_set1_ps(1.0), temp), *(__m512*)(A[y]));
			temp1 = (__m512*)(A[y] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_mul_ps(temp, temp1[j]);
		}
		temp2 = (__m512*)(A[y] + m1);
		for (x = y + 1; x < A.rows; ++x) {
			temp = _mm512_set1_ps(-A[x][y]);
			*(__m512*)(A[x]) = _mm512_add_ps(*(__m512*)(A[x]), _mm512_mul_ps(_mm512_mask_blend_ps(mask1, _mm512_setzero_ps(), temp), *(__m512*)(A[y])));
			temp1 = (__m512*)(A[x] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_add_ps(temp1[j], _mm512_mul_ps(temp, temp2[j]));
		}
	}
	for (y = s; y < A.cols - 1; ++y) {
		n1 = (A.cols - y - 1) / 16 + 1;
		m1 = A.cols - 16 * n1;
		j = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.rows; ++i) {
			t = fabs(A[i][y]);
			if (t > big) {
				j = i;
				big = t;
			}
		}
		if (j != y) {
			ret *= -A[j][y];
			temp = _mm512_set1_ps(1.0 / A[j][y]);

			temp1 = (__m512*)(A[j] + m1);
			temp2 = (__m512*)(A[y] + m1);
			for (i = 0; i < n1; ++i) {
				copy = _mm512_mul_ps(temp, temp1[i]);
				temp1[i] = temp2[i];
				temp2[i] = copy;
			}
		}
		else {
			ret *= A[y][y];
			temp = _mm512_set1_ps(1.0 / A[y][y]);
			temp1 = (__m512*)(A[y] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_mul_ps(temp, temp1[j]);
		}
		temp2 = (__m512*)(A[y] + m1);
		for (x = y + 1; x < A.rows; ++x) {
			temp = _mm512_set1_ps(-A[x][y]);
			temp1 = (__m512*)(A[x] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_add_ps(temp1[j], _mm512_mul_ps(temp, temp2[j]));
		}
	}
	y = A.cols - 1;

	return ret * A[y][y];
}

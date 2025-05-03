
template <typename T>
void __SV_dqds_LR(T*& p, T*& q, int m, int n, double eps) {

	T t, d, s = 0;
	int i;
	while (m <= n) {
		if (q[n - 1] < eps) {
			--n;
			/*
			s = p[n] - p[n + 1];
			d = p[m] - s;
			for (i = m; i < n; ++i) {
				p[i] = d + q[i];
				t = p[i + 1] / p[i];
				q[i] = q[i] * t;
				d = d * t - s;
			}
			p[n] = d;
			*/
			continue;
		}
		if (q[m] < eps) {
			++m;
			continue;
		}
		d = p[m];
		for (i = m; i < n; ++i) {
			p[i] = d + q[i];
			t = p[i + 1] / p[i];
			q[i] = q[i] * t;
			d = d * t;
		}
		p[n] = d;
	}
	//*/
}

template<typename T>
void __SV_dqds(MAT<T>& A, MAT<T>& S, double eps) {
	int i = 0;
	int s;
	int r;

	memcpy(S.data, A.data, sizeof(T) * S.rows * S.cols);

	T* w, * v;
	if (S.rows > S.cols) {
		//T* w = (T*)alloca(sizeof(T) * S.rows);
		s = sizeof(T) * (S.cols - 1) / 64 + 1;
		w = new T[s * 64 / sizeof(T)];
		v = new T[s * 64 / sizeof(T)];
		s = S.cols - 1;

		for (r = 0; r < s; ++r) {
			__SV_HouseHolder_left(S, w, v, r, r);
			__SV_HouseHolder_right(S, w, r, r + 1);
		}
		__SV_HouseHolder_left(S, w, v, s, s);

		T* p(S[S.rows - 2]);
		T* q(S[S.rows - 1]);
		for (r = 0; r < S.cols; ++r) {
			p[r] = S[r][r] * S[r][r];
		}
		for (r = 1; r < S.cols; ++r) {
			q[r - 1] = S[r - 1][r] * S[r - 1][r];
		}
		__SV_dqds_LR(p, q, 0, A.cols - 1, eps);

		for (r = 0; r < S.cols; ++r) {
			S[r][r] = sqrt(p[r]);
		}
		for (r = 1; r < S.cols; ++r) {
			S[r - 1][r] = 0;
		}
		for (r = 0; r < S.cols; ++r) {
			p[r] = 0;
		}
	}
	else {
		//T* w = (T*)alloca(sizeof(T) * S.cols);
		s = sizeof(T) * (S.cols - 1) / 64 + 1;
		w = new T[s * 64 / sizeof(T)];
		v = new T[s * 64 / sizeof(T)];
		s = S.rows - 1;

		for (r = 0; r < s; ++r) {
			__SV_HouseHolder_right(S, w, r, r);
			__SV_HouseHolder_left(S, w, v, r, r + 1);
		}
		__SV_HouseHolder_right(S, w, s, s);

		T* p(S.data);
		T* q(p + S.rows);
		for (r = 0; r < S.rows; ++r) {
			p[r] = S[r][r] * S[r][r];
		}
		for (r = 1; r < S.rows; ++r) {
			q[r - 1] = S[r][r - 1] * S[r][r - 1];
		}

		__SV_dqds_LR(p, q, 0, S.rows - 1, eps);

		for (r = 0; r < S.rows; ++r) {
			S[r][r] = sqrt(p[r]);
		}
		for (r = 1; r < S.rows; ++r) {
			S[r][r - 1] = 0;
		}
		for (r = 1; r < S.rows; ++r) {
			p[r] = 0;
		}
	}
	delete[] w;
	delete[] v;
}
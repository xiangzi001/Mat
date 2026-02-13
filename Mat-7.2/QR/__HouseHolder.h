
template<typename T>
inline void __ApplyHouseHolderOnTheRight(MAT<T>& A, T* w, T* v, int r) {
    ZeroMemory(v, sizeof(T) * A.rows);
    for (int i = r; i < A.cols; ++i) {
        __mixed3(v, A[i], w[i], v, A.rows);
    }
    for (int i = r; i < A.cols; ++i) {
        __mixed3(A[i], v, -w[i], A[i], A.rows);
    }
}
template<typename T>
inline void __ApplyHouseHolderOnTheleft(MAT<T>& A, T* w, int r) {
    for (int i = 0; i < A.cols; ++i) {
        __mixed3(A[i] + r, w + r, -__dot(w + r, A[i] + r, A.rows - r), A[i] + r, A.rows - r);
    }
}
template<typename t>
inline void __ApplyHouseHolderOnTheRight(MAT<complex<t>>& A, complex<t>* w, complex<t>* v, int r) {
    ZeroMemory(v, sizeof(complex<t>) * A.rows);
    for (int i = r; i < A.cols; ++i) {
        __mixed3(v, A[i], w[i], v, A.rows);
    }
    for (int i = r; i < A.cols; ++i) {
        __mixed3(A[i], v, -w[i].conj(), A[i], A.rows);
    }
}

template<typename T>
inline void __ApplyHouseHolderOnTheRight2(MAT<T>& A, T* w, T* v, int m, int n) {
    for (int i = m; i < A.cols; ++i) {
        v[i] = __dot(A[i] + n, w + n, A.rows - n);
    }
    for (int i = n; i < A.cols; ++i) {
        __mixed3(A[i] + m, v + m, -w[i], A[i] + m, A.rows - m);
    }
}
template<typename T>
inline void __ApplyHouseHolderOnTheleft2(MAT<T>& A, T* w, int m, int n) {
    for (int i = m; i < A.cols; ++i) {
        __mixed3(A[i] + n, w + n, -__dot(w + n, A[i] + n, A.rows - n), A[i] + n, A.rows - n);
    }
}
template<typename t>
inline void __ApplyHouseHolderOnTheRight2(MAT<complex<t>>& A, complex<t>* w, complex<t>* v, int m, int n) {
    for (int i = m; i < A.cols; ++i) {
        v[i] = __dot(A[i] + n, w + n, A.rows - n);
    }
    for (int i = n; i < A.cols; ++i) {
        __mixed3(A[i] + m, v + m, -w[i].conj(), A[i] + m, A.rows - m);
    }
}

template<typename T>
inline void __ApplyHouseHolderOnTheRight3(MAT<T>& A, T* w, T* v, int m, int n) {
    ZeroMemory(v, sizeof(T) * A.rows);
    for (int i = n; i < A.cols; ++i) {
        __mixed3(v + m, A[i] + m, w[i], v + m, A.rows - m);
    }
    for (int i = n; i < A.cols; ++i) {
        __mixed3(A[i] + m, v + m, -w[i], A[i] + m, A.rows - m);
    }
}

template<typename T>
void __QR_HouseHolder(MAT<T>& Q, MAT<T>& R, T* w, T* v) {
    int n = 0;
    if (R.rows > R.cols) {
        n = R.cols;
    }
    else {
        n = R.rows;
    };
    int r;

    T cr, hr;

    for (r = 0; r < n; ++r) {
        memcpy(w + r, R[r] + r, sizeof(T) * (R.rows - r));
        cr = sqrt(__dot(w + r, w + r, R.rows - r));

        if (w[r] < 0.0)cr = -cr;
        w[r] += cr;
        hr = 1.0 / sqrt(cr * w[r]);

        __times(w + r, hr, w + r, R.rows - r);
        __ApplyHouseHolderOnTheRight(Q, w, v, r);
        __ApplyHouseHolderOnTheleft2(R, w, r, r);
    }
}

template<typename t>
void __QR_HouseHolder(MAT<complex<t>>& Q, MAT<complex<t>>& R, complex<t>* w, complex<t>* v) {
    int n = 0;
    if (R.rows > R.cols) {
        n = R.cols;
    }
    else {
        n = R.rows;
    };
    int r;

    complex<t> sum, _sign;
    t cr, hr;

    for (r = 0; r < n - 1; ++r) {
        memcpy(w + r, R[r] + r, sizeof(t) * (R.rows - r) * 2);
        cr = sqrt(__dot((t*)(w + r), (t*)(w + r), (R.rows - r) * 2));

        _sign = sign(w[r]) * cr;
        hr = 1 / sqrt(cr * cr - times_conj(_sign, w[r]).real);
        w[r] -= _sign;

        __times((t*)(w + r), hr, (t*)(w + r), (R.rows - r) * 2);
        __ApplyHouseHolderOnTheRight(Q, w, v, r);
        __ApplyHouseHolderOnTheleft2(R, w, r, r);
    }
}

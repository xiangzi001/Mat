template<typename T>
inline void __ApplyGivensOnTheRight(T* __restrict a, T* __restrict b, T c, T s, int len) {
    T ai, bi;
    for (int i = 0; i < len; ++i) {
        ai = a[i];
        bi = b[i];
        a[i] = ai * c + bi * s;
        b[i] = -ai * s + bi * c;
    }
}

template<typename t>
inline void __ApplyGivensOnTheRight(complex<t>* __restrict a, complex<t>* __restrict b, t c, complex<t> s, int len) {
    int i;
    complex<t> _s(s.conj()), ai, bi;
    for (int i = 0; i < len; ++i) {
        ai = a[i];
        bi = b[i];
        a[i] = ai * c + bi * s;
        b[i] = -ai * _s + bi * c;
    }
}

template<typename T>
void __QR_Givens(MAT<T>& Q, MAT<T>& R) {
    T temp, c, s;
    T* r;

    int i, j, k;
    for (j = 0; j < R.cols; ++j) {
        for (i = R.rows - 1; i > j; i--) {
            //double theta = atan2(R[i][j], R[i - 1][j]);
            //double c = cos(theta);
            //double s = sin(theta);
            temp = R(i, j) / R(i - 1, j);
            c = 1 / sqrt(temp * temp + 1);
            s = temp * c;

            for (k = 0; k < R.cols; ++k) {
                r = R[k];
                temp = r[i - 1] * c + r[i] * s;
                r[i] = -r[i - 1] * s + r[i] * c;
                r[i - 1] = temp;
            }
            __ApplyGivensOnTheRight(Q[i - 1], Q[i], c, s, Q.rows);
        }
    }
}


template<typename t>
void __QR_Givens(MAT<complex<t>>& Q, MAT<complex<t>>& R) {
    t c;
    complex<t> temp, s, _s;
    complex<t>* r;

    int i, j, k;
    for (j = 0; j < R.cols; ++j) {
        for (i = R.rows - 1; i > j; --i) {
            temp = R(i, j) / R(i - 1, j);
            c = 1 / sqrt(temp.len2() + 1);
            s = temp * c;
            _s = s.conj();

            for (k = 0; k < R.cols; ++k) {
                r = R[k];
                temp = r[i - 1] * c + r[i] * _s;
                r[i] = -r[i - 1] * s + r[i] * c;
                r[i - 1] = temp;
            }
            __ApplyGivensOnTheRight(Q[i - 1], Q[i], c, s, Q.rows);
        }
    }
}

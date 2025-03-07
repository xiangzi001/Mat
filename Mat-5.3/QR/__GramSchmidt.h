
template<typename T>
void __QR_GramSchmidt(MAT<T>& Q, MAT<T>& R) {
    int i, j, k;
    T len;

    for (j = 0; j < Q.cols; ++j) {
        for (i = 0; i < j; ++i) {
            len = 0;
            for (k = 0; k < Q.rows; ++k) {
                len += Q[k][i] * Q[k][j];
            }
            for (k = 0; k < Q.rows; ++k) {
                Q[k][j] -= Q[k][i] * len;
            }
            R[i][j] = len;
        }

        len = 0;
        for (k = 0; k < Q.rows; ++k) {
            len += Q[k][j] * Q[k][j];
        }
        len = sqrt(len);
        R[j][j] = len;
        len = 1 / len;
        for (k = 0; k < Q.rows; ++k) {
            Q[k][j] *= len;
        }
    }
}


template<typename T>
void __QR_GramSchmidt(MAT<complex<T>>& Q, MAT<complex<T>>& R) {
    int i, j, k;
    complex<T> len;

    for (j = 0; j < Q.cols; ++j) {
        for (i = 0; i < j; ++i) {
            len = 0;
            for (k = 0; k < Q.rows; ++k) {
                len += Q[k][i].conj() * Q[k][j];
            }
            for (k = 0; k < Q.rows; ++k) {
                Q[k][j] -= Q[k][i] * len;
            }
            R[i][j] = len;
        }

        len = 0;
        for (k = 0; k < Q.rows; ++k) {
            len += Q[k][j].len2();
        }
        len = sqrt(len);
        R[j][j] = len;
        len = 1 / len;
        for (k = 0; k < Q.rows; ++k) {
            Q[k][j] *= len;
        }
    }
}
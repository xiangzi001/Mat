
template<typename T>
void __QR_Givens(MAT<T>& Q, MAT<T>& R) {
    T temp, t, c, s;

    for (int j = 0; j < R.cols; ++j) {
        for (int i = R.rows - 1; i > j; i--) {
            //double theta = atan2(R[i][j], R[i - 1][j]);
            //double c = cos(theta);
            //double s = sin(theta);
            t = R[i][j] / R[i - 1][j];
            c = 1 / sqrt(t * t + 1);
            s = t * c;

            for (int k = 0; k < R.cols; ++k) {
                temp = R[i - 1][k] * c + R[i][k] * s;
                R[i][k] = -R[i - 1][k] * s + R[i][k] * c;
                R[i - 1][k] = temp;
            }

            for (int k = 0; k < R.rows; ++k) {
                temp = Q[k][i - 1] * c + Q[k][i] * s;
                Q[k][i] = -Q[k][i - 1] * s + Q[k][i] * c;
                Q[k][i - 1] = temp;
            }

        }
    }
}


template<typename T>
void __QR_Givens(MAT<complex<T>>& Q, MAT<complex<T>>& R) {
    T c;
    complex<T> temp, t, s, _s;
    for (int j = 0; j < R.cols; ++j) {
        for (int i = R.rows - 1; i > j; --i) {
            if (R[i][j].is_zero()) {
                continue;
            }
            //double theta = atan2(R[i][j], R[i - 1][j]);
            //double c = cos(theta);
            //double s = sin(theta);
            t = R[i][j] / R[i - 1][j];
            c = 1 / sqrt(t.real * t.real + t.imag * t.imag + 1);
            s = t * c;
            _s = s.conj();

            for (int k = j; k < R.cols; ++k) {
                temp = R[i - 1][k] * c + R[i][k] * _s;
                R[i][k] = R[i][k] * c - R[i - 1][k] * s;
                R[i - 1][k] = temp;
            }

            for (int k = 0; k < R.rows; ++k) {
                temp = Q[k][i - 1] * c + Q[k][i] * s;
                Q[k][i] = Q[k][i] * c - Q[k][i - 1] * _s;
                Q[k][i - 1] = temp;
            }

        }
    }
}

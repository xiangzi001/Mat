
template<typename T>
void QR_HouseHolder(MAT<complex<T>>& Q, MAT<complex<T>>& R, complex<T>* w) {
    int n = 0;
    if (R.rows > R.cols) {
        n = R.cols;
    }
    else {
        n = R.rows;
    };
    int i, j, k, r;

    complex<T> sum, _sign;
    T temp, cr, hr;

    for (r = 0; r < n-1; ++r) {

        temp = 0;
        for (k = r; k < R.rows; ++k) {
            temp += R[k][r].len2();
        }
        if (temp == 0) {
            continue;
        }
        cr = sqrt(temp);
        _sign = sign(R[r][r]) * cr;

        //if (R[r][r].real < 0.0)cr = -cr;
        hr = 1 / (cr * cr - times_conj(_sign, R[r][r]).real);

        //定义w
        w[r] = R[r][r] - _sign;
        for (i = r + 1; i < R.rows; ++i)
        {
            w[i] = R[i][r];
        }

        for (i = 0; i < R.rows; ++i)//定义Q
        {
            sum = 0;
            for (j = r; j < n; ++j)
                sum += Q[i][j] * w[j];
            sum = sum * hr;
            for (j = r; j < R.cols; ++j)
            {
                Q[i][j] -= times_conj(sum, w[j]);
            }
        }
        for (j = 0; j < R.cols; ++j)//定义R
        {
            sum = 0;
            for (i = r; i < n; ++i)
                sum += times_conj(R[i][j], w[i]);
            sum = sum * hr;
            for (i = r; i < R.rows; ++i)
                R[i][j] -= w[i] * sum;
        }

    }
}

template<typename T>
void QR_Givens(MAT<complex<T>>& Q, MAT<complex<T>>& R) {
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
            c = 1 / sqrt(t.real*t.real + t.imag*t.imag + 1);
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

template<typename T>
void QR(MAT<complex<T>>& A, MAT<complex<T>>& Q, MAT<complex<T>>& R, QR_type type) {
    int s = A.cols * A.rows;
    int i;
    for (i = 0; i < s; ++i) {
        R.data[i] = A.data[i];
    }
    s = Q.cols * Q.rows;
    for (i = 0; i < s; ++i) {
        Q.data[i] = 0;
    }
    for (i = 0; i < Q.rows; ++i) {
        Q[i][i] = 1;
    }

    switch (type)
    {
    case _QR_HouseHolder: {
        complex<T>* w = new complex<T>[R.cols];
        QR_HouseHolder(Q, R, w);
        delete[] w;
        break;
    }
    case _QR_Givens: {
        QR_Givens(Q, R);
        break;
    }
    default:
        break;
    }
}



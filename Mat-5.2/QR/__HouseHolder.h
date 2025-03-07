
template<typename T>
void __QR_HouseHolder(MAT<T>& Q, MAT<T>& R, T* w) {
    int n = 0;
    if (R.rows > R.cols) {
        n = R.cols;
    }
    else {
        n = R.rows;
    };
    int i, j, k, r;

    T sum, cr, hr;

    for (r = 0; r < n; ++r) {

        sum = 0;
        for (k = r; k < R.rows; ++k)
            sum += R[k][r] * R[k][r];
        cr = sqrt(sum);

        if (R[r][r] < 0.0)cr = -cr;
        hr = 1 / (cr * (cr + R[r][r]));

        //定义w
        w[r] = R[r][r] + cr;
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
                Q[i][j] = Q[i][j] - w[j] * sum;
            }
        }
        for (j = 0; j < R.cols; ++j)//定义R
        {
            sum = 0;
            for (i = r; i < n; ++i)
                sum += R[i][j] * w[i];
            sum = sum * hr;
            for (i = r; i < R.rows; ++i)
                R[i][j] = R[i][j] - w[i] * sum;
        }

    }
}


template<typename T>
void __QR_HouseHolder(MAT<complex<T>>& Q, MAT<complex<T>>& R, complex<T>* w) {
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

    for (r = 0; r < n - 1; ++r) {

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

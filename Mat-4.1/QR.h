
template<typename T>
void QR_HouseHolder(MAT<T>& Q, MAT<T>& R, T*w) {
    int n = 0;
    if (R.rows > R.cols) {
        n = R.cols;
    }
    else {
        n = R.rows;
    };
    int i, j, k, r;

    T sum, cr, hr;

    for (r = 0; r < n; r++) {

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
void QR_Givens(MAT<T>& Q, MAT<T>& R) {
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
void QR_GramSchmidt(MAT<T>& Q, MAT<T>& R) {
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
            len += Q[k][j]* Q[k][j];
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
void QR(MAT<T>& A, MAT<T>& Q, MAT<T>& R, QR_type type) {
    switch (type)
    {
    case _QR_HouseHolder: {
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

        T* w = new T[R.cols];
        QR_HouseHolder(Q, R, w);
        delete[] w;
        break;
    }
    case _QR_Givens: {
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
        QR_Givens(Q, R);
        break;
    }
    case _QR_GramSchmidt: {
        int s = A.cols * A.rows;
        int i,j;
        for (i = 0; i < s; ++i) {
            R.data[i] = 0;
        }
        for (i = 0; i < Q.rows; ++i) {
            for (j = 0; j < Q.cols; ++j) {
                Q[i][j] = A[i][j];
            }
        }
        QR_GramSchmidt(Q, R);
        break;
    }
    default:
        break;
    }
}




template<typename T>
void __QR_HouseHolder(MAT<T>& Q, MAT<T>& R, T* w, T* v) {
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

        for (i = r; i < R.rows; ++i){
            w[i] = R[i][r];
        }
        sum = 0;
        for (k = r; k < R.rows; ++k)
            sum += w[k] * w[k];
        cr = sqrt(sum);

        if (w[r] < 0.0)cr = -cr;
        w[r] = R[r][r] + cr;
        hr = 1.0 / sqrt(cr * w[r]);

        for (i = r; i < R.rows; ++i){
            w[i] *= hr;
        }

        for (i = 0; i < R.rows; ++i)//定义Q
        {
            sum = 0;
            for (j = r; j < n; ++j)
                sum += Q[i][j] * w[j];
            for (j = r; j < R.cols; ++j)
            {
                Q[i][j] = Q[i][j] - w[j] * sum;
            }
        }
        for (j = 0; j < R.cols; ++j) {
            v[j] = 0;
        }
        for (i = r; i < R.rows; ++i) {
            for (j = 0; j < R.cols; ++j) {
                v[j] += w[i] * R[i][j];
            }
        }
        for (i = r; i < R.rows; ++i) {
            for (j = 0; j < R.cols; ++j) {
                R[i][j] -= w[i] * v[j];
            }
        }
    }
}

void __QR_HouseHolder(MAT<double>& Q, MAT<double>& R, double* w, double* v) {
    int N = R.rows;
    if (R.rows > R.cols) {
        N = R.cols;
    }

    int i, j, k, r;
    int n;
    double L;

    __m512d sum;
    __m512d* temp1;
    __m512d* temp2;

    ZeroMemory(w, sizeof(double) * (R.rows + 7));
    ZeroMemory(v, sizeof(double) * (R.cols + 7));

    for (r = 0; r < N; ++r) {
        n = (R.rows - r - 1) / 8 + 1;
        temp1 = (__m512d*)(w + r);

        for (i = r; i < R.rows; ++i) {
            w[i] = R[i][r];
        }

        sum = _mm512_setzero_pd();
        for (k = 0; k < n; ++k)
            sum = _mm512_add_pd(sum, _mm512_mul_pd(temp1[k], temp1[k]));
        L = sqrt(_mm512_reduce_add_pd(sum));

        if (w[r] < 0.0)L = -L;
        w[r] += L;
        sum = _mm512_set1_pd(1.0 / sqrt(L * w[r]));

        for (k = 0; k < n; ++k)
            temp1[k] = _mm512_mul_pd(sum, temp1[k]);

        for (i = 0; i < Q.rows; ++i) {
            temp2 = (__m512d*)(Q[i] + r);
            sum = _mm512_setzero_pd();
            for (j = 0; j < n; ++j)
                sum = _mm512_add_pd(sum, _mm512_mul_pd(temp1[j], temp2[j]));
            sum = _mm512_set1_pd(_mm512_reduce_add_pd(sum));

            for (j = 0; j < n; ++j) {
                temp2[j] = _mm512_sub_pd(temp2[j], _mm512_mul_pd(sum, temp1[j]));
            }
        }

        ZeroMemory(v, sizeof(double) * R.cols);
        n = (R.cols - 1) / 8 + 1;
        temp1 = (__m512d*)(v);

        for (i = r; i < R.rows; ++i) {
            temp2 = (__m512d*)(R[i]);
            sum = _mm512_set1_pd(w[i]);
            for (j = 0; j < n; ++j) {
                temp1[j] = _mm512_add_pd(temp1[j], _mm512_mul_pd(temp2[j], sum));
            }
        }
        ZeroMemory(v + R.cols, sizeof(double) * 7);
        for (i = r; i < R.rows; ++i) {
            temp2 = (__m512d*)(R[i]);
            sum = _mm512_set1_pd(w[i]);
            for (j = 0; j < n; ++j) {
                temp2[j] = _mm512_sub_pd(temp2[j], _mm512_mul_pd(temp1[j], sum));
            }
        }
        w[r] = 0;
    }
}

void __QR_HouseHolder(MAT<float>& Q, MAT<float>& R, float* w, float* v) {
    int N = R.rows;
    if (R.rows > R.cols) {
        N = R.cols;
    }

    int i, j, k, r;
    int n;
    float L;

    __m512 sum;
    __m512* temp1;
    __m512* temp2;

    ZeroMemory(w, sizeof(float) * (R.rows + 15));
    ZeroMemory(v, sizeof(float) * (R.cols + 15));

    for (r = 0; r < N; ++r) {
        n = (R.rows - r - 1) / 16 + 1;
        temp1 = (__m512*)(w + r);

        for (i = r; i < R.rows; ++i) {
            w[i] = R[i][r];
        }

        sum = _mm512_setzero_ps();
        for (k = 0; k < n; ++k)
            sum = _mm512_add_ps(sum, _mm512_mul_ps(temp1[k], temp1[k]));
        L = sqrt(_mm512_reduce_add_ps(sum));

        if (w[r] < 0.0)L = -L;
        w[r] += L;
        sum = _mm512_set1_ps(1.0 / sqrt(L * w[r]));

        for (k = 0; k < n; ++k)
            temp1[k] = _mm512_mul_ps(sum, temp1[k]);

        for (i = 0; i < Q.rows; ++i) {
            temp2 = (__m512*)(Q[i] + r);
            sum = _mm512_setzero_ps();
            for (j = 0; j < n; ++j)
                sum = _mm512_add_ps(sum, _mm512_mul_ps(temp1[j], temp2[j]));
            sum = _mm512_set1_ps(_mm512_reduce_add_ps(sum));

            for (j = 0; j < n; ++j) {
                temp2[j] = _mm512_sub_ps(temp2[j], _mm512_mul_ps(sum, temp1[j]));
            }
        }

        ZeroMemory(v, sizeof(float) * R.cols);
        n = (R.cols - 1) / 16 + 1;
        temp1 = (__m512*)(v);

        for (i = r; i < R.rows; ++i) {
            temp2 = (__m512*)(R[i]);
            sum = _mm512_set1_ps(w[i]);
            for (j = 0; j < n; ++j) {
                temp1[j] = _mm512_add_ps(temp1[j], _mm512_mul_ps(temp2[j], sum));
            }
        }
        ZeroMemory(v + R.cols, sizeof(float) * 15);
        for (i = r; i < R.rows; ++i) {
            temp2 = (__m512*)(R[i]);
            sum = _mm512_set1_ps(w[i]);
            for (j = 0; j < n; ++j) {
                temp2[j] = _mm512_sub_ps(temp2[j], _mm512_mul_ps(temp1[j], sum));
            }
        }
        w[r] = 0;
    }
}


template<typename T>
void __QR_HouseHolder(MAT<complex<T>>& Q, MAT<complex<T>>& R, complex<T>* w, complex<T>* v) {
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

        for (i = r; i < R.rows; ++i) {
            w[i] = R[i][r];
        }
        temp = 0;
        for (k = r; k < R.rows; ++k) {
            temp += w[k].len2();
        }
        cr = sqrt(temp);
        _sign = sign(R[r][r]) * cr;
        hr = 1 / sqrt(cr * cr - times_conj(_sign, R[r][r]).real);

        w[r] -= _sign;
        for (i = r; i < R.rows; ++i) {
            w[i] *= hr;
        }


        for (i = 0; i < R.rows; ++i)//定义Q
        {
            sum = 0;
            for (j = r; j < n; ++j)
                sum += Q[i][j] * w[j];
            for (j = r; j < R.cols; ++j)
            {
                Q[i][j] -= times_conj(sum, w[j]);
            }
        }
        for (j = 0; j < R.cols; ++j) {
            v[j] = 0;
        }
        for (i = r; i < R.rows; ++i) {
            for (j = 0; j < R.cols; ++j) {
                v[j] += times_conj(R[i][j], w[i]);
            }
        }
        for (i = r; i < R.rows; ++i) {
            for (j = 0; j < R.cols; ++j) {
                R[i][j] -= w[i] * v[j];
            }
        }
    }
}


template<typename T>
void __EV_Q_Givens(T* a, T* b, int m, int n, double eps) {

    T t, temp, temp_2, c2, s2, cs;
    T B(0);
    int i, k;

    while (n > m) {
        if (fabs(b[m]) < eps) {
            ++m;
            continue;
        }
        if (fabs(b[n - 1]) < eps) {
            --n;
            continue;
        }
        for (i = m + 1; i < n - 1; ++i) {
            if (fabs(b[i]) < eps) {
                __EV_Q_Givens(a, b, m, i, eps);
                m = i + 1;
            }
        }

        t = b[m] / (a[m] - a[n]);
        c2 = 1 / (t * t + 1);
        s2 = 1 - c2;
        cs = t * c2;
        temp = a[m];
        temp_2 = a[m + 1];

        a[m] = c2 * temp + s2 * temp_2 + 2 * cs * b[m];
        a[m + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[m];
        b[m] = -cs * temp + cs * temp_2 + (c2 - s2) * b[m];
        c2 = sqrt(c2);
        s2 = t * c2;
        B = s2 * b[m + 1];
        b[m + 1] = c2 * b[m + 1];

        for (i = m + 1; i < n; ++i) {
            t = B / b[i - 1];
            c2 = 1 / (t * t + 1);
            s2 = 1 - c2;
            cs = t * c2;
            temp = a[i];
            temp_2 = a[i + 1];

            a[i] = c2 * temp + s2 * temp_2 + 2 * cs * b[i];
            a[i + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[i];
            b[i] = -cs * temp + cs * temp_2 + (c2 - s2) * b[i];
            c2 = sqrt(c2);
            s2 = t * c2;
            b[i - 1] = c2 * b[i - 1] + s2 * B;
            B = s2 * b[i + 1];
            b[i + 1] = c2 * b[i + 1];
        }
    }
}

template<typename T>
void EV_Q(MAT<T>& A, MAT<T>& E, double eps) {
    memcpy(E.data, A.data, sizeof(T) * E.rows * E.cols);

    int i;
    int s;

    T* w = new T[E.rows + 64 / sizeof(T) - 1];
    T* v = new T[E.cols + 64 / sizeof(T) - 1];
    __EV_HouseHolder_R(E, w, v);
    delete[] w;
    delete[] v;

    T* a(E.data);
    T* b(E.data + E.cols);
    for (i = 1; i < E.cols; ++i) {
        a[i] = E[i][i];
    }
    s = E.cols - 1;
    for (i = 1; i < s; ++i) {
        b[i] = E[i + 1][i];
    }
    __EV_Q_Givens(a, b, 0, E.rows - 1, eps);
    for (i = 1; i < E.cols; ++i) {
        E[i][i] = a[i];
    }

    for (i = 1; i < E.rows; ++i) {
        E[i - 1][i] = 0;
        E[i][i - 1] = 0;
    }
    for (i = 2; i < E.cols; ++i) {
        a[i] = 0;
    }
}

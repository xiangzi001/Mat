template<typename T>
void __Solve_direct_UX(__BlockMat<T>& U, __BlockMat<T>& X) {
    T* x;
    int i;
    for (int r = 0; r < X.cols; ++r) {
        x = X[r];
        for (i = U.cols - 1; i >= 0; --i) {
            x[i] /= U(i, i);
            __mixed(x, U[i], -x[i], x, i);
        }
    }
}

template <typename T>
void __Solve_direct_LU_X(MAT<T>& A, MAT<T>& X, T* __restrict A_pack, T* __restrict B_pack) {
    int r, i, j, l, m, n;
    __BlockMat<T> _L, _U, _A;

    constexpr int MC = __BlockSize_Mul_MC;
    constexpr int NC = __BlockSize_Mul_NC;
    constexpr int K = __BlockSize_LU;

    for (r = 0; r < A.cols; r += K) {
        l = (r + K > A.cols) ? (A.cols - r) : K;

        if (r + l < X.rows) {
            _L.set(A, r + l, r, X.rows - (r + l), l);
            __pack_A_block<T, T>(_L.data, _L._rows, X.rows - (r + l), l, A_pack);
        }

        for (j = 0; j < X.cols; j += NC) {
            n = (j + NC > X.cols) ? (X.cols - j) : NC;

            _L.set(A, r, r, A.rows - r, l);
            _U.set(X, r, j, l, n);
            __Solve_direct_LX(_L, _U);
            __pack_B_block<T, T>(_U[0], _U._rows, l, n, B_pack);

            for (i = r + l; i < X.rows; i += MC) {
                m = (i + MC > X.rows) ? (X.rows - i) : MC;

                _A.set(X, i, j, m, n);
                constexpr int MR = __micro_kernel_mr<T, T, T>();
                const T* A_base = A_pack + (size_t)((i - (r + l)) / MR) * (size_t)l * MR;
                __mulAB_packed_n<T>(A_base, B_pack, _A.data, _A._rows,
                    _A.rows, _A.cols, l);
            }
        }
    }

    r = (X.rows - 1) / K;
    for (r = r * K; r >= 0; r -= K) {
        n = (r + K < X.rows) ? K : (X.rows - r);
        if (r > 0) {
            _U.set(A, 0, r, r, n);
            __pack_A_block<T, T>(_U.data, _U._rows, r, n, A_pack);
        }

        for (j = 0; j < X.cols; j += NC) {
            m = (j + NC > X.cols) ? (X.cols - j) : NC;

            _U.set(A, r, r, n, n);
            _L.set(X, r, j, n, m);
            __Solve_direct_UX(_U, _L);
            __pack_B_block<T, T>(_L[0], _L._rows, n, m, B_pack);

            for (i = 0; i < r; i += MC) {
                int mi = (i + MC > r) ? (r - i) : MC;

                _A.set(X, i, j, mi, m);
                constexpr int MR = __micro_kernel_mr<T, T, T>();
                const T* A_base = A_pack + (size_t)(i / MR) * (size_t)n * MR;
                __mulAB_packed_n<T>(A_base, B_pack, _A.data, _A._rows,
                    _A.rows, _A.cols, n);
            }
        }
    }
}


template<typename T>
MAT<T> Solve_direct_narrow(MAT<T> A, MAT<T>& B)
{
    const int n = A.rows;
    const int nc = B.cols;
    MAT<T> b(n, nc);
    if (n == 0 || nc == 0) return b;

    constexpr int K = __BlockSize_LU;
    const int s = (A.rows > A.cols) ? A.cols : A.rows;

    int* pos = new int[n];
    for (int i = 0; i < n; ++i) pos[i] = i;
    if (s <= K) {
        __BlockMat<T> _L, _U;
        _L.set(A, 0, 0, A.rows, s);
        _U.set(A, 0, 0, A.rows, s);
        __PLU2(pos, _L, _U);
    }
    else {
        constexpr int NC = __BlockSize_Mul_NC;
        constexpr int MR = __micro_kernel_mr<T, T, T>();
        constexpr int NR = __micro_kernel_nr<T, T, T>();
        const size_t a_bytes = (size_t)((A.rows + MR - 1) / MR) * K * MR;
        const size_t b_bytes = (size_t)((NC + NR - 1) / NR) * K * NR;
        T* A_pack = (T*)_mm_malloc(sizeof(T) * a_bytes, 64);
        T* B_pack = (T*)_mm_malloc(sizeof(T) * b_bytes, 64);
        __PLU(pos, A, A, A_pack, B_pack);
        _mm_free(A_pack);
        _mm_free(B_pack);
    }

    for (int j = 0; j < nc; ++j) {
        const T* src = B.data + j * B.rows;
        T* dst = b.data + j * b.rows;
        for (int i = 0; i < n; ++i) 
            dst[i] = src[pos[i]];
    }
    delete[] pos;

    T* Xd = b.data;
    const int ldX = b.rows;
    T* Ad = A.data;
    const int ldA = A.rows;

    for (int j = 0; j < nc; ++j) {
        T* x = Xd + j * ldX;
        for (int k = 0; k < n - 1; ++k) {
            __mixed(x + k + 1, Ad + k * ldA + k + 1, -x[k], x + k + 1, n - k - 1);
        }
    }

    for (int j = 0; j < nc; ++j) {
        T* x = Xd + j * ldX;
        for (int r = n - 1; r > 0; --r) {
            x[r] /= Ad[r + r * ldA];
            __mixed(x, Ad + r * ldA, -x[r], x, r);
        }
        x[0] /= Ad[0];
    }

    return b;
}


template<typename T>
MAT<T> Solve_direct(MAT<T> A, MAT<T>& B) {
    MAT<T> b(B.rows, B.cols);
    int i, j;

    if (B.cols <= 4) {
        return Solve_direct_narrow(A, B);
    }

    constexpr int K = __BlockSize_LU;
    const int s = (A.rows > A.cols) ? A.cols : A.rows;

    int* pos = new int[B.rows];
    for (i = 0; i < A.rows; ++i) {
        pos[i] = i;
    }

    T* Temp1, * Temp2;

    if (s <= K) {
        __BlockMat<T> _L, _U, _X;

        _L.set(A, 0, 0, A.rows, s);
        _U.set(A, 0, 0, A.rows, s);
        __PLU2(pos, _L, _U);

        for (j = 0; j < B.cols; ++j) {
            Temp1 = b[j];
            Temp2 = B[j];
            for (i = 0; i < B.rows; ++i) {
                Temp1[i] = Temp2[pos[i]];
            }
        }
        delete[] pos;

        _L.set(A, 0, 0, s, s);
        _U.set(A, 0, 0, s, s);
        _X.set(b, 0, 0, b.rows, b.cols);
        __Solve_direct_LX(_L, _X);
        __Solve_direct_UX(_U, _X);

        return b;
    }

    constexpr int NC = __BlockSize_Mul_NC;
    constexpr int MR = __micro_kernel_mr<T, T, T>();
    constexpr int NR = __micro_kernel_nr<T, T, T>();

    const size_t a_pack_size = (size_t)((A.rows + MR - 1) / MR) * K * MR;
    const size_t b_pack_size = (size_t)((NC + NR - 1) / NR) * K * NR;

    T* A_pack = (T*)_mm_malloc(sizeof(T) * a_pack_size, 64);
    T* B_pack = (T*)_mm_malloc(sizeof(T) * b_pack_size, 64);

    __PLU(pos, A, A, A_pack, B_pack);

    for (j = 0; j < B.cols; ++j) {
        Temp1 = b[j];
        Temp2 = B[j];
        for (i = 0; i < B.rows; ++i) {
            Temp1[i] = Temp2[pos[i]];
        }
    }
    delete[] pos;

    __Solve_direct_LU_X(A, b, A_pack, B_pack);

    _mm_free(A_pack);
    _mm_free(B_pack);

    return b;
}
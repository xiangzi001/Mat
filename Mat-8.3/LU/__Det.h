template <typename T>
void __Det(T& ret, __BlockMat<T>& A, int cL, int cU) {
    int r, i;
    T* Temp;
    for (r = 0; r < A.cols; ++r) {
        Temp = A[r];
        for (i = 0; i < r; ++i) {
            __mixed(Temp + i + 1, A[i] + i + 1, -Temp[i], Temp + i + 1, A.rows - i - 1);
        }

        int j = r;
        auto temp1 = abs(A(r, r)), temp2 = temp1;
        for (i = r + 1; i < A.rows; ++i) {
            temp2 = abs(A(i, r));
            if (temp2 > temp1) {
                temp1 = temp2;
                j = i;
            }
        }
        if (j != r) {
            A.swap_rows(r, j, cL, cU);
            ret = -ret;
        }

        __times(Temp + r + 1, (T)1 / Temp[r], A[r] + r + 1, A.rows - r - 1);
    }
}template<typename T>
T Det(MAT<T> A) {
    T ret = (T)1;
    constexpr int K = __BlockSize_LU;
    const int s = (A.rows > A.cols) ? A.cols : A.rows;

    if (s <= K) {
        __BlockMat<T> _A;
        _A.set(A, 0, 0, A.rows, s);
        __Det(ret, _A, 0, A.cols);
        for (int i = 0; i < s; ++i) {
            ret = ret * A(i, i);
        }
        return ret;
    }

    int r, i, j, m, n;
    __BlockMat<T> _L, _U, _A;

    constexpr int MC = __BlockSize_Mul_MC;
    constexpr int NC = __BlockSize_Mul_NC;
    constexpr int MR = __micro_kernel_mr<T, T, T>();
    constexpr int NR = __micro_kernel_nr<T, T, T>();

    const size_t a_pack_size = (size_t)((A.rows + MR - 1) / MR) * K * MR;
    const size_t b_pack_size = (size_t)((NC + NR - 1) / NR) * K * NR;

    T* A_pack = (T*)_mm_malloc(sizeof(T) * a_pack_size, 64);
    T* B_pack = (T*)_mm_malloc(sizeof(T) * b_pack_size, 64);

    for (r = 0; r + K <= A.cols; r += K) {
        _A.set(A, r, r, A.rows - r, K);
        __Det(ret, _A, -r, A.cols - r);

        if (r + K < A.cols) {
            _L.set(A, r + K, r, A.rows - (r + K), K);
            __pack_A_block<T, T>(_L.data, _L._rows, A.rows - (r + K), K, A_pack);

            for (j = r + K; j < A.cols; j += NC) {
                n = (j + NC > A.cols) ? (A.cols - j) : NC;

                _L.set(A, r, r, A.rows - r, K);
                _U.set(A, r, j, K, n);
                __Solve_direct_LX(_L, _U);
                __pack_B_block<T, T>(_U[0], _U._rows, K, n, B_pack);

                for (i = r + K; i < A.rows; i += MC) {
                    m = (i + MC > A.rows) ? (A.rows - i) : MC;
                    _A.set(A, i, j, m, n);
                    constexpr int MR = __micro_kernel_mr<T, T, T>();
                    const T* A_base = A_pack + (size_t)((i - (r + K)) / MR) * K * MR;
                    __mulAB_packed_n<T>(A_base, B_pack, _A.data, _A._rows,
                        _A.rows, _A.cols, K);
                }
            }
        }
    }

    if (r < A.cols) {
        int rem = A.cols - r;
        _A.set(A, r, r, A.rows - r, rem);
        __Det(ret, _A, -r, A.cols - r);
    }

    for (r = 0; r < A.cols; ++r) {
        ret = ret * A(r, r);
    }

    _mm_free(A_pack);
    _mm_free(B_pack);
    return ret;
}


template<typename T, typename t>
inline void __copy(T* dst, const t* src, int len) {
	for (int i = 0; i < len; ++i) {
		dst[i] = (T)(src[i]);
	}
}
template<typename T>
inline void __copy(T* dst, const T* src, int len) {
	memcpy(dst, src, sizeof(T) * len);
}

template<typename T>
inline void __set0(T* dst, int len) {
	for (int i = 0; i < len; ++i) {
		dst[i] = T(0);
	}
}
inline void __set0(int* dst, int len) {
	memset(dst, 0, sizeof(int) * len);
}
inline void __set0(long long* dst, long long len) {
	memset(dst, 0, sizeof(long long) * len);
}
inline void __set0(float* dst, float len) {
	memset(dst, 0, sizeof(float) * len);
}
inline void __set0(double* dst, double len) {
	memset(dst, 0, sizeof(double) * len);
}

#include "__Matrix.h"
#include "__Vector.h"
#include "__Block.h"

#include "__Calculate.h"
#include "__times.h"

#include "__Norm.h"
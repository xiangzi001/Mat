

template<typename T, typename t>
inline void __copy(T* dst, const t* src, int len) {
	for (int i = 0; i < len; ++i) {
		dst[i] = src[i];
	}
}
template<typename T>
inline void __copy(T* dst, const T* src, int len) {
	memcpy(dst, src, sizeof(T) * len);
}

template<typename T>
inline void __set0(T* dst, int len) {
	memset(dst, 0, sizeof(T) * len);
}


#include "__Complex.h"

#include "__Matrix.h"
#include "__Vector.h"
#include "__Block.h"

#include "__Calculate.h"
#include "__times.h"

#include "__Norm.h"
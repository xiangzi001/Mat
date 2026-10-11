
//constexpr

template<typename T>
VEC<T> Rotate(const VEC<T>& v, const VEC<T>& n, double angle) {

	T len2 = n.data[0] * n.data[0] + n.data[1] * n.data[1] + n.data[2] * n.data[2];
	T c = cos(angle);
	T s = sin(angle) / sqrt(len2);
	T temp = (1 - c) * Dot(n, v) / len2;

	VEC<T> ret(3);
	ret[0] = v.data[0] * c + n.data[0] * temp + (n.data[1] * v.data[2] - n.data[2] * v.data[1]) * s;
	ret[1] = v.data[1] * c + n.data[1] * temp + (n.data[2] * v.data[0] - n.data[0] * v.data[2]) * s;
	ret[2] = v.data[2] * c + n.data[2] * temp + (n.data[0] * v.data[1] - n.data[1] * v.data[0]) * s;

	return ret;
}

template<typename T>
MAT<T> Rotate(const VEC<T>& n, double angle) {
	T len2 = n.data[0] * n.data[0] + n.data[1] * n.data[1] + n.data[2] * n.data[2];

	T c = cos(angle);
	T c1 = (1 - c) / len2;
	T s = sin(angle) / sqrt(len2);

	MAT<T> ret(3, 3);

	ret[0][0] = c + c1 * n.data[0] * n.data[0];
	ret[0][1] = c1 * n.data[0] * n.data[1] + s * n.data[2];
	ret[0][2] = c1 * n.data[0] * n.data[2] - s * n.data[1];

	ret[1][0] = c1 * n.data[0] * n.data[1] - s * n.data[2];
	ret[1][1] = c + c1 * n.data[1] * n.data[1];
	ret[1][2] = c1 * n.data[1] * n.data[2] + s * n.data[0];

	ret[2][0] = c1 * n.data[0] * n.data[2] + s * n.data[1];
	ret[2][1] = c1 * n.data[1] * n.data[2] - s * n.data[0];
	ret[2][2] = c + c1 * n.data[2] * n.data[2];

	return ret;
}
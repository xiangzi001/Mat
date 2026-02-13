

template<typename t>
t Dot(const VEC<t>& x, const VEC<t>& y) {
	t ret(0);

	for (int i = 0; i < x.rows; ++i) {
		ret += x.data[i] * y.data[i];
	}

	return ret;
}

template<typename t>
t Angle(const VEC<t>& x, const VEC<t>& y) {
	t d(0), lx(0), ly(0);
	int i;

	for (i = 0; i < x.rows; ++i) {
		d += x.data[i] * y.data[i];
	}
	for (i = 0; i < x.rows; ++i) {
		lx += x.data[i] * x.data[i];
	}
	for (i = 0; i < y.rows; ++i) {
		ly += y.data[i] * y.data[i];
	}

	return acos(d / sqrt(lx * ly));
}

template<typename T>
VEC<T> Cross(const VEC<T>& x, const VEC<T>& y) {
	VEC<T> ret(3);

	ret[0] = x.data[1] * y.data[2] - x.data[2] * y.data[1];
	ret[1] = x.data[2] * y.data[0] - x.data[0] * y.data[2];
	ret[2] = x.data[0] * y.data[1] - x.data[1] * y.data[0];

	return ret;
}

template<typename T>
MAT<T> Cross(const VEC<T>& x) {
	MAT<T> ret(3, 3);

	ret[0][0] = 0; ret[1][0] = -x.data[2]; ret[2][0] = x.data[1];
	ret[0][1] = x.data[2]; ret[1][1] = 0; ret[2][1] = -x.data[0];
	ret[0][2] = -x.data[1]; ret[1][2] = x.data[0]; ret[2][2] = 0;

	return ret;
}
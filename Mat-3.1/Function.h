
template<class T>
class Func {
public:
	T* data;
	int times;
	Func() :
		data(0),
		times(0)
	{
	}
	Func(int t) :
		data(new T[t + 1]),
		times(t)
	{
	}
	Func(int t, T* d) :
		data(new T[t + 1]),
		times(t)
	{
		for (int i = 0; i <= t; ++i) {
			data[i] = d[i];
		}
	}
	Func(int t, T value, ...) :
		data(new T[t + 1]),
		times(t)
	{
		va_list num;
		va_start(num, value);
		data[0] = value;
		for (int i = 1; i <= t; ++i) {
			data[i] = va_arg(num, T);
		}
		va_end(num);
	}
	~Func() {
		delete[] data;
	}

	template<class _T>
	void set(int t) {
		delete[] data;
		data = new T[t + 1];
		times = t;
	}

	template<class _T>
	void set(int t, _T* d) {
		delete[] data;
		data = new T[t + 1];
		times = t;

		for (int i = 0; i <= t; ++i) {
			data[i] = d[i];
		}
	}

	void set(int t, T value, ...)
	{
		delete[] data;
		data = new T[t + 1];
		times = t;

		va_list num;
		va_start(num, value);
		data[0] = value;
		for (int i = 1; i <= t; ++i) {
			data[i] = va_arg(num, T);
		}
		va_end(num);
	}

	T operator [] (int n) {
		return data[n];
	}
	template<class t>
	t operator () (t x) {
		t ret(0), temp(1);
		for (int i = 0; i < times; ++i) {
			ret = ret + data[i] * temp;
			temp *= x;
		}
		ret = ret + data[times] * temp;

		return ret;
	}
	template<class _T>
	Func operator * (Func<_T> b) {
		Func ret(times + b.times);

		int i, j;
		for (i = 0; i <= ret.times; ++i) {
			ret.data[i] = 0;
		}
		for (i = 0; i <= times; ++i) {
			for (j = 0; j <= b.times; ++j) {
				ret.data[i + j] += data[i] * b.data[j];
			}
		}

		b.data = 0;
		return ret;
	}
	template<class _T>
	Func operator + (Func<_T> b) {
		if (times < b.times) {
			Func ret(b.times);

			int i(0);
			for (; i <= times; ++i) {
				ret.data[i] = data[i] + b.data[i];
			}
			for (; i <= b.times; ++i) {
				ret.data[i] = b.data[i];
			}

			b.data = 0;
			return ret;
		}
		else {
			Func ret(times);

			int i(0);
			for (; i <= b.times; ++i) {
				ret.data[i] = data[i] + b.data[i];
			}
			for (; i <= times; ++i) {
				ret.data[i] = data[i];
			}

			b.data = 0;
			return ret;
		}
	}

	template<class _T>
	void operator = (Func<_T> b) {
		delete[] data;
		times = b.times;
		data = new T[times + 1];

		for (int i = 0; i <= times; ++i) {
			data[i] = b.data[i];
		}
		b.data = 0;
	}
};

template<class T> 
Func<T> func(int t, T value, ...){
	Func<T> ret(t);

	va_list num;
	va_start(num, value);
	ret.data[0] = value;
	for (int i = 1; i <= t; ++i) {
		ret.data[i] = va_arg(num, T);
	}
	va_end(num);

	return ret;
}



#include <functional>
#include <unordered_map>
#include <unordered_set>

struct vec3i {
	vec3i() {
		x = 0;
		y = 0;
		z = 0;
	}
	vec3i(int x_, int y_, int z_) {
		x = x_;
		y = y_;
		z = z_;
	}
	inline vec3i operator+(const vec3i& other) const {
		return vec3i(x+other.x, y+other.y, z+other.z);
	}
	inline vec3i operator-(const vec3i& other) const {
		return vec3i(x-other.x, y-other.y, z-other.z);
	}
	inline bool operator==(const vec3i& other) const {
		if ((x == other.x) && (y == other.y) && (z == other.z))
			return true;
		return false;
	}
	int x;
	int y;
	int z;
};

namespace std {

template <>
struct hash<vec3i> : public std::unary_function<vec3i, size_t> {
	size_t operator()(const vec3i& v) const {
		//TODO larger prime number (64 bit) to match size_t
		const size_t p0 = 73856093;
		const size_t p1 = 19349669;
		const size_t p2 = 83492791;
		const size_t res = ((size_t)v.x * p0)^((size_t)v.y * p1)^((size_t)v.z * p2);
		return res;
	}
};

}

template<class T>
class SparseGrid3 {
public:
	typedef typename std::unordered_map<vec3i, T, std::hash<vec3i>>::iterator iterator;
	typedef typename std::unordered_map<vec3i, T, std::hash<vec3i>>::const_iterator const_iterator;
	iterator begin() {return m_Data.begin();}
	iterator end() {return m_Data.end();}
	const_iterator begin() const {return m_Data.begin();}
	const_iterator end() const {return m_Data.end();}
	
	SparseGrid3(float maxLoadFactor = 0.6, size_t reserveBuckets = 64) {
		m_Data.reserve(reserveBuckets);
		m_Data.max_load_factor(maxLoadFactor);
	}

  size_t size() const {
    return m_Data.size();
  }

	void clear() {
		m_Data.clear();
	}

	bool exists(const vec3i& i) const {
		return (m_Data.find(i) != m_Data.end());
	}

	bool exists(int x, int y, int z) const {
		return exists(vec3i(x, y, z));
	}

	const T& operator()(const vec3i& i) const {
		return m_Data.find(i)->second;
	}

	//! if the element does not exist, it will be created with its default constructor
	T& operator()(const vec3i& i) {
		return m_Data[i];
	}

	const T& operator()(int x, int y, int z) const {
		return (*this)(vec3i(x,y,z));
	}
	T& operator()(int x, int y, int z) {
		return (*this)(vec3i(x,y,z));
	}

	const T& operator[](const vec3i& i) const {
		return (*this)(i);
	}
	T& operator[](const vec3i& i) {
		return (*this)(i);
	}

protected:
	std::unordered_map<vec3i, T, std::hash<vec3i>> m_Data;
};


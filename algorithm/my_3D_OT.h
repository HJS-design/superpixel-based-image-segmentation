#pragma once








class MyOTClass_empty
{
public:
	MyOTClass_empty() {
		return;
	};
	~MyOTClass_empty() { return; };
	inline double variance(int i, int j) { return 0; };
	inline double distance(std::vector<unsigned int>& vec1, std::vector<unsigned int>& vec2) {
		return 0;
	};
	inline double distance(int i, int j) { return 0; };
	inline double variance(std::vector<unsigned int>& vec1) { return 0; };
	inline void merge(int i, int j) {return ; };
	inline double variance(int i) { return 0; };

};


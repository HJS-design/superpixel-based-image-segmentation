#pragma once

#include "pch.hpp"
using namespace cv;




//#define outputfile

#ifdef outputfile
std::ofstream ff(R"(aa.txt)");
#endif // outputfile



class My_3D_OT_for_variance
{
public:

	
	My_3D_OT_for_variance(Mat& squared_each_cell, Mat& sum_each_cell, std::vector<double>& area_each_cell);
	~My_3D_OT_for_variance();
	inline double variance(int i, int j);
	inline double variance(int i);
	inline double distance(int i, int j);
	inline void merge(int i, int j);
	
	Mat squared_each_cell;
	Mat sum_each_cell;
	std::vector<double> area_each_cell;

};



My_3D_OT_for_variance::~My_3D_OT_for_variance()
{
}


inline void My_3D_OT_for_variance::merge(int i, int j)
{
	area_each_cell[i] += area_each_cell[j];
	squared_each_cell.row(i) += squared_each_cell.row(j);
	sum_each_cell.row(i) += sum_each_cell.row(j);

}




inline My_3D_OT_for_variance::My_3D_OT_for_variance(Mat& sum_each_cell_out, Mat& squared_each_cell_out, std::vector<double>& area_each_cell_out)
{
	
	squared_each_cell = squared_each_cell_out;
	sum_each_cell = sum_each_cell_out;
	area_each_cell = std::move(area_each_cell_out);
	assert(area_each_cell.size() == sum_each_cell.rows);
	assert(sum_each_cell.cols == squared_each_cell.cols);
	assert(std::ranges::fold_left(area_each_cell | std::views::transform([](double a) {return a >= 0; }), true, std::logical_and()));

	//assert(cv::norm(sum_each_cell.mul(sum_each_cell) - squared_each_cell, cv::NORM_INF) <= 1e-5);

}



inline double My_3D_OT_for_variance::variance(int i, int j)
{
	
	double d1 = 0, d2 = 0, total_area = 0, * p;
	total_area = (area_each_cell[i] + area_each_cell[j]);

	p = sum_each_cell.ptr<double>(i);
	d2 = std::transform_reduce(p, p + sum_each_cell.cols, sum_each_cell.ptr<double>(j), 0.0, std::plus<double>(), [&d1](double a, double b) { d1 = a + b; return d1 * d1; });
	assert(std::abs(d2- std::transform_reduce(p, p + sum_each_cell.cols, sum_each_cell.ptr<double>(j), 0.0, std::plus<double>(), [](double a, double b) { return (a+b) * (a+b); }))<=1e-8);
	d2 /= (total_area * total_area);
#ifdef outputfile
	ff << sum_each_cell.row(i) << sum_each_cell.row(j) << "\n";
	ff << squared_each_cell.row(i) << squared_each_cell.row(j) << "\n--------\n";
#endif // outputfile

	

	p = squared_each_cell.ptr<double>(i);
	d1 = std::transform_reduce(p, p + squared_each_cell.cols, squared_each_cell.ptr<double>(j), 0.0, std::plus<double>(), std::plus<double>());
	d1 /= total_area;
	
	
	d1 -= d2;
	assert(d1 + 1e-8 >= 0);
	return d1;

	




}

inline double My_3D_OT_for_variance::variance(int i)
{
	double d1 = 0, d2 = 0, total_area = 0,*p;
		
		total_area = area_each_cell[i];
		p = squared_each_cell.ptr<double>(i);
		d1 = std::reduce(p,p+ squared_each_cell.cols)/ total_area;
		p = sum_each_cell.ptr<double>(i);
		d2 = std::transform_reduce(p, p + sum_each_cell.cols, 0.0, std::plus<double>(), [](double a) {return a * a; });
		d2 /= (total_area * total_area);
		d1 -= d2;
		assert(d1 + 1e-8 >= 0);
		return d1;
	
}

inline double My_3D_OT_for_variance::distance(int i, int j)
{
	return variance(i, j);
}





#include "mex.h" /* Always include this */
#include "mat.h"
#include "matrix.h"
#include <cfloat>
#include <cmath> 
#include <cstdio>
#include <filesystem>
#include <type_traits>
#include <unordered_set>
#include <valarray>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <format>
#include <numbers>
#include "cstring"
#include "tools.h"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "emd.h"

//#define debug_flag
//#define only_use_CV_regional_term
//#define use_linear_OT_functional
//#define use_OT_weight_in_functional
//#define edge_linear

//#define interval_based_on_variance
//#define no_image_boundary_edge_term

//#define 3D_OT
//#define add_interval_penalty_to_functional_no_area
#define add_interval_penalty_to_functional_with_area
#define use_squared_ot
#define interval_based_on_minimum
//#define use_pure_OT_no_functional
using namespace std::views;
using namespace std::ranges;
using namespace cv;
//double  threshold_area_out;
//num of target (merged) superpixel
double expect_area_num_out;
//: region term + edge_lambda_out* edge term
double edge_lambda_out;
int channel_out;
int image_channel_out;
int marker_channel_out;
double left_bins = 0;
double right_bins = 0;
//std::ofstream output_file(R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\h&e_image\aa.txt)", std::ofstream::app);
inline std::pair<double, double> get_mean_variance_from_pdf(std::array<double, 256>& bins) {
	double mean = 0, variance = 0, d = 0;
	assert(std::abs(std::reduce(bins.begin(), bins.end()) - 1) < 1e-8);
	for (auto [x, dy] : enumerate(bins)) {
		d = x * dy;
		mean += d;
		variance += d * x;
	}
	variance = variance - mean * mean;
	assert(mean >= 0 & mean <= 255);
	assert(variance >= 0);
	return { mean,variance };

}
inline double OT_dis_from_bins_to_linear(std::array<double, 256>& bins, double b) {
	// the inverse cdf is a*y+b;
	//below is OT
	/*fmt::print("b: {} a+b: {}\n",b,a+b);*/
	/*assert(b >= 0 && a + b <= 255);*/
	assert(std::abs(std::reduce(bins.begin(), bins.end()) - 1) < 1e-8);
	/*assert(a >0);*/
	double i1;
	double ot = 0;
	//i1 is the final OT

	double* p = bins.data();
	for (int x = 0; x < 256; x++) {
		i1 = b - x;
		ot += i1 * i1 * p[x];

	}

	/*fmt::print("normal {}", ot);*/

#if defined(use_squared_ot)
	return ot;
#else // (use_squared_ot)
	return sqrt(ot);
#endif

	
}
inline double OT_dis_from_bins_to_linear(std::array<double, 256>& bins, double a, double b) {
	// the inverse cdf is a*y+b;
	//below is OT
	/*fmt::print("b: {} a+b: {}\n",b,a+b);*/
	/*assert(b >= 0 && a + b <= 255);*/
	assert(std::abs(std::reduce(bins.begin(), bins.end()) - 1) < 1e-8);
	assert(a > 0);
	double i1, i2;
	double ot = 0;
	//i1 is the final OT
	double y = 0;
	double* p = bins.data();
	for (int x = 0; x < 256; x++) {

		i1 = a * y + b - x;
		y += p[x];
		i2 = a * y + b - x;
		ot += (i2 * i2 * i2 - i1 * i1 * i1) / (3 * a);
	}

	/*fmt::print("normal {}", ot);*/
	assert(std::abs(y - 1) < 1e-10);


#if defined(use_squared_ot)
	return ot;
#else // (use_squared_ot)
	return sqrt(ot);
#endif
}

inline std::pair<double,double> dis_to_closet_uniform_pdf(std::array<double, 256>& bins,double mu) {


	double i1 = 0, i2 = 0, a = 0, b = 0, y = 0, dy, dy2;
	assert(std::abs(std::reduce(bins.begin(), bins.end()) - 1) < 1e-8);




	



	y = 0;
	/*fmt::print("left {}, right {}\n", a,b);*/
	for (int x = 0; x < 256; x++) {
		dy2 = y;
		y += (dy = bins[x]);
		i1 += x * dy;
		dy2 = y * y - dy2 * dy2;
		i2 += x * dy2;
	}
	i2 /= 2;

	// mid= x;
	a =  - 6 * i1 + 12 * i2;
	a /= (1 + 12*mu);
	b = i1 - a / 2;
	/*fmt::println("c1 {} c_variance {}",a,2*sqrt(get_mean_variance_from_pdf(bins).second));*/

	assert(std::abs(a / 2 + b - i1)<1e-8);
	assert(a >= 0);

	//---------------------
	/*fmt::print("i1 {}, i2 {}\n",i1,i2);*/
	//i1 = std::max(b, 0.0);
	//i2 = std::min(255.0, a + b);
	//b = i1;
	//a = i2 - b;



	/*fmt::println("i1 {}, i2 {},  a {} b {}\n",i1,i2, a,b);*/
	//-----------------------
	/*st = std::round(b);
	end = std::round(a+b);*/
	y = (a < 0.001) ? OT_dis_from_bins_to_linear(bins, b) : OT_dis_from_bins_to_linear(bins, a, b);
#ifdef add_interval_penalty_to_functional_no_area
	return {  y,a };
#else
	return { y+mu*a*a ,-1 };
#endif // add_interval_penalty_to_functional_no_area

	


	

}

struct log_file
{
	std::forward_list<double> region;
	std::forward_list<double> edge;
	std::string debug_file{ R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\h&e_image)" };
}log_file;
struct MyStruct
{
	double similar;
	unsigned int x, y;
	unsigned int record_size_x, record_size_y;
};
std::function<bool(MyStruct&, MyStruct&)> cmp = [](MyStruct& left, MyStruct& right) { return left.similar > right.similar; };


struct dsu {
	std::priority_queue<MyStruct, std::vector<MyStruct>, std::function<bool(MyStruct&, MyStruct&)>> Q;
	std::vector<unsigned int> pa;
	std::vector<unsigned int> size;
	//size: number of cells (not area) of each set.
	//  We use it to obtain the variance E(x^2)-(E(x))^2
	std::vector<std::array<double, 3>>& cell_feature;
	//length, area, x^2
	//intensity_sum is composed of the information of the image and the distance array. 
	//The channel number (the size of std::vector<double>) is (image channel) + (distance function channel) 
	std::vector<cv::Mat>& label_2_o1;
	std::vector<cv::Mat> label_2_o1_sum_in_each_ehannel;
	std::vector<unsigned char>& label_2_marker;
	// marker_center is of size (marker number) * (distance function channel) 
	std::vector<cv::Mat>& marker_2_center;
	std::vector<unsigned char> region_near_boundary;
	//x in all channels;
	//double is the probability of this edge.
	std::vector<std::unordered_map<unsigned int, double>> edge_each_cell;
	//-----------------------
	/*std::forward_list<double> incre;*/
	//---------------------
	std::array<double, 256> store_bins;
	std::vector<std::array<double, 3>> x1_array;
	std::vector<std::array<double, 3>> x2_array;
	std::vector<std::array<double, 3>>  x_array;

	//double threshold_area = threshold_area_out;
	const double total_area;
	const unsigned int expect_cell_num = expect_area_num_out;
	const double edge_lambda = edge_lambda_out;
	const int channel = channel_out;
	//in total length, all edges are counted twice and the image bounday is counted once. 
	double total_length;

	unsigned int now_cell_num;

	//--------------------------
	//size_ is the number of regions rather than the size of images.
	explicit dsu(unsigned int size_, std::vector<std::array<double, 3>>& cell_feature_out, std::vector<cv::Mat>& intensity_sum_out, std::vector<std::tuple<unsigned int, unsigned int, double>>& adjacent_, std::vector<unsigned char>& marker_each_ccl_out, std::vector<cv::Mat>& marker_center_out,std::vector<unsigned char> & region_near_boundary_out) :
		pa(size_),
		label_2_o1(intensity_sum_out),
		size(size_, 1),
		edge_each_cell(size_),
		cell_feature(cell_feature_out),
		Q(cmp),
		now_cell_num(size_),
		total_area(fold_left(elements<1>(cell_feature), 0, std::plus<double>())),
		label_2_marker(marker_each_ccl_out),
		marker_2_center(marker_center_out),
		x1_array(256),
		x2_array(256),
		x_array(512),
		region_near_boundary(region_near_boundary_out)
	{
		total_length = fold_left(elements<0>(cell_feature), 0, std::plus<double>());
		std::ranges::iota(pa, 0);
		for (auto [i, all_o1] : enumerate(label_2_o1)) {
			label_2_o1_sum_in_each_ehannel.emplace_back(Mat(1, 256, CV_64FC1));
			auto& aa = label_2_o1_sum_in_each_ehannel.back();
			cv::reduce(all_o1, aa, 0, cv::REDUCE_SUM);
			assert(aa.total() == 256);
		}
		unsigned char mi, mj;
		for (auto [i, j, d] : adjacent_) {
			mi = label_2_marker[i];
			mj = label_2_marker[j];
			edge_each_cell[i][j] = d;
			edge_each_cell[j][i] = d;
			if (mi == mj || !mi || !mj) {
				Q.emplace(MyStruct{ total_energy(i,j),i,j,1,1 });
			}
		}

	}
	template<int caseid, typename mytype>
	inline void  pdf_2_cdf(mytype&& x, std::vector<std::array<double, 3Ui64>>& xi_array)
	{
		xi_array.clear();
		double s = cv::sum(x)[0], sum = 0;
		for (auto [id, a] : views::counted((double*)x.data, 256) | views::enumerate) {
			if (a > 0) {
				if constexpr (caseid == 1)
					xi_array.emplace_back(std::array<double, 3>{ (sum += a) / s, double(id), 300 });
				else
					xi_array.emplace_back(std::array<double, 3>{ (sum += a) / s, 300, double(id) });
			}
		}
	}

	inline void get_prefix_sum_uniform(std::vector<std::array<double, 3Ui64>>& xi_array, double start, double end) {
		xi_array.clear();
		double step = 1 / (end - start + 1), sum = 0;
		for (int id = start; id <= end; id++) {
			xi_array.emplace_back(std::array<double, 3>{ sum += step, 300, double(id) });
		}

		assert(fabs(xi_array.back()[0] - 1) < 1e-7);

	}
	inline void get_end_points(std::vector<std::array<double, 3Ui64>>& xi_array, double& st, double& end) {




		auto it = std::find_if(xi_array.begin(), xi_array.end(), [](std::array<double, 3Ui64>& v) { return v[0] > left_bins; });
		st = (*it)[1];

		auto debug_1 = [](std::vector<std::array<double, 3Ui64>>& v, int id, double thre) {
			bool flag = true;
			flag &= v[id][0] > thre;
			if (id > 0)
				flag &= v[id - 1][0] <= thre;
			return flag;
			};
		assert(debug_1(xi_array, std::distance(xi_array.begin(), it), left_bins));

		it = std::ranges::find_if(it, xi_array.end(), [](std::array<double, 3Ui64>& v) { return v[0] >= right_bins; });
		end = (*it)[1];

		auto debug_2 = [](std::vector<std::array<double, 3Ui64>>& v, int id, double thre) {
			bool flag = true;
			flag &= v[id][0] >= thre;
			if (id > 0)
				flag &= v[id - 1][0] < thre;
			return flag;
			};
		assert(debug_2(xi_array, std::distance(xi_array.begin(), it), right_bins));



	}
	inline std::array<double, 2> get_variance_from_bins(Mat& x) {
		//x is CV_64FC1
		auto aa = views::counted((double*)x.data, 256) | views::enumerate | views::filter([](auto a) {auto [v, count] = a; return count > 0; });

		double x2 = 0, x1 = 0, area = 0;
		for (auto [v, count] : aa) {
			x2 += v * v * count;
			x1 += v * count;
			area += count;
		}
		x1 /= area;
		x2 /= area;
		return { x1, x2 - x1 * x1 };
	}

	inline double get_OT_DIS(std::vector<std::array<double, 3Ui64>>& X_array)
	{
		double sum = 0;
		double d = 0, last_y = 0;
		for (auto& it : X_array) {
			auto& [y, x0, x1] = it;
			d = x0 - x1;
			//fmt::print("x:{} y:{} p:{} last_p:{} con:{}\n", x0, x1, y, last_y, d * d * (y - last_y));
			sum += d * d * (y - last_y);
			last_y = y;
		}
		/*return sum;*/
#if defined(use_squared_ot)
		return sum;
#else // (use_squared_ot)
		return sqrt(sum);
#endif
		/*return std::pow(sum, 0.25);*/
		/*return std::cbrt(sum);*/
	}

	template<typename mytype>
	inline std::pair<double,double> getvariance(mytype&& bins,double mu=left_bins) {
		//type is Mat
		assert(bins.total() == 256);
#if defined(only_use_CV_regional_term)
		double s = cv::sum(bins)[0], * p = (double*)bins.data;
		for (int i = 0; i < 256; i++) {
			store_bins[i] = p[i] / s;
		}
		auto [mean, variance] = get_mean_variance_from_pdf(store_bins);
		return { variance ,-1};
#elif defined(interval_based_on_minimum)&&defined(interval_based_on_variance)
		double s = cv::sum(bins)[0], * p = (double*)bins.data;
		for (int i = 0; i < 256; i++) {
			store_bins[i] = p[i] / s;
		}
		auto [mean, variance] = get_mean_variance_from_pdf(store_bins);
		variance = sqrt(variance);
		double st = mean - mu * variance;
		double end = mean + mu * variance;
		for (int i = 0; i < st; i++) store_bins[i] = 0;
		for (int i = std::ceil(end); i < 256; i++) store_bins[i] = 0;
		s = std::reduce(store_bins.begin(), store_bins.end());
		for (double& d : store_bins) d /= s;
		return dis_to_closet_uniform_pdf(store_bins, mu);
#elif defined(interval_based_on_minimum)
		double s = cv::sum(bins)[0], * p = (double*)bins.data;
		for (int i = 0; i < 256; i++) {
			store_bins[i] = p[i] / s;
		}

		return dis_to_closet_uniform_pdf(store_bins, mu);

		
#elif defined(interval_based_on_variance)
		double s = cv::sum(bins)[0], * p = (double*)bins.data;
		for (int i = 0; i < 256; i++) {
			store_bins[i] = p[i] / s;
		}
		auto [mean, variance] = get_mean_variance_from_pdf(store_bins);
		variance = sqrt(variance);
		double st = mean - mu * variance;
		double end = mean + mu * variance;
		/*fmt::print("mean {} end {} st {} end {}\n",mean,variance, st,end);*/
		double v1 = end - st < 1e-3 ? OT_dis_from_bins_to_linear(store_bins, st) : OT_dis_from_bins_to_linear(store_bins, end - st, st);
		assert(v1 > 0);
		return { v1 ,-1};
#else
		// bins are CV64FC1.





		pdf_2_cdf<1>(bins, x1_array);


		

		double st;
		double end;

		/*st = std::max(std::round(mm - h), double(0));
		end = std::min(std::round(mm + h), double(255));*/



		get_end_points(x1_array, st, end);

		get_prefix_sum_uniform(x2_array, st, end);


		/*printf("x1_array\n");
		for (auto& bb : x1_array)
			print("{} {} {}\n", bb[0], bb[1], bb[2]);
		printf("x2_array\n");
		for (auto& bb : x2_array)
			print("{} {} {}\n", bb[0], bb[1], bb[2]);*/

		x_array.resize(x1_array.size() + x2_array.size());
		std::ranges::merge(x1_array, x2_array, x_array.data(), [](std::array<double, 3>& a, std::array<double, 3>& b) ->bool {return a[0] < b[0]; });




		//fmt::print("x_array before\n {}\n----------\n", x_array);




		double x0 = x1_array.back()[1];
		double x1 = x2_array.back()[2];
		for (auto& [y, tx0, tx1] : x_array | views::reverse) {
			x0 = tx0 = std::min(x0, tx0);
			x1 = tx1 = std::min(x1, tx1);
		}
		//fmt::print("x_array after\n {}\n----------\n", x_array);


		/*auto re = ranges::unique(x_array, [](std::array<double, 3>& a, std::array<double, 3>& b) {return a[0] == b[0]; });
		x_array.erase(re.begin(), re.end());*/

		/*fmt::print("x_array_unique\n {}\n", x_array);*/

		assert(st <= end);

		double s = cv::sum(bins)[0], * p = (double*)bins.data;
		for (int i = 0; i < 256; i++) {
			store_bins[i] = p[i] / s;
		}
		double v1 = end - st < 1e-3 ? OT_dis_from_bins_to_linear(store_bins, st) : OT_dis_from_bins_to_linear(store_bins, end - st, st);

		double v2 = get_OT_DIS(x_array);
		assert(abs(v1 - v2) < 1);
		/*if(std::abs(end-st)<1)*/
			/*fmt::println("st {} end {} v1 {} v2 {}",st,end, v1, v2);*/
#ifdef use_linear_OT_functional
		return { v1 ,-1};
#else
		return { v2 ,-1};
#endif // use_bins_OT_functional
#endif // defined(interval_based_on_minimum)&&defined(interval_based_on_variance)





	}


	inline double displace_variance(double area, Mat&& o1, Mat&& marker) {
		assert(o1.total() == marker_channel_out && marker.total() == marker_channel_out);
		auto t = marker * area - o1;
		return t.dot(t) / area;
	}
	template<typename mytype1, typename mytype2>
	inline double dis_between_two_bins(mytype1&& bins1, mytype2&& bins2) {
		pdf_2_cdf<1>(bins1, x1_array);
		pdf_2_cdf<2>(bins2, x2_array);

		x_array.resize(x1_array.size() + x2_array.size());
		std::ranges::merge(x1_array, x2_array, x_array.data(), [](std::array<double, 3>& a, std::array<double, 3>& b) ->bool {return a[0] < b[0]; });




		//fmt::print("x_array before\n {}\n----------\n", x_array);




		double x0 = x1_array.back()[1];
		double x1 = x2_array.back()[2];
		for (auto& [y, tx0, tx1] : x_array | views::reverse) {
			x0 = tx0 = std::min(x0, tx0);
			x1 = tx1 = std::min(x1, tx1);
		}





		double v2 = get_OT_DIS(x_array);
		return v2;



	}

	unsigned int find(unsigned int x) { return pa[x] == x ? x : pa[x] = find(pa[x]); }
	inline double reional_term_direct_dis(int i, int j) {
		auto [temp1, areai, o2i] = cell_feature[i];
		auto [temp2, areaj, o2j] = cell_feature[j];
		assert(!label_2_marker[i] && !label_2_marker[j]);
		double sum = 0;
		for (int x = 0; x < channel; x++) {
			sum += dis_between_two_bins(label_2_o1[x].row(i), label_2_o1[x].row(j));

		}

		return sum;
	}
	inline double regional_term(int i, int j) {
		auto [temp1, areai, o2i] = cell_feature[i];
		auto [temp2, areaj, o2j] = cell_feature[j];
		int m1 = label_2_marker[i];
		int m2 = label_2_marker[j];
		int marker = std::max(m1,m2);
		assert(m1 == m2 || !m1 || !m2);
		double sum = 0;
		for (int x = 0; x < channel; x++) {
			if (x >= image_channel_out && marker){
				Mat marker_bins = marker_2_center[x - image_channel_out].row(marker - 1);
				Mat sum_ij = label_2_o1[x].row(i) + label_2_o1[x].row(j);
				sum += dis_between_two_bins(sum_ij, marker_bins)* (areai + areaj);
				sum -= dis_between_two_bins(label_2_o1[x].row(i), marker_bins)*areai;
				sum-= dis_between_two_bins(label_2_o1[x].row(j), marker_bins)*areaj;
				continue;
			}

#if  defined(use_OT_weight_in_functional)
			cv::Mat bs = label_2_o1[x].row(i) + label_2_o1[x].row(j);
			cv::Mat bi = label_2_o1[x].row(i);
			cv::Mat bj = label_2_o1[x].row(j);
			double weighti = dis_between_two_bins(label_2_o1_sum_in_each_ehannel[x], bi);
			double weightj = dis_between_two_bins(label_2_o1_sum_in_each_ehannel[x], bj);
			double sum_weight = dis_between_two_bins(label_2_o1_sum_in_each_ehannel[x], bs);

			sum += getvariance(bs).first /sum_weight - getvariance(bi).first /weighti - getvariance(bj).first/weightj;
			/*sum += getvariance(bs) / (1 + sum_weight) - getvariance(bi) / (1 + weighti) - getvariance(bj) / (1 + weightj);*/

			/*sum += getvariance(bs)* sum_weight - getvariance(bi)* weighti - getvariance(bj)* weightj;*/
			//sum += getvariance(bs)*exp(-sum_weight) - getvariance(bi) * exp(-weighti) - getvariance(bj) * exp(-weightj);
			/*sum *= (areai + areaj);*/
#elif defined(add_interval_penalty_to_functional_no_area)
			cv::Mat bs = label_2_o1[x].row(i) + label_2_o1[x].row(j);
			auto [value, a] = getvariance(bs, left_bins / (areai + areaj));
			sum += value*(areai + areaj)+left_bins*a*a;
			auto  [value1, a1] = getvariance(label_2_o1[x].row(i), left_bins / areai);
			sum-= value1* areai+ left_bins * a1 * a1;
			auto  [value2, a2] = getvariance(label_2_o1[x].row(j), left_bins / areaj);
			sum -= value2 * areaj + left_bins * a2 * a2;
#else
			cv::Mat bs = label_2_o1[x].row(i) + label_2_o1[x].row(j);
			sum += getvariance(bs).first * (areai + areaj) - getvariance(label_2_o1[x].row(i)).first * areai - getvariance(label_2_o1[x].row(j)).first * areaj;

#endif 








			

		}
		

		return sum/total_area;
	};
	inline double edge_term_3(int i, int j) {
		//this simplify edge_term. But due to float error, the result does not equal to it.
		assert(pa[i] == i);
		assert(pa[j] == j);
		assert(edge_each_cell[i][j] == edge_each_cell[j][i]);
		auto [li, areai, o2i] = cell_feature[i];
		auto [lj, areaj, o2j] = cell_feature[j];
		double lk = li + lj - 2 * edge_each_cell[i][j];


#if defined(edge_linear)
		
#else
		lk *= lk;
		lj *= lj;
		li *= li;
#endif

#ifdef no_image_boundary_edge_term
		if (region_near_boundary[i]) {
			li = 4 * std::numbers::pi * areai; lk = 4 * std::numbers::pi * (areai + areaj);
		}
			
		if (region_near_boundary[j]) {
			lj = 4 * std::numbers::pi * areaj; lk = 4 * std::numbers::pi * (areai + areaj);
		}
#endif // no_image_boundary_edge_term


		return (lk - li - lj) / total_area;


		


	}

	double total_energy(int i, int j) {
		//assert(fabs(edge_term_3(i, j) - edge_term(i, j))<1e-10);
#ifdef use_pure_OT_no_functional
		double v1 = reional_term_direct_dis(i, j);
#else
		double v1 = regional_term(i, j);
#endif // use_pure_OT



		double v2 = edge_term_3(i, j);

		//fmt::println("v1 {} v2 {} nv2 {} ",v1,v2, edge_lambda * v2);
		return v1 + edge_lambda * v2;

	}

	void unite(unsigned int x, unsigned int y) {
		x = find(x), y = find(y);
		if (x == y) return;
		//log_file.region.push_front(regional_term(x, y));
		//log_file.edge.push_front(edge_lambda * edge_term_3(x, y));
		if (size[x] < size[y]) std::swap(x, y);
		/*mexPrintf("before: %d %d\n", marker_each_ccl[x], marker_each_ccl[y]);*/
		pa[y] = x;
		size[x] += size[y];
		// the code below is an additional step which is not the standard disjoint set.
		unsigned char mx, my, mz;
		mx = label_2_marker[x];
		my = label_2_marker[y];

		/*output_file << std::format("x: {}, y: {}, mx: {}, my: {}\n",x,y,mx,my);*/
		assert(mx == my || !mx || !my);
		if (my > 0)
			label_2_marker[x] = my;
		/*mexPrintf("after: %d %d\n---------\n", marker_each_ccl[x], marker_each_ccl[y]);*/
		total_length -= 2 * edge_each_cell[x][y];
		cell_feature[x][0] += cell_feature[y][0] - 2 * edge_each_cell[x][y];
		cell_feature[x][1] += cell_feature[y][1];
		cell_feature[x][2] += cell_feature[y][2];
		for (int i = 0; i < channel; i++) {
			label_2_o1[i].row(x) += label_2_o1[i].row(y);
		}

		// For each edge [x,z], add z to x's neighbor list, and change z's neighbor list accordingly.
		// since y will be removed, you can remove all its information from z's neighbor list.
		for (auto [z, p] : edge_each_cell[y]) {
			edge_each_cell[z][x] = (edge_each_cell[x][z] += p);
			edge_each_cell[z].erase(y);
		}
		edge_each_cell[y].clear();
		//if x (id: 0) has the  neighbors 1,2
		// y (id: 1) has the neighbor 3 0
		// then after this, x has the neighbors 1 2 3 0
		// so there is certainly a duplication.
		edge_each_cell[x].erase(x);


		//have finished the unite. But now you should maintain the priority_queue
		mx = label_2_marker[x];
		assert(pa[x] == x);
		for (auto [z, p] : edge_each_cell[x]) {
			assert(pa[z] == z);
			mz = label_2_marker[z];
			if (!mx || !mz || mz == mx)
				Q.push({ total_energy(x,z),x,z,size[x],size[z] });
		}







	}
	void merge() {
		unsigned int x, y;
		double similar;
		unsigned char mx, my;
		/*mexPrintf("merge_in\n");*/
		while (!Q.empty()) {
			auto [similar_recorded, i1, i2, record_size_1, record_size_2] = Q.top();
			Q.pop();
			//i1 may not equal x
			x = find(i1);
			y = find(i2);
			if (x == y)
				continue;
			//only merge small cell
			/*if ((cell_feature[x][0] > threshold_area) && (cell_feature[y][0] > threshold_area))
				continue;*/
			mx = label_2_marker[x];
			my = label_2_marker[y];
			if (!mx || !my || my == mx) {
				//the similar stored in queue may not be the same as the current similar. Because you repeatedly merge but the similar value is not updated. So you should use record_size to compare the recorded size and the actual size.
				if (record_size_1 != size[x] || record_size_2 != size[y]) {
					/*mexPrintf("sorry\n");*/
					continue;
				}
				unite(x, y);
				/*incre.push_front(similar_recorded);*/
				now_cell_num--;
				if (now_cell_num <= expect_cell_num)
					return;
			}
		}

	}
	std::vector<unsigned int> count_numer() {
		std::vector<unsigned int> count(pa.size(), 0);
		unsigned int n = 1;
		unsigned int size = pa.size();
		unsigned int x, id;
		for (unsigned int i = 0; i < size; i++) {
			x = find(i);
			id = count[x];
			if (id)
				count[i] = id;
			else {
				count[i] = n;
				count[x] = n++;
			}
		}
		for (auto& aa : count) aa--;
		return count;
	}
};
void get_cell_information(const mxArray* img, std::vector<std::array<double, 3>>& cell_feature, std::vector<cv::Mat>& label_2_o1, std::vector<unsigned int>& label) {
	//length, area,  x ^ 2
	memset(cell_feature.data(), 0, cell_feature.size() * sizeof(cell_feature[0]));
	auto shape = mxGetDimensions(img);
	const uint32_T channel = mxGetNumberOfDimensions(img) > 2 ? shape[2] : 1;
	const uint32_T size_slice = shape[0] * shape[1];
	int region_id = 0;
	short* img_p = (short*)mxGetPr(img);
	double f1;
	for (auto id : label) cell_feature[id][1]++;
	std::cout << "image_size:" << size_slice * channel << "\n";
	for (int j = 0; j < channel; j++) {
		Mat& label_2_o1_reference = label_2_o1[j];
		for (int i = 0; i < size_slice; i++)
		{
			f1 = *(img_p++);
			region_id = label[i];
			cell_feature[region_id][2] += f1 * f1;
			label_2_o1_reference.at<double>(region_id, int(f1))++;
			//output_file << label_2_o1_reference << "\n";
		}
	}



};
void region_near_boundary_fun(std::vector<unsigned int>& label, std::array<int, 2>&& dimension, std::vector<unsigned char>& output) {
	auto [ni, nj] = dimension;
	
	Mat storelabel(nj, ni, CV_64FC1);
	std::copy_n(label.data(),nj*ni, (double*)storelabel.data);
	storelabel.row(0).forEach<double>([&output](double a, const int* position) {output[a] = 1; });
	storelabel.row(nj-1).forEach<double>([&output](double a, const int* position) {output[a] = 1; });
	storelabel.col(0).forEach<double>([&output](double a, const int* position) {output[a] = 1; });
	storelabel.col(ni - 1).forEach<double>([&output](double a, const int* position) {output[a] = 1; });

}
void merge_region(const mxArray* img, const unsigned char* marker, std::vector<cv::Mat>& marker_2_center, const uint32_T* nearest_segments, uint32_T* output) {


	auto shape = mxGetDimensions(img);
	int img_m = shape[0]; /* Get the dimensions of A */
	int img_n = shape[1];
	const int channel = channel_out;
	//img ranges from 0 to N
	std::vector<unsigned int> label(img_m * img_n);
	unsigned int size_region;

	//	{
	//std::ofstream os("E:\\matlabcode\\superpixel\\Brain_Tumor_Detection\\code\\label_before",std::ofstream::binary);
	//	os.write((char*)nearest_segments,label.size()*sizeof(unsigned int));
	//	}


	size_region = connected_component_labelling(nearest_segments, { img_m,img_n }, label.data());
	assert(*max_element(label) + 1 == size_region);
	assert(*min_element(label) == 0);

	std::vector<unsigned char> region_near_boundary(size_region, 0);
	region_near_boundary_fun(label, { img_m,img_n }, region_near_boundary);

	/*tonii(label.data(), { img_m,img_n,1 }, R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\h&e_image\label.nii)");
	tonii(region_near_boundary.data(), { int(size_region),1,1 }, R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\h&e_image\region_near_boundary.nii)");*/
	//tonii(label.data(), { img_m,img_n,1 }, std::format(R"({}\label{}.nii)",log_file.debug_file,expect_area_num_out).c_str());


//	{
//std::ofstream os("E:\\matlabcode\\superpixel\\Brain_Tumor_Detection\\code\\label_after", std::ofstream::binary);
//	os.write((char*)label.data(), label.size() * sizeof(unsigned int));
//	}


	//{
	//	//tonii<vtkUnsignedIntArray>(label.data(), { img_n,img_m  ,1 }, (des / "label_before.nii").string().c_str());
	//	const auto [min, max] = std::minmax_element(std::begin(label), std::end(label));
	//	mexPrintf("after_merge label min:%d max:%d\n", *min, *max);
	//}


	//length, area, x^2




	auto debug_marker_each_ccl = [](int size_region, const unsigned char* marker, std::vector<unsigned int>& label) ->bool {
		std::valarray<unsigned char> mmin((unsigned char)255, size_region);
		std::valarray<unsigned char> mmax((unsigned char)(0), size_region);
		unsigned char m;
		for (auto [i, l] : enumerate(label)) {
			m = marker[i];
			if (m) {
				mmin[l] = std::min(mmin[l], m);
				mmax[l] = std::max(mmax[l], m);
			}
		}
		mmin[mmin == 255] = 0;
		auto aa = mmin == mmax;
		bool bb = aa.min();
		return bb;

		};
	//cell_feature : perimeter, area, x^2
	std::vector<std::array<double, 3>> cell_feature(size_region);

	std::vector<cv::Mat> label_2_o1;
	label_2_o1.reserve(channel);
	for (int i = 0; i < channel; i++) {
		label_2_o1.emplace_back(Mat(size_region, 256, CV_64FC1, double(0)));
	}
	get_cell_information(img, cell_feature, label_2_o1, label);


	std::vector<unsigned char> label_2_marker(size_region, 0);
	assert(debug_marker_each_ccl(size_region, marker, label));
	for (auto [i, l] : enumerate(label)) {
		label_2_marker[l] = std::max(marker[i], label_2_marker[l]);
	}
	auto debug_label_2_marker = [](std::vector<unsigned char>& label_2_marker, int marker_size) {
		std::valarray<bool>v(false, marker_size);
		for (auto aa : label_2_marker) {
			v[aa] = true;
		}
		return v.min();
		};
	assert(debug_label_2_marker(label_2_marker, (*std::max_element(marker, marker + img_m * img_n)) + 1));
	//memset(intensity_sum.data(),0, intensity_sum.size()*channel*sizeof(double));





	auto debug_intensity_sum = [&channel](std::vector<cv::Mat>& label_2_o1, const mxArray* img) {

		auto shape = mxGetDimensions(img);
		int img_m = shape[0]; /* Get the dimensions of A */
		int img_n = shape[1];
		int slice = img_m * img_n;

		cv::Mat cell_mat(1, 256, CV_64FC1, double(0));
		Mat sum_in_reduce;
		for (int i = 0; i < channel; i++) {
			cv::reduce(label_2_o1[i], sum_in_reduce, 0, REDUCE_SUM);
			cell_mat += sum_in_reduce;
		}
		//cell_mat  row vector
		//std::ofstream ff(R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\code1\aa.txt)");
		//ff << label_2_o1<<"\n---------\n";
		cv::Mat img_mat(1, 256, CV_64FC1, double(0));

		short* pimg = ((short*)mxGetPr(img));

		for (int i = 0; i < slice * channel; i++) {
			img_mat.at<double>(0, int(pimg[i]))++;
		}
		/*output_file << img_mat << "\n---------\n";*/
		/*output_file << cell_mat.t() << "\n---------\n";*/
		//img_mat col vector
		Mat temp = cell_mat - img_mat;
		//std::cout << cell_mat << "\n";
		//std::cout << img_mat << "\n";
		double tt = norm(temp, NORM_INF);
		/*output_file << tt;*/
		return tt < 1e-7;

		};
	assert(debug_intensity_sum(label_2_o1, img));

	/*std::vector<double> aa(2, 100);
	auto bb=cv::Mat(aa);*/


	auto debug_cell_feature = [&cell_feature, &channel](const mxArray* img)->bool {
		std::valarray<double> on_image(double(0), 2);

		std::for_each_n((short*)mxGetPr(img), mxGetNumberOfElements(img), [&on_image](short a) {on_image += {1, double(a* a)}; });
		on_image[0] /= channel;
		std::valarray<double> on_cell(double(0), 2);
		std::for_each(cell_feature.begin(), cell_feature.end(), [&on_cell](auto& a) {on_cell += {a[1], a[2]}; });
		return std::abs(on_image - on_cell).max() < 1e-6;
		};

	assert(debug_cell_feature(img));



	// if 1 and 2 are adjacent, this vector will only have [1,2] rather than [1,2] and [2,1]
	std::vector<std::tuple<unsigned int, unsigned int, double>> edge_feature = adjacent_in_label(label, { img_m,img_n }, size_region, cell_feature);

	//assert(std::fabs(std::ranges::fold_left(adjacent | std::views::elements<2>, 0, std::plus<double>()) - 1) < 1e-7);
	assert(fabs(2 * fold_left(edge_feature | elements<2>, 0, std::plus<double>()) + img_m * 2 + img_n * 2 - fold_left(cell_feature | elements<0>, 0, std::plus<double>())) < 1e-7);
	/*output_file << 2 * fold_left(edge_feature | elements<2>, 0, std::plus<double>()) + img_m * 2 + img_n * 2 << ' ' << fold_left(cell_feature | elements<0>, 0, std::plus<double>()) << '\n';*/









	dsu dsu_region(size_region, cell_feature, label_2_o1, edge_feature, label_2_marker, marker_2_center, region_near_boundary);
	//marker_each_ccl: ccl->marker
	//marker_center: marker->dis [1,2,3]->[p1,p2,p3]
	assert(dsu_region.label_2_marker.size() == (*max_element(label)) + 1);
	assert(*max_element(dsu_region.label_2_marker) == *std::max_element(marker, marker + img_m * img_n));
	if (marker_channel_out)
	{
		assert(marker_2_center[0].rows == *std::max_element(marker, marker + img_m * img_n));
		assert(marker_2_center[0].cols == 256);
		assert(marker_2_center.size() == marker_channel_out);
	}


	assert(dsu_region.total_area * channel == mxGetNumberOfElements(img));
	assert(dsu_region.expect_cell_num == expect_area_num_out);
	assert(dsu_region.total_area == fold_left(elements<1>(dsu_region.cell_feature), 0, std::plus<double>()));
	assert(dsu_region.total_length == fold_left(elements<0>(dsu_region.cell_feature), 0, std::plus<double>()));
	/*output_file << fold_left(dsu_region.edge_each_cell|join | elements<1>, 0, std::plus<double>()) + img_m * 2 + img_n * 2 << ' ' << dsu_region.total_length<<'\n';*/
	assert(fold_left(dsu_region.edge_each_cell | join | elements<1>, 0, std::plus<double>()) + img_m * 2 + img_n * 2 == dsu_region.total_length);
	int i3, i4, j1, j2;
	double f;


	dsu_region.merge();
	assert(fold_left(dsu_region.edge_each_cell | join | elements<1>, 0, std::plus<double>()) + img_m * 2 + img_n * 2 == dsu_region.total_length);

	auto debug_area_in_cell_feature = [](int total_area, std::vector<std::array<double, 3>>& cell_feature, std::vector<unsigned int>& pa) {
		double sum = 0;
		for (auto [i, z] : enumerate(zip(pa, cell_feature))) {
			auto [a, b] = z;
			if (i == a)
				sum += b[1];
		}
		return sum == total_area;
		};
	assert(debug_area_in_cell_feature(dsu_region.total_area, dsu_region.cell_feature, dsu_region.pa));


	auto debug_length_in_cell_feature = [](std::vector<std::array<double, 3>>& cell_feature, std::vector<unsigned int>& pa) {
		double sum = 0;
		for (auto [i, z] : enumerate(zip(pa, cell_feature))) {
			auto [a, b] = z;
			if (i == a)
				sum += b[0];
		}
		return sum;
		};
	assert(debug_length_in_cell_feature(dsu_region.cell_feature, dsu_region.pa) == fold_left(dsu_region.edge_each_cell | join | elements<1>, 0, std::plus<double>()) + img_m * 2 + img_n * 2);

	auto debug_x2_in_cell_feature = [](std::vector<unsigned int>& pa, std::vector<std::array<double, 3>>& cell_feature, const mxArray* img) {
		double sum1 = 0;
		for (auto [i, z] : enumerate(zip(pa, cell_feature))) {
			auto [p, f] = z;
			if (i == p) sum1 += f[2];
		}
		Mat mm(1, mxGetNumberOfElements(img), CV_64FC1);
		std::copy_n((short*)mxGetPr(img), mxGetNumberOfElements(img), (double*)(mm.data));
		double sum2 = mm.dot(mm);
		return  fabs(sum1 - sum2) < 1e-7;
		};
	assert(debug_x2_in_cell_feature(dsu_region.pa, dsu_region.cell_feature, img));


	auto debug_x1 = [](std::vector<unsigned int>& pa, std::vector<Mat>& cell_o1, short* img, int slice, int channel) {
		Mat sum_x1(1, 256, CV_64FC1, double(0));
		for (int i = 0; i < channel; i++) {
			for (auto [j, p] : enumerate(pa)) {
				if (j == p) sum_x1 += cell_o1[i].row(j);
			}
		}

		Mat sum_x2(1, 256, CV_64FC1, double(0));
		for (int i = 0; i < slice * channel; i++) {
			sum_x2.at<double>(0, int(img[i]))++;
		}

		//output_file << sum_x1 << "\n-----------\n" << sum_x2 << "\n-----------\n";
		return norm(sum_x1 - sum_x2, NORM_INF) < 1e-7;
		};

	assert(debug_x1(dsu_region.pa, dsu_region.label_2_o1, (short*)mxGetPr(img), img_m * img_n, channel));
#ifdef debug_flag
	/*{
		auto temp = log_file.region | to<std::vector>();
		tonii<vtkDoubleArray>(temp.data(), { (int)temp.size(),1,1 }, R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_term\code\region.nii)");
	}*/
	//{
	//	auto temp = log_file.region | to<std::vector>();
	//	tonii(temp.data(), { (int)temp.size(),1,1 }, (log_file.debug_file+R"(\region.nii)").c_str());
	//}

	/*{
		auto temp = log_file.edge | to<std::vector>();
		tonii<vtkDoubleArray>(temp.data(), { (int)temp.size(),1,1 }, R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_term\code\edge.nii)");
	}*/
	//{
	//	auto temp = log_file.edge | to<std::vector>();
	//	tonii(temp.data(), { (int)temp.size(),1,1 }, (log_file.debug_file + R"(\edge.nii)").c_str());
	//}
#endif // debug_flag
//#ifdef debug_flag
//	{
//		std::vector<double> temp = dsu_region.incre | std::ranges::to<std::vector<double>>();
//		std::ofstream(R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\code\energy_increament)",std::ofstream::binary).write((char*)temp.data(),temp.size()*sizeof(double));
//		
//	}
//#endif // debug_flag



	std::vector<unsigned int> old_label_2_new_label = dsu_region.count_numer();

	//{
	//	//this will be a 3*m*n
	//	std::ofstream os("E:\\matlabcode\\superpixel\\Brain_Tumor_Detection\\code\\cell_feature", std::ofstream::binary); 
	//	std::vector<std::array<double,3>> temp;
	//	temp.reserve(img_m * img_n);
	//	for (unsigned int aa : label) {
	//		temp.push_back(dsu_region.cell_feature[dsu_region.pa[aa]]);
	//	}
	//	os.write((char*)temp.data(), temp.size() * sizeof(std::array<double, 3>));
	//}


	std::transform(label.begin(), label.end(), output, [&old_label_2_new_label](unsigned int aa) {return old_label_2_new_label[aa]; });



}
void mexFunction(int nlhs, mxArray* plhs[],       /* Outputs */
	int nrhs, const mxArray* prhs[]) /* Inputs */
{	//prhs
	//0: img (n channel)+ distance (m channel)  (short) ;
	//1: marker (uint8) (to generate distance function): 0, 1, 2, 3, ..... M. 
	//2: marker center in the distance array (double): C1, C2, c3,....CM. Note 0 is background, which does not have a center. The length is M*m 
	//3: label (uint32);
	//4: some parameters (double)
		// 0) expected region number;
		// 1) edge term parameter;




	/*pmat = matOpen("aa.mat", "w");*/

#ifdef debug_flag
	std::filesystem::create_directories(log_file.debug_file);
#endif // debug_flag


	auto shape = mxGetDimensions(prhs[0]);

	int img_m = shape[0]; /* Get the dimensions of A */
	int img_n = shape[1];
	channel_out = mxGetNumberOfDimensions(prhs[0]) > 2 ? shape[2] : 1;



	UINT32_T* ptr_my = (UINT32_T*)mxGetPr(prhs[3]);

	/*{
		auto [mmin, mmax] = std::minmax_element(ptr_my, ptr_my + mxGetNumberOfElements(prhs[3]));
		mexPrintf("before merge min:%d max:%d\n", *mmin, *mmax);
	}*/
	unsigned char* marker = (unsigned char*)mxGetPr(prhs[1]);


	marker_channel_out = *max_element(marker, marker + img_m * img_n);
	assert(marker_channel_out <= channel_out);
	double* temp = (double*)mxGetPr(prhs[4]);
	//0: required number of merged super pixels
	//1: edge term coefficient
	expect_area_num_out = temp[0];
	edge_lambda_out = temp[1];
	left_bins = temp[2];
	right_bins = temp[3];
	image_channel_out = channel_out - marker_channel_out;


	std::vector<cv::Mat> distance_bins_of_markers_each_channel;
	distance_bins_of_markers_each_channel.reserve(marker_channel_out);
	//marker_center: 0 

	if (marker_channel_out) {
		double* ptr = (double*)mxGetPr(prhs[2]);
		for (int i = 0; i < marker_channel_out; i++) {
			distance_bins_of_markers_each_channel.emplace_back(Mat(marker_channel_out, 256, CV_64FC1, ptr));
			ptr += marker_channel_out * 256;
		}
	}
	if (marker_channel_out) {
		assert(mxGetNumberOfElements(prhs[2]) == distance_bins_of_markers_each_channel.size() * distance_bins_of_markers_each_channel[0].rows * distance_bins_of_markers_each_channel[0].cols);
	}


	plhs[0] = mxCreateNumericMatrix(img_m, img_n, mxUINT32_CLASS, mxREAL);
	UINT32_T* output = (UINT32_T*)mxGetPr(plhs[0]);


	merge_region(prhs[0], marker, distance_bins_of_markers_each_channel, ptr_my, output);


}
#pragma once
#include "My_3D_OT_for_OT.h"
#include "My_3D_OT_for_variance.h"
#include "my_3D_OT.h"

#include "tools.h"




//#include "emd.h"
//#include "emd.c"

//#include "emd.c"


//#include <glpk.h>
// 
//#define only_use_CV_regional_term
//#define use_linear_OT_functional
//#define use_OT_weight_in_functional
//#define edge_linear

//#define interval_based_on_variance
//#define no_image_boundary_edge_term

//#define only_output_small_homo

//#define use_color_based_weight
//#define only_use_merged_similirity

//#define output_debug_3

#ifdef output_debug_3
std::ofstream ff(R"(E:\temp\ff.txt)");
#endif

//#define use_homo_all
//#define use_all_neighboorhood_of_a_region

//#define use_all_neighboorhood_of_a_region_1
//#define use_complete_outside_energy


#ifdef use_all_neighboorhood_of_a_region
int two_inside_model = 0;
int two_outside_model = 0;
int one_outside_model = 0;
bool use_area;
std::vector<double> parameters(4);
#else
int two_inside_model = 1;
bool use_area = false;
#endif // #define


#define output_label_each_step

#ifdef output_label_each_step
const int  step_to_output = 9999;
std::filesystem::path des;
int output_label_each_step_int = 0;
#endif // output_label_each_step


//#define add_interval_penalty_to_functional_no_area
//#define add_interval_penalty_to_functional_with_area
//#define use_squared_ot
//#define interval_based_on_minimum
//#define use_pure_OT_no_functional
using namespace std::views;
using namespace std::ranges;
using namespace cv;
//double  threshold_area_out;
//num of target (merged) superpixel
double expect_area_num_out;
//: region term + edge_lambda_out* edge term
double edge_lambda_out=0;

int count_index_number_3D = 0;
//#define ouput_use_boundary_based_OT
#ifdef ouput_use_boundary_based_OT
std::ofstream ff(R"(E:\temp\ff.txt)");
#endif // ouput_use_boundary_based_OT



//#define output_des

//#ifdef output_des
//std::vector<unsigned int> label_output_des;
//int num_now_output_des;
//std::string next_save_string;
//std::string des = R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\h&e_image_3D\save_data20)";
//#endif // output_des

#ifdef output_des
std::vector<unsigned int> label_output_des;
int num_now_output_des;
std::string next_save_string;
std::string des = R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\h&e_image_3D\save_data200)";
#endif // output_des

//#define outputfile
// 
//std::ofstream outputfile1(R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\h&e_image_3D\aa.txt)", std::ios::app);
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

inline std::pair<double, double> dis_to_closet_uniform_pdf(std::array<double, 256>& bins, double mu) {


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
	a = -6 * i1 + 12 * i2;
	a /= (1 + 12 * mu);
	b = i1 - a / 2;
	/*fmt::println("c1 {} c_variance {}",a,2*sqrt(get_mean_variance_from_pdf(bins).second));*/

	assert(std::abs(a / 2 + b - i1) < 1e-8);
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
	return { y,a };
#else
	return { y + mu * a * a ,-1 };
#endif // add_interval_penalty_to_functional_no_area






}


























struct MyStruct
{
	double region_term;
	unsigned int x, y;
	unsigned int record_size_x, record_size_y;
};
std::function<bool(MyStruct&, MyStruct&)> cmp = [](MyStruct& left, MyStruct& right) { return left.region_term > right.region_term; };

#ifdef use_all_neighboorhood_of_a_region
struct edge_information
{
	double length = 0;
	double merged_energy_inside = 0;
	double merged_energy_outside = 0;
	//total_area means area of of i j and all k.
};
#else
struct edge_information
{
	double length = 0;
	double merged_energy_inside = 0;
};
#endif // use_all_neighboorhood_of_a_region

#ifdef use_all_neighboorhood_of_a_region
struct cell_information {
	double length = 0;
	double area = 0;
	double homo_in = 0;
	double homo_out = 0;
};
#else
struct cell_information {
	double length = 0;
	double area = 0;
	double homo_in = 0;
};
#endif // use_all_neighboorhood_of_a_region


template <typename MyOTClass1, typename MyOTClass2>
struct dsu {
	std::priority_queue<MyStruct, std::vector<MyStruct>, std::function<bool(MyStruct&, MyStruct&)>> Q;
	std::vector<unsigned int> pa;
	std::vector<unsigned int> size;
	std::vector<unsigned int> recorded_edge;
	//size: number of cells (not area) of each set.
	//  We use it to obtain the variance E(x^2)-(E(x))^2
	//length, area, x^2
	std::vector<cell_information> cell_feature;
	//length, area, x^2
	//intensity_sum is composed of the information of the image and the distance array. 
	//The channel number (the size of std::vector<double>) is (image channel) + (distance function channel) 

#ifdef use_OT_weight_in_functional
	std::vector<cv::Mat> label_2_o1_sum_in_each_ehannel;// label_2_o1_sum_in_each_ehannel[channel]=Mat(1,256);  sum all color information in a channel. only use in use_OT_weight_in_functional; 
#endif // use_OT_weight_in_functional


	std::vector<unsigned char>& label_2_marker;
	// marker_2_center type int
	MyOTClass1& data_class;
	MyOTClass2& dis_class;
	//The first double is the probability (length) of this edge.
	// the second double is the OT homogeneous of the union of cell i and cell j.
	// that is, if z=(x union y), this is the OT_dis(z,\bar{z})


	std::vector<std::unordered_map<unsigned int, edge_information>> edge_each_cell;


	//std::vector<unsigned int> vec1_for_use_all_neighboorhood_of_a_region;
	//std::vector<unsigned int> vec2_for_use_all_neighboorhood_of_a_region;

#ifdef use_all_neighboorhood_of_a_region

	std::vector<std::array<unsigned int, 2>> vec_for_edge;
#endif // use_all_neighboorhood_of_a_region
	//-----------------------
	/*std::forward_list<double> incre;*/
	//---------------------


	//double threshold_area = threshold_area_out;
	const unsigned int expect_cell_num = expect_area_num_out;
	const double edge_lambda = edge_lambda_out;
	//in total length, all edges are counted twice and the image bounday is counted once. 
	double total_length;
	const double total_area;
	unsigned int now_cell_num;

	//--------------------------
	//size_ is the number of regions rather than the size of images.
	explicit dsu(unsigned int size_, std::vector<std::array<double, 2>>& cell_feature_out, std::vector<std::tuple<unsigned int, unsigned int, double>>& adjacent_, std::vector<unsigned char>& marker_each_ccl_out, MyOTClass1& data_OT_out, MyOTClass2& dis_OT_out) :
		pa(size_),
		size(size_, 1),
		recorded_edge(size_*2),
		edge_each_cell(size_),
		Q(cmp),
		now_cell_num(size_),
		label_2_marker(marker_each_ccl_out),
		data_class(data_OT_out),
		total_length(fold_left(cell_feature_out | views::elements<0>, 0, std::plus<double>())),
		total_area(fold_left(cell_feature_out | views::elements<1>, 0, std::plus<double>())),
		dis_class(dis_OT_out)
	{
		cell_feature.reserve(cell_feature_out.size());
		for (auto& [length, area] : cell_feature_out) {
			cell_feature.emplace_back(length, area);
		}
		std::ranges::iota(pa, 0);

		recorded_edge.clear();

#ifdef use_all_neighboorhood_of_a_region
		vec1_for_use_all_neighboorhood_of_a_region.reserve(size_ * 5);
		vec2_for_use_all_neighboorhood_of_a_region.reserve(size_ * 5);
		vec_for_edge.reserve(adjacent_.size() * 5);
#endif // use_all_neighboorhood_of_a_region



		//------------edge
		for (auto [count, edge] : enumerate(adjacent_)) {
			auto [i, j, d] = edge;
			assert(!edge_each_cell[i].contains(j));
			assert(!edge_each_cell[j].contains(i));

			edge_each_cell[i][j] = { d };
			edge_each_cell[j][i] = { d };
		}
		if constexpr (typeid(data_class) == typeid(My_3D_OT_for_variance)) {
			assert(size_ == data_OT_out.sum_each_cell.rows);
			assert(size_ == data_class.sum_each_cell.rows);
		}
		else if  constexpr(typeid(data_class) == typeid(My_3D_OT_for_OT)){
			assert(size_ == data_OT_out.color_index.rows);
			assert(size_ == data_class.color_index.rows);
		}






		//-----------------------cell
		for (int i = 0; i < size_; i++) {
			//cell_feature[i].homo_in = calculate_cell_energy_inside(i);
#ifdef use_homo_all
			cell_feature[i].homo_out = calculate_cell_energy_outside(i);
#endif // use_homo_all

		}

















		unsigned char mi, mj;
		for (auto [count, edge] : enumerate(adjacent_)) {


			auto [i, j, d] = edge;
			assert(edge_each_cell[i].contains(j));
			assert(edge_each_cell[j].contains(i));
			mi = label_2_marker[i];
			mj = label_2_marker[j];
			if (!(mi == mj || !mi || !mj)) {

				continue;
			}
			auto& edgei = edge_each_cell[i][j];
			auto& edgej = edge_each_cell[j][i];







			edgej.merged_energy_inside = edgei.merged_energy_inside = calculate_merged_energy_inside(i, j);
#ifdef use_all_neighboorhood_of_a_region
			edgej.merged_energy_outside = edgei.merged_energy_outside = calculate_merged_energy_outside(i, j);
#endif // use_all_neighboorhood_of_a_region














		}


		for (auto [i, j, _] : adjacent_) {
			mi = label_2_marker[i];
			mj = label_2_marker[j];
			if (!(mi == mj || !mi || !mj)) {

				continue;
			}
			Q.emplace(MyStruct{ total_energy(i,j),i,j,1,1 });
		}


#ifdef output_des
		save_image_for_debug();
#endif // output_des	




	}



	inline void merged_points_from_point(int i, std::vector<std::array<unsigned int, 2>>& vec) {
		// here vec stores the edge id of adjacent edges of i
		vec.clear();
		assert(i == pa[i]);
		for (auto [z, p] : edge_each_cell[i]) {
			assert(z == pa[z]);
			assert(z != i);
			for (auto [zz, pp] : edge_each_cell[z]) {
				assert(zz == pa[zz]);
				assert(zz != z);
				if (zz != i)
					vec.emplace_back(std::array<unsigned int, 2>{z, zz});
			}
		}
	}
	inline void point_from_merged_points(int i, int j, std::vector<unsigned int>& vec) {
		// <i,j> is the merged cell
		// vec may have repeated elements
		assert(i == pa[i]);
		assert(j == pa[j]);
		vec.clear();
#ifdef output_debug_2
		ff << fmt::format("point_from_merged_points {} {} ", i, j);
		ff << "point_from_merged_points i\n";
		for (auto [z, p] : edge_each_cell[i]) {
			auto [a, b, c] = p;
			ff << fmt::format("({} {})-{} {} {} {}\n", i, z, pa[z], a, b, c);
		}
		ff << "point_from_merged_points j\n";
		for (auto [z, p] : edge_each_cell[j]) {
			auto [a, b, c] = p;
			ff << fmt::format("({} {})-{} {} {} {}\n", j, z, pa[z], a, b, c);
		}



		std::cout << fmt::format("point_from_merged_points {} {} ", i, j);
		std::cout << "point_from_merged_points i\n";
		for (auto [z, p] : edge_each_cell[i]) {
			auto [a, b, c] = p;
			std::cout << fmt::format("({} {})-{} {} {} {}\n", i, z, pa[z], a, b, c);
		}
		std::cout << "point_from_merged_points j\n";
		for (auto [z, p] : edge_each_cell[j]) {
			auto [a, b, c] = p;
			std::cout << fmt::format("({} {})-{} {} {} {}\n", j, z, pa[z], a, b, c);
		}
#endif
		for (auto [k, p] : edge_each_cell[i]) {
			assert(k == pa[k]);
			assert(k != i);
			if (k == j) continue;
			vec.push_back(k);
		}
		for (auto [k, p] : edge_each_cell[j]) {
			assert(k == pa[k]);
			assert(k != j);
			if (k == i) continue;
			vec.push_back(k);
		}

	}
	double calculate_merged_energy_inside(unsigned int  i, unsigned int j)
	{
		double dis = 0;
		if (two_inside_model == 0)
			dis = data_class.variance(i, j) + dis_class.variance(i, j);
		else {
			
			dis = data_class.distance(i,j) + dis_class.distance(i,j);
		}
		double ss = cell_feature[i].area + cell_feature[j].area;
		ss /= total_area;
		return use_area ? dis * ss : dis;
	}
#ifdef use_all_neighboorhood_of_a_region
	inline double calculate_merged_energy_outside(int i, int j) {
		point_from_merged_points(i, j, vec1_for_use_all_neighboorhood_of_a_region);

		std::sort(vec1_for_use_all_neighboorhood_of_a_region.begin(), vec1_for_use_all_neighboorhood_of_a_region.end());
		auto it = std::unique(vec1_for_use_all_neighboorhood_of_a_region.begin(), vec1_for_use_all_neighboorhood_of_a_region.end());
		vec1_for_use_all_neighboorhood_of_a_region.erase(it, vec1_for_use_all_neighboorhood_of_a_region.end());
		double ss = cell_feature[i].area + cell_feature[j].area;
		for (auto k : vec1_for_use_all_neighboorhood_of_a_region) {

			ss += cell_feature[k].area;
		}
		assert(fold_left(vec1_for_use_all_neighboorhood_of_a_region | views::transform([i, j](unsigned int a) {return a != i && a != j; }), true, std::logical_and()));
		double d = 0;
		if (two_outside_model == 0) {
			vec1_for_use_all_neighboorhood_of_a_region.push_back(i);
			vec1_for_use_all_neighboorhood_of_a_region.push_back(j);
			d = data_class.OT_3D_distance(vec1_for_use_all_neighboorhood_of_a_region) + dis_class.OT_3D_distance(vec1_for_use_all_neighboorhood_of_a_region);
		}
		else {
			vec2_for_use_all_neighboorhood_of_a_region.clear();
			vec2_for_use_all_neighboorhood_of_a_region.push_back(i);
			vec2_for_use_all_neighboorhood_of_a_region.push_back(j);
			d = data_class.OT_3D_distance(vec1_for_use_all_neighboorhood_of_a_region, vec2_for_use_all_neighboorhood_of_a_region) + dis_class.OT_3D_distance(vec1_for_use_all_neighboorhood_of_a_region, vec2_for_use_all_neighboorhood_of_a_region);
		}





		ss /= total_area;
		return use_area ? d * ss : d;
	}
#endif // use_all_neighboorhood_of_a_region



#ifdef use_homo_all
	inline double calculate_cell_energy_outside(int i) {
		vec1_for_use_all_neighboorhood_of_a_region.clear();
		vec1_for_use_all_neighboorhood_of_a_region.push_back(i);
		double ss = 0;
		double d = 0;
		ss = cell_feature[i].area;
		for (auto& [j, _] : edge_each_cell[i]) {
			assert(j != i);
			ss += cell_feature[j].area;
		}

		if (one_outside_model == 0) {
			for (auto& [j, _] : edge_each_cell[i]) {
				assert(j != i);
				vec1_for_use_all_neighboorhood_of_a_region.push_back(j);
			}
			d = data_class.OT_3D_distance(vec1_for_use_all_neighboorhood_of_a_region) + dis_class.OT_3D_distance(vec1_for_use_all_neighboorhood_of_a_region);
		}
		else {
			vec2_for_use_all_neighboorhood_of_a_region.clear();
			for (auto& [j, _] : edge_each_cell[i]) {
				assert(j != i);
				vec2_for_use_all_neighboorhood_of_a_region.push_back(j);
			}
			d = data_class.OT_3D_distance(vec1_for_use_all_neighboorhood_of_a_region, vec2_for_use_all_neighboorhood_of_a_region) + dis_class.OT_3D_distance(vec1_for_use_all_neighboorhood_of_a_region, vec2_for_use_all_neighboorhood_of_a_region);
		}

		ss /= total_area;
		return use_area ? d * ss : d;
	}

#endif
	inline double calculate_cell_energy_inside(int i) {
		double d = data_class.variance(i) + dis_class.variance(i);
		double ss = cell_feature[i].area;
		ss /= total_area;
		return use_area ? d * ss : d;
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
#ifdef output_des
	void save_image_for_debug() {
		double d = 0;
		for (auto& a : label_output_des) a = pa[a];
		tonii(label_output_des.data(), { 200,201,1 }, std::string(fmt::format(R"({}\label_output_des_{}.nii)", des, now_cell_num)).c_str());
		std::ofstream ff(std::string(fmt::format(R"({}\output_{}.txt)", des, now_cell_num)).c_str(), std::ios::binary);
		ff << "\n\nhomogeniousity\n\n";
		for (auto [i, a] : enumerate(homogeniousity)) ff << fmt::format("{} : {}\n", i, a);
		ff << "\n\edge_each_cell\n\n";
		std::forward_list<double> temp;
		for (auto [i, map] : enumerate(edge_each_cell)) {
			for (auto [j, p] : map) {
				auto [a, b] = p;
				auto [temp1, areai, o2i] = cell_feature[i];
				auto [temp2, areaj, o2j] = cell_feature[j];

				ff << fmt::format("( {:>3}  {:>3} ): {:>15.6}", i, j, b);
				d = b * (areai + areaj) - areai * homogeniousity[i] - areaj * homogeniousity[j];
				temp.push_front(d);
				ff << fmt::format("|| {:>15.6}*{:>15.6}-{:>15.6}*{:>15.6}-{:>15.6}*{:>15.6}={:>15.6} \n", b, (areai + areaj), homogeniousity[i], areai, homogeniousity[j], areaj, d);
			}
		}
		temp.sort();
		for (auto aa : temp) ff << aa << ' ';
		ff << "------------\n\n\n";
	}
#endif // output_des








	unsigned int find(unsigned int x) { return pa[x] == x ? x : pa[x] = find(pa[x]); }

	inline double edge_term_3(int i, int j) {
		//this simplify edge_term. But due to float error, the result does not equal to it.
		assert(pa[i] == i);
		assert(pa[j] == j);
		//assert(edge_each_cell[i][j] == edge_each_cell[j][i]);


#if defined(use_all_neighboorhood_of_a_region)
		double li = cell_feature[i].length;
		double lj = cell_feature[j].length;
		double lk = li + lj - 2 * (edge_each_cell[i][j]).length;
		double areak = cell_feature[i].area + cell_feature[j].area;
		return lk * lk / areak;


#elif defined(no_image_boundary_edge_term)
		double li = cell_feature[i].length;
		double lj = cell_feature[j].length;
		double lk = li + lj - 2 * (edge_each_cell[i][j]).length;
		if (region_near_boundary[i]) {
			li = 4 * std::numbers::pi * areai; lk = 4 * std::numbers::pi * (areai + areaj);
		}

		if (region_near_boundary[j]) {
			lj = 4 * std::numbers::pi * areaj; lk = 4 * std::numbers::pi * (areai + areaj);
		}
		return (lk - li - lj) / total_area;
#else
		double li = cell_feature[i].length;
		double lj = cell_feature[j].length;
		double lk = li + lj - 2 * (edge_each_cell[i][j]).length;
		return lk * lk / total_area;
#endif










	}
	double total_energy(int i, int j) {


		assert(edge_each_cell[j][i].merged_energy_inside == edge_each_cell[i][j].merged_energy_inside);
	


		double v1 = edge_each_cell[j][i].merged_energy_inside;





		double v2 = edge_term_3(i, j);

		return v1 + edge_lambda * v2;



	}
	inline void create_new_remove_old(int x, int y) {
#ifdef output_debug_2
		ff << fmt::format("create_new_remove_old {}-{}\n", x, y);
		for (auto [z, p] : edge_each_cell[x]) {
			auto [a, b, c] = p;
			ff << fmt::format("({} {})-{} {} {} {}\n", x, z, pa[z], a, b, c);
		}


		std::cout << fmt::format("create_new_remove_old {}-{}\n", x, y);
		for (auto [z, p] : edge_each_cell[x]) {
			auto [a, b, c] = p;
			std::cout << fmt::format("({} {})-{} {} {} {}\n", x, z, pa[z], a, b, c);
		}
#endif


		// For each edge [x,z], add z to x's neighbor list, and change z's neighbor list accordingly.
		// since y will be removed, you can remove all its information from z's neighbor list.
		for (auto [z, p] : edge_each_cell[y]) {
			//pay attention to this place
			auto it = edge_each_cell[x].insert({ z,p });
			if (it.second) {
				;
			}
			else {
				(it.first->second).length += (p.length);
			}
			edge_each_cell[z][x] = it.first->second;
			//std::get<0>(edge_each_cell[z][x]) = (std::get<0>(edge_each_cell[x][z]) += std::get<0>(p));

			edge_each_cell[z].erase(y);
		}
		edge_each_cell[y].clear();
		//if x (id: 0) has the  neighbors 1,2
		// y (id: 1) has the neighbor 3 0
		// then after this, x has the neighbors 1 2 3 0
		// so there is certainly a duplication.
		edge_each_cell[x].erase(x);
#ifdef output_debug_2
		std::cout << "after\n";
		for (auto [z, p] : edge_each_cell[x]) {
			auto [a, b, c] = p;
			fmt::println("({} {})-{} {} {} {}", x, z, pa[z], a, b, c);
		}
#endif // output_debug_2



		assert(x == pa[x]);
		assert(x == pa[y]);
		assert(edge_each_cell[y].empty());
		assert(fold_left(edge_each_cell[x] | elements<0> | views::transform([this, x](auto z) { return z == pa[z]; }), true, std::logical_and()));
	}

	void unite(unsigned int x, unsigned int y) {
		x = find(x), y = find(y);
		assert(x!=y);

		recorded_edge.push_back(x);
		recorded_edge.push_back(y);



		if (size[x] < size[y]) std::swap(x, y);
		//------------move information
		pa[y] = x;
		size[x] += size[y];

		// additional custom move step.
		data_class.merge(x, y);
		dis_class.merge(x, y);
		label_2_marker[x] = std::max(label_2_marker[x], label_2_marker[y]);
		total_length -= 2 * (edge_each_cell[x][y].length);
		cell_feature[x].length += cell_feature[y].length - 2 * (edge_each_cell[x][y].length);
		cell_feature[x].area += cell_feature[y].area;
		cell_feature[x].homo_in = edge_each_cell[x][y].merged_energy_inside;
#ifdef use_all_neighboorhood_of_a_region
		cell_feature[x].homo_out = edge_each_cell[x][y].merged_energy_outside;
#endif 

		if constexpr (typeid(data_class) == typeid(My_3D_OT_for_variance)) {
			assert(data_class.area_each_cell[x] == cell_feature[x].area);
		}





		assert(label_2_marker[x] == label_2_marker[y] || !label_2_marker[x] || !label_2_marker[y]);
		if constexpr (typeid(data_class) == typeid(My_3D_OT_for_OT)) {
			assert(data_class.marker[x] == label_2_marker[x]);
			assert( data_class.marker[y]== label_2_marker[y]);
		}
		if constexpr (typeid(dis_class) == typeid(My_3D_OT_for_OT)) {
			assert(dis_class.marker[x] == label_2_marker[x]);
			assert( dis_class.marker[y] == label_2_marker[y]);
		}
		




		/*fmt::print("before {} {}\n", fold_left(edge_each_cell | join | views::transform([](auto a) {return a.second.length; }), 0, std::plus<double>()), total_length);*/
		//----------after move,  apply new relationship
		create_new_remove_old(x, y);
		/*fmt::print("after {} {}\n", fold_left(edge_each_cell | join | views::transform([](auto a) {return a.second.length; }), 0, std::plus<double>()), total_length);*/

		// define new energy
		//have finished the unite. But now you should maintain the priority_queue

		assert(pa[x] == x);

		unsigned char mx = label_2_marker[x];
		unsigned char mz;
		for (auto& [z, p] : edge_each_cell[x]) {
			assert(pa[z] == z);
			assert(z != x);
			mz = label_2_marker[z];
			if (!mx || !mz || mz == mx) {
#ifdef use_all_neighboorhood_of_a_region
				edge_each_cell[z][x].merged_energy_outside = p.merged_energy_outside = calculate_merged_energy_outside(x, z);
#endif // use_all_neighboorhood_of_a_region


				(edge_each_cell[z][x].merged_energy_inside) = (p.merged_energy_inside) = calculate_merged_energy_inside(x, z);
			}
		}

#ifdef use_complete_outside_energy
		merged_points_from_point(x, vec_for_edge);
		remove_duplicate_my(vec_for_edge);

		for (auto [x, z] : vec_for_edge) {
			mx = label_2_marker[x];
			mz = label_2_marker[z];
			if (!mx || !mz || mz == mx) {
				edge_each_cell[z][x].merged_energy_outside = edge_each_cell[x][z].merged_energy_outside = calculate_merged_energy_outside(x, z);
			}
		}
#endif // use_complete_outside_energy




#ifdef output_debug_3
		ff << fmt::format("\n\n\n merging {} and {} now_cell {}\n\n", x, y, now_cell_num);
		std::ofstream ff(fmt::format(R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\H&E_image_3\16\{}.pa)", now_cell_num), std::ios::binary);
		auto temp = views::iota(0) | take(pa.size()) | views::transform([this](auto a) ->unsigned int {return find(a); }) | to<std::vector>();
		ff.write((char*)temp.data(), pa.size() * sizeof(pa[0]));
		ff.close();
#endif // output_debug_3


		mx = label_2_marker[x];
		for (auto [z, _] : edge_each_cell[x]) {
			mz = label_2_marker[z];
			if (!mx || !mz || mz == mx)
				Q.push({ total_energy(x,z),x,z,size[x],size[z]});

		}




	}
	void merge() {
		unsigned int x, y;
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

#ifdef output_label_each_step
				if (output_label_each_step_int&& now_cell_num < step_to_output) {
					std::ofstream ff(des / (fmt::format("{}.bin", now_cell_num).c_str()), std::ios::binary);
					std::vector<unsigned int> vec = count_numer();
					
					unsigned int temp[3] = { vec.size(), cell_feature.size(), cell_feature.size() };
					ff.write((char*)(&similar_recorded), sizeof(double));
					ff.write((char*)temp, 3 * sizeof(unsigned int));
					ff.write((char*)vec.data(), vec.size() * sizeof(unsigned int));
					auto aa = views::iota(0) | take(cell_feature.size()) | views::transform([this](auto i) ->double {return cell_feature[find(i)].homo_in; }) | to<std::vector<double>>();
					ff.write((char*)aa.data(), aa.size() * sizeof(double));
#ifdef use_all_neighboorhood_of_a_region
					auto bb = views::iota(0) | take(cell_feature.size()) | views::transform([this](auto i) ->double {return cell_feature[find(i)].homo_out; }) | to<std::vector<double>>();
					ff.write((char*)bb.data(), bb.size() * sizeof(double));
#endif // use_all_neighboorhood_of_a_region


					ff.close();
				}
#endif // output_label_each_step
				//fmt::println("{}\n",now_cell_num);
#ifdef output_des
				save_image_for_debug();
#endif // output_des

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
	std::vector<unsigned int> count_number_remove_background() {
		std::vector<unsigned int> count(pa.size(), 0);
		unsigned int n = 1;
		unsigned int size = pa.size();
		unsigned int x, id;
		for (unsigned int i = 0; i < size; i++) {
			x = find(i);
			id = count[x];
			if (id)
				count[i] = id;
			else if (cell_feature[x].homo_in < 500) {
				count[i] = n;
				count[x] = n++;
			}
		}
		return count;
	}
	void store_homo(double* output) {
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
				count[x] = n;
				output[n - 1] = cell_feature[x].homo_in;
				n++;

			}
		}
		for (auto& aa : count) aa--;

	}
};
#pragma once
#include "network_simplex_simple.h"
#include "pch.hpp"
#include "my_3D_OT.h"
using namespace cv;
//#define OT_linear_programming

//#define use_boundary_based_OT
//#define use_boundary_based_OT_variance

//#define only_use_merged_similirity_for_test

//#define OT_fast_OT_gaussian_pdf_only_use_center_with_nozero_count
//#define OT_fast_OT_gaussian_pdf

//#define OT_fast_OT_pure_distance_no_center



//#define OT_3D_rectangle

//#define OT_fast_OT_target_only_use_center_with_nozero_count

//#define OT_3D_use_identity_matrix_as_cov
//#define OT_use_0nor8_in_target_pdf

#define OT_fast_OT
#define OT_3D



//#define outputfile

#ifdef outputfile
std::ofstream ff(R"(aa.txt)");
#endif // outputfile


void get_source_array(Mat& count, std::vector<unsigned short>& source_for_OT_3D, std::vector<double>& weight_source) {
	source_for_OT_3D.clear();
	weight_source.clear();
	double* p = (double*)count.data;
	for (int i = 0; i < count.total(); i++) {
		if (p[i]) {
			source_for_OT_3D.push_back(i);
			weight_source.push_back(p[i]);
		}
	}
#ifndef OT_fast_OT_pure_distance_no_center
	assert(std::abs(std::reduce(weight_source.begin(), weight_source.end()) - 1) < 1e-5);
#endif // !OT_fast_OT_pure_distance_no_center



	assert(std::transform_reduce(weight_source.begin(), weight_source.end(), true, std::logical_and<>{}, [](double a) {return a >= 0; }));

}
void get_mean_and_cov_variance(Mat& color, Mat& count_nor, Mat& mean, Mat& cov_variance) {
	//std::cout << " color " << color << "\n";

	//std::cout << " count " << count << "\n";

	Mat count_nor_repeat = cv::repeat(count_nor, 1, color.cols);
	cv::reduce(count_nor_repeat.mul(color), mean, 0, cv::REDUCE_SUM);

#ifdef OT_3D_use_identity_matrix_as_cov
	cov_variance = cv::Mat::eye(color.cols, color.cols, CV_64FC1);
#else
	Mat mean_repeat = cv::repeat(mean, color.rows, 1);
	MatExpr color_remove_mean = color - mean_repeat;
	cov_variance = color_remove_mean.mul(count_nor_repeat).t() * color_remove_mean;
#endif // OT_3D_use_identity_matrix_as_cov


	//if (cv::determinant(cov_variance) < 0) {
	//	std::cout << cov_variance << "\n";
	//	std::cout << cv::determinant(cov_variance) << "\n";

	//}

	assert(cov_variance.at<double>(0, 0) + 1e-7 >= 0);


	//#ifdef outputfile
	//	outputfile1 << "count_nor:\n" << count_nor << '\n';
	//	outputfile1 << "mean:\n" << mean << '\n';
	//	outputfile1 << "cov_variance_before:\n" << cov_variance << '\n';
	//#endif // outputfile

	if (std::abs(cv::determinant(cov_variance)) < 1e-6)
		cov_variance = cv::Mat::eye(cov_variance.rows, cov_variance.rows, CV_64FC1);
	assert(cv::determinant(cov_variance) + 1e-8 >= 0);
	//#ifdef outputfile
	//	outputfile1 << "cov_variance_after:\n" << cov_variance << '\n';
	//#endif // outputfile

	cov_variance = cov_variance.inv();

	//#ifdef outputfile
	//
	//	outputfile1 << "inv_cov_variance:\n" << cov_variance << '\n';
	//#endif // outputfile
		/*std::cout << cov_variance << "\n";*/




}
void get_target_array(Mat& color, Mat& count_nor, std::vector<unsigned short>& target_for_OT_3D, std::vector<double>& weight_target, double range_threshold) {

	Mat mean;
	Mat cov_variance;

	get_mean_and_cov_variance(color, count_nor, mean, cov_variance);

	//std::cout << "cov_variance: \n" << cov_variance << "\n";


	assert(cov_variance.rows == cov_variance.cols);
	assert(mean.rows == 1 && mean.cols == cov_variance.cols);


#if defined(OT_fast_OT_gaussian_pdf_only_use_center_with_nozero_count)
	target_for_OT_3D.clear();
	weight_target.clear();
	double d = 0, * p = (double*)count_nor.data;
	MatExpr temp;
	assert(count_nor.total() == count_nor.rows);
	for (int i = 0; i < count_nor.rows; i++) {
		d = p[i];
		if (d > 0) {
			temp = color.row(i) - mean;
			d = (temp * cov_variance).dot(temp);
			d = exp(-d / 2);
			target_for_OT_3D.push_back(i);
			weight_target.push_back(d);
		}
	}
	d = std::reduce(weight_target.begin(), weight_target.end());
	for (double& a : weight_target) a /= d;

#else
	std::vector<std::pair<double, unsigned short>> sort_dis;

	target_for_OT_3D.clear();
	weight_target.clear();
	MatExpr temp;
	sort_dis.clear();
	double d = 0;
	for (int i = 0; i < color.rows; i++) {
		temp = color.row(i) - mean;
		sort_dis.emplace_back((temp * cov_variance).dot(temp), i);
	}

	std::sort(sort_dis.begin(), sort_dis.end());
	//#ifdef outputfile
	//	outputfile1 << "sort_dis\n";
	//	for (auto [a, b] : sort_dis)
	//		outputfile1 << fmt::format("({} {} {})", a, b, count_nor.at<double>(b));
	//
	//#endif // outputfile




		/*fmt::println("sort:\n {}\n", sort_dis);*/
	double* p = (double*)count_nor.data;
	unsigned short id = 0;
	double count_d = 0;
	double sum = 0;

	for (int i = 0; i < color.rows; i++) {
		id = sort_dis[i].second;
		count_d = p[id];
		assert(count_d >= 0);
#ifdef OT_fast_OT_target_only_use_center_with_nozero_count
		if (std::abs(count_d) < 1e-10)
			continue;
#endif // OT_fast_OT_target_only_use_center_with_nozero_count



		sum += count_d;

		if (sum >= range_threshold) {
			target_for_OT_3D.push_back(id);
			weight_target.push_back(count_d - sum + range_threshold);
			break;
		}

		//fmt::println("i {} count {} sum {}",i, count_d, sum);
		target_for_OT_3D.push_back(id);
		weight_target.push_back(count_d);
	}
	assert(std::abs(std::reduce(weight_target.begin(), weight_target.end()) - range_threshold) < 1e-8);
	//#ifdef outputfile
	//
	//	outputfile1 << "\ntarget_for_OT_3D:\n";
	//	for (auto aa : target_for_OT_3D) outputfile1 << aa << ' ';
	//	outputfile1 << "\n";
	//
	//	outputfile1 << "weight_target_before:\n";
	//	for (auto aa : weight_target) outputfile1 << aa << ' ';
	//	outputfile1 << "\n";
	//#endif // outputfile

#ifdef OT_fast_OT_gaussian_pdf
	for (auto [id, w] : std::views::zip(target_for_OT_3D, weight_target)) {
		temp = color.row(id) - mean;
		d = (temp * cov_variance).dot(temp);
		d = exp(-d / 2);
		w = d;
	}
	d = std::reduce(weight_target.begin(), weight_target.end());
	for (double& a : weight_target) a /= d;
#else
	sum = double(1) / double(weight_target.size());
	for (auto& a : weight_target) a = sum;
#endif // OT_fast_OT_gaussian_pdf



	/*sum = std::reduce(weight_target.begin(),weight_target.end());
	sum /= weight_target.size();
	for (auto& a : weight_target) a = sum;*/

	/*assert(sum > 0);*/


	///*assert(sum > 0);*/
	//for (auto& a : weight_target)
	//	a /= sum;

#endif // OT_fast_OT_gaussian_pdf
	assert(std::transform_reduce(weight_target.begin(), weight_target.end(), true, std::logical_and<>{}, [](double a) {return a >= 0; }));
	assert(std::abs(std::reduce(weight_target.begin(), weight_target.end()) - 1) < 1e-5);
#ifdef OT_use_0nor8_in_target_pdf
	for (auto& a : weight_target) a *= 0.8;
#endif // OT_use_0.8_in_target_pdf

}
void get_sorted_center_and_weight(Mat& weight_repeat, Mat& color, Mat& color_sorted) {
	//weight_repeat: 3*n
	//color:n*3

	Mat merged_mat;
	Mat channels[2] = { color.t(), weight_repeat };
	cv::merge(channels, 2, merged_mat);
	cv::Vec2d* s;
	for (int i = 0; i < merged_mat.rows; i++) {
		s = merged_mat.ptr<Vec2d>(i);
		std::sort(reinterpret_cast<std::pair<double, double>*>(s), reinterpret_cast<std::pair<double, double>*>(s + merged_mat.cols));
	}

	cv::split(merged_mat, channels);
	color_sorted = channels[0];
	weight_repeat = channels[1];
}
void get_rectangel_region(Mat& weight_repeat, Mat& color_sorted, Mat& region_lower_coordinates, Mat& region_upper_coordinates, Mat& mean, double regularization) {
	//color_sorted: 3*n
	//weight_repeat:3*n
	cv::reduce(color_sorted.mul(weight_repeat), mean, 1, cv::REDUCE_SUM);
#ifdef outputfile
	ff << "mean:\n" << mean << "\n";
#endif // outputfile



	double* p;
	for (int i = 0; i < weight_repeat.rows; i++) {
		p = weight_repeat.ptr<double>(i);
		std::inclusive_scan(p, p + weight_repeat.cols, p);
	}

#ifdef outputfile
	ff << "cdf:\n" << weight_repeat << "\n";
#endif // outputfile






	weight_repeat = weight_repeat.mul(weight_repeat);
#ifdef outputfile
	ff << "cdf^2:\n" << weight_repeat << "\n";
#endif // outputfile


	for (int i = 0; i < weight_repeat.rows; i++) {
		p = weight_repeat.ptr<double>(i);
		std::adjacent_difference(p, p + weight_repeat.cols, p);
	}
#ifdef outputfile
	ff << "cdf^2_diff:\n" << weight_repeat << "\n";
#endif // outputfile


	Mat i2;
	cv::reduce(color_sorted.mul(weight_repeat), i2, 1, cv::REDUCE_SUM);
	i2 /= 2;
#ifdef outputfile
	ff << "i2:\n" << i2 << "\n";
#endif // outputfile


	MatExpr a = -6 * mean + 12 * i2;
	//std::cout << "-6*" << mean << "+12*{}" << i2<<"\n";
	//std::cout << a << "\n";
	a = a / (1 + 12 * regularization);
	//std::cout << a << "\n";
	MatExpr b = mean - a / 2;
#ifdef outputfile
	ff << "a: \n" << a << "\n\n";
	ff << "b: \n" << b << "\n\n";
#endif // outputfile

	region_lower_coordinates = cv::repeat(b.t(), color_sorted.cols, 1);
	region_upper_coordinates = cv::repeat(a.t() + b.t(), color_sorted.cols, 1);
}
void get_weight_and_id_from_rectangle_region(Mat& color, Mat& count_nor, Mat& region_lower_coordinates, Mat& region_upper_coordinates, std::vector<unsigned short>& target_for_OT_3D, std::vector<double>& weight_target) {
	Mat mask;
	cv::reduce((color >= region_lower_coordinates) & (color <= region_upper_coordinates), mask, 1, cv::REDUCE_MIN);
	//std::cout << "mask:" << type2str(mask.type()) << "\n" << mask << "\n";
	target_for_OT_3D.clear();
	weight_target.clear();
	unsigned char* mask_p = mask.ptr<unsigned char>();
	double* w_p = count_nor.ptr<double>();
	for (int i = 0; i < color.rows; i++) {
		if (mask_p[i]) {
			target_for_OT_3D.push_back(i);
			weight_target.push_back(1);
		}
	}
	double s = double(1) / double(weight_target.size());
	for (double& a : weight_target) a = s;
}
void get_target_array_rectangle_region(Mat& color, Mat& count_nor, std::vector<unsigned short>& target_for_OT_3D, std::vector<double>& weight_target, double range_threshold, Mat& mean) {
	//color: n* 3  
	//count_nor: n*1

	Mat weight_repeat = cv::repeat(count_nor.t(), color.cols, 1);
	Mat color_sorted;
#ifdef outputfile
	ff << "weight_repeat_before:\n " << weight_repeat << "\n\n";
#endif // outputfile


	get_sorted_center_and_weight(weight_repeat, color, color_sorted);
#ifdef outputfile
	ff << "color:\n " << color << "\n\n";
	ff << "count_nor:\n " << count_nor << "\n\n";
	ff << "weight_repeat_after:\n " << weight_repeat << "\n\n";
	ff << "color_sorted: \n" << color_sorted << "\n\n";
#endif // outputfile

	Mat region_lower_coordinates;
	Mat region_upper_coordinates;

	get_rectangel_region(weight_repeat, color_sorted, region_lower_coordinates, region_upper_coordinates, mean, range_threshold);

#ifdef outputfile
	ff << "region_lower_coordinates: \n" << region_lower_coordinates << "\n\n";
	ff << "region_upper_coordinates: \n" << region_upper_coordinates << "\n\n";
#endif // outputfile

	get_weight_and_id_from_rectangle_region(color, count_nor, region_lower_coordinates, region_upper_coordinates, target_for_OT_3D, weight_target);




	assert(weight_target.size() ? std::transform_reduce(weight_target.begin(), weight_target.end(), true, std::logical_and<>{}, [](double a) {return a >= 0; }) : 1);
	assert(weight_target.size() ? std::abs(std::reduce(weight_target.begin(), weight_target.end()) - 1) < 1e-5 : 1);
}


using namespace lemon;
typedef int64_t arc_id_type; // {short, int, int64_t} ; Should be able to handle (n1*n2+n1+n2) with n1 and n2 the number of nodes (INT_MAX = 46340^2, I64_MAX = 3037000500^2)
typedef double supply_type; // {float, double, int, int64_t} ; Should be able to handle the sum of supplies and *should be signed* (a demand is a negative supply)
typedef double cost_type;  // {float, double, int, int64_t} ; Should be able to handle (number of arcs * maximum cost) and *should be signed* 
typedef FullBipartiteDigraph Digraph;
DIGRAPH_TYPEDEFS(FullBipartiteDigraph);

class My_3D_OT_for_OT
{
public:

	My_3D_OT_for_OT(Mat& color_index_out, Mat& color_out, Mat& marker_center_out, Mat& stablize_out, Mat& pairwise_dis_for_3D_ot_out, std::vector<unsigned char>& marker_out, double range_threshold_out);
	~My_3D_OT_for_OT();
	inline double variance(int i, int j);
	inline double variance(int i);
	void sum_of_pdfs(std::vector<unsigned int>& vec1);
	inline double distance(int i, int j);
	inline double distance(std::vector<unsigned int>& vec1, std::vector<unsigned int>& vec2);
	inline double variance(std::vector<unsigned int>& vec1);
	inline void merge(int i, int j);
	inline void output_all_variables();
	Mat pairwise_dis_for_3D_ot;// double 
	Mat color;       //c * 3(or 4, 5 ,6)  (double)
	Mat color_index; //region* c (int)
	Mat marker_center; //(2 or 3) * c  (int)
	Mat stablize;  // (int) some numbers * c; 
	std::vector<unsigned char> marker; //region
private:
	double range_threshold;
	std::vector<unsigned short> continous_id_to_true_id_source;
	std::vector<unsigned short> continous_id_to_true_id_target;
	std::vector<double> weight_source;
	std::vector<double> weight_target;
	std::vector<double> dis_from_source_to_a_single_point;
	Mat color_nor;
	double calculate_OT_use_some_methods();
	void get_pairwise_distance();
	double variance_from_pdf();
	template<typename mytype>
	void obtain_normilized_pdf(mytype&& color_index_out);
	template<typename mytype, typename mytype1>
	void obtain_normilized_pdf(mytype&& color_index_out1, mytype1&& color_index_out2);
};



My_3D_OT_for_OT::~My_3D_OT_for_OT()
{
}

inline double My_3D_OT_for_OT::calculate_OT_use_some_methods()
{

#ifdef OT_fast_OT
	assert(weight_source.size() == continous_id_to_true_id_source.size());
	assert(weight_target.size() == continous_id_to_true_id_target.size());
	int n1 = weight_source.size();
	int n2 = weight_target.size();
	Digraph di(n1, n2);
	NetworkSimplexSimple<Digraph, supply_type, cost_type, arc_id_type> net(di, true, n1 + n2, n1 * n2);
	arc_id_type idarc = 0;
	for (int i = 0; i < n1; i++) {
		for (int j = 0; j < n2; j++) {
			cost_type d = pairwise_dis_for_3D_ot.at<double>(continous_id_to_true_id_source[i], continous_id_to_true_id_target[j]);
			net.setCost(di.arcFromId(idarc), d);
			idarc++;
		}
	}
	for (auto& a : weight_target) a = -a;
	net.supplyMap(weight_source.data(), n1, weight_target.data(), n2);

	int ret = net.run();

	double resultdist = net.totalCost();  // resultdist is the EMD


	//#ifdef outputfile
	//	outputfile1 << "z: " << resultdist << "\n";
	//#endif // outputfile
	return resultdist;
#endif // OT_fast_OT


#ifdef  OT_linear_programming

	glp_prob* lp;
	lp = glp_create_prob();
	glp_set_obj_dir(lp, GLP_MAX);
	glp_add_rows(lp, weight_source.size() * weight_target.size());
	glp_add_cols(lp, weight_source.size() + weight_target.size());

	ia.resize(1);
	ja.resize(1);
	ar.resize(1);

	int count = 1;
	double d = 0;
	for (int i = 0; i < weight_source.size(); i++)
		for (int j = 0; j < weight_target.size(); j++) {
			d = pairwise_dis_for_3D_ot.at<double>(continous_id_to_true_id_source[i], continous_id_to_true_id_target[j]);
#ifdef outputfile

			outputfile1 << "d: " << d << "\n";


#endif // outputfile
			//glp_set_row_bnds(lp,count , GLP_UP, 0.0,d );
			glp_set_row_bnds(lp, count, GLP_FX, d, d);
			ia.push_back(count);
			ja.push_back(i + 1);
			ar.push_back(1);

			ia.push_back(count++);
			ja.push_back(weight_source.size() + j + 1);
			ar.push_back(1);

		}

	glp_load_matrix(lp, count - 1, ia.data(), ja.data(), ar.data());



	count = 1;
	for (int i = 0; i < weight_source.size(); i++) {
		glp_set_col_bnds(lp, count, GLP_FR, 0.0, 0.0);
		glp_set_obj_coef(lp, count++, weight_source[i]);
	}
	for (int i = 0; i < weight_target.size(); i++) {
		glp_set_col_bnds(lp, count, GLP_FR, 0.0, 0.0);
		glp_set_obj_coef(lp, count++, weight_target[i]);
	}

	glp_smcp parm;
	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_OFF;
	/*parm.meth = GLP_PRIMAL;*/



	//parm.meth = GLP_DUAL;
	glp_simplex(lp, &parm);




	double z = glp_get_obj_val(lp);
	glp_delete_prob(lp);
	//std::ofstream ff(R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\h&e_image_3D\aa.txt)",std::ios::app);
	//ff << z << '\n';
	//std::cout << z<<"\n";
	assert(z >= 0);
	//fmt::println("z in function {}", z);

#ifdef outputfile
	outputfile1 << "z: " << z << "\n";
#endif // outputfile

	return z;
#endif //  OT_linear_programming
}

inline void My_3D_OT_for_OT::get_pairwise_distance()
{
	MatExpr temp;
	double d = 0;
	for (int i = 0; i < color.rows; i++) {
		for (int j = 0; j < color.rows; j++) {
			temp = color.row(i) - color.row(j);

			d = temp.dot(temp);
			pairwise_dis_for_3D_ot.at<double>(i, j) = d;

		}
	}
}

inline double My_3D_OT_for_OT::variance_from_pdf()
{
	assert(color_nor.cols == 1);


	MatExpr mean = color_nor.t() * color;
	//assert(mean.rows == 1);
	Mat temp;
	cv::reduce(color - cv::repeat(mean, color.rows, 1), temp, 1, cv::REDUCE_SUM2);

	return temp.dot(color_nor);

}


inline void My_3D_OT_for_OT::merge(int i, int j)
{
	assert(marker[i] == marker[j] || !marker[i] || !marker[j]);
	

		color_index.row(i) += color_index.row(j);
		marker[i] = std::max(marker[i], marker[j]);
	

}

inline void My_3D_OT_for_OT::output_all_variables()
{
	std::ofstream ff(R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\H&E_image_3\16\new.txt)");
	ff << "pairwise_dis_for_3D_ot\n\n" << pairwise_dis_for_3D_ot << "\n\n";
	ff << "color\n\n" << color << "\n\n";
	ff << "color_index\n\n" << color_index << "\n\n";
	ff << "marker_center\n\n" << marker_center << "\n\n";
	ff << "stablize\n\n" << stablize << "\n\n";
	ff << "marker\n\n" << fmt::format("{}", marker) << "\n\n";
}


inline My_3D_OT_for_OT::My_3D_OT_for_OT(Mat& color_index_out, Mat& color_out, Mat& marker_center_out, Mat& stablize_out, Mat& pairwise_dis_for_3D_ot_out, std::vector<unsigned char>& marker_out, double range_threshold_out)
{
	
	range_threshold = range_threshold_out;
	continous_id_to_true_id_source.reserve(color_out.rows);
	continous_id_to_true_id_target.reserve(color_out.rows);
	weight_source.reserve(color_out.rows);
	weight_target.reserve(color_out.rows);
	dis_from_source_to_a_single_point.reserve(color_index_out.rows);
	if (color_out.rows > 0)
		color_nor = Mat(color_out.rows, 1, CV_64FC1);
	stablize = stablize_out;
	color_index = color_index_out;
	marker_center = marker_center_out;

	color = color_out;

	pairwise_dis_for_3D_ot = pairwise_dis_for_3D_ot_out;
	/*pairwise_dis_for_3D_ot = Mat(color.rows, color.rows, CV_64FC1);
	get_pairwise_distance();*/
	marker = std::move(marker_out);
	assert(stablize.empty() ? color_index.cols : stablize.cols == color_index.cols == marker_center.empty() ? color_index.cols : marker_center.cols == color.rows == pairwise_dis_for_3D_ot.rows == pairwise_dis_for_3D_ot.cols);
	assert(color_index.rows == marker.size());
	assert(color_nor.rows == color_index.cols);


	assert(std::transform_reduce((double*)pairwise_dis_for_3D_ot.data, (double*)pairwise_dis_for_3D_ot.data + pairwise_dis_for_3D_ot.total(), true, std::logical_and<>{}, [](double a) {return a >= 0; }));


	//std::cout << pairwise_dis_for_3D_ot;



}




template<typename mytype>
inline void My_3D_OT_for_OT::obtain_normilized_pdf(mytype&& color_index_out)
{
	std::copy_n(color_index_out.begin<int>(), color_index_out.total(), color_nor.begin<double>());
	color_nor /= cv::sum(color_nor)[0];



}

template<typename mytype, typename mytype1>
inline void My_3D_OT_for_OT::obtain_normilized_pdf(mytype&& color_index_out1, mytype1&& color_index_out2)
{
	std::copy_n(color_index_out1.begin<int>(), color_index_out1.total(), color_nor.begin<double>());
	std::transform(color_nor.begin<double>(), color_nor.end<double>(), color_index_out2.begin<int>(), color_nor.begin<double>(), [](double a, int b) {return a + b; });
	color_nor /= cv::sum(color_nor)[0];
}



inline double My_3D_OT_for_OT::variance(int i, int j)
{
	
	assert(marker[i] == marker[j] || !marker[i] || !marker[j]);

	
#ifdef use_boundary_based_OT_variance

		if (j == -1) {
			std::copy_n(color_index.ptr<int>(i), color_index.cols, color_nor.begin<double>());
		}
		else
			std::transform(color_index.ptr<int>(i), color_index.ptr<int>(i) + color_index.cols, color_index.ptr<int>(j), color_nor.begin<double>(), [](int a, int b)->double {return a + b; });
		color_nor /= cv::sum(color_nor)[0];
		return variance_from_pdf();
#else
		int marker_index = std::max(marker[i], j >= 0 ? marker[j] : unsigned char(0));
		//color_index_local's type is  int   1* c
			//color is double              c* 3

		if (j < 0)
			obtain_normilized_pdf(color_index.row(i));
		else {
			obtain_normilized_pdf(color_index.row(i), color_index.row(j));
		}





		get_source_array(color_nor, continous_id_to_true_id_source, weight_source);









		if (marker_index) {

			obtain_normilized_pdf(marker_center.row(marker_index - 1));
			assert(color_nor.cols == 1);
			get_source_array(color_nor, continous_id_to_true_id_target, weight_target);
			//fmt::println("{}", continous_id_to_true_id_target.size());
		}
		else {
#ifdef OT_3D_rectangle
			Mat mean;
			get_target_array_rectangle_region(color, color_nor, continous_id_to_true_id_target, weight_target, range_threshold, mean);
			assert(mean.cols == 1);
			assert(weight_target.size() == continous_id_to_true_id_target.size());

			if (weight_target.empty()) {
				mean = mean.t();
				MatExpr temp;
				dis_from_source_to_a_single_point.clear();
				weight_target.push_back(1);
				for (auto i : continous_id_to_true_id_source) {
					temp = color.row(i) - mean;
					dis_from_source_to_a_single_point.push_back(temp.dot(temp));
				}
			}


#ifdef outputfile
			ff << fmt::format("target_for_OT_3D: {}\n {}\n", continous_id_to_true_id_target.size(), continous_id_to_true_id_target);
			ff << fmt::format("weight_target: {}\n {}\n", weight_target.size(), weight_target);
#endif // outputfile
#else
			get_target_array(color, color_nor, continous_id_to_true_id_target, weight_target, range_threshold);
#endif // OT_3D_rectangle

		}

		double d1;
#ifdef OT_3D_rectangle
		if (continous_id_to_true_id_target.empty())
			d1 = calculate_OT_from_pdf_to_mean();
		else
			d1 = calculate_OT_use_some_methods();
#else
		d1 = calculate_OT_use_some_methods();
#endif // OT_3D_rectangle



		double d2 = d1;
		if (!stablize.empty()) {
			obtain_normilized_pdf(stablize.row(0));
			get_source_array(color_nor, continous_id_to_true_id_target, weight_target);
			d2 = calculate_OT_use_some_methods();
		}
		return std::min(d1, d2);
#endif // use_boundary_based_OT_variance


	



}

inline double My_3D_OT_for_OT::variance(int i)
{

	return variance(i, -1);
	
}



inline void My_3D_OT_for_OT::sum_of_pdfs(std::vector<unsigned int>& vec1)
{

	memset(color_nor.data, 0, color_nor.rows * sizeof(double));
	assert(color_nor.total() == color_nor.rows);
	for (auto i : vec1)
		std::transform(color_nor.begin<double>(), color_nor.end<double>(), color_index.ptr<int>(i), color_nor.begin<double>(), [](double a, int b) {return a + b; });


	color_nor /= cv::sum(color_nor)[0];
}

inline double My_3D_OT_for_OT::distance(int i, int j)
{
	assert(marker[i] == marker[j] || !marker[i] || !marker[j]);

	int marker_index = std::max(marker[i],  marker[j] );
	if (marker_index > 0) return variance(i, j);

	obtain_normilized_pdf(color_index.row(i));
	get_source_array(color_nor, continous_id_to_true_id_source, weight_source);

	obtain_normilized_pdf(color_index.row(j));
	assert(color_nor.cols == 1);
	get_source_array(color_nor, continous_id_to_true_id_target, weight_target);
	return  calculate_OT_use_some_methods();
}

inline double My_3D_OT_for_OT::distance(std::vector<unsigned int>& vec1, std::vector<unsigned int>& vec2)
{
	

	//this seems wrong , because vec1 and vec2 should have the same marker.
	unsigned char marker_index = std::ranges::max(vec1 | std::views::transform([this](unsigned int i) {return marker[i]; }));

	if (marker_index) {
		obtain_normilized_pdf(marker_center.row(marker_index - 1));
		assert(color_nor.cols == 1);
	}
	else {
		sum_of_pdfs(vec1);

	}
	get_source_array(color_nor, continous_id_to_true_id_source, weight_source);

	marker_index = std::ranges::max(vec2 | std::views::transform([this](unsigned int i) {return marker[i]; }));
	if (marker_index) {
		obtain_normilized_pdf(marker_center.row(marker_index - 1));
		assert(color_nor.cols == 1);

	}
	else {
		sum_of_pdfs(vec2);

	}

	get_source_array(color_nor, continous_id_to_true_id_target, weight_target);


	return calculate_OT_use_some_methods();
}

inline double My_3D_OT_for_OT::variance(std::vector<unsigned int>& vec1)
{
	
	sum_of_pdfs(vec1);
	get_source_array(color_nor, continous_id_to_true_id_source, weight_source);

	unsigned char marker_index = std::ranges::max(vec1 | std::views::transform([this](unsigned int i) {return marker[i]; }));

	if (marker_index) {
		obtain_normilized_pdf(marker_center.row(marker_index - 1));
		assert(color_nor.cols == 1);
		get_source_array(color_nor, continous_id_to_true_id_target, weight_target);
		//fmt::println("{}", continous_id_to_true_id_target.size());
	}
	else {
		get_target_array(color, color_nor, continous_id_to_true_id_target, weight_target, range_threshold);
	}

	return calculate_OT_use_some_methods();
}

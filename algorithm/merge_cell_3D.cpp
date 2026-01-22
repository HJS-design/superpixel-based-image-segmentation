
#include "my_merge_class.h"


void get_cell_information(const mxArray* img, std::vector<std::array<double, 3>>& cell_feature, const std::vector<unsigned int>& label) {
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
		for (int i = 0; i < size_slice; i++)
		{
			f1 = *(img_p++);
			region_id = label[i];
			cell_feature[region_id][2] += f1 * f1;
			//output_file << label_2_o1_reference << "\n";
		}
	}



};
void region_near_boundary_fun(const std::vector<unsigned int>& label, std::array<int, 2>&& dimension, std::vector<unsigned char>& output) {
	auto [ni, nj] = dimension;

	Mat storelabel(nj, ni, CV_64FC1);
	std::copy_n(label.data(), nj * ni, (double*)storelabel.data);
	storelabel.row(0).forEach<double>([&output](double a, const int* position) {output[a] = 1; });
	storelabel.row(nj - 1).forEach<double>([&output](double a, const int* position) {output[a] = 1; });
	storelabel.col(0).forEach<double>([&output](double a, const int* position) {output[a] = 1; });
	storelabel.col(ni - 1).forEach<double>([&output](double a, const int* position) {output[a] = 1; });

}
void count_center_in_region(Mat& image_index, const std::vector<unsigned int>& label, Mat& index_count) {
	// unsigned short, unsigned int, int
	for (auto [ind, l] : zip(label, views::counted((unsigned short*)image_index.data, image_index.total()))) {
		index_count.at<int>(ind, l)++;
	}

}

template <typename MyOTClass1, typename MyOTClass2>
void merge_region(  std::vector<unsigned char>& label_2_marker, MyOTClass1& data_class, MyOTClass2& dis_class, std::vector<std::array<double, 2>> &cell_feature, std::vector<std::tuple<unsigned int, unsigned int, double>> &edge_feature,  mxArray* plhs[]) {
	// if 1 and 2 are adjacent, edge_feature will only have [1,2] rather than [1,2] and [2,1]

	// image_index : unsigned short
	
	//img ranges from 0 to N
	unsigned int size_region=label_2_marker.size();

	//	{
	//std::ofstream os("E:\\matlabcode\\superpixel\\Brain_Tumor_Detection\\code\\label_before",std::ofstream::binary);
	//	os.write((char*)nearest_segments,label.size()*sizeof(unsigned int));
	//	}






	/*tonii(label.data(), { img_m,img_n,1 },R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\h&e_image_3D\label.nii)");*/












	/*std::ofstream ff(R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\h&e_image_3D\aa.txt)", std::ios::app);
	ff << pairwise_dis_for_3D_ot << '\n';*/


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




	auto debug_marker_each_ccl = [](int size_region, const unsigned char* marker, const std::vector<unsigned int>& label) ->bool {
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





	


	
	
	//memset(intensity_sum.data(),0, intensity_sum.size()*channel*sizeof(double));






	/*std::vector<double> aa(2, 100);
	auto bb=cv::Mat(aa);*/








	//assert(std::fabs(std::ranges::fold_left(adjacent | std::views::elements<2>, 0, std::plus<double>()) - 1) < 1e-7);
	assert(fabs(2 * fold_left(edge_feature | elements<2>, 0, std::plus<double>())  - fold_left(cell_feature | elements<0>, 0, std::plus<double>())) < 1e-5);
	/*output_file << 2 * fold_left(edge_feature | elements<2>, 0, std::plus<double>()) + img_m * 2 + img_n * 2 << ' ' << fold_left(cell_feature | elements<0>, 0, std::plus<double>()) << '\n';*/



	



	dsu dsu_region(size_region, cell_feature, edge_feature, label_2_marker, data_class, dis_class);
	//marker_each_ccl: ccl->marker
	//marker_center: marker->dis [1,2,3]->[p1,p2,p3]
	assert(dsu_region.label_2_marker.size() == size_region);
	assert(*max_element(dsu_region.label_2_marker) == *max_element(label_2_marker));




	assert(dsu_region.expect_cell_num == expect_area_num_out);
	assert(dsu_region.total_area == fold_left(dsu_region.cell_feature | views::transform([](auto& a) {return a.area; }), 0, std::plus<double>()));
	assert(dsu_region.total_length == fold_left(dsu_region.cell_feature | views::transform([](auto& a) {return a.length; }), 0, std::plus<double>()));
	/*output_file << fold_left(dsu_region.edge_each_cell|join | elements<1>, 0, std::plus<double>()) + img_m * 2 + img_n * 2 << ' ' << dsu_region.total_length<<'\n';*/

	/*fmt::println("{} {}\n", fold_left(dsu_region.edge_each_cell | join | elements<1> | views::transform([](auto& a) {return a.length; }), 0, std::plus<double>()) + img_m * 2 + img_n * 2, dsu_region.total_length);*/
	/*fmt::print("{} {} \n", fold_left(dsu_region.edge_each_cell | join | elements<1> | views::transform([](auto& a) {return a.length; }), 0, std::plus<double>()), dsu_region.total_length);*/

	assert(std::abs(fold_left(dsu_region.edge_each_cell | join | elements<1> | views::transform([](auto& a) {return a.length; }), 0, std::plus<double>())  - dsu_region.total_length)<=1e-5);



	dsu_region.merge();
	assert(std::abs(fold_left(dsu_region.edge_each_cell | join | elements<1> | views::transform([](auto& a) {return a.length; }), 0, std::plus<double>()) -dsu_region.total_length)<1e-8);

	auto debug_area_in_cell_feature = [](double total_area, std::vector<cell_information>& cell_feature, std::vector<unsigned int>& pa) {
		double sum = 0;
		for (int i = 0; i < pa.size();i++) {
			if(i==pa[i])
				sum += cell_feature[i].area;
		}
		return std::abs(sum - total_area);
		};
	assert(debug_area_in_cell_feature(dsu_region.total_area, dsu_region.cell_feature, dsu_region.pa)<1e-7);


	auto debug_length_in_cell_feature = [](std::vector<cell_information>& cell_feature, std::vector<unsigned int>& pa) {
		double sum = 0;
		for (int i = 0; i < pa.size(); i++) {
			if (i == pa[i])
				sum += cell_feature[i].length;
		}
		return sum;
		};
	assert(std::abs(debug_length_in_cell_feature(dsu_region.cell_feature, dsu_region.pa) - fold_left(dsu_region.edge_each_cell | join | elements<1> | views::transform([](auto& a) {return a.length; }), 0, std::plus<double>()))<1e-8);

	




//#ifdef debug_flag
//	{
//		std::vector<double> temp = dsu_region.incre | std::ranges::to<std::vector<double>>();
//		std::ofstream(R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\code\energy_increament)",std::ofstream::binary).write((char*)temp.data(),temp.size()*sizeof(double));
//		
//	}
//#endif // debug_flag

	
#ifdef only_output_small_homo
	std::vector<unsigned int> old_label_2_new_label = dsu_region.count_number_remove_background();
#else
	std::vector<unsigned int> old_label_2_new_label = dsu_region.count_numer();
#endif // only_output_small_homo
	plhs[0] = mxCreateNumericMatrix(label_2_marker.size(), 1, mxUINT32_CLASS, mxREAL);
	UINT32_T* output = (UINT32_T*)mxGetPr(plhs[0]);
	std::copy(old_label_2_new_label.begin(), old_label_2_new_label.end(), output);

	plhs[1] = mxCreateNumericMatrix(dsu_region.recorded_edge.size(), 1, mxUINT32_CLASS, mxREAL);
	 output = (UINT32_T*)mxGetPr(plhs[1]);
	std::copy(dsu_region.recorded_edge.begin(), dsu_region.recorded_edge.end(), output);
	

}

My_3D_OT_for_variance construct_my_ot_class(const mxArray* arr, std::vector<std::array<double, 2>> &cell_feature) {

	unsigned int m, k;
	double* p;

	p = mxGetPr(arr);
	m = *p;
	p++;
	k = *p;
	p++;

	
	Mat sum_each(m,k,CV_64FC1,p);
	p += m * k;
	Mat sq_each(m, k, CV_64FC1, p);
	std::vector<double> area;
	area.reserve(cell_feature.size());;

	for (auto [length, t] : cell_feature) {
		area.push_back(t);
	}

	return My_3D_OT_for_variance(sum_each,sq_each,area);
}

My_3D_OT_for_OT construct_my_ot_class(const mxArray* arr, const mxArray* markerpdf_out, const mxArray* label2marker_out, Mat& stabilize) {

	unsigned int m, n, k;
	double threshold;
	double* p;

	p = mxGetPr(arr);
	m = *p;
	p++;
	n = *p;
	p++;
	k = *p;
	p++;
	threshold = *p;
	p++;
	Mat index;
	Mat(m, n, CV_64FC1, p).convertTo(index, CV_32SC1);
	p += m * n;
	Mat color(n, k, CV_64FC1, p);
	p += n * k;
	Mat dis(n, n, CV_64FC1, p);
	Mat markerpdf;
	if (!mxIsEmpty(markerpdf_out)) {
		markerpdf = cv::Mat(mxGetDimensions(markerpdf_out)[1], mxGetDimensions(markerpdf_out)[0], CV_32SC1, (int*)mxGetPr(markerpdf_out));
	}
	std::vector<unsigned char> label2marker(reinterpret_cast<unsigned char*>(mxGetPr(label2marker_out)), reinterpret_cast<unsigned char*>(mxGetPr(label2marker_out)) + mxGetNumberOfElements(label2marker_out));


	return My_3D_OT_for_OT(index, color, markerpdf, stabilize, dis, label2marker, threshold);
}

void get_function(const mxArray* arr_out,std::vector<std::array<double, 2>> &cell_feature,
	std::vector<std::tuple<unsigned int, unsigned int, double>> &edge_feature) {
	double* p = mxGetPr(arr_out);
	unsigned int cell_size, edge_size;
	//size is not row size, not numel()
	cell_size = *p;
	p++;
	edge_size = *p;
	p++;
	assert(cell_size==cell_feature.size());

	memcpy(cell_feature.data(),p,cell_size*sizeof(double)*2);
	p += cell_size*2;


	edge_feature.reserve(edge_size);
	
	for (int i = 0; i < edge_size; i++,p+=3) {
		edge_feature.emplace_back(p[0],p[1], p[2]);
		
			/*fmt::print("{} {} {}\n", std::get<0>(*tp), std::get<1>(*tp), std::get<2>(*tp));*/
	}

}
void mexFunction(int nlhs, mxArray* plhs[],       /* Outputs */
	int nrhs, const mxArray* prhs[]) /* Inputs */
{	//prhs
	
	//0: label to marker (uint8) 
	//1: marker pdf M (int). M(i,j)=c; the i-th (0, 1, 2, ... M-1) marker has c (c is int number) color j (j means j-th predefined color center). The type is int
	
	//2: region term: (double) m, n,  k, threshold, an array with size m*n*sizeof(double) (index), an array with size n*k*sizeof(double), an array of size n*n(double). they are index, color, and color distance.  
	//3: selective term: (double)  m', n',  k' , an array with size m'*n'*sizeof(double), an array with size n'*k'*sizeof(double), an array of size n'*n'(double). they are index, color, and color distance.  
	// 4: std::vector<std::array<double, 2>> cell_feature(*std::max_element(label.begin(),label.end()));
	//std::vector<std::tuple<unsigned int, unsigned int, double>> edge_feature;
	// cell_feature is perimeter id and area.
	// edge_feature is edge [i,j] and its gradient length.
	//5: some parameters (double)
		// 0) expected region number;
		// 1) edge term parameter;



// if define start_from_scratch
	// m (double) k (double) m*k sum each cell m*k squared each cell

	/*pmat = matOpen("aa.mat", "w");*/

#ifdef debug_flag
	std::filesystem::create_directories(log_file.debug_file);
#endif // debug_flag

	


	double* temp = (double*)mxGetPr(prhs[5]);
	//0: required number of merged super pixels
	//1: edge term coefficient
	expect_area_num_out = temp[0];
	edge_lambda_out = temp[1];
	int start_from_scratch = temp[2];
	output_label_each_step_int = temp[3];
	if(mxGetNumberOfElements(prhs[5])>4)
		use_area = temp[4] > 0;
	
	


	std::vector<unsigned char> label_2_marker(reinterpret_cast<unsigned char*>(mxGetPr(prhs[0])), reinterpret_cast<unsigned char*>(mxGetPr(prhs[0])) + mxGetNumberOfElements(prhs[0]));

	assert(*min_element(label_2_marker) == 0);



	


#ifdef use_all_neighboorhood_of_a_region
	one_outside_model = temp[2];
	two_inside_model = temp[3]; 
	two_outside_model = temp[4];
	use_area = temp[5];

	parameters[0] = (temp[6]);
	parameters[1] = (temp[7]);
	parameters[2] = (temp[8]);
	parameters[3] = (temp[9]);
#endif // use_all_neighboorhood_of_a_region





	Mat stabilize;
	/*if (nrhs >= 8) {
		cv::Mat(mxGetDimensions(prhs[7])[1], mxGetDimensions(prhs[7])[0], CV_32SC1, (double*)mxGetPr(prhs[7])).copyTo(stabilize);
		assert(stabilize.rows == 1);
	
	}*/
	


#ifdef output_label_each_step
	if (nrhs >= 7) {
		char*p = mxArrayToString(prhs[6]);
		des = std::string(p);
		//des = std::filesystem::path(std::string_view(p, n));
		std::cout << "output_file" << des.string() << "\n";
	}
#endif // output_label_each_step


	// length area
	std::vector<std::array<double, 2>> cell_feature(label_2_marker.size());
	std::vector<std::tuple<unsigned int, unsigned int, double>> edge_feature;
	// i j -length
	static_assert(std::is_trivially_copyable<std::array<double, 2>>::value);
	get_function(prhs[4], cell_feature, edge_feature);
	

	assert(fold_left(cell_feature | views::transform([](auto a) {auto [b, c] = a; return c >= 0 && b >= 0; }), true, std::logical_and()));
	assert(fold_left(edge_feature | views::transform([](auto a) {auto [b, c, d] = a; return (c >= 0) && (b >= 0) && (d + 1e-8 >= 0); }), true, std::logical_and()));

	assert(std::ranges::min(edge_feature | views::transform([](auto a) {auto [b, c, d] = a; return std::array<unsigned int, 2>{b, c}; }) | views::join) == 0);
	assert(std::ranges::max(edge_feature | views::transform([](auto a) {auto [b, c, d] = a; return std::array<unsigned int, 2>{b, c}; }) | views::join) == (label_2_marker.size()-1));

	


	


	if (mxIsEmpty(prhs[3])) {
		MyOTClass_empty dis_OT;
		
		if (start_from_scratch) {
			My_3D_OT_for_variance data_OT = construct_my_ot_class(prhs[2], cell_feature);
			merge_region(label_2_marker, data_OT, dis_OT, cell_feature, edge_feature, plhs);
		}
		else {
			My_3D_OT_for_OT data_OT = construct_my_ot_class(prhs[2], prhs[1], prhs[0], stabilize);
			merge_region(label_2_marker, data_OT, dis_OT, cell_feature, edge_feature, plhs);
		}
	}
	else {
		My_3D_OT_for_OT dis_OT = construct_my_ot_class(prhs[3], prhs[1], prhs[0], stabilize);
		if (start_from_scratch) {
			My_3D_OT_for_variance data_OT = construct_my_ot_class(prhs[2], cell_feature);
			merge_region(label_2_marker, data_OT, dis_OT, cell_feature, edge_feature, plhs);
		}
		else {
			My_3D_OT_for_OT data_OT = construct_my_ot_class(prhs[2], prhs[1], prhs[0], stabilize);
			merge_region(label_2_marker, data_OT, dis_OT, cell_feature, edge_feature, plhs);
		}
	}



}
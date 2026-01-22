#pragma once
// C++ program for the above approach


#include "pch.hpp"
#include "disjointset.h"

template<typename type>
void remove_duplicate_my(std::vector<std::array<type, 2>>& vec) {
	for (auto& [a, b] : vec) {
		if (a > b)
			std::swap(a, b);
	}
	assert(std::ranges::fold_left(vec | std::views::transform([](auto a) {return a[0] < a[1]; }), true, std::logical_and()));
	std::sort(vec.begin(), vec.end());
	auto it=std::unique(vec.begin(), vec.end());
	vec.erase(it,vec.end());

}

template<typename type,typename output_type>
int connected_component_labelling(const type* mat, std::array<int, 2>&& dimension, output_type* out_bw) {
	auto [ni, nj] = dimension;
	int slicen = ni * nj;
	cDisjointSet ds;

	ds.Reset();
	int index = 1;
	std::vector<node*> labels(slicen);

	type current_node ; 
	type pre_node;
	pre_node = mat[0];
	labels[0] = ds.MakeSet(0);
	int i, j, k;
	for (i = 1; i < ni; i++) {
		current_node = mat[index];
		labels[index] = pre_node==current_node ? labels[index - 1] : ds.MakeSet(0);
		pre_node=current_node ;
		index++;
	}
	for (j = 1; j < nj; j++) {
		current_node = mat[index];
		
		labels[index] = mat[index - ni]== current_node ? labels[index - ni] : ds.MakeSet(0);
		pre_node = current_node;
		index++;
		for (i = 1; i < ni; i++) {
			current_node = mat[index];
			
			labels[index] = mat[index - ni]== current_node ?  (pre_node== current_node ? (ds.Union(labels[index - 1], labels[index - ni]), labels[index - 1]) : (labels[index - ni])) : (pre_node==current_node ? (labels[index - 1]) : (ds.MakeSet(0)));
			pre_node = current_node;
			index++;
		}
	}

	int ss = ds.Reduce();
	//printf("%d\n", ss);


	for (int i = 0; i < slicen; i++, out_bw++) {
		*out_bw = ds.Find(labels[i])->i;
	}
	//if id=[0,1,2]
	//ss =3.
	return ss;
}

std::vector<std::tuple<unsigned int, unsigned int,double>> adjacent_in_label(const std::vector<unsigned int>& label, std::array<int, 2>&& dimension,int size_label,std::vector<std::array<double, 3>> &cell_feature) {
	using namespace std::views;
	using namespace std::ranges;
	//use index, index-ni, index-1 to construct the adjacent graph.
	auto [ni, nj] = dimension;
	std::forward_list<std::pair<unsigned int, unsigned int>> adjacent;
	//size_label is the number of regions, but this may be out of memory.  
	int a, b;
	for (int j = 0; j < nj; j++) {
		for (int i = 0; i < ni-1; i++) {
			a = label[j * ni + i];
			b = label[j * ni + i+1];
			if(a<b)
				adjacent.emplace_front(a ,b  );
			else if(a>b)
				adjacent.emplace_front(b, a);
		}
	}
	for (int j = 0; j < nj-1; j++) {
		for (int i = 0; i < ni; i++) {
			a = label[j * ni + i];
			b = label[j * ni + i+ni];
			if (a < b)
				adjacent.emplace_front(a, b);
			else if (a > b)
				adjacent.emplace_front(b, a);
		}
	}
	for (int j : {0,nj-1})
		for (int i = 0; i < ni; i++) 
			cell_feature[label[j * ni + i]][0]++;
	for (int j = 0; j < nj; j++)
		for (int i : {0,ni-1})
			cell_feature[label[j * ni + i]][0]++;
	std::vector<std::pair<unsigned int, unsigned int>> adj_vec(std::begin(adjacent), std::end(adjacent));
	
	for (auto [a,b]: adj_vec) {
		cell_feature[a][0]++;
		cell_feature[b][0]++;
	}
	/*std::ofstream(R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_term\code\adj_all)", std::ofstream::binary).write((char*)adj_vec.data(), adj_vec.size() * sizeof(adj_vec[0]));*/

	std::ranges::sort(adj_vec);
	auto cv = adj_vec | std::views::chunk_by(std::ranges::equal_to{});
	/*double sum = adj_vec.size();
	assert(sum= std::ranges::fold_left(cv | std::views::transform([](auto xx) {return xx.size(); }), 0, std::plus<double>()));*/
	std::vector<std::tuple<unsigned int, unsigned int,double>> edge_and_count;
	edge_and_count.reserve(adj_vec.size());
	for (auto xx : cv) {
		auto [a, b] = xx[0];
		//edge_and_count.emplace_back(a,b,xx.size()/sum);
		edge_and_count.emplace_back(a, b, xx.size());
	}
	return edge_and_count;
}






void fromnii(const char* outfile, vtkSmartPointer<vtkImageData>& data) {
	vtkSmartPointer<vtkNIFTIImageReader> reader = vtkSmartPointer<vtkNIFTIImageReader>::New();
	reader->SetFileName(outfile);
	reader->Update();
	data = reader->GetOutput();
};





template <typename T>
void tonii(T* data, std::array<int, 3>&& dimension, const char* outfile) {
	auto [ni, nj, nk] = dimension;
	const int loop_ub = ni * nj * nk;
	int sz[] = { ni,nj,nk };
	vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
	imageData->SetDimensions(sz);
	imageData->AllocateScalars(vtkTypeTraits<T>::VTKTypeID(), 1);

	T* ptr = static_cast<T*>(imageData->GetScalarPointer());
	std::copy_n(data, loop_ub, ptr);

	vtkSmartPointer<vtkNIFTIImageWriter> writer = vtkSmartPointer<vtkNIFTIImageWriter>::New();
	writer->SetFileName(outfile);
	writer->SetInputData(imageData);
	writer->Write();
}
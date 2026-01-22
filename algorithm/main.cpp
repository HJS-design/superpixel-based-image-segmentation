#define slic
//#define segmentation
//#define segmentation_OT_3

#ifdef slic
#include "Power_SLIC.cpp"
#endif // slic

#ifdef segmentation
#include "merge_cell.cpp"
#endif // segmentation

#ifdef segmentation_OT_3
#include "merge_cell_3D.cpp"
#endif //  segmentation_OT_3

int main(int nrhs, char** prhs_main) {

#ifdef segmentation_OT_3
	//0: distance (m channel)  (short) ;
	//1: marker (uint8) (to generate distance function): 0, 1, 2, 3, ..... M. 
	//2: marker center in the distance array (double): C1, C2, c3,....CM. Note 0 is background, which does not have a center. The length is M*m
	//3: label (uint32);
	//4: img index (unsigned short) with the same 2D size as the distance array. It is the index into some predefined 3D color. 0 , 2, 3, .... n-1 
	//5: predefined color (double). n*3;
	//6: some parameters (double)
		// 0) expected region number;
		// 1) edge term parameter;



	const char* name;

	/*pmat = matOpen("aa.mat", "w");*/



	MATFile* pmat = matOpen(R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\h&e_image_3D\testdata.mat)", "r");

#ifdef debug_flag
	std::filesystem::create_directories(log_file.debug_file);
#endif // debug_flag
	const mxArray* prhs[7];
	for (int i = 0; i < 7; i++) {
		prhs[i] = matGetNextVariable(pmat, &name);
		std::cout << name << "\n";
	}
	mxArray* plhs[2];
	mexFunction(1, plhs, 4, prhs);
	matClose(pmat);

	pmat = matOpen(R"(E:\matlabcode\superpixel\Brain_Tumor_Detection\add_edge_selective\h&e_image_3D\output.mat)", "w");

	matPutVariable(pmat, "region", plhs[0]);

	matClose(pmat);
#endif // segmentation_OT_3


#ifdef segmentation
	//prhs
	//0: img (n channel)+ distance (m channel)  (short) ;
	//1: marker (uint8) (to generate distance function): 0, 1, 2, 3, ..... M. 
	//2: marker center in the distance array (double): C1, C2, c3,....CM. Note 0 is background, which does not have a center. The length is M*m 
	//3: label (uint32);
	//4: some parameters (double)
		// 0) expected region number;
		// 1) edge term parameter;


	const char* name;

	/*pmat = matOpen("aa.mat", "w");*/


	std::filesystem::path des(prhs_main[1]);
	MATFile* pmat = matOpen((des/"testdata.mat").string().c_str(), "r");

#ifdef debug_flag
	std::filesystem::create_directories(log_file.debug_file);
#endif // debug_flag
	const mxArray* prhs[5];
	for (int i = 0; i < 5; i++) {
		prhs[i] = matGetNextVariable(pmat, &name);
		std::cout << name << "\n";
	}
	mxArray* plhs[2];
	mexFunction(1, plhs, 4, prhs);
	matClose(pmat);

	pmat = matOpen((des / "output.mat").string().c_str(), "w");

	matPutVariable(pmat, "region", plhs[0]);

	matClose(pmat);

#endif
#ifdef slic
	const char* name;

	/*pmat = matOpen("aa.mat", "w");*/



	MATFile* pmat = matOpen(R"(E:\matlabcode\superpixel\code\testdata.mat)", "r");

#ifdef debug_flag
	std::filesystem::create_directories(log_file.debug_file);
#endif // debug_flag
	const mxArray* prhs[5];
	prhs[0] = matGetVariable(pmat,"img");
	prhs[1] = matGetVariable(pmat, "k");
	prhs[2] = matGetVariable(pmat, "par1");
	prhs[3] = matGetVariable(pmat, "par2");
	prhs[4] = matGetVariable(pmat, "par3");
	/*for (int i = 0; i < 5; i++) {
		prhs[i] = matGetNextVariable(pmat, &name);
		std::cout << name << "\n";
	}*/
	matClose(pmat);
	mxArray* plhs[2];
	mexFunction(1, plhs, 4, prhs);
	

	pmat = matOpen(R"(E:\matlabcode\superpixel\code\output.mat)", "w");

	matPutVariable(pmat, "region", plhs[0]);

	matClose(pmat);

#endif // slic

	//	MATFile* pmat = matOpen(R"(E:\matlabcode\superpixel\underwater_image\code\testdata.mat)", "r");
	//
	//#ifdef debug_flag
	//	std::filesystem::create_directories(log_file.debug_file);
	//#endif // debug_flag
	//	const mxArray* prhs[5];
	//	const char* name;
	//	for (int i = 0; i < 2; i++) {
	//		prhs[i] = matGetNextVariable(pmat, &name);
	//		std::cout << name << "\n";
	//	}
	//	cv::Mat bins(1, 256, CV_64FC1, mxGetPr(prhs[0]));
	//	matClose(pmat);
	//	/*dis_to_closet_uniform_pdf(bins);*/
	//	double* p = mxGetPr(prhs[1]);
	//	OT_dis_from_bins_to_linear(bins, p[1] - p[0] , p[0]);
}
cmake_minimum_required(VERSION 3.20)
project(my_mex_project)
# Set the C++ standard you want to use (C++11 in this case)
set(CMAKE_CXX_STANDARD 23)
#the key aspect of cmake is this line
add_definitions(/DMATLAB_MEX_FILE) #define matlab macros
add_definitions(/DMX_COMPAT_32)
# Find the required Matlab package
# Find MATLAB
   find_package(VTK)

# Find Matlab
find_package(Matlab REQUIRED COMPONENTS  ENG_LIBRARY MAT_LIBRARY)
find_package(OpenCV REQUIRED)
find_package(fmt CONFIG REQUIRED)

add_library(new_target disjointset.cpp)
#target_sources(new_target PUBLIC  full_bipartitegraph.h network_simplex_simple.h tools.h disjointset.h)
# Add your source file

#target_link_libraries(level_set_4_files_parallel TBB::tbb)

set(mylist 
${VTK_USE_FILE}
)
foreach(i ${mylist})
	message(include\  ${i})
endforeach()
message(------\n------\n------\n------\n)
target_include_directories(new_target PUBLIC 
${mylist}
)


set(mylist 
 ${VTK_LIBRARIES}
 ${OpenCV_LIBS}
 ${Matlab_LIBRARIES}
 fmt::fmt
)

															 

foreach(i ${mylist})
	message(libraries\  ${i})
endforeach()
message(------\n------\n------\n------\n)

target_link_libraries(new_target ${mylist})
target_include_directories(new_target PUBLIC ${Matlab_INCLUDE_DIRS})
#target_precompile_headers(new_target PUBLIC pch.hpp)


matlab_add_mex(NAME merge_cell SRC merge_cell.cpp)
matlab_add_mex(NAME merge_cell_3D SRC merge_cell_3D.cpp)



add_executable(seg_3D main1.cpp)
add_executable(seg_1D main.cpp)
target_link_libraries(merge_cell new_target)
target_link_libraries(seg_1D new_target)
target_link_libraries(seg_3D new_target)
target_link_libraries(merge_cell_3D new_target)


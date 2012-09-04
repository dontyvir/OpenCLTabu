/*
 * ParallelSolver.cpp
 *
 *  Created on: 31-08-2012
 *      Author: donty
 */

#include "ParallelSolver.h"

namespace tabu {

ParallelSolver::ParallelSolver(unsigned int max_iterations,
		int diversification_param, unsigned int n_machines,
		unsigned int n_parts, unsigned int n_cells,
		unsigned int max_machines_cell, Matrix* incidence_matrix,
		int tabu_turns) :
		Solver(max_iterations, diversification_param, n_machines, n_parts,
				n_cells, max_machines_cell, incidence_matrix, tabu_turns) {

}

ParallelSolver::~ParallelSolver() {
}

int ParallelSolver::OpenCL_init() {

    cl_int err;

    // Platform info
    std::vector<cl::Platform> platforms;

    //HelloCL! Getting Platform Information
    err = cl::Platform::get(&platforms);
    if(err != CL_SUCCESS)
    {
        std::cout << "Platform::get() failed (" << err << ")" << std::endl;
        return SDK_FAILURE;
    }

    std::vector<cl::Platform>::iterator i;
    if(platforms.size() > 0)
    {
        for(i = platforms.begin(); i != platforms.end(); ++i)
        {
            if(!strcmp((*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str(), "Advanced Micro Devices, Inc."))
            {
                break;
            }
        }
    }
    if(err != CL_SUCCESS)
    {
        std::cout << "Platform::getInfo() failed (" << err << ")" << std::endl;
        return SDK_FAILURE;
    }

    /*
     * If we could find our platform, use it. Otherwise pass a NULL and get whatever the
     * implementation thinks we should be using.
     */

    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(*i)(), 0 };

    // Creating a context AMD platform
    cl::Context context(CL_DEVICE_TYPE_CPU, cps, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Context::Context() failed (" << err << ")\n";
        return SDK_FAILURE;
    }

    //Getting device info
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (err != CL_SUCCESS) {
        std::cout << "Context::getInfo() failed (" << err << ")\n";
        return SDK_FAILURE;
    }
    if (devices.size() == 0) {
        std::cout << "No device available\n";
        return SDK_FAILURE;
    }

    // Loading and compiling CL source
    streamsdk::SDKFile file;
    if (!file.open("TabuSolver_Kernels.cl")) {
         std::cout << "We couldn't load CL source code\n";
         return SDK_FAILURE;
    }
    cl::Program::Sources sources(1, std::make_pair(file.source().data(), file.source().size()));

    cl::Program program = cl::Program(context, sources, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Program::Program() failed (" << err << ")\n";
        return SDK_FAILURE;
    }

    err = program.build(devices);
    if (err != CL_SUCCESS) {

        if(err == CL_BUILD_PROGRAM_FAILURE)
        {
            cl::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);

            std::cout << " \n\t\t\tBUILD LOG\n";
            std::cout << " ************************************************\n";
            std::cout << str.c_str() << std::endl;
            std::cout << " ************************************************\n";
        }

        std::cout << "Program::build() failed (" << err << ")\n";
        return SDK_FAILURE;
    }

    this->kernel(program, "sum", &err);
    if (err != CL_SUCCESS) {
        std::cout << "Kernel::Kernel() failed (" << err << ")\n";
        return SDK_FAILURE;
    }
    if (err != CL_SUCCESS) {
        std::cout << "Kernel::setArg() failed (" << err << ")\n";
        return SDK_FAILURE;
    }

    this->queue(context, devices[0], 0, &err);
    if (err != CL_SUCCESS) {
        std::cout << "CommandQueue::CommandQueue() failed (" << err << ")\n";
        return SDK_FAILURE;
    }

    // Done Passed
    return SDK_SUCCESS;

}

void ParallelSolver::init() {

	Solver::init();
	OpenCL_init();

	// precómputo de vectores

	parts_machines(n_machines,std::vector<int>());
	for(unsigned int i=0;i<n_machines;i++){
		for(unsigned int j=0;j<n_parts;j++){
			if(incidence_matrix->getMatrix()[i][j] == 1){
				parts_machines[i].push_back(j);
			}
		}
	}
}

long ParallelSolver::get_cost(Solution* solution) {

	long cost = 0;

	std::vector<std::vector<int> > machines_cells(n_cells,std::vector<int>());
	for(unsigned int k=0;k<n_cells;k++){
		for(unsigned int i=0;i<n_machines;i++){
			if(solution->cell_vector[i] == (signed int)k){
				machines_cells[k].push_back(i);
			}
		}
	}

	for(unsigned int i=0;i<n_machines;i++){ // sum i=1...M
		for(unsigned int j=0;j<parts_machines[i].size();j++){
			for(unsigned int k=0;k<n_cells;k++){ // sum k=1...C
				for(unsigned int i_=0;i_<machines_cells[k].size();i_++){

					int i__ = machines_cells[k][i_];
					int j_ = parts_machines[i][j];

					int different = i__ - i;
					different = ((different | (~different + 1)) >> 31) & 1;

					cost += different * incidence_matrix->getMatrix()[i__][j_];

				}
			}
		}
	}


	for(unsigned int i=0;i<n_machines;i++){ // sum i=1...M
		for(unsigned int j=0;j<n_parts;j++){ // sum j=1...P
			if(incidence_matrix->getMatrix()[i][j] == 1){ // a_ij = 1

				//parts_machines


				for(unsigned int k=0;k<n_cells;k++){ // sum k=1...C
					if(solution->cell_vector[i] != (signed int)k){ //(1 - y_ik)


						//machines_cells

						//(z_jk)
						for(unsigned int i_=0;i_<n_machines;i_++){


							if( i != i_ &&
								solution->cell_vector[i_] == (signed int)k
								&& incidence_matrix->getMatrix()[i_][j] == 1
							){
								cost++;
							}
						}

					}
				}
			}
		}
	}




    cl_int err;

    // Running CL program
    err = queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(4, 4), cl::NDRange(2, 2)
    );

    if (err != CL_SUCCESS) {
        std::cout << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
       return SDK_FAILURE;
    }

    err = queue.finish();
    if (err != CL_SUCCESS) {
        std::cout << "Event::wait() failed (" << err << ")\n";
    }

    return 0;
}

} /* namespace tabu */

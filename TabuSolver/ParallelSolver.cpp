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

inline int different(int a,int b){

	int diff = a - b;

	diff = ((diff | (~diff + 1)) >> 31) & 1;

	return diff;
}

inline int equal(int a,int b){

	int equal = a - b;

	equal = ((equal | (~equal + 1)) >> 31) & 0;

	return equal;
}

inline int zero(int a){

	int zero = ((a | (~a + 1)) >> 31) & 0;
	return zero;
}

VariableMatrix* ParallelSolver::vector_to_var_mat(
		std::vector<std::vector<int> > vector, int max_cols) {

	VariableMatrix *mat = new VariableMatrix();

	int *storage = new int[vector.size()*max_cols];
	mat->rows = vector.size();
	mat->cols = new int[vector.size()];

	for(int i=0;i<(signed int)vector.size();i++){
		std::copy(vector[i].begin(), vector[i].begin()+vector[i].size(), storage+i*max_cols);
		mat->cols[i] = vector[i].size();
	}
	mat->storage = storage;

	return mat;
}

StaticMatrix* ParallelSolver::matrix_to_StaticMatrix(Matrix *mat, int size) {

	StaticMatrix *smat = new StaticMatrix();

	int *storage = new int[mat->rows*size];
	smat->rows = mat->rows;
	smat->cols = mat->cols;

	for(int i=0;i<(signed int)mat->rows;i++){
		std::copy(mat->storage[i].begin(), mat->storage[i].begin()+mat->storage[i].size(), storage+i*size);
	}
	smat->storage = storage;

	return smat;
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
        exit(SDK_FAILURE);
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
        exit(SDK_FAILURE);
    }

    /*
     * If we could find our platform, use it. Otherwise pass a NULL and get whatever the
     * implementation thinks we should be using.
     */

    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(*i)(), 0 };

    // Creating a context AMD platform
    context = cl::Context(CL_DEVICE_TYPE_CPU, cps, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Context::Context() failed (" << err << ")\n";
        exit(SDK_FAILURE);
    }

    //Getting device info
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (err != CL_SUCCESS) {
        std::cout << "Context::getInfo() failed (" << err << ")\n";
        exit(SDK_FAILURE);
    }
    if (devices.size() == 0) {
        std::cout << "No device available\n";
        exit(SDK_FAILURE);
    }

    // Loading and compiling CL source
    streamsdk::SDKFile file;
    if (!file.open("TabuSolver_Kernels.cl")) {
         std::cout << "We couldn't load CL source code\n";
         exit(SDK_FAILURE);
    }


    cl::Program::Sources sources(1, std::make_pair(file.source().data(), file.source().size()));

    cl::Program program = cl::Program(context, sources, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Program::Program() failed (" << err << ")\n";
        exit(SDK_FAILURE);
    }

    err = program.build(devices,"-x clc++");
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
        exit(SDK_FAILURE);
    }

    kernel = cl::Kernel(program, "get_cost", &err);
    if (err != CL_SUCCESS) {
        std::cout << "Kernel::Kernel() failed (" << err << ")\n";
        exit(SDK_FAILURE);
    }
    if (err != CL_SUCCESS) {
        std::cout << "Kernel::setArg() failed (" << err << ")\n";
        exit(SDK_FAILURE);
    }

    queue = cl::CommandQueue(context, devices[0], 0, &err);
    if (err != CL_SUCCESS) {
        std::cout << "CommandQueue::CommandQueue() failed (" << err << ")\n";
        exit(SDK_FAILURE);
    }

    // fixed input buffers

    buf_cl_params = cl::Buffer(context,
    					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    					sizeof(ClParams),
    					NULL,
    					&err);

    buf_parts_machines_storage = cl::Buffer(context,
    					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    					sizeof(int)*n_parts*n_machines,
    					NULL,
    					&err);
    buf_parts_machines_lengths = cl::Buffer(context,
    					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    					sizeof(int)*n_machines,
    					NULL,
    					&err);

    buf_incidence_matrix = cl::Buffer(context,
    					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    					sizeof(StaticMatrix),
    					NULL,
    					&err);
    buf_incidence_matrix_storage = cl::Buffer(context,
    					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    					sizeof(int)*n_parts*n_machines,
    					NULL,
    					&err);

    cl_int status;
    cl::Event writeEvt;

    ClParams *params = new ClParams();
    params->max_machines_cell = max_machines_cell;
    params->n_cells = n_cells;
    params->n_machines = n_machines;
    params->n_parts = n_parts;

    status = queue.enqueueWriteBuffer(
    		buf_cl_params,
    		CL_FALSE,
    		0,
    		sizeof(ClParams),
    		params,
    		NULL,
    		&writeEvt);

    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_max_machines_cell)");

    VariableMatrix *var_parts_machines = vector_to_var_mat(parts_machines,n_machines);
    status = queue.enqueueWriteBuffer(
    		buf_parts_machines_storage,
    		CL_FALSE,
    		0,
    		sizeof(int)*n_parts*n_machines,
    		var_parts_machines->storage,
    		NULL,
    		&writeEvt);

    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_parts_machines_storage)");
    status = queue.enqueueWriteBuffer(
    		buf_parts_machines_lengths,
    		CL_FALSE,
    		0,
    		sizeof(int)*n_machines,
    		var_parts_machines->cols,
    		NULL,
    		&writeEvt);

    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_parts_machines_storage)");


	StaticMatrix *static_incidence = matrix_to_StaticMatrix(incidence_matrix, n_parts);
    status = queue.enqueueWriteBuffer(
    		buf_incidence_matrix,
    		CL_FALSE,
    		0,
    		sizeof(StaticMatrix),
    		static_incidence,
    		NULL,
    		&writeEvt);

    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_incidence_matrix)");
    status = queue.enqueueWriteBuffer(
    		buf_incidence_matrix_storage,
    		CL_FALSE,
    		0,
    		sizeof(int)*n_parts*n_machines,
    		static_incidence->storage,
    		NULL,
    		&writeEvt);

    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_incidence_matrix_storage)");



    status = queue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    // out buffer
    buf_out_cost = cl::Buffer(context,
    					CL_MEM_READ_WRITE,
    					sizeof(long),
    					NULL, &err);

    // variable input buffers
    buf_machines_in_cells_storage = cl::Buffer(context,
    					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    					sizeof(int)*n_machines*n_cells,
    					NULL,
    					&err);
    buf_machines_in_cells_lengths = cl::Buffer(context,
    					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    					sizeof(int)*n_cells,
    					NULL,
    					&err);

    buf_machines_not_in_cells_storage = cl::Buffer(context,
    					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    					sizeof(int)*n_cells*n_machines,
    					NULL,
    					&err);
    buf_machines_not_in_cells_lengths = cl::Buffer(context,
    					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    					sizeof(int)*n_cells,
    					NULL,
    					&err);

    // Done Passed
    //OpenCL_initialized = true;
    return SDK_SUCCESS;

}

long ParallelSolver::get_cpu_cost(Solution* solution) {

	long cost = 0;

	std::vector<std::vector<int> > machines_in_cells(n_cells,std::vector<int>());
	std::vector<std::vector<int> > machines_not_in_cells(n_cells,std::vector<int>());
	for(unsigned int k=0;k<n_cells;k++){
		for(unsigned int i=0;i<n_machines;i++){
			if(solution->cell_vector[i] == (signed int)k)
				machines_in_cells[k].push_back(i);
			else
				machines_not_in_cells[k].push_back(i);
		}
	}

	for(unsigned int k=0;k<n_cells;k++){ // sum k=1...C // celdas

		//------------penalizacion y_ik <= Mmax------------------

		int machines_cell = machines_in_cells[k].size();

		int difference = machines_cell - max_machines_cell;
		int sign = 1 ^ ((unsigned int)difference >> 31); // if difference > 0 sign = 1 else sign = 0
		cost += difference * n_parts * sign;

		// -------------------------------------------------------

		int machines_not_in = machines_not_in_cells[k].size();

		for(int i_n=0;i_n<machines_not_in;i_n++){ // máquinas no en celda

			int i = machines_not_in_cells[k][i_n]; // máquina no en celda
			int machines_in = machines_in_cells[k].size();
			int parts = parts_machines[i].size();

			for(int j_=0;j_<parts;j_++){ // partes de máquinas no en la celda

				int j = parts_machines[i][j_]; // parte de máquina no en celda

				for(int i_in=0;i_in<machines_in;i_in++){

					int i_ = machines_in_cells[k][i_in]; // máquina en celda

					// costo+ si la máquina en celda tiene la parte que también es de la máquina no en celda
					cost += incidence_matrix->getMatrix()[i_][j];
				}
			}
		}
	}

	for(unsigned int j=0;j<n_parts;j++){

		int in_cells = 0;

		for(unsigned int k=0;k<n_cells;k++){

			int machines_in = machines_in_cells[k].size();

			for(int i_in=0;i_in<machines_in;i_in++){

				int i_ = machines_in_cells[k][i_in]; // máquina en celda
				int parts = parts_machines[i_].size();

				for(int j_in=0;j_in<parts;j_in++){ // partes de máquinas en la celda

					int j_ = parts_machines[i_][j_in]; // parte de máquina en celda

					in_cells += equal(j,j_);
				}

			}
		}

		// cost+=n_parts*in_cells para in_cells != 1
		cost += (in_cells - 1) * n_parts * (-1*zero(in_cells));
	}

	return cost;
}

void ParallelSolver::init() {

	// precómputo de vector
	parts_machines.assign(n_machines,std::vector<int>());
	for(unsigned int i=0;i<n_machines;i++){
		for(unsigned int j=0;j<n_parts;j++){
			if(incidence_matrix->getMatrix()[i][j] == 1){
				parts_machines[i].push_back(j);
			}
		}
	}
	OpenCL_init();
	Solver::init();
}

long ParallelSolver::get_cost(Solution* solution) {

	long cost=0;

	std::vector<std::vector<int> > machines_in_cells(n_cells,std::vector<int>());
	std::vector<std::vector<int> > machines_not_in_cells(n_cells,std::vector<int>());
	for(unsigned int k=0;k<n_cells;k++){
		for(unsigned int i=0;i<n_machines;i++){
			if(solution->cell_vector[i] == (signed int)k)
				machines_in_cells[k].push_back(i);
			else
				machines_not_in_cells[k].push_back(i);
		}
	}

    cl_int err;

    /////////////////////////runCLKernels////////////////////////////////////////
    cl_int status;
    cl_int eventStatus = CL_QUEUED;

    cl::Event writeEvt;

    VariableMatrix *var_machines_in_cells = vector_to_var_mat(machines_in_cells,n_cells);
    status = queue.enqueueWriteBuffer(
    		buf_machines_in_cells_storage,
    		CL_FALSE,
    		0,
    		sizeof(int)*n_cells*n_machines,
    		var_machines_in_cells->storage,
    		NULL,
    		&writeEvt);

    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_machines_in_cells)");
    status = queue.enqueueWriteBuffer(
    		buf_machines_in_cells_lengths,
    		CL_FALSE,
    		0,
    		sizeof(int)*n_cells,
    		var_machines_in_cells->cols,
    		NULL,
    		&writeEvt);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_machines_in_cells)");


    VariableMatrix *var_machines_not_in_cells = vector_to_var_mat(machines_not_in_cells,n_cells);
    status = queue.enqueueWriteBuffer(
    		buf_machines_not_in_cells_storage,
    		CL_FALSE,
    		0,
    		sizeof(int)*n_cells*n_machines,
    		var_machines_not_in_cells->storage,
    		NULL,
    		&writeEvt);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_machines_not_in_cells)");
    status = queue.enqueueWriteBuffer(
    		buf_machines_not_in_cells_lengths,
    		CL_FALSE,
    		0,
    		sizeof(int)*n_cells,
    		var_machines_not_in_cells->cols,
    		NULL,
    		&writeEvt);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_machines_not_in_cells)");



    status = queue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = writeEvt.getInfo<cl_int>(
                    CL_EVENT_COMMAND_EXECUTION_STATUS,
                    &eventStatus);
        CHECK_OPENCL_ERROR(status, "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");
    }

    //////////////////////////////////////////////////////////////////

    status = kernel.setArg(0, &buf_out_cost);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_out_cost)");

    status = kernel.setArg(1, &buf_cl_params);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_cl_params)");

    status = kernel.setArg(2, &buf_parts_machines_storage);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_parts_machines_storage)");

    status = kernel.setArg(3, &buf_parts_machines_lengths);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_parts_machines_lengths)");

    status = kernel.setArg(4, &buf_machines_in_cells_storage);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_machines_in_cells_storage)");

    status = kernel.setArg(5, &buf_machines_in_cells_lengths);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_machines_in_cells_lengths)");

    status = kernel.setArg(6, &buf_machines_not_in_cells_storage);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_machines_not_in_cells_storage)");

    status = kernel.setArg(7, &buf_machines_not_in_cells_lengths);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_machines_not_in_cells_lengths)");

    status = kernel.setArg(8, &buf_incidence_matrix);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_incidence_matrix)");

    status = kernel.setArg(9, &buf_incidence_matrix_storage);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_incidence_matrix_storage)");

    cl::NDRange globalThreads(2, 2);
    cl::NDRange localThreads(2, 2);
    cl::Event ndrEvt;

    // Running CL program
    err = queue.enqueueNDRangeKernel(
        kernel, 3, globalThreads, localThreads, NULL, &ndrEvt
    );

    if (err != CL_SUCCESS) {
        std::cout << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
       return SDK_FAILURE;
    }

    ////////////////////////////////////////////////////////////////////////

    status = queue.flush();
     CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

     eventStatus = CL_QUEUED;
     while(eventStatus != CL_COMPLETE)
     {
         status = ndrEvt.getInfo<cl_int>(
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     &eventStatus);
         CHECK_OPENCL_ERROR(status, "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");
     }

     // Enqueue readBuffer
     cl::Event readEvt;
     status = queue.enqueueReadBuffer(
    		 	 buf_out_cost,
                 CL_FALSE,
                 0,
                 sizeof(long),
                 (void *)&cost,
                 NULL,
                 &readEvt);
     CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueReadBuffer failed. (outputImageBuffer)");

     status = queue.flush();
     CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

     eventStatus = CL_QUEUED;
     while(eventStatus != CL_COMPLETE)
     {
         status = readEvt.getInfo<cl_int>(
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     &eventStatus);
         CHECK_OPENCL_ERROR(status, "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");
     }

    ////////////////////////////////////////////////////////////////////////

    return cost;
}

} /* namespace tabu */

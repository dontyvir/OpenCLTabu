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

	//cl cleanup



}

void print_array(int *array,int size){

	for(int i=0;i<size;i++){
		std::cout << array[i] << " ";
	}
	std::cout << std::endl;
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

	int vector_size = vector.size();

	int *storage = new cl_int[vector_size*max_cols];
	mat->rows = vector.size();
	mat->cols = new cl_int[vector_size]; // cols en cada fila

	memset(storage,0,sizeof(cl_int)*vector_size*max_cols);
	memset(mat->cols,0,sizeof(cl_int)*vector_size);

	for(int i=0;i<(signed int)vector_size;i++){
		std::copy(vector[i].begin(), vector[i].begin()+vector[i].size(), storage+(i*max_cols));
		mat->cols[i] = vector[i].size();
	}
	mat->storage = storage;

    //-------------------------------------------------------------

    print_array(storage, vector.size()*max_cols);
    print_array(mat->cols, vector.size());

	return mat;
}

StaticMatrix* ParallelSolver::matrix_to_StaticMatrix(Matrix *mat) {

	StaticMatrix *smat = new StaticMatrix();

	int rows = mat->rows;
	int cols = mat->cols;

	int *storage = new cl_int[rows*cols];
	smat->rows = rows;
	smat->cols = cols;

	memset(storage,0,sizeof(cl_int)*rows*cols);

	for(int i=0;i<(signed int)mat->rows;i++){
		std::copy(mat->storage[i].begin(), mat->storage[i].begin()+mat->storage[i].size(), storage+(i*cols));
	}
	smat->storage = storage;

    print_array(storage, rows*cols);

	return smat;
}

int ParallelSolver::OpenCL_init() {

    cl_int err;

    // Platform info
    std::vector<cl::Platform> platforms;

    //Getting Platform Information
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

    queue = cl::CommandQueue(context, devices[0], 0, &err);
    if (err != CL_SUCCESS) {
        std::cout << "CommandQueue::CommandQueue() failed (" << err << ")\n";
        exit(SDK_FAILURE);
    }

    // fixed input buffers

    buf_cl_params = cl::Buffer(context,
    					CL_MEM_READ_ONLY,
    					sizeof(ClParams),
    					NULL,
    					&err);

    buf_parts_machines_storage = cl::Buffer(context,
    					CL_MEM_READ_ONLY,
    					sizeof(int)*n_parts*n_machines,
    					NULL,
    					&err);
    buf_parts_machines_lengths = cl::Buffer(context,
    					CL_MEM_READ_ONLY,
    					sizeof(int)*n_machines,
    					NULL,
    					&err);

    buf_incidence_matrix_storage = cl::Buffer(context,
    					CL_MEM_READ_ONLY,
    					sizeof(int)*n_parts*n_machines,
    					NULL,
    					&err);

    // out buffer
    buf_out_cost = cl::Buffer(context,
    					CL_MEM_WRITE_ONLY,
    					sizeof(cl_uint),
    					NULL, &err);

    cl_int status;
    cl::Event writeEvt;

    cl_uint cost = 0;
    status = queue.enqueueWriteBuffer(
    		buf_out_cost,
    		CL_FALSE,
    		0,
    		sizeof(cl_uint),
    		(void *)&cost,
    		NULL,
    		&writeEvt);

    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_out_cost)");

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
    		(void *)params,
    		NULL,
    		&writeEvt);

    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_max_machines_cell)");

    std::cout << "parts_machines" << std::endl;
    VariableMatrix *var_parts_machines = vector_to_var_mat(parts_machines,n_machines);
    std::cout << "enqueue parts_machines" << std::endl;
    status = queue.enqueueWriteBuffer(
    		buf_parts_machines_storage,
    		CL_FALSE,
    		0,
    		sizeof(int)*n_parts*n_machines,
    		(void *)(var_parts_machines->storage),
    		NULL,
    		&writeEvt);

    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_parts_machines_storage)");
    status = queue.enqueueWriteBuffer(
    		buf_parts_machines_lengths,
    		CL_FALSE,
    		0,
    		sizeof(int)*n_machines,
    		(void *)(var_parts_machines->cols),
    		NULL,
    		&writeEvt);

    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_parts_machines_storage)");

    std::cout << "incidence_matrix" << std::endl;
	StaticMatrix *static_incidence = matrix_to_StaticMatrix(incidence_matrix);
    std::cout << "enqueue incidence" << std::endl;
    status = queue.enqueueWriteBuffer(
    		buf_incidence_matrix_storage,
    		CL_FALSE,
    		0,
    		sizeof(int)*n_parts*n_machines,
    		(void *)(static_incidence->storage),
    		NULL,
    		&writeEvt);

    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_incidence_matrix_storage)");

    status = queue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");
    std::cout << "flush" << std::endl;

    std::cout << "delete" << std::endl;
    delete[] static_incidence->storage;
    delete static_incidence;

    delete[] var_parts_machines->storage;
    delete[] var_parts_machines->cols;
    delete var_parts_machines;

    delete params;

    // variable input buffers
    buf_machines_in_cells_storage = cl::Buffer(context,
    					CL_MEM_READ_ONLY,
    					sizeof(cl_int)*n_machines*n_cells,
    					NULL,
    					&err);
    buf_machines_in_cells_lengths = cl::Buffer(context,
    					CL_MEM_READ_ONLY,
    					sizeof(cl_int)*n_cells,
    					NULL,
    					&err);

    buf_machines_not_in_cells_storage = cl::Buffer(context,
    					CL_MEM_READ_ONLY,
    					sizeof(cl_int)*n_cells*n_machines,
    					NULL,
    					&err);
    buf_machines_not_in_cells_lengths = cl::Buffer(context,
    					CL_MEM_READ_ONLY,
    					sizeof(cl_int)*n_cells,
    					NULL,
    					&err);
    std::cout << "return" << std::endl;
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

	cl_uint cost=0;

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

    std::cout << "machines_in_cells" << std::endl;
    VariableMatrix *var_machines_in_cells = vector_to_var_mat(machines_in_cells,n_machines);
    status = queue.enqueueWriteBuffer(
    		buf_machines_in_cells_storage,
    		CL_FALSE,
    		0,
    		sizeof(int)*n_cells*n_machines,
    		(void *)(var_machines_in_cells->storage),
    		NULL,
    		&writeEvt);

    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_machines_in_cells)");
    status = queue.enqueueWriteBuffer(
    		buf_machines_in_cells_lengths,
    		CL_FALSE,
    		0,
    		sizeof(int)*n_cells,
    		(void *)(var_machines_in_cells->cols),
    		NULL,
    		&writeEvt);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_machines_in_cells)");

    std::cout << "machines_not_in_cells" << std::endl;
    VariableMatrix *var_machines_not_in_cells = vector_to_var_mat(machines_not_in_cells,n_machines);
    std::cout << "enqueue machines_not_in_cells" << std::endl;
    status = queue.enqueueWriteBuffer(
    		buf_machines_not_in_cells_storage,
    		CL_FALSE,
    		0,
    		sizeof(cl_int)*n_cells*n_machines,
    		(void *)(var_machines_not_in_cells->storage),
    		NULL,
    		&writeEvt);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteBuffer() failed. (buf_machines_not_in_cells)");
    std::cout << "enqueue lengths machines_not_in_cells" << std::endl;

    status = queue.enqueueWriteBuffer(
    		buf_machines_not_in_cells_lengths,
    		CL_FALSE,
    		0,
    		sizeof(cl_int)*n_cells,
    		(void *)(var_machines_not_in_cells->cols),
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

    std::cout << "delete" << std::endl;
    delete[] var_machines_not_in_cells->storage;
    delete[] var_machines_not_in_cells->cols;
    delete var_machines_not_in_cells;

    delete[] var_machines_in_cells->storage;
    delete[] var_machines_in_cells->cols;
    delete var_machines_in_cells;

    // set kernel args

    status = kernel.setArg(0, sizeof(cl_uint*),&buf_out_cost);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_out_cost)");

    status = kernel.setArg(1, sizeof(ClParams*), &buf_cl_params);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_cl_params)");

    status = kernel.setArg(2, sizeof(cl_int *),&buf_parts_machines_storage);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_parts_machines_storage)");

    status = kernel.setArg(3, sizeof(cl_int*),&buf_parts_machines_lengths);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_parts_machines_lengths)");

    status = kernel.setArg(4, sizeof(cl_int*),&buf_machines_in_cells_storage);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_machines_in_cells_storage)");

    status = kernel.setArg(5, sizeof(cl_int*),&buf_machines_in_cells_lengths);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_machines_in_cells_lengths)");

    status = kernel.setArg(6, sizeof(cl_int*),&buf_machines_not_in_cells_storage);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_machines_not_in_cells_storage)");

    status = kernel.setArg(7, sizeof(cl_int*),&buf_machines_not_in_cells_lengths);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_machines_not_in_cells_lengths)");

    status = kernel.setArg(8, sizeof(cl_int*),&buf_incidence_matrix_storage);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_incidence_matrix_storage)");

    cl::NDRange globalThreads(n_machines, n_parts, n_cells );
    //cl::NDRange localThreads(1, 1, 1);
    cl::Event ndrEvt;

    // Running CL program
    err = queue.enqueueNDRangeKernel(
        kernel,cl::NullRange, globalThreads, cl::NullRange, 0, &ndrEvt
    );

    if (err != CL_SUCCESS) {
        std::cout << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
       return SDK_FAILURE;
    }

    std::cout << "flush kernel completion" << std::endl;
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
     std::cout << "enqueue read buffer" << std::endl;
     // Enqueue readBuffer
     cl::Event readEvt;
     status = queue.enqueueReadBuffer(
    		 	 buf_out_cost,
                 CL_FALSE,
                 0,
                 sizeof(cl_uint),
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

    return (long)cost;
}

} /* namespace tabu */

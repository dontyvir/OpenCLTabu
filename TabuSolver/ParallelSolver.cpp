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

	out_cost = NULL;
	params = NULL;
	gsol = NULL;
	min_i = 0;
	min_cost = UINT_MAX;

}

ParallelSolver::~ParallelSolver() {

	delete params;
//	delete[] gsol;
//	delete[] out_cost;
}

void print_array(int *array,int size){

	for(int i=0;i<size;i++){
		std::cout << array[i] << " ";
	}
	std::cout << std::endl;
}

void print_vector(std::vector<std::vector<int> > vector){

	for(std::vector<int>::size_type i = 0; i != vector.size(); i++) {
		for(std::vector<int>::size_type j = 0; j != vector[i].size(); j++) {
			std::cout << vector[i][j] << " ";
		}
	}
	std::cout << std::endl;
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

    //print_array(storage, rows*cols);

	return smat;
}

int ParallelSolver::OpenCL_init() {

    cl_int err;

    cl_device_type dev_type = CL_DEVICE_TYPE_CPU;

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
        	printf("%s\n",(*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str());
        	if(!strcmp((*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str(), "Advanced Micro Devices, Inc."))
			//if(!strcmp((*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str(), "Intel(R) Corporation"))
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
    context = cl::Context(dev_type, cps, NULL, NULL, &err);
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
        exit(SDK_FAILURE);
    }

    kernel_local_search = cl::Kernel(program, "local_search", &err);
    if (err != CL_SUCCESS) {
        std::cout << "Kernel::Kernel() failed (" << err << ")\n";
        exit(SDK_FAILURE);
    }

    kernel_cost = cl::Kernel(program, "costs", &err);
    if (err != CL_SUCCESS) {
        std::cout << "Kernel::Kernel() failed (" << err << ")\n";
        exit(SDK_FAILURE);
    }

    kernel_pen_Mmax = cl::Kernel(program, "penalizaciones_Mmax", &err);
    if (err != CL_SUCCESS) {
        std::cout << "Kernel::Kernel() failed (" << err << ")\n";
        exit(SDK_FAILURE);
    }

    kernel_cost_min = cl::Kernel(program, "mejor_solucion", &err);
    if (err != CL_SUCCESS) {
        std::cout << "Kernel::Kernel() failed (" << err << ")\n";
        exit(SDK_FAILURE);
    }

    // queue profiling enabled
    queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        std::cout << "CommandQueue::CommandQueue() failed (" << err << ")\n";
        exit(SDK_FAILURE);
    }

    // fixed input buffers

    params = new ClParams();
    params->max_machines_cell = max_machines_cell;
    params->n_cells = n_cells;
    params->n_machines = n_machines;
    params->n_parts = n_parts;

    buf_cl_params = cl::Buffer(context,
    					CL_MEM_USE_HOST_PTR,
    					sizeof(ClParams),
    					(void *)params,
    					&err);

	StaticMatrix *static_incidence = matrix_to_StaticMatrix(incidence_matrix);
    buf_incidence_matrix = cl::Buffer(context,
    					CL_MEM_USE_HOST_PTR,
    					sizeof(cl_int)*n_parts*n_machines,
    		    		(void *)(static_incidence->storage),
    					&err);

    // out buffer
    //out_cost = new cl_uint[n_machines*n_machines];
    //memset(out_cost,0,sizeof(cl_uint)*n_machines*n_machines);

    buf_out_cost = cl::Buffer(context,
    					CL_MEM_ALLOC_HOST_PTR,
    					sizeof(cl_uint)*n_machines*n_machines,
    					NULL, &err);

    buf_cur_sol = cl::Buffer(context,
    					CL_MEM_USE_HOST_PTR,
    					sizeof(cl_int)*n_machines,
    					current_solution,
    					&err);

    //gsol = new cl_int[n_machines*n_machines*n_machines];

    buf_gsol = cl::Buffer(context,
    					CL_MEM_ALLOC_HOST_PTR,//CL_MEM_USE_PERSISTENT_MEM_AMD,
    					sizeof(cl_int)*n_machines*n_machines*n_machines,
    					NULL,
    					&err);

    buf_min_i = cl::Buffer(context,
    					CL_MEM_USE_HOST_PTR,
    					sizeof(cl_uint),
    					&min_i,
    					&err);

    buf_min_cost = cl::Buffer(context,
    					CL_MEM_USE_HOST_PTR,
    					sizeof(cl_uint),
    					&min_cost,
    					&err);

    cl_int status;

    status = queue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    // set kernel args

    // kernel local search

    status = kernel_local_search.setArg(0, sizeof(ClParams*),&buf_cl_params);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_cl_params)");

    status = kernel_local_search.setArg(1, sizeof(cl_uint*),&buf_cur_sol);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_cur_sol)");

    status = kernel_local_search.setArg(2, sizeof(cl_int)*n_machines, NULL);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_lsol)");

    status = kernel_local_search.setArg(3, sizeof(cl_int*),&buf_gsol);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_gsol)");


    // kernel cost
    status = kernel_cost.setArg(0, sizeof(cl_uint*),&buf_out_cost);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_out_cost)");

    status = kernel_cost.setArg(1, sizeof(ClParams*), &buf_cl_params);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_cl_params)");

    status = kernel_cost.setArg(2, sizeof(cl_int*),&buf_incidence_matrix);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_incidence_matrix)");

    status = kernel_cost.setArg(3, sizeof(cl_int*),&buf_gsol);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_gsol)");

    status = kernel_cost.setArg(4, sizeof(cl_int)*n_machines,NULL);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (parts_cells)");

    // kernel penalización Mmax
    status = kernel_pen_Mmax.setArg(0, sizeof(cl_uint*),&buf_out_cost);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_out_cost)");

    status = kernel_pen_Mmax.setArg(1, sizeof(ClParams*), &buf_cl_params);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_cl_params)");

    status = kernel_pen_Mmax.setArg(2, sizeof(cl_int*),&buf_gsol);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_gsol)");

    status = kernel_pen_Mmax.setArg(3, sizeof(cl_int*), NULL);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_machines_cell)");


    // kernel reducción costo mínimo
    status = kernel_cost_min.setArg(0, sizeof(cl_uint*),&buf_out_cost);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_out_cost)");

    status = kernel_cost_min.setArg(1, sizeof(cl_int*),&buf_gsol);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_gsol)");

    status = kernel_cost_min.setArg(2, sizeof(cl_uint*), &buf_min_i);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_min_i)");

    status = kernel_cost_min.setArg(3, sizeof(cl_uint*), &buf_min_cost);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (buf_min_cost)");


    //std::cout << "delete" << std::endl;
    delete[] static_incidence->storage;
    delete static_incidence;

    //std::cout << "return" << std::endl;
    return SDK_SUCCESS;

}

void ParallelSolver::init() {

	OpenCL_init();
	Solver::init();
}

int ParallelSolver::local_search(){

	// moves : vector permutation
    cl_int err;
    /////////////////////////runCLKernels////////////////////////////////////////

    cl::Event writeEvt1,writeEvt2,writeEvt3,writeEvt4;

    out_cost = (cl_uint *)queue.enqueueMapBuffer(buf_out_cost,CL_FALSE,CL_MAP_WRITE,0,sizeof(cl_uint)*n_machines*n_machines,NULL,&writeEvt2,&err);
    CHECK_OPENCL_ERROR(err, "CommandQueue::enqueueMapBuffer failed. (buf_out_cost)");

    gsol = (cl_int *)queue.enqueueMapBuffer(buf_gsol,CL_FALSE,CL_MAP_WRITE,0,sizeof(cl_int)*n_machines*n_machines*n_machines,NULL,&writeEvt1,&err);
    CHECK_OPENCL_ERROR(err, "CommandQueue::enqueueMapBuffer failed. (buf_gsol)");

    err = queue.flush();
    CHECK_OPENCL_ERROR(err, "cl::CommandQueue.flush failed.");

    std::vector<cl::Event> write_events;
    write_events.push_back(writeEvt1);
    write_events.push_back(writeEvt2);
    cl::WaitForEvents(write_events);

    // llenar buffer gsol
    for(unsigned int i=0;i<n_machines*n_machines;i++){
    	memcpy(gsol+(i*n_machines),current_solution->cell_vector,sizeof(cl_int)*n_machines);
    }

    //reset cost buffer
    memset(out_cost,0,sizeof(cl_uint)*n_machines*n_machines);
    min_cost = UINT_MAX;

	err = queue.enqueueUnmapMemObject(buf_gsol,gsol,NULL,&writeEvt3);
    CHECK_OPENCL_ERROR(err, "CommandQueue::enqueueUnMapMemObject failed. (buf_gsol)");

	err = queue.enqueueUnmapMemObject(buf_out_cost,out_cost,NULL,&writeEvt4);
    CHECK_OPENCL_ERROR(err, "CommandQueue::enqueueUnMapMemObject failed. (buf_out_cost)");

    err = queue.flush();
    CHECK_OPENCL_ERROR(err, "cl::CommandQueue.flush failed.");

	write_events.clear();
    write_events.push_back(writeEvt3);
    write_events.push_back(writeEvt4);
    cl::WaitForEvents(write_events);

    cl::NDRange globalThreads_local_search(n_machines,n_machines);
    cl::NDRange globalThreads_cost(n_machines*n_machines*n_machines, n_machines,n_parts);
    cl::NDRange localThreads_cost(n_machines,n_machines,1);
    cl::NDRange globalThreads_pen_Mmax(n_cells,n_machines*n_machines,n_machines);
    cl::NDRange localThreads_pen_Mmax(1,1,n_machines);
    cl::NDRange globalThreads_cost_min(n_machines*n_machines);
    cl::Event ndrEvt;
    cl::Event kernel_local_search_evt;
    cl::Event kernel_costos_evt;
    cl::Event kernel_penMmax_evt;
    cl::Event kernel_cost_min_evt;

    // kernel de costos ---------------------------------
    err = queue.enqueueNDRangeKernel(
    		kernel_local_search,cl::NullRange, globalThreads_local_search, cl::NullRange, NULL, &kernel_local_search_evt
    );

    if (err != CL_SUCCESS) { std::cout << "CommandQueue::enqueueNDRangeKernel() failed (" << err << ")\n"; }

    err = queue.enqueueNDRangeKernel(
    		kernel_cost,cl::NullRange, globalThreads_cost, localThreads_cost, 0, &kernel_costos_evt
    );

    if (err != CL_SUCCESS) { std::cout << "CommandQueue::enqueueNDRangeKernel() failed (" << err << ")\n"; }

    err = queue.enqueueNDRangeKernel(
    		kernel_pen_Mmax,cl::NullRange, globalThreads_pen_Mmax, localThreads_pen_Mmax, 0, &kernel_penMmax_evt
    );

    if (err != CL_SUCCESS) { std::cout << "CommandQueue::enqueueNDRangeKernel() failed (" << err << ")\n"; }

    err = queue.enqueueNDRangeKernel(
    		kernel_cost_min,cl::NullRange, globalThreads_cost_min, cl::NullRange, 0, &kernel_cost_min_evt
    );

    if (err != CL_SUCCESS) { std::cout << "CommandQueue::enqueueNDRangeKernel() failed (" << err << ")\n"; }

    err=queue.flush();
    CHECK_OPENCL_ERROR(err, "cl::CommandQueue.flush failed.");

    std::vector<cl::Event> events;
    events.push_back(kernel_local_search_evt);
    events.push_back(kernel_costos_evt);
    events.push_back(kernel_penMmax_evt);
    events.push_back(kernel_cost_min_evt);

    cl::WaitForEvents(events);

     Solution *local_best = new Solution(n_machines);
     // Enqueue readBuffer
     cl::Event readEvt3;

     gsol = (cl_int *)queue.enqueueMapBuffer(buf_gsol,CL_FALSE,CL_MAP_READ,0,sizeof(cl_int)*n_machines*n_machines*n_machines,NULL,&readEvt3,&err);
     CHECK_OPENCL_ERROR(err, "CommandQueue::enqueueMapBuffer failed. (buf_min_cost)");

     err = queue.flush();
     CHECK_OPENCL_ERROR(err, "cl::CommandQueue.flush failed.");

     std::vector<cl::Event> read_events;
     read_events.push_back(readEvt3);

     cl::WaitForEvents(read_events);

     memcpy(local_best->cell_vector,gsol+min_i,sizeof(cl_int)*n_machines);

     //TODO
	 printf("min_cost: %i\n",min_cost);
     min_cost = get_cost(local_best);

     //local_best->cost = get_cost(local_best);
     local_best->cost = min_cost;

 	 err = queue.enqueueUnmapMemObject(buf_gsol,gsol,NULL,&readEvt3);
     CHECK_OPENCL_ERROR(err, "CommandQueue::enqueueUnMapMemObject failed. (buf_gsol)");

     err = queue.flush();
     CHECK_OPENCL_ERROR(err, "cl::CommandQueue.flush failed.");

	 delete current_solution;

	 current_solution = local_best;

	if(min_cost < (uint)global_best_cost){
		delete global_best;
		global_best = current_solution->clone();
		global_best_cost = min_cost;
	}

     read_events.clear();
     read_events.push_back(readEvt3);

     cl::WaitForEvents(read_events);

	 return 0;
}

} /* namespace tabu */

/*
 * ParallelSolver.h
 *
 *  Created on: 31-08-2012
 *      Author: donty
 */

#ifndef PARALLELSOLVER_H_
#define PARALLELSOLVER_H_
#define __NO_STD_STRING

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <CL/cl.hpp>
#include <SDKUtil/SDKCommon.hpp>
#include <SDKUtil/SDKFile.hpp>
#include <SDKUtil/SDKApplication.hpp>
#include "Solver.h"

namespace tabu {

class StaticMatrix {

	public:
		cl_int rows;
		cl_int cols;
		cl_int *storage;
};

class VariableMatrix {

	public:
		cl_int rows;
		cl_int *cols; // length = row
		cl_int *storage;
};

typedef struct cl_params {

	cl_int n_cells;
	cl_int n_parts;
	cl_int n_machines;
	cl_int max_machines_cell;
} ClParams;

class ParallelSolver: public tabu::Solver {
public:
	ParallelSolver(unsigned int max_iterations, int diversification_param,
			unsigned int n_machines, unsigned int n_parts, unsigned int n_cells,
			unsigned int max_machines_cell, Matrix *incidence_matrix,
			int tabu_turns);
	virtual ~ParallelSolver();

private:
	cl::Context context;
	cl::Kernel kernel;
	cl::Kernel kernel1;
	cl::CommandQueue queue;
	long get_cost(Solution *solution);
	long get_cpu_cost(Solution *solution);
	int OpenCL_init();
	void init();
	VariableMatrix *vector_to_var_mat(std::vector<std::vector<int> > vector,int max_cols);
	StaticMatrix *matrix_to_StaticMatrix(Matrix *mat);
	std::vector<std::vector<int> > parts_machines;

	cl::Buffer buf_out_cost;
    cl::Buffer buf_cl_params;

    cl::Buffer buf_parts_machines_storage;
    cl::Buffer buf_parts_machines_lengths;

    cl::Buffer buf_machines_in_cells_storage;
    cl::Buffer buf_machines_in_cells_lengths;

    cl::Buffer buf_machines_not_in_cells_storage;
    cl::Buffer buf_machines_not_in_cells_lengths;

    cl::Buffer buf_incidence_matrix_storage;
};

} /* namespace tabu */
#endif /* PARALLELSOLVER_H_ */

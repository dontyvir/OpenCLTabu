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
		int getElem(int row, int col);
		int *getRow(int row);
		int rows;
		int cols;
		int *storage;
};

class VariableMatrix {

	public:
		int getElem(int row, int col);
		int *getRow(int row);
		int rows;
		int *cols; // length = row
		int *storage;
};

typedef struct cl_params {

	int n_cells;
	int n_parts;
	int n_machines;
	int max_machines_cell;
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
	cl::CommandQueue queue;
	long get_cost(Solution *solution);
	long get_cpu_cost(Solution *solution);
	int OpenCL_init();
	void init();
	VariableMatrix *vector_to_var_mat(std::vector<std::vector<int> > vector,int max_cols);
	StaticMatrix *matrix_to_StaticMatrix(Matrix *mat, int size);
	std::vector<std::vector<int> > parts_machines;

	cl::Buffer buf_out_cost;
    cl::Buffer buf_cl_params;

    cl::Buffer buf_parts_machines_storage;
    cl::Buffer buf_parts_machines_lengths;

    cl::Buffer buf_machines_in_cells_storage;
    cl::Buffer buf_machines_in_cells_lengths;

    cl::Buffer buf_machines_not_in_cells_storage;
    cl::Buffer buf_machines_not_in_cells_lengths;

    cl::Buffer buf_incidence_matrix;
    cl::Buffer buf_incidence_matrix_storage;
};

} /* namespace tabu */
#endif /* PARALLELSOLVER_H_ */

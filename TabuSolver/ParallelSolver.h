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
	std::vector<std::vector<int> > parts_machines;
};

} /* namespace tabu */
#endif /* PARALLELSOLVER_H_ */

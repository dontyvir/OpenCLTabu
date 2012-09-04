/*
 * ParallelSolver.h
 *
 *  Created on: 31-08-2012
 *      Author: donty
 */

#ifndef PARALLELSOLVER_H_
#define PARALLELSOLVER_H_

#include "Solver.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <SDKFile.hpp>
#include <SDKCommon.hpp>
#include <SDKApplication.hpp>

#define __NO_STD_STRING

#include <CL/cl.hpp>

namespace tabu {

class ParallelSolver: public tabu::Solver {
public:
	ParallelSolver(unsigned int max_iterations, int diversification_param,
			unsigned int n_machines, unsigned int n_parts, unsigned int n_cells,
			unsigned int max_machines_cell, Matrix *incidence_matrix,
			int tabu_turns);
	virtual ~ParallelSolver();

private:
	cl::Kernel kernel;
	cl::CommandQueue queue;
	long get_cost(Solution *solution);
	int OpenCL_init();
	void init();
	std::vector<std::vector<int> > parts_machines;
};

} /* namespace tabu */
#endif /* PARALLELSOLVER_H_ */

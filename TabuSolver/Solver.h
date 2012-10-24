/*
 * Solver.h
 *
 *  Created on: 14-08-2012
 *      Author: donty
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include "Solution.h"
#include "Matrix.h"
#include "TabuList.h"
#include <climits>
#include <iostream>
#include <fstream>

#ifndef SOLVER_H_
#define SOLVER_H_

namespace tabu {

class Solver {
public:
	Solver(unsigned int max_iterations, int diversification_param,
			unsigned int n_machines, unsigned int n_parts, unsigned int n_cells,
			unsigned int max_machines_cell, Matrix *incidence_matrix,
			int tabu_turns);
	virtual ~Solver();
	void set_incidence_matrix(Matrix *incidence_matrix);
	bool es_factible(Solution *solution);
	int get_costo_real(Solution *solution);
	Solution *solve();
	Matrix *incidence_matrix;
	Solution *current_solution;
	Solution *global_best;
	int global_best_cost;

protected:
	TabuList *tabu_list;
	int tabu_turns;
	int tabu_list_max_length;
	unsigned int n_iterations;
	int diversification_param;
	unsigned int max_iterations;
	unsigned int n_machines;
	unsigned int n_parts;
	unsigned int max_machines_cell;
	virtual void init();
	void local_search();
	void global_search();
	unsigned int n_cells;
	virtual long get_cost(Solution *solution);
	void print_solution(Solution *sol);
	void print_file_solution(unsigned int iteration, Solution *sol, std::ofstream &file);
	double iter_cost_time;
	double total_cost_time;
	std::vector<std::vector<int> > parts_machines;
};

} /* namespace tabu */
#endif /* SOLVER_H_ */

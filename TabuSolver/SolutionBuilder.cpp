/*
 * SolutionBuilder.cpp
 *
 *  Created on: 14-08-2012
 *      Author: donty
 */

#include "SolutionBuilder.h"

namespace tabu {

SolutionBuilder::SolutionBuilder(unsigned int m) {
	this->m = m;
}

SolutionBuilder::~SolutionBuilder() {
}

Solution *SolutionBuilder::create_solution() {

	Solution *sol = new Solution(m);
	return sol;
}

Solution *SolutionBuilder::copy_solution(Solution *solution) {
	Solution *new_sol = solution;
	return new_sol;
}

void SolutionBuilder::destroy_solution(Solution *solution) {
	delete solution;
}

} /* namespace tabu */

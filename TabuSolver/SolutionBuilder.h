/*
 * SolutionBuilder.h
 *
 *  Created on: 14-08-2012
 *      Author: donty
 */

#ifndef SOLUTIONBUILDER_H_
#define SOLUTIONBUILDER_H_

#include "Solution.h"

namespace tabu {

class SolutionBuilder {
public:
	SolutionBuilder(unsigned int m);
	virtual ~SolutionBuilder();
	Solution *create_solution();
	Solution *copy_solution(Solution *solution);
	void destroy_solution(Solution *solution);
private:
	unsigned int m;
};

} /* namespace tabu */
#endif /* SOLUTIONBUILDER_H_ */

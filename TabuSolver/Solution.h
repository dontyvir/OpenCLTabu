/*
 * Solution.h
 *
 *  Created on: 14-08-2012
 *      Author: donty
 */

#include <iostream>

#ifndef SOLUTION_H_
#define SOLUTION_H_

namespace tabu {

class Solution {
public:
	Solution(unsigned int n_machines);
	virtual ~Solution();
	int *cell_vector;
	void init();
	int exchange(unsigned int i,unsigned int j);
	Solution *clone();
	int cost;
private:
	unsigned int n_machines;

};

} /* namespace tabu */
#endif /* SOLUTION_H_ */

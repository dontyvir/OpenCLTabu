/*
 * Solution.cpp
 *
 *  Created on: 14-08-2012
 *      Author: donty
 */

#include "Solution.h"

namespace tabu {

/* +----------------------+
 * | k_ |  k_  |  k_ | k_ |
 * +----------------------+
 * |_______n_machines_____|
 */

Solution::Solution(unsigned int n_machines) {
	this->n_machines = n_machines;
	this->cell_vector = new int[n_machines];
	//std::fill( this->cell_vector, this->cell_vector+n_machines, -1 );
}

Solution::~Solution() {
	delete[] this->cell_vector;
}

void Solution::init(){

}

int Solution::exchange(unsigned int i, unsigned int j) {
	int ret = 0;
	if(i!=j && i > 0 && j > 0 && i < n_machines && j < n_machines &&
			(cell_vector[i] != cell_vector[j])
	){
		int aux = cell_vector[i];
		cell_vector[i] = cell_vector[j];
		cell_vector[j] = aux;
		ret = 1;
	}
	return ret;
}

Solution* Solution::clone() {

	Solution *new_sol = new Solution(this->n_machines);
//	std::copy(this->cell_vector, this->cell_vector+n_machines,
//				new_sol->cell_vector);

	for(unsigned int i=0;i<n_machines;i++)
		new_sol->cell_vector[i] = this->cell_vector[i];

	return new_sol;
}

} /* namespace tabu */

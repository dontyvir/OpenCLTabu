/*
 * TabuList.cpp
 *
 *  Created on: 24-08-2012
 *      Author: donty
 */

#include "TabuList.h"

namespace tabu {

TabuList::TabuList(unsigned int n_machines, int tabu_turns) {
	this->n_machines = n_machines;
	this->tabu_turns = tabu_turns;
}

TabuList::~TabuList() {

}

bool TabuList::is_tabu(Solution *sol) {

	bool tabu = false;

	int *comparison_vector = sol->cell_vector;

	std::list<Tabu_item>::iterator itr;
	for ( itr = tabu_list.begin(); itr != tabu_list.end(); itr++)
	{
		int *solution_vector = itr->solution->cell_vector;
		unsigned int i=0;
		for(i=0;i<n_machines;i++){
			if(solution_vector[i] != comparison_vector[i])
				break;
		}

		if(i==n_machines){
			tabu = true;
			break;
		}
	}

	return tabu;
}

void TabuList::update_tabu() {
	std::list<Tabu_item>::iterator itr;
	for ( itr = tabu_list.begin(); itr != tabu_list.end(); itr++ )
	{
		itr->turns--;
		if(itr->turns == 0){
			itr = tabu_list.erase(itr);
		}
	}
}

void TabuList::add_tabu(Solution* sol) {

	Tabu_item *item = new Tabu_item();
	item->solution = sol->clone();
	item->turns = tabu_turns;
	tabu_list.push_back(*item);
}

tabu_item::~tabu_item(){
	delete this->solution;
}

} /* namespace tabu */

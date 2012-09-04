/*
 * TabuList.h
 *
 *  Created on: 24-08-2012
 *      Author: donty
 */

#ifndef TABULIST_H_
#define TABULIST_H_

#include "Solution.h"
#include <list>

namespace tabu {

typedef struct tabu_item {

public:
	Solution *solution;
	int turns;
	~tabu_item();

} Tabu_item;

class TabuList {
public:
	TabuList(unsigned int n_machines, int tabu_turns);
	virtual ~TabuList();
	bool is_tabu(Solution *sol);
	void update_tabu();
	void add_tabu(Solution *sol);
private:
	std::list<Tabu_item> tabu_list;
	unsigned int n_machines;
	int tabu_turns;
};

} /* namespace tabu */
#endif /* TABULIST_H_ */

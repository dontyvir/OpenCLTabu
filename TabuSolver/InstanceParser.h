/*
 * InstanceParser.h
 *
 *  Created on: 14-08-2012
 *      Author: donty
 */

#ifndef INSTANCEPARSER_H_
#define INSTANCEPARSER_H_

#include "Matrix.h"

namespace tabu {

class InstanceParser {
public:
	InstanceParser();
	virtual ~InstanceParser();
	Matrix *parse_input(const char *filename);
	int machines;
	int parts;
};

} /* namespace tabu */
#endif /* INSTANCEPARSER_H_ */

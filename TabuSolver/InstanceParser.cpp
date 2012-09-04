/*
 * InstanceParser.cpp
 *
 *  Created on: 14-08-2012
 *      Author: donty
 */

#include "InstanceParser.h"
#include <iostream>
#include <stdio.h>

namespace tabu {

InstanceParser::InstanceParser() {
	machines = 0;
	parts = 0;
}

InstanceParser::~InstanceParser() {
}

Matrix *InstanceParser::parse_input(const char *filename) {

	FILE *file;
	if ((file = fopen(filename, "r")) == NULL)
	        return NULL;

	int filas = 0;
	int columnas = 0;

	fscanf(file, "%i %i", &filas, &columnas);
	Matrix *mat = new Matrix(filas,columnas);

	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			int buffer = 0;

			fscanf(file, "%i", &buffer);
			mat->getMatrix()[i][j] = buffer;
		}
	}

	fclose(file);

	this->parts = columnas;
	this->machines = filas;
	return mat;
}

} /* namespace tabu */

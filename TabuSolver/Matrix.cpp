/*
 * Matrix.cpp
 *
 *  Created on: 23-08-2012
 *      Author: donty
 */

#include "Matrix.h"

namespace tabu {

Matrix::Matrix(const int _rows, const int _cols) : rows(_rows), cols(_cols)
{
	std::vector<std::vector<int> > storage(_rows,std::vector<int>(_cols,0));
	this->storage = storage;
}


Matrix::~Matrix() {
}

std::vector<int> &Matrix::operator [](int x) {
	return this->storage[x];
}

std::vector<std::vector<int> >& Matrix::getMatrix() {
	return storage;
}

} /* namespace tabu */

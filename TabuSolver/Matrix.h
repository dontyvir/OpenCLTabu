/*
 * Matrix.h
 *
 *  Created on: 23-08-2012
 *      Author: donty
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <vector>

namespace tabu {

class Matrix {
public:
	Matrix(const int _rows, const int _cols);
	virtual ~Matrix();
	std::vector<int>& operator [] (int x);
	std::vector<std::vector<int> > &getMatrix();
private:
	std::vector<std::vector<int> > storage;
	size_t rows;
	size_t cols;
};

} /* namespace tabu */
#endif /* MATRIX_H_ */

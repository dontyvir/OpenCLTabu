
class SquareMatrix {
	
	private:
		int **storage;
	public:
		SquareMatrix(int rows, int cols);
		int getElem(int row, int col);
		int *getRow(int row);
		int rows;
		int cols;
};

SquareMatrix::SquareMatrix(int rows, int cols){
	
}

int SquareMatrix::getElem(int row, int col){
	return storage[row][col];
}

int *SquareMatrix::getRow(int row){
	return storage[row];
}

class VariableMatrix {

	private:
		int **storage;
		
	public:
		VariableMatrix(int rows, int* cols);
		int getElem(int row, int col);
		int *getRow(int row);
		int rows;
		int *cols; // length = row
};

VariableMatrix::VariableMatrix(int rows, int* cols){
	
}

int VariableMatrix::getElem(int row, int col){
	return storage[row][col];
}

int *VariableMatrix::getRow(int row){
	return storage[row];
}

__kernel void
get_cost(__global int *input,
		__global int *output,
		__global int *max_machines_cell,
		__global int *n_parts,
		__global VariableMatrix *parts_machines,
		__global VariableMatrix *machines_in_cells,
		__global VariableMatrix *machines_not_in_cells,
		__global SquareMatrix *incidence_matrix
)
{
    uint k = get_global_id(0); //machines_not_in_cells.size();
    uint i_n = get_global_id(1); // machines_not_in_cells[k].size();
    uint j_ = get_global_id(2); // parts_machines[i].size();
    
    long cost = 0;

	//------------penalizacion y_ik <= Mmax------------------

	int machines_cell = machines_in_cells->cols[k];

	int difference = machines_cell - (*max_machines_cell);
	int sign = 1 ^ ((unsigned int)difference >> 31); // if difference > 0 sign = 1 else sign = 0
	cost += difference * (*n_parts) * sign;

	// -------------------------------------------------------

	//int machines_not_in = machines_not_in_cells->cols[k];

	// máquinas no en celda

	int i = machines_not_in_cells->getElem(k,i_n); // máquina no en celda
	int machines_in = machines_in_cells->cols[k];
	//int parts = parts_machines->cols[i];

	int j = parts_machines->getElem(i,j_); // parte de máquina no en celda

	for(int i_in=0;i_in<machines_in;i_in++){

		int i_ = machines_in_cells->getElem(k,i_in); // máquina en celda

		// costo+ si la máquina en celda tiene la parte que también es de la máquina no en celda
		cost += incidence_matrix->getElem(i_,j);
	}
	
    *output=cost;
}

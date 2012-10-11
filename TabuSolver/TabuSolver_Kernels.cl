
typedef struct cl_params {

	int n_cells;
	int n_parts;
	int n_machines;
	int max_machines_cell;
} ClParams;

__kernel void
get_cost(
		__global int *cost_out,
		__global ClParams *params,
		
		__global int *parts_machines_storage,
		__global int *parts_machines_lengths,
		
		__local int *machines_in_cells_storage,
		__local int *machines_in_cells_lengths,
		
		__local int *machines_not_in_cells_storage,
		__local int *machines_not_in_cells_lengths,
		
		__global int *incidence_matrix_storage

)
{
    uint k = get_global_id(0); //machines_not_in_cells.size();
    uint i_n = get_global_id(1); // machines_not_in_cells[k].size();
    uint j_ = get_global_id(2); // parts_machines[i].size();
    
    long cost = 0;

    // buffer_matrix[row][col] = buffer[row*max_cols+col]
    
    
	//------------penalizacion y_ik <= Mmax------------------

	int machines_cell = machines_in_cells_lengths[k];

	int difference = machines_cell - params->max_machines_cell;
	int sign = 1 ^ ((unsigned int)difference >> 31); // if difference > 0 sign = 1 else sign = 0
	cost += difference * (params->n_parts) * sign;

	// -------------------------------------------------------

	//int machines_not_in = machines_not_in_cells->cols[k];

	// máquinas no en celda
	
	int i = machines_not_in_cells_storage[k*params->n_cells+i_n]; // máquina no en celda
	int machines_in = machines_in_cells_storage[k];
	//int parts = parts_machines->cols[i];

	int j = parts_machines_storage[i*params->n_cells+j_]; // parte de máquina no en celda

	for(int i_in=0;i_in<machines_in;i_in++){

		int i_ = machines_in_cells_storage[k*params->n_machines+i_in]; // máquina en celda

		// costo+ si la máquina en celda tiene la parte que también es de la máquina no en celda
		cost += incidence_matrix_storage[i_*params->n_parts+j];
	}
	
	*cost_out=cost;
}

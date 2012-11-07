
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

typedef struct cl_params {

	int n_cells;
	int n_parts;
	int n_machines;
	int max_machines_cell;
} ClParams;

//--------------------------------------------------------------------------------------
__kernel void cost_2(
		__global unsigned int *cost_out,
		__global ClParams *params,
		__global int *incidence_matrix,
		__global int *solution
){
	
	long cost = 0;

	uint i = get_global_id(0);// sum i=1...M
	uint j = get_globa_id(1);// sum j=1...P
	uint k = get_global_id(2); // sum k=1...C
					
	//(z_jk)
	for(unsigned int i_=0;i_<n_machines;i_++){
				
		cost += (
					(incidence_matrix[i*n_parts+j] == 1)
					* (solution[i] != (signed int)k)
					* (i != i_)
					* (solution[i_] == (signed int)k)
					* (incidence_matrix->getMatrix()[i_*n_parts+j] == 1);
				)
	}
	
	atom_add (cost_out,cost);
}

__kernel void penalization_Mmax_2(
		__global unsigned int *cost_out,
		__global ClParams *params,
		__global int *machines_in_cells_lengths){
	
    uint k = get_global_id(0);

	int machines_cell = 0;
	for(unsigned int i=0;i<n_machines;i++){
		if(solution->cell_vector[i] == (signed int)k){
			machines_cell++;
		}
	}
	// y_ik <= Mmax
	if((unsigned int)machines_cell > max_machines_cell){
		cost += (machines_cell - max_machines_cell) * n_parts;
	}
    
}

//--------------------------------------------------------------------------------------


__kernel void cost(
		__global unsigned int *cost_out,
		__global ClParams *params,
		__global int *parts_machines_storage,
		__global int *parts_machines_lengths,
		__global int *machines_in_cells_storage,
		__global int *machines_in_cells_lengths,
		__global int *machines_not_in_cells_storage,
		__global int *machines_not_in_cells_lengths,
		__global int *incidence_matrix_storage,
		__global int *parts_cells
){

	uint k = get_global_id(1); // celda	
	int machines_in = machines_in_cells_lengths[k];
	uint i_in = get_global_id(2);//índice máquina en celda
	
	if(!(i_in < machines_in))
		return;

	int cost = 0;
	
	int n_parts = params->n_parts;
	int n_machines = params->n_machines;
	
	uint j = get_global_id(0); //parte

	//i_in : índice máquina en celda
	
	int i_ = machines_in_cells_storage[k*n_machines+i_in]; // máquina en celda
	
	if(incidence_matrix_storage[i_*n_parts+j]!=1)
		return;
	
	// parte j en celda k
	if(parts_cells[j]!=-1)
		return;
	
	atom_xchg(parts_cells+j, k);
	
	int machines_not_in = machines_not_in_cells_lengths[k];
	
	for(int i_n=0;i_n<machines_not_in;i_n++){ // máquinas no en celda
	
		int i = machines_not_in_cells_storage[k*n_machines+i_n]; // máquina no en celda
		
		// elemento excepcional (parte celda k también en otra celda)						
		cost += incidence_matrix_storage[i*n_parts+j];
	}

	atom_add (cost_out,cost);
}

/**
 * penalizaciones por número de máquinas en celda mayor a máximo
 * 
 */
__kernel void penalization_Mmax(
		__global unsigned int *cost_out,
		__global ClParams *params,
		__global int *machines_in_cells_lengths){
	
    uint k = get_global_id(0);
    
	int machines_in_cell = machines_in_cells_lengths[k];
	int n_parts = params->n_parts;
	
	int difference = machines_in_cell - params->max_machines_cell;

	unsigned int cost = (difference+n_parts) * (difference > 0); // costo+ por cada máquina en celda mayor que el máximo especificado
	
	atom_add (cost_out,cost);
}

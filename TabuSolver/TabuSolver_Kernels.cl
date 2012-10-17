
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

typedef struct cl_params {

	int n_cells;
	int n_parts;
	int n_machines;
	int max_machines_cell;
} ClParams;

inline int equal(int a,int b){

	int equal = a - b;

	equal = ((equal | (~equal + 1)) >> 31) & 0;

	return equal;
}

inline int zero(int a){

	int zero = ((a | (~a + 1)) >> 31) & 0;
	return zero;
}

/**
 * costo base 
 * 
 */
__kernel void cost(
		__global unsigned int *cost_out,
		__global ClParams *params,
		__global int *parts_machines_storage,
		__global int *parts_machines_lengths,
		__global int *machines_in_cells_storage,
		__global int *machines_in_cells_lengths
){
	
    uint i_in = get_global_id(0);
    uint j = get_global_id(1);
    uint k = get_global_id(2);
	
	int machines_in = machines_in_cells_lengths[k];
	if(!(i_in < machines_in)){
		return;
	}
	
	int n_parts = params->n_parts;
	int in_cells = 0;
	
	int i_ = machines_in_cells_storage[k*machines_in+i_in]; // máquina en celda
	int parts = parts_machines_lengths[i_];

	for(int j_in=0;j_in<parts;j_in++){ // partes de máquinas en la celda

		int j_ = parts_machines_storage[i_*parts+j_in]; // parte de máquina en celda

		in_cells += equal(j,j_);
	}

	// cost+=n_parts*in_cells para in_cells != 1
    unsigned int cost = (in_cells - 1) * n_parts * (-1*zero(in_cells));
    atom_add (cost_out,cost);
}

/**
 * penalizaciones 
 * 
 */
__kernel void penalization(
		__global unsigned int *cost_out,
		__global ClParams *params,
		__global int *parts_machines_storage,
		__global int *parts_machines_lengths,
		__global int *machines_in_cells_storage,
		__global int *machines_in_cells_lengths,
		__global int *machines_not_in_cells_storage,
		__global int *machines_not_in_cells_lengths,
		__global int *incidence_matrix_storage
){
	
    uint i_n = get_global_id(0);
    uint k = get_global_id(1);
    
	int machines_not_in_cell = machines_not_in_cells_lengths[k];
	
	if(!(i_n < machines_not_in_cell)){
		return;
	}

	int n_parts = params->n_parts;
    unsigned int cost = 0;	

	int machines_not_in = machines_not_in_cells_lengths[k];

	int i = machines_not_in_cells_storage[k*machines_not_in+i_n]; // máquina no en celda
	int machines_in = machines_in_cells_lengths[k];
	int parts = parts_machines_lengths[i];

	
	for(int j_=0;j_<parts;j_++){ // partes de máquinas no en la celda

		int j = parts_machines_storage[i*parts+j_]; // parte de máquina no en celda

		for(int i_in=0;i_in<machines_in;i_in++){

			int i_ = machines_in_cells_storage[k*machines_in+i_in]; // máquina en celda

			// costo+ si la máquina en celda tiene la parte que también es de la máquina no en celda
			cost += incidence_matrix_storage[i_*n_parts+j];
		}
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
	int sign = 1 ^ ((unsigned int)difference >> 31); // if difference > 0 sign = 1 else sign = 0
	
	unsigned int cost = difference * n_parts * sign;
	
	atom_add (cost_out,cost);
}

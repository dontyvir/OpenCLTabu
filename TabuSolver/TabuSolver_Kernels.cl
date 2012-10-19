
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
	int n_machines = params->n_machines;
	int in_cells = 0;
	
	int i_ = machines_in_cells_storage[k*n_machines+i_in]; // índice matriz incidencia (i_) máquina (i_in) en celda (k)
	int parts = parts_machines_lengths[i_]; // número de partes para la máquina (i_)

	for(int j_in=0;j_in<parts;j_in++){ // índice j_in de partes de máquina (i_) en la celda (k)

		int j_ = parts_machines_storage[i_*n_parts+j_in]; // índice matriz incidencia (j_) de parte parte de máquina (i_) en celda (k)

		in_cells += (j==(unsigned)j_); // si la parte j se encuentra en la celda k
	}

	// cost+=n_parts*in_cells para in_cells != 1
    unsigned int cost = (in_cells - 1) * n_parts * (in_cells > 0); // costo+ si la parte se encuentra en más de una celda
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
    
	int machines_not_in = machines_not_in_cells_lengths[k];
	
	if(!(i_n < machines_not_in)){
		return;
	}

	int n_parts = params->n_parts;
	int n_machines = params->n_machines;
    unsigned int cost = 0;
    
	int i = machines_not_in_cells_storage[k*n_machines+i_n]; // máquina (i_n) no en celda (k)
	int machines_in = machines_in_cells_lengths[k]; // máquinas en celda (k)
	int parts = parts_machines_lengths[i]; // número de partes para la máquina (i)
	
	for(int j_=0;j_<parts;j_++){ // índice de partes de máquina (i) no en la celda (k)

		int j = parts_machines_storage[i*n_parts+j_]; // parte (j) de máquina (i) no en celda (k)

		for(int i_in=0;i_in<machines_in;i_in++){ // índice de máquinas (i_in) en celda (k) 

			int i_ = machines_in_cells_storage[k*n_machines+i_in]; // índice de la matriz de incidencia  máquina (i_in) en celda (k) 
			
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
	//int sign = 1 ^ ((unsigned int)difference >> 31); // if difference > 0 sign = 1 else sign = 0
	
	//unsigned int cost = difference * n_parts * sign;
	unsigned int cost = difference * n_parts * (difference > 0); // costo+ por cada máquina en celda mayor que el máximo especificado
	
	atom_add (cost_out,cost);
}


#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

typedef struct cl_params {

	const int n_cells;
	const int n_parts;
	const int n_machines;
	const int max_machines_cell;
} ClParams;


__kernel void local_search(
		__constant ClParams *params,
		__constant int *solution,
		__local int *lsol, // sizeof(int)*n_machines*work_group_size
		__global int *gsol // sizeof(int)*n_machines*n_machines
){

	uint i = get_global_id(0);// n_machines
	uint j = get_global_id(1);// n_machines

	int n_machines = params->n_machines;
	
	__local int *lsol_item = lsol+(n_machines*get_local_id(0));
	
	// copiar a memoria local
	event_t copy_evt;
	copy_evt = async_work_group_copy(lsol_item, solution, n_machines, copy_evt);
	wait_group_events(1, &copy_evt);

	// intercambio
	int tmp = lsol_item[j];
	lsol_item[j] = lsol_item[i];
	lsol_item[i] = tmp;
	
	//subir a memoria global
	copy_evt = async_work_group_copy(gsol+((n_machines*i)+j), lsol_item, n_machines, copy_evt);
	wait_group_events(1, &copy_evt);	
	
}


__kernel void costs(
		__global unsigned int *cost_out,
		__const ClParams *params,
		__const int *incidence_matrix,
		__global int *gsol
){
	long cost = 0;

	uint i = get_global_id(0);// sum i=1...M
	uint j = get_global_id(1);// sum j=1...P
	uint k = get_global_id(2); // sum k=1...C

	__global int *solution = gsol+(n_machines*i+j);
	
	int n_machines = params->n_machines;
	int n_parts = params->n_parts;
	
	//(z_jk)
	for(unsigned int i_=0;i_<n_machines;i_++){
				
		cost += (
					(incidence_matrix[i*n_parts+j] == 1)
					* (solution[i] != (signed int)k)
					* (i != i_)
					* (solution[i_] == (signed int)k)
					* (incidence_matrix[i_*n_parts+j] == 1)
				);
	}
	
	atom_add (cost_out,cost);
}

//--------------------------------------------------------------------------------------
__kernel void cost(
		__global unsigned int *cost_out,
		__constant ClParams *params,
		__constant int *incidence_matrix,
		__constant int *solution
){
	
	long cost = 0;

	uint i = get_global_id(0);// sum i=1...M
	uint j = get_global_id(1);// sum j=1...P
	uint k = get_global_id(2); // sum k=1...C

	int n_machines = params->n_machines;
	int n_parts = params->n_parts;
	
	//(z_jk)
	for(unsigned int i_=0;i_<n_machines;i_++){
				
		cost += (
					(incidence_matrix[i*n_parts+j] == 1)
					* (solution[i] != (signed int)k)
					* (i != i_)
					* (solution[i_] == (signed int)k)
					* (incidence_matrix[i_*n_parts+j] == 1)
				);
	}
	
	atom_add (cost_out,cost);
}

__kernel void penalization_Mmax(
		__global unsigned int *cost_out,
		__constant ClParams *params,
		__constant int *solution){
	
    uint k = get_global_id(0);

	long cost = 0;
    
	int n_machines = params->n_machines;
	int n_parts = params->n_parts;
    int max_machines_cell = params->max_machines_cell;
	
	int machines_cell = 0;
	for(unsigned int i=0;i<n_machines;i++){
		machines_cell+=(solution[i] == (signed int)k);
	}
	// y_ik <= Mmax
	cost += ( (machines_cell - max_machines_cell) * n_parts ) * ((unsigned int)machines_cell > max_machines_cell);
    
	atom_add (cost_out,cost);
}

//--------------------------------------------------------------------------------------


__kernel void cost_(
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
__kernel void penalization_Mmax_(
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

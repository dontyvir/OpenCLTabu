
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

typedef struct cl_params {

	int n_cells;
	int n_parts;
	int n_machines;
	int max_machines_cell;
} ClParams;


__kernel void local_search(
		__constant ClParams *params,
		__global int *solution,
		__local int *lsol, // sizeof(int)*n_machines
		__global int *gsol // sizeof(int)*n_machines*n_machines
){
	
	int n_machines = params->n_machines;
	
	uint sol_offset = ((get_global_id(0)*n_machines)+get_global_id(1))*n_machines;
	uint i = get_global_id(0);// n_machines
	uint j = get_global_id(1);// n_machines
	
	int tmp = gsol[sol_offset+i];
	gsol[sol_offset+i] = gsol[sol_offset+j];
	gsol[sol_offset+j] = tmp;
	
}

__kernel void costs(
		__global uint *cost_out,
		__constant ClParams *params,
		__constant int *incidence_matrix,
		__global int *gsol,
		__local int *parts_cells // n_machines*n_machines
){
	uint cost = 0;
	
	uint j = get_global_id(2);// sum j=1...P
	uint i_sol = get_global_id(0);// sum i=1...M 0...(n_machines*n_machines*n_machines)-1
	uint i_ = get_global_id(1); // sum i_=1...M

	int n_machines = params->n_machines;
	int n_parts = params->n_parts;
	uint sol_offset = i_sol/n_machines;
	
	__global int *solution = gsol+sol_offset;

	int i = i_sol%(n_machines);

//	if(get_local_id(0)==0&&get_local_id(1)==0&&get_local_id(2)==0){
//		parts_cells[0] = 0;
//		parts_cells[1] = 0;
//		parts_cells[2] = 0;
//		parts_cells[3] = 0;
//		parts_cells[4] = 0;
//		parts_cells[5] = 0;
//		parts_cells[6] = 0;
//		parts_cells[7] = 0;
//		parts_cells[8] = 0;
//		parts_cells[9] = 0;
//		parts_cells[10] = 0;
//		parts_cells[11] = 0;
//		parts_cells[12] = 0;
//		parts_cells[13] = 0;
//		parts_cells[14] = 0;
//		parts_cells[15] = 0;
//	}
		
	//barrier( CLK_LOCAL_MEM_FENCE );
	
	if(incidence_matrix[i*n_parts+j] == 1){
	
		if(solution[i] != solution[i_]){
				
			if(incidence_matrix[i_*n_parts+j] == 1){

				if(parts_cells[i] != -1){
					
//						for(int il=0;il<16;il++){
//							if(il!=i)atomic_xchg(parts_cells+il, -1);
//								//parts_cells[il] = (il!=i)?-1:0;
//						}

						cost++;
				}
			}
		}	
	}

	
	//atom_add (cost_out+sol_offset,cost);
	cost_out[sol_offset] += cost;
}

__kernel void penalizaciones_Mmax(
		__global uint *cost_out,
		__constant ClParams *params,
		__global int *gsol,
		__local uint *machines_cell){
	
    uint sol_offset = get_global_id(1); // n_machines * n_machines
    uint k = get_global_id(0); // cell
    uint i = get_local_id(2); // local work group cell 0 ... n_machines
    uint cost = 0;
    
	if(i == 0)
		*machines_cell = 0;
	
	barrier( CLK_LOCAL_MEM_FENCE );
	
	int n_machines = params->n_machines;
	int n_parts = params->n_parts;
    int max_machines_cell = params->max_machines_cell;
	
    __global int *item = gsol+(sol_offset*n_machines)+i;

    if(*item == (int)k)
    	atomic_add(machines_cell,1);
    
	barrier( CLK_LOCAL_MEM_FENCE );

		int _machines_cell = *machines_cell; // cached private?
		
		if(_machines_cell > max_machines_cell){
			cost = ( (_machines_cell - max_machines_cell) * n_parts ); 

	    	atom_add (cost_out+sol_offset,cost);
		}

}

__kernel void mejor_solucion(
		__global uint *cost_out,
		__global int *gsol,
		__global uint *best_i,
		__global uint *best_cost)
{

	uint i = get_global_id(0); // n_machines * n_machines

	atomic_min(best_cost,cost_out[i]);
	
	barrier( CLK_GLOBAL_MEM_FENCE );
	
	if(*best_cost == cost_out[i])
		atomic_xchg(best_i,i);
}

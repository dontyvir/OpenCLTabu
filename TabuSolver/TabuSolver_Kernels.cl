
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
		__local int *lsol, // sizeof(int)*n_machines*work_group_size
		__global int *gsol // sizeof(int)*n_machines*n_machines
){
	
	int n_machines = params->n_machines;
	
	uint sol_offset = ((get_global_id(0)*n_machines)+get_global_id(1))*n_machines;
	uint i = get_global_id(0);// n_machines
	uint j = get_global_id(1);// n_machines
	
	int tmp = gsol[sol_offset+i];
	gsol[sol_offset+i] = gsol[sol_offset+j];
	gsol[sol_offset+j] = tmp;
	
//	if(get_global_id(0) == 0){
//		printf("%u %u %u\n",sol_offset,i,j);
//		printf("solution: %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n",
//				gsol[sol_offset], gsol[sol_offset+1],gsol[sol_offset+2],gsol[sol_offset+3],
//				gsol[sol_offset+4], gsol[sol_offset+5],gsol[sol_offset+6],gsol[sol_offset+7],
//				gsol[sol_offset+8], gsol[sol_offset+9],gsol[sol_offset+10],gsol[sol_offset+11],
//				gsol[sol_offset+12], gsol[sol_offset+13],gsol[sol_offset+14],gsol[sol_offset+15]
//		);
//	}
	
}


__kernel void costs(
		__global uint *cost_out,
		__constant ClParams *params,
		__constant int *incidence_matrix,
		__global int4 *gsol
){
	uint cost = 0;

	uint i = get_global_id(0);// sum i=1...M 0...(n_machines*n_machines*n_machines/4)-1
	uint j = get_global_id(1);// sum j=1...P
	uint i_ = get_global_id(2); // sum i_=1...M

	int n_machines = params->n_machines;
	int n_parts = params->n_parts;
	
	uint sol_offset = i/(n_machines/4); // 0 ... n_machines*n_machines-1
	uint item = get_local_id(0); // 0 .. 3
	uint4 mat_i; // incidence matrix offset	
	mat_i.s0 = item*4;
	mat_i.s1 = item*4+1;
	mat_i.s2 = item*4+2;
	mat_i.s3 = item*4+3;
	
//	if(get_group_id(1) == 0 && get_group_id(2) == 0)
//		printf("%u %u %u %u %u\n",i,j,k,sol_offset,item);
	
	__global int4 *solution = gsol+(sol_offset*4); // == (int)gsol[sol_offset*16]

	//(z_jk)
						
	cost += (
				(incidence_matrix[mat_i.s0*n_parts+j] == 1)
				* (solution[item].s0 != solution[i_].s0)// item k != i_ k
				* (mat_i.s0 != i_)
				* (incidence_matrix[i_*n_parts+j] == 1)
			);

	cost += (
				(incidence_matrix[mat_i.s1*n_parts+j] == 1)
				* (solution[item].s1 != solution[i_].s1)// item k != i_ k
				* (mat_i.s1 != i_)
				* (incidence_matrix[i_*n_parts+j] == 1)
			);
	
	cost += (
				(incidence_matrix[mat_i.s2*n_parts+j] == 1)
				* (solution[item].s2 != solution[i_].s2)// item k != i_ k
				* (mat_i.s2 != i_)
				* (incidence_matrix[i_*n_parts+j] == 1)
			);
	
	cost += (
				(incidence_matrix[mat_i.s3*n_parts+j] == 1)
				* (solution[item].s3 != solution[i_].s3) // item k != i_ k
				* (mat_i.s3 != i_)
				* (incidence_matrix[i_*n_parts+j] == 1)
			);
	
	atom_add (cost_out+sol_offset,cost);

}

__kernel void penalizaciones_Mmax(
		__global uint *cost_out,
		__constant ClParams *params,
		__global int4 *gsol,
		__local uint *machines_cell){
	
    uint sol_offset = get_global_id(0); // n_machines * n_machines
    uint k = get_global_id(1); // cell
    uint lid = get_local_id(2); // local work group cell
    uint sub_sol_offset = get_global_id(2);
    uint cost = 0;

//	if(get_group_id(0) == 1 && get_group_id(1) == 1)
//	printf("%u %u %u %u %u %u %u\n", get_global_id(0), get_global_id(1),
//			get_global_id(2), get_local_size(0), get_local_size(1), get_local_size(2), get_local_id(2));
	
	if(lid == 0)
		*machines_cell = 0;

	barrier( CLK_LOCAL_MEM_FENCE );
	
	int n_machines = params->n_machines;
	int n_parts = params->n_parts;
    int max_machines_cell = params->max_machines_cell;
	
    __global int4 *solution = gsol+sol_offset+sub_sol_offset;
    
//	if(get_group_id(0) < 2 && get_group_id(1) < 2){
//	//printf("vector: %u %u %u %u\n", (*solution).s0, (*solution).s1, (*solution).s2, (*solution).s3);
//    printf("solution: %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n",
//    		gsol[sol_offset].s0, gsol[sol_offset].s1,gsol[sol_offset].s2,gsol[sol_offset].s3,
//    		gsol[sol_offset+1].s0, gsol[sol_offset+1].s1,gsol[sol_offset+1].s2,gsol[sol_offset+1].s3,
//    		gsol[sol_offset+2].s0, gsol[sol_offset+2].s1,gsol[sol_offset+2].s2,gsol[sol_offset+2].s3,
//    		gsol[sol_offset+3].s0, gsol[sol_offset+3].s1,gsol[sol_offset+3].s2,gsol[sol_offset+3].s3
//    );
//	}
    /* 
     * k = 0 | offset = 0 | sub_offset = 0
     *  								 1
     *  		....					 2
     *  								 3
     * k = 0 | offset = n-1| sub_offset = ....
     * 
     */
    
	atomic_add(machines_cell,((*solution).s0 == (int)k));
	atomic_add(machines_cell,((*solution).s1 == (int)k));
	atomic_add(machines_cell,((*solution).s2 == (int)k));
	atomic_add(machines_cell,((*solution).s3 == (int)k));
    
	barrier( CLK_LOCAL_MEM_FENCE );
	
//	if(get_group_id(0) == 1 && get_group_id(1) == 1)
//	printf("%u\n", *machines_cell);
	
	if(lid == 0) {
		//printf("%u\n", *machines_cell);
		int past_max = (*machines_cell > max_machines_cell);
		cost = ( (*machines_cell - max_machines_cell) * n_parts ) * past_max; 
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

//	if(cost_out[i] < 20)
//		printf("%i\n", cost_out[i]);
	atomic_min(best_cost,cost_out[i]);
	atomic_xchg(best_i,i);
}

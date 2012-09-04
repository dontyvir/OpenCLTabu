

__kernel void
get_cost(global float * vec, int *cost)
{
    int i =  get_global_id(0);
    int j =  get_global_id(1);
    int k =  get_global_id(2);

    long cost = 0;
    //precompute
    //if(incidence_matrix->getMatrix()[i][j] == 1)
	//if(solution->cell_vector[i] != (signed int)k){ //(1 - y_ik)
    
    
	for(unsigned int i=0;i<n_machines;i++){ // sum i=1...M
		for(unsigned int j=0;j<n_parts;j++){ // sum j=1...P
			
			
			
			if(incidence_matrix->getMatrix()[i][j] == 1){ // a_ij = 1
				
				
				
				for(unsigned int k=0;k<n_cells;k++){ // sum k=1...C
					if(solution->cell_vector[i] != (signed int)k){ //(1 - y_ik)
						//(z_jk)
						for(unsigned int i_=0;i_<n_machines;i_++){
							if( i != i_ &&
								solution->cell_vector[i_] == (signed int)k
								&& incidence_matrix->getMatrix()[i_][j] == 1
							){
								cost++;
							}
						}

					}
				}
				
				
				
			}
			
			
			
		}
	}

	// infactible penalization

	//z_jk = 1
	for(unsigned int j=0;j<n_parts;j++){
		int in_cells = 0;
		for(unsigned int k=0;k<n_cells;k++){ // sum k=1...C
			//(z_jk)
			for(unsigned int i_=0;i_<n_machines;i_++){
				if(solution->cell_vector[i_] == (signed int)k
					&& 	incidence_matrix->getMatrix()[i_][j] == 1
				){
					in_cells++;
					break;
				}
			}
		}
		if(in_cells > 1)
			cost += (in_cells - 1) * n_parts;
		else
			cost += (1 - in_cells) * n_parts;
	}


	for(unsigned int k=0;k<n_cells;k++){
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
    
    
}

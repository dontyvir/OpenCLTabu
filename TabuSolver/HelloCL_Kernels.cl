//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//

__kernel void
hello()
{
	/*
	Just a stub kernel. 
	*/
	
    size_t i =  get_global_id(0);
    size_t j =  get_global_id(1);
}

kernel void sum(global float * vec, global float * sum)
{

   int i = get_global_id(0);
   sum[0] += vec[i];

}
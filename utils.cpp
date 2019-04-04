/**
 * @file    utils.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements common utility/helper functions.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "utils.h"
#include <iostream>

/*********************************************************************
 *                 Implement your own functions here                 *
 *********************************************************************/

// ...

void get_first_col_ranks(int ranks[], int dim,  MPI_Comm comm){
    // Store all the coords of first col
    int col_coords[dim][2]; 
    for(int i = 0; i < dim; i++){
        col_coords[i][0] = i;
        col_coords[i][1] = 0;
    }

    // Get all the ranks of first col
    for(int i = 0; i < dim; i++){
        MPI_Cart_rank(comm, col_coords[i], &ranks[i]);
    }    

    return;
}

int get_col_elem_num(const int row_idx, const int dim, const int total_num){
 	if(row_idx < (total_num % dim))
 		return ceil(total_num*1.0/ dim);
 	else 
    	return floor(total_num*1.0/dim);
}

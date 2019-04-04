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

void get_first_col_row_ranks(int ranks[], int dim,  MPI_Comm comm, int type){

    // Store all the coords of first col | row
    int coords[dim][2]; 
    for(int i = 0; i < dim; i++){
        coords[i][0] = type == COL ? i : 0;
        coords[i][1] = type == COL ? 0 : i;
    }

    // Get all the ranks of first col
    for(int i = 0; i < dim; i++){
        MPI_Cart_rank(comm, coords[i], &ranks[i]);
    }    

    return;
}

int get_cell_elem_num(const int idx, const int dim, const int total_num){
 	if(idx < (total_num % dim))
 		return ceil(total_num*1.0/ dim);
 	else 
    	return floor(total_num*1.0/dim);
}

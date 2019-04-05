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

int get_cell_elem_num(const int idx, const int dim, const int n){
 	if(idx < (n % dim))
 		return ceil(n*1.0 / dim);
 	else 
    	return floor(n*1.0 / dim);
}

int get_idx_frow_row_col(int row, int col, int n){
	return row * n + col;
}

int get_elem_idx_from_dim_idx(int idx, int dim, int n){
	int num = idx * floor(n / dim);
	// if n mod dim = 1, 
	// which means idx = 0 processor has one more data, 
	// and idx = 1 processor starts from idx * floor(n/dim) + 1
	// if n mod dim = 3, idx = 3, then return 3 * floor(n/dim) + 3

	if(idx > 0 && (idx - 1) < (n % dim)){
		num += idx;
	}
	return num;
}

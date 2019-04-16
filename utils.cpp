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

void get_diag_ranks(int ranks[], int dim, MPI_Comm comm) {

	// Store all the coordinates of the diagonal elements (ii,ii)
	int coords[dim][2]; // dim: number of rows of processors in the grid
	for (int ii = 0; ii < dim; ii++) {
		coords[ii][0] = coords[ii][1] = ii;
	}

	// Get all the ranks of the diagonal elements
	for (int ii = 0; ii < dim; ii++) {
		MPI_Cart_rank(comm, coords[ii], &ranks[ii]);
	}

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
    // n = 4, dim = 3, 
    // idx = 0, should return 0 (has element idx 0, 1)
    // idx = 1, should return 2 (has element idx, 2)
    // idx = 2, should return 3 (has element idx, 3)

    // n = 5, dim = 3, 
    // idx = 0, should return 0 (has element idx 0, 1)
    // idx = 1, should return 2 (has element idx, 2, 3)
    // idx = 2, should return 4 (has element idx, 4)

    int retidx = idx * floor(n*1.0/dim);
    retidx += (idx < (n % dim) ? idx : (n % dim));
	return retidx;
}


/*For tests*/
void print_sent_matrix(double* A, int n){
    printf("Print input matrix:");
    for(int i = 0; i < n*n; i++){
        if(i % n == 0){
            printf("\n");
        }
        printf("%.2f, ", A[i]);
    }   
    printf("\n");
}

void print_sent_vector(double* x, int n){
    printf("Print input vector: \n");
    for(int i = 0; i < n; i++){
        printf("%.2f, ", x[i]);
    }   
    printf("\n");
}


// cmd: mpirun -np 9 --oversubscribe  ./jacobi -n 5
void test_distribute_matrix(double* local_A, int n, MPI_Comm comm){
    if(local_A){
        int dims[2];
        int periods[2];
        int coords[2];
        MPI_Cart_get(comm, 2, dims, periods, coords);

        int row_elem_num = get_cell_elem_num(coords[0], dims[0], n);
        int col_elem_num = get_cell_elem_num(coords[1], dims[1], n);

        int elem_num = row_elem_num * col_elem_num;
        printf("(%d, %d) matrix elem_num %d \n",coords[0], coords[1], elem_num);
        for(int i = 0; i < elem_num; i++)
            printf("%.2f, ", local_A[i]);
        printf("\n");
    }
}

void test_distribute_vector(double* local_b, int n, MPI_Comm comm){
    if(local_b){
        int dims[2];
        int periods[2];
        int coords[2];
        MPI_Cart_get(comm, 2, dims, periods, coords);

        int elem_num = get_cell_elem_num(coords[0], dims[0], n);
        printf("(%d, %d) vector elem_num %d \n",coords[0], coords[1], elem_num);
    }
}

void test_gather_vector(double* x, int n, MPI_Comm comm){
    if(x){
        int myrank;
        MPI_Comm_rank(comm, &myrank);
        if(myrank == 0){
            for(int i = 0; i < n; i++){
                printf("%.2f, ", x[i]);
            }        
            printf("\n");            
        }
    }
}

// Subtracts two arrays of size n: arr1 - arr2 = arr3
void matrix_subtract(double* arr1, double* arr2, double* arr3, int n) {
	for (int ii = 0; ii < n; ii++) {
		arr3[ii] = arr1[ii] - arr2[ii];
	}
}

// Calculates the l2 norm of arr1, an array of size n
double l2_norm(double* arr1, int n) {
	double l2 = 0.;
	for (int ii = 0; ii < n; ii++) {
		l2 += pow(arr1[ii],2);
	}
	l2 = sqrt(l2);
	return l2;
}

/**
 * @file    utils.h
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements common utility/helper functions.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

/*********************************************************************
 *             You can add new functions to this header.             *
 *********************************************************************/

#ifndef UTILS_H
#define UTILS_H

#include <mpi.h>
#include <cmath>
#include <stdexcept>

#define ROW 0
#define COL 1
/*********************************************************************
 * DO NOT CHANGE THE FUNCTION SIGNATURE OF THE FOLLOWING 3 FUNCTIONS *
 *********************************************************************/

inline int block_decompose(const int n, const int p, const int rank)
{
    return n / p + ((rank < n % p) ? 1 : 0);
}

inline int block_decompose(const int n, MPI_Comm comm)
{
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    return block_decompose(n, p, rank);
}

inline int block_decompose_by_dim(const int n, MPI_Comm comm, int dim)
{
    // get dimensions
    int dims[2];
    int periods[2];
    int coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    return block_decompose(n, dims[dim], coords[dim]);
}


/*********************************************************************
 *                  DECLARE YOUR OWN FUNCTIONS HERE                  *
 *********************************************************************/

// ...

// Get the first column processor ranks
void get_first_col_row_ranks(int ranks[], int dim,  MPI_Comm comm, int type);

// Get the diagonal processor ranks
void get_diag_ranks(int ranks[], int dim, MPI_Comm comm);

// Get the element number based on ith row
int get_cell_elem_num(const int idx, const int dim, const int n);

// Get serialized idx from rol and col
int get_idx_frow_row_col(int row, int col, int n);

// Get the accumulated idx from dim
int get_elem_idx_from_dim_idx(int idx, int dim, int n);


/*Test functions*/
void print_sent_matrix(double* A, int n);
void print_sent_vector(double* x, int n);

// cmd: mpirun -np 9 --oversubscribe  ./jacobi -n 5
void test_distribute_matrix(double* local_A, int n, MPI_Comm comm);
void test_distribute_vector(double* local_b, int n, MPI_Comm comm);
void test_gather_vector(double* x, int n, MPI_Comm comm);

// Helper functions for sequential calculations
void matrix_subtract(double* arr1, double* arr2, double* arr3, int n);
double l2_norm(double* arr1, int n);

#endif // UTILS_H

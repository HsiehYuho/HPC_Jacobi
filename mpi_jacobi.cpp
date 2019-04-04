/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>

/*
 * TODO: Implement your solutions here
 */


void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
    // Get the process0 with all input vector, then distribute to the first column of the processors
    // Not consider log(p) implementation because the number of communication is at most 8 (64^0.5)

    int myrank;

    MPI_Comm_rank(comm, &myrank);

    int dims[2];
    int periods[2];
    int coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);

    int dim = dims[0]; // num of row processor of cartesian grid

    // calculate the first column's ranks
    int *col_ranks = new int[dim];
    get_first_col_row_ranks(col_ranks, dim, comm, COL);

    // (0,0) processor as sender sends to all first column processors
    if(myrank == col_ranks[0]){
        // idx of the first send-out element
        int prev_sent_idx = 0; 
        for(int i = 0; i < dim; i++){ 
            int elem_num = get_cell_elem_num(i, dim, n);
            double* buf = &input_vector[prev_sent_idx];
            MPI_Send(buf, elem_num, MPI_DOUBLE, col_ranks[i], 222, comm);

            prev_sent_idx += elem_num;
        }
    }

    // First column of processor will receive portion of input_vector
    for(int i = 0; i < dim; i++){
        // current processor is one of (x,0) 
        // will receive corresponding number of elements
        if(myrank == col_ranks[i]){
            MPI_Status stat;
            int elem_num = get_cell_elem_num(i, dim, n);
            double *buf = new double[elem_num];
            MPI_Recv(buf, elem_num, MPI_DOUBLE, col_ranks[0], 222, comm, &stat);
            *local_vector = buf;
        }
    }
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{ 
    int myrank;

    MPI_Comm_rank(comm, &myrank);

    int dims[2];
    int periods[2];
    int coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);

    int dim = dims[0]; // num of rol of cartesian grid

    int *col_ranks = new int[dim];
    get_first_col_row_ranks(col_ranks, dim, comm, COL);

    // Each first-col processor sends elements back
    for(int i = 0; i < dim; i++){
        if(myrank == col_ranks[i]){
            int elem_num = get_cell_elem_num(i, dim, n);
            MPI_Send(local_vector, elem_num, MPI_DOUBLE, col_ranks[0], 222, comm);
        }
    }

    // Processor (0,0) collects all the elements to output_vector
    if(myrank == col_ranks[0]){
        int last_recv_idx = 0; 
        for(int i = 0; i < dim; i++){
            MPI_Status stat;

            int elem_num = get_cell_elem_num(i, dim, n);
            double *buf = &output_vector[last_recv_idx];
            MPI_Recv(buf, elem_num, MPI_DOUBLE, col_ranks[i], 222, comm, &stat);
            last_recv_idx += elem_num;
        }        
    }
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    // TODO

}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    // TODO
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    // TODO
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    // TODO
}


// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);
    
    /*For test*/
    if(local_b){
        int dims[2];
        int periods[2];
        int coords[2];
        MPI_Cart_get(comm, 2, dims, periods, coords);

        int elem_num = get_cell_elem_num(coords[0], dims[0], n);
        printf("(%d, %d) elem_num %d \n",coords[0], coords[1], elem_num);
    }

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);

    /*For test*/
    // if(x){
    //     int myrank;
    //     MPI_Comm_rank(comm, &myrank);
    //     if(myrank == 0){
    //         for(int i = 0; i < n; i++){
    //             printf("%.2f, ", x[i]);
    //         }        
    //         printf("\n");            
    //     }
    // }
}

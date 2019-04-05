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

    delete[] col_ranks;
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
    delete[] col_ranks;
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    int myrank, rank00;
    int coords00[2] = {0,0};
    MPI_Comm_rank(comm, &myrank);
    MPI_Cart_rank(comm, coords00, &rank00);

    int dims[2];
    int periods[2];
    int coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);

    // Send the element from (0,0) to all the cells
    if(myrank == rank00){
        for(int row = 0; row < dims[0]; row++){
            for(int col = 0; col < dims[1]; col++){

                int row_elem_num = get_cell_elem_num(row, dims[0], n);
                int col_elem_num = get_cell_elem_num(col, dims[1], n);
                int elem_num = row_elem_num * col_elem_num;

                int row_start = get_elem_idx_from_dim_idx(row, dims[0], n);
                int row_end = (row_start + row_elem_num);
                int col_start = get_elem_idx_from_dim_idx(col, dims[1], n);
                int col_end = (col_start + col_elem_num);

                if(row_end > n || col_end > n){
                    printf("Element idx across boundary \n");
                    exit(EXIT_FAILURE);
                }

                // Serialize the data 
                double* buf = new double[elem_num];
                int buf_idx = 0;
                
                for(int i = row_start; i < row_end; i++){
                    for(int j = col_start; j < col_end; j++){
                        int idx = get_idx_frow_row_col(i, j, n);
                        buf[buf_idx++] = input_matrix[idx];
                    }
                }

                // Send out the data to correpsponding destination
                int dest_coords[2] = {row, col};
                int dest_rank = 0;
                MPI_Cart_rank(comm, dest_coords, &dest_rank);

                MPI_Send(buf, elem_num, MPI_DOUBLE, dest_rank, 222, comm);
            }
        }        
    }

    // Each processor expects to receive its own data
    for(int row = 0; row < dims[0]; row++){
        for(int col = 0; col < dims[1]; col++){
            int dest_coords[2] = {row, col};
            int dest_rank = 0;
            MPI_Cart_rank(comm, dest_coords, &dest_rank);

            if(myrank == dest_rank){
                MPI_Status stat;

                int row_elem_num = get_cell_elem_num(row, dims[0], n);
                int col_elem_num = get_cell_elem_num(col, dims[1], n);
                int elem_num = row_elem_num * col_elem_num;
    
                double *buf = new double[elem_num];
                MPI_Recv(buf, elem_num, MPI_DOUBLE, rank00, 222, comm, &stat);

                *local_matrix = buf;
                return;
            }
        }
    }
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    int myrank;
    MPI_Comm_rank(comm, &myrank);

    int dims[2];
    int periods[2];
    int coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    
    // (x,0) processor has col_vector and send it to (i,i) cell
    if(coords[1] == 0){
        int dest_rank = 0;
        int dest_coords[2] = {coords[0],coords[0]};
        MPI_Cart_rank(comm, dest_coords, &dest_rank);
        
        int vec_elem_num = get_cell_elem_num(coords[0], dims[0], n);
        MPI_Send(col_vector, vec_elem_num, MPI_DOUBLE, dest_rank, 222, comm);
    }

    // (x,x) processor expects to receive data from (x,0) then broadcast to sub-communicators
    int vec_elem_num = get_cell_elem_num(coords[0], dims[0], n);

    if(coords[0] == coords[1]){
        MPI_Status stat; 

        int source_rank = 0;
        int source_coords[2] = {coords[0],0};
        MPI_Cart_rank(comm, source_coords, &source_rank);
        MPI_Recv(row_vector, vec_elem_num, MPI_DOUBLE, source_rank, 222, comm, &stat);
    }

    // May not be necessary, but just in case
    MPI_Barrier(comm);

    // Set the keep each row but drop column
    int remains_dim[2] = {1,0}; 

    // Create sub communicators
    MPI_Comm col_comm;
    MPI_Cart_sub(comm, remains_dim, &col_comm);

    // Broadcast vec to same column cells
    int root_rank = 0;
    int root_coords[1] = {coords[1]};
    MPI_Cart_rank(col_comm, root_coords, &root_rank);
    MPI_Bcast(row_vector, vec_elem_num, MPI_DOUBLE, root_rank, col_comm);

    MPI_Comm_free(&col_comm);
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    // TODO
    int myrank;
    MPI_Comm_rank(comm, &myrank);

    int dims[2];
    int periods[2];
    int coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);

    int vec_elem_num = get_cell_elem_num(coords[0], dims[0], n);
    double *row_vec = new double[vec_elem_num];
    // First transpose and bcast vector to get the local vector
    // only (x,0) processor has local_x
    transpose_bcast_vector(n, local_x, row_vec, comm);
    printf("(%d, %d): \n", coords[0], coords[1]);
    for(int i = 0; i < vec_elem_num; i++)
        printf("%.2f, ", row_vec[i]);
    printf("\n");

}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    // TODO
}

void print_sent_matrix(double* A, int n){
    printf("Print input matrix: \n");
    for(int i = 0; i < n*n; i++){
        printf("%.2f, ", A[i]);
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

// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // /*For test*/
    // int coords00[2] = {0,0};
    // int myrank = 0, rank00 = 0;
    // MPI_Cart_rank(comm, coords00, &rank00);
    // MPI_Comm_rank(comm, &myrank);

    // if(myrank == rank00){
    //     print_sent_matrix(A, n);
    // }

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
    
    /*** TEMP ***/
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_b, local_y, comm);
    /*** TEMP ***/

    // allocate local result space
    // double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    // distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    // gather_vector(n, local_x, x, comm);
}

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

#include <cstring>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <string.h>

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

    MPI_Request req;

    // (0,0) processor as sender sends to all first column processors
    if(myrank == col_ranks[0]){
        // idx of the first send-out element
        int prev_sent_idx = 0; 
        for(int i = 0; i < dim; i++){ 
            int elem_num = get_cell_elem_num(i, dim, n);
            double* buf = &input_vector[prev_sent_idx];
            MPI_Isend(buf, elem_num, MPI_DOUBLE, col_ranks[i], 222, comm, &req);

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

    MPI_Request req;

    // Each first-col processor sends elements back
    for(int i = 0; i < dim; i++){
        if(myrank == col_ranks[i]){
            int elem_num = get_cell_elem_num(i, dim, n);
            MPI_Isend(local_vector, elem_num, MPI_DOUBLE, col_ranks[0], 222, comm, &req);
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

    MPI_Request req;

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

                MPI_Isend(buf, elem_num, MPI_DOUBLE, dest_rank, 222, comm, &req);
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

    MPI_Request req; 

    // (x,0) processor has col_vector and send it to (i,i) cell
    if(coords[1] == 0){
        

        int dest_rank = 0;
        int dest_coords[2] = {coords[0],coords[0]};
        MPI_Cart_rank(comm, dest_coords, &dest_rank);
        
        int vec_elem_num = get_cell_elem_num(coords[0], dims[0], n);
        MPI_Isend(col_vector, vec_elem_num, MPI_DOUBLE, dest_rank, 222, comm, &req);
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
    // Ref: https://www3.nd.edu/~zxu2/acms60212-40212-S12/Lec-08-1.pdf
    int remains_dim[2] = {1,0}; 

    // Create sub communicators
    MPI_Comm col_comm;
    MPI_Cart_sub(comm, remains_dim, &col_comm);

    // Broadcast vec to same column cells
    int root_rank = 0;
    int root_coords[1] = {coords[1]};
    MPI_Cart_rank(col_comm, root_coords, &root_rank);

    // Note, vec element will be function of b where b is the column index (a,b)
    vec_elem_num = get_cell_elem_num(coords[1], dims[1], n);

    MPI_Bcast(row_vector, vec_elem_num, MPI_DOUBLE, root_rank, col_comm);

    MPI_Comm_free(&col_comm);
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    int myrank;
    MPI_Comm_rank(comm, &myrank);

    int dims[2];
    int periods[2];
    int coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);

    // Note, vec element will be function of b where b is the column index (a,b)
    int vec_elem_num = get_cell_elem_num(coords[1], dims[1], n);
    double *row_vec = new double[vec_elem_num];

    // First transpose and bcast vector to get the local vector
    transpose_bcast_vector(n, local_x, row_vec, comm);

    // Compute the local matrix and row_vec multiplication
    int row_elem_num = get_cell_elem_num(coords[0], dims[0], n);
    int col_elem_num = get_cell_elem_num(coords[1], dims[1], n);

    // Clear y array before use
    double* buf = new double[row_elem_num];
    memset(buf, 0, row_elem_num * sizeof(*buf));

    for(int i = 0; i < row_elem_num; i++){
        for(int j = 0; j < col_elem_num; j++){
            int idx = get_idx_frow_row_col(i, j, col_elem_num);
            buf[i] += (local_A[idx] * row_vec[j]);
        }
    }

    // Set the keep each column but drop row
    int remains_dim[2] = {0,1}; 

    // Create row sub-commiunicators
    MPI_Comm row_comm;
    MPI_Cart_sub(comm, remains_dim, &row_comm);

    // Collect vec to the same row, first-column (x,0) cell
    // The root rank is 0 always
    int root_rank = 0;
    int root_coords[1] = {0};
    MPI_Cart_rank(row_comm, root_coords, &root_rank);

    memset(local_y, 0, row_elem_num * sizeof(*local_y));

    // MPI_Reduce to collect (sum) every local_y to first column of cells 
    MPI_Reduce(buf, local_y, row_elem_num, MPI_DOUBLE, MPI_SUM, root_rank, row_comm);

    delete[] buf;
    MPI_Comm_free(&row_comm);

}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    int myrank;
    MPI_Comm_rank(comm, &myrank);
    int dims[2];
	int periods[2];
	int mycoords[2];
	MPI_Cart_get(comm, 2, dims, periods, mycoords);

	int local_n = get_cell_elem_num(mycoords[0], dims[0], n);
	int local_m = get_cell_elem_num(mycoords[1], dims[1], n);

	// Find D and R, send D to first col
	double local_D[local_n], local_R[local_n*local_n];
	int col_ranks[local_n];
	get_first_col_row_ranks(col_ranks, dims[0], comm, COL);
	if (mycoords[0] == mycoords[1]) {
		// global diags are stored in diag procs
		// diag procs are always square matrices (n x n)
		// global diags are stored in diag of diag procs
		for (int row = 0; row < local_n; row++) {
			for (int col = 0; col < local_n; col++) {
				if (row == col) {
					local_D[row] = local_A[row*local_n+col];
				} else {
					local_R[row*local_n+col] = local_A[row*local_n+col];
				}
			}
		}
		// Send D to first column proc with the same row number
		if (myrank != 0) { // Do not send if = proc0 because it already has what it needs
			MPI_Send(local_D, local_n, MPI_DOUBLE, col_ranks[mycoords[0]], 123, comm);
			local_D = NULL;
		}
	}

	// Receive D if proc is in the first column
	int diag_ranks[local_n];
	get_diag_ranks(diag_ranks, dims[0], comm);
	for (int ii = 1; ii < dims[0]; ii++) { // skip proc 0
		if(myrank == col_ranks[ii]){
			MPI_Status stat;
			MPI_Recv(D, local_n, MPI_DOUBLE, diag_ranks[ii], 123, comm, &stat);
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
    
    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}

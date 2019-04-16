/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"
#include "utils.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>
#include <limits>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    matrix_vector_mult(n, n, A, x, y);
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
	for (int row = 0; row < n; row++) {
    	y[row] = 0; // reset
		for (int col = 0; col < m; col++) {
    		y[row]+=A[row*m+col] * x[col]; // convert 2D coordinate to 1D: [row][col] = row*m+col
    	}
    }
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
    double D[n] = {};
    double R[n*n] = {}; // D_inv is diag elements of the inverse of D, a diagonal matrix
    for (int row = 0; row < n; row++) {
    	for (int col = 0; col < n; col++) {
    		if (row == col) {
    			D[row] = A[row*n+col];
    		} else {
    			R[row*n+col] = A[row*n+col];
    		}
    	}
    }

//    matrix_vector_mult(n, A, x, Ax);
//    matrix_subtract(Ax, b, AxMinusb, n);
    // Ax = [0,0,...,0] because x is a vector of zeros
    // Ax-b = -b, negative cancels when norming
    double l2 = l2_norm(b, n);
	int iter = 0;
	double Rx[n] = {};
	double bMinusRx[n] = {};
	double Ax[n] = {};
	double AxMinusb[n] = {};
    while (iter++ < max_iter && l2 > l2_termination) {
    	// Update x <- D^-1 * (b-Rx)
    	matrix_vector_mult(n, R, x, Rx);
    	matrix_subtract(b, Rx, bMinusRx, n);
    	for (int ii = 0; ii < n; ii++) {
    		x[ii] = bMinusRx[ii]/D[ii];
    	}

    	// Recalculate Ax-b
    	matrix_vector_mult(n, A, x, Ax);
    	matrix_subtract(Ax, b, AxMinusb, n);
    	l2 = l2_norm(AxMinusb, n);
    }
}

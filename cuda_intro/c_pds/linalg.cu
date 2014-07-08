// linear algebra subroutines
// Ben Cumming @ CSCS
extern "C"{
#include "linalg.h"
#include "operators.h"
#include "stats.h"
}

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// NOTE: there are lots of ACC todays - ignore those, as this time we're going for CUDA!

// Always use these (and if you want, have a fancier version that you can 
//  turn off for production runs)
#ifndef NDEBUG
#define CUDA_ERR_CHECK(x) \
	do { cudaError_t err = x; if (err != cudaSuccess) { \
		fprintf (stderr, "Error \"%s\" at %s:%d \n", \
		 cudaGetErrorString(err), \
		__FILE__, __LINE__); exit(-1); \
	}} while (0);
#else
#define CUDA_ERR_CHECK(x) \
        do{ } while(0);
#endif

// We define a block size which should be generally efficient
#define BLOCKSIZE 128

// A convenience to allow for breaking things into blocks
#define ROUNDUP(n, width) (((n) + (width) - 1) & ~unsigned((width) - 1))


int cg_initialized = 0;
double *r = NULL, *Ap = NULL, *p = NULL;
double *Fx = NULL, *Fxold = NULL, *v = NULL, *xold = NULL; // 1d


// initialize temporary storage fields used by the cg solver
// I do this here so that the fields are persistent between calls
// to the CG solver. This is useful if we want to avoid malloc/free calls
// on the device for the OpenACC implementation (feel free to suggest a better
// method for doing this)
void cg_init(const int N)
{
    Ap    = (double*) malloc(N*sizeof(double));
    r     = (double*) malloc(N*sizeof(double)); 
    p     = (double*) malloc(N*sizeof(double));
    Fx    = (double*) malloc(N*sizeof(double));
    Fxold = (double*) malloc(N*sizeof(double));
    v     = (double*) malloc(N*sizeof(double));
    xold  = (double*) malloc(N*sizeof(double));

    cg_initialized = 1;
}

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 reductions
////////////////////////////////////////////////////////////////////////////////

// computes the inner product of x and y
// x and y are vectors on length N
double ss_dot(const double* x, const double* y, const int N)
{
    double result = 0;
	int i;
//
// TODO:  ACC PARALLEL region: 
//          Variables X and Y should be present
// TODO:  ACC LOOP with a reduction of variable ACC

    for (i = 0; i < N; i++)
        result += x[i] * y[i];

// TODO:  end ACC PARALLEL region: 

    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 2 * N;
    
    return result;
}

// computes the 2-norm of x
// x is a vector on length N
double ss_norm2(const double* x, const int N)
{
    double result = 0;
	int i;
// TODO:  ACC PARALLEL region: 
//          Variable X should be present
// TODO:  ACC LOOP with a reduction of variable ACC
    for (i = 0; i < N; i++)
        result += x[i] * x[i];

    result = sqrt(result);

    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 2 * N;
    
    return result;
}

// sets entries in a vector to value
// x is a vector on length N
// value is th
void ss_fill(double* x, const double value, const int N)
{
	int i;
// TODO:  ACC PARALLEL region: 
    for (i = 0; i < N; i++)
        x[i] = value;
}

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 vector-vector operations
////////////////////////////////////////////////////////////////////////////////

// computes y := alpha*x + y
// x and y are vectors on length N
// alpha is a scalar

// CUDA kernel
__global__ void ss_axpy_kernel(double* y, const double alpha, const double* x, const int N)
{
	//compute the index into the 1D array
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//perform the axpy (at this index only!)
	y[idx] += alpha * x[idx];
}

void ss_axpy(double* y, const double alpha, const double* x, const int N)
{

	// Allocate arrays on the Device
	double* y_d;
	double* x_d;
	CUDA_ERR_CHECK( cudaMalloc(&x_d,    N * sizeof(double)) );
	CUDA_ERR_CHECK( cudaMalloc(&y_d,    N * sizeof(double)) );

	// Transfer Data to Device
	CUDA_ERR_CHECK( cudaMemcpy(x_d, x, N * sizeof(double), cudaMemcpyHostToDevice) );
	CUDA_ERR_CHECK( cudaMemcpy(y_d, y, N * sizeof(double), cudaMemcpyHostToDevice) );

	// Replace kernel with call to a CUDA kernel
	ss_axpy_kernel<<<ROUNDUP(N,BLOCKSIZE)/BLOCKSIZE,BLOCKSIZE>>>(y_d,alpha,x_d,N) ;
	CUDA_ERR_CHECK( cudaGetLastError() ); // kernels don't return errors, so get it this way

	// Wait
	CUDA_ERR_CHECK( cudaDeviceSynchronize() );

	// Transfer data back from th GPU to the CPU array
	CUDA_ERR_CHECK( cudaMemcpy(y, y_d, N * sizeof(double), cudaMemcpyDeviceToHost) );

	// Deallocate arrays on the GPU
	CUDA_ERR_CHECK( cudaFree(x_d) );
	CUDA_ERR_CHECK( cudaFree(y_d) );

	// record the number of floating point oporations
	flops_blas1 = flops_blas1 + 2 * N;
}

// computes y = x + alpha*(l-r)
// y, x, l and r are vectors of length N
// alpha is a scalar
void ss_add_scaled_diff(double* y, const double* x, const double alpha,
    const double* l, const double* r, const int N)
{
	int i;
// TODO:  ACC PARALLEL region: 
    for (i = 0; i < N; i++)
        y[i] = x[i] + alpha * (l[i] - r[i]);

    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 3 * N;
}

// computes y = alpha*(l-r)
// y, l and r are vectors of length N
// alpha is a scalar
void ss_scaled_diff(double* y, const double alpha,
    const double* l, const double* r, const int N)
{
	int i;
// TODO:  ACC PARALLEL region: 
    for (i = 0; i < N; i++)
        y[i] = alpha * (l[i] - r[i]);

    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 2 * N;
}

// computes y := alpha*x
// alpha is scalar
// y and x are vectors on length n
void ss_scale(double* y, const double alpha, double* x, const int N)
{
	int i;
// TODO:  ACC PARALLEL region: 
    for (i = 0; i < N; i++)
        y[i] = alpha * x[i];

    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + N;
}

// computes linear combination of two vectors y := alpha*x + beta*z
// alpha and beta are scalar
// y, x and z are vectors on length n
void ss_lcomb(double* y, const double alpha, double* x, const double beta,
    const double* z, const int N)
{
	int i;
// TODO:  ACC PARALLEL region: 
    for (i = 0; i < N; i++)
        y[i] = alpha * x[i] + beta * z[i];

    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 3 * N;
}

// copy one vector into another y := x
// x and y are vectors of length N
void ss_copy(double* y, const double* x, const int N)
{
	int i;
// TODO:  ACC PARALLEL region: 
    for (i = 0; i < N; i++)
        y[i] = x[i];
}

// conjugate gradient solver
// solve the linear system A*x = b for x
// the matrix A is implicit in the objective function for the diffusion equation
// the value in x constitute the "first guess" at the solution
// x(N)
// ON ENTRY contains the initial guess for the solution
// ON EXIT  contains the solution
void ss_cg(double* x, const double* b, const int maxiters, const double tol, int* success)
{
    //struct discretization_t* options = data::options;

    // this is the dimension of the linear system that we are to solve
    int N = options.N;

    if (!cg_initialized)
    {
        printf("INITIALIZING CG STATE\n");
        cg_init(N);
    }

    // epslion value use for matrix-vector approximation
    double eps     = 1.e-8;
    double eps_inv = 1. / eps;

    // allocate memory for temporary storage
    ss_fill(Fx,    0.0, N);
    ss_fill(Fxold, 0.0, N);
    ss_copy(xold, x, N);

    // matrix vector multiplication is approximated with
    // A*v = 1/epsilon * ( F(x+epsilon*v) - F(x) )
    //     = 1/epsilon * ( F(x+epsilon*v) - Fxold )
    // we compute Fxold at startup
    // we have to keep x so that we can compute the F(x+exps*v)
    diffusion(x, Fxold);

    // v = x + epsilon*x
    ss_scale(v, 1.0 + eps, x, N);

    // Fx = F(v)
    diffusion(v, Fx);

    // r = b - A*x
    // where A*x = (Fx-Fxold)/eps
    ss_add_scaled_diff(r, b, -eps_inv, Fx, Fxold, N);

    // p = r
    ss_copy(p, r, N);

    // rold = <r,r>
    double rold = ss_dot(r, r, N), rnew = rold;

    // check for convergence
    *success = 0;
    if (sqrt(rold) < tol)
    {
        *success = 1;
        return;
    }

    int iter;
    for (iter = 1; iter <= maxiters; iter++)
    {
        // Ap = A*p
        ss_lcomb(v, 1.0, xold, eps, p, N);
        diffusion(v, Fx);
        ss_scaled_diff(Ap, eps_inv, Fx, Fxold, N);

        // alpha = rold / p'*Ap
        double alpha = rold / ss_dot(p, Ap, N);

        // x += alpha*p
        ss_axpy(x, alpha, p, N);

        // r -= alpha*Ap
        ss_axpy(r, -alpha, Ap, N);

        // find new norm
        rnew = ss_dot(r, r, N);

        // test for convergence
        if (sqrt(rnew) < tol)
        {
            *success = 1;
            break;
        }

        // p = r + rnew.rold * p
        ss_lcomb(p, 1.0, r, rnew / rold, p, N);

        rold = rnew;
    }
    iters_cg += iter;

    if (!*success)
    {
        fprintf(stderr, "ERROR: CG failed to converge after %d iterations\n", maxiters);
        fprintf(stderr, "       achived tol = %E, required tol = %E\n", sqrt(rnew), tol);
    }
}


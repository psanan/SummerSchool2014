// *****************************************
// operators.f90
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#ifndef OPERATORS_H
#define OPERATORS_H

#include "check.h"
#include "data.h"
#include "stats.h"

#define U(j,i)    up[(i) + (j)*nx]
#define S(j,i)    sp[(i) + (j)*nx]
#define X(j,i) x_old[(i) + (j)*nx]

// Target: Given that up and sp data are already on GPU, implement a set of
// GPU kernels for diffusion operator
// Things to note:
// 1) Look into how kernels in linalg are implemented
// 2) Note the code which provides kernels with optimal compute grid configs
// 3) Note options structure has cpu:: and gpu:: versions
// 4) Note you can turn every loop of diffusion into separate GPU kernel
// and run all of them asynchronously.
// 5) We won't use the template parameters (these are for vectorizing within each thread)
namespace gpu
{
	namespace diffusion_interior_grid_points_kernel
	{
		template<short V, typename T> 
		__global__ void kernel(const double* const __restrict__ up, double* __restrict__ sp)
		{
      using namespace gpu;

			// recover global indices
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			const int j = blockIdx.y * blockDim.y + threadIdx.y;

			// Recover parameters from options (already copied to global GPU memory)
			const double alpha = options.alpha;
			const double dxs = 1000.*options.dx*options.dx;
			const int nx = options.nx;
			const int ny = options.ny;

      // return if out of range 
      if ((i < 1) || (i > nx-1) || (j < 1) || (j > ny-1)){
        return;
      }

			// Apply stencil
			S(j, i) = -(4. + alpha)*U(j,i)    // central point
                                + U(j,i-1) + U(j,i+1) // east and west
                                + U(j-1,i) + U(j+1,i) // north and south

                                + alpha*X(j,i)
                                + dxs*U(j,i)*(1.0 - U(j,i));
		}

		config_t config;
	}

	namespace diffusion_east_west_boundary_points_kernel
	{
		template<short V, typename T>
		__global__ void kernel(const double* const __restrict__ up, double* __restrict__ sp)
		{
      using namespace gpu;

			// recover global index
			const int i_east = options.nx - 1;
			const int i_west = 0;
			const int j = blockIdx.y * blockDim.y + threadIdx.y + 1; //+1  due to details of allocation (see main.cu)

			// Recover parameters from options (already copied to global GPU memory)
			const double alpha = options.alpha;
			const double dxs = 1000.*options.dx*options.dx;
			const int nx = options.nx;
			const int ny = options.nx;

      // return if out of range 
      if (j > ny-1){
        return;
      }
			// Apply stencil
			{ // East Boundary
				const int i = i_east;
				S(j, i) = -(4. + alpha) * U(j,i)
					+ U(j, i - 1) + U(j - 1, i) + U(j + 1, i)
					+ alpha*X(j, i) + bndE[j]
					+ dxs * U(j, i) * (1.0 - U(j, i));
			}

			{ // West Boundary
				const int i = i_west;
				S(j, i) = -(4. + alpha) * U(j, i)
					+ U(j, i + 1) + U(j - 1, i) + U(j + 1, i)

					+ alpha*X(j, i) + bndW[j]
					+ dxs*U(j, i) * (1.0 - U(j, i));
			}
		}

		config_t config;
	}

	namespace diffusion_north_south_boundary_points_kernel
	{
		template<short V, typename T>
		__global__ void kernel(const double* const __restrict__ up, double* __restrict__ sp)
		{
      using namespace gpu;

			// recover global indices
			const int i = blockIdx.x * blockDim.x + threadIdx.x + 1; //+1 since memory was allocated that way
			const int j_north = options.ny - 1;
			const int j_south = 0;

			// Recover parameters from options (already copied to global GPU memory)
			const double alpha = options.alpha;
			const double dxs = 1000.*options.dx*options.dx;
			const int nx = options.nx;

      // Return if out of range
      if (i > nx-1){
        return;
      }
			// Apply stencils
			{ //North Boundary
				const int j = j_north;
				S(j, i) = -(4. + alpha) * U(j, i)
					+ U(j, i - 1) + U(j, i + 1) + U(j - 1, i)
					+ alpha*X(j, i) + bndN[i]
					+ dxs * U(j, i) * (1.0 - U(j, i));
			}

			{ // South Boundary
				const int j = j_south;
				S(j, i) = -(4. + alpha) * U(j, i)
					+ U(j, i - 1) + U(j, i + 1) + U(j + 1, i)
					+ alpha * X(j, i) + bndS[i]
					+ dxs * U(j, i) * (1.0 - U(j, i));
			}
		}

		config_t config;
	}

	namespace diffusion_corner_points_kernel
	{
		__global__ void kernel(const double* const __restrict__ up, double* __restrict__ sp)
		{
      using namespace gpu;

			// recover global indices
			const int i_east = options.nx -1;
			const int i_west = 0;
			const int j_north = options.ny - 1;
			const int j_south = 0;

			// Recover parameters from options (already copied to global GPU memory)
			const double alpha = options.alpha;
			const double dxs = 1000.*options.dx*options.dx;
			const int nx = options.nx;

			// Apply stencils
      {
        const int i = i_west;
        const int j = j_north;
        S(j, i) = -(4. + alpha) * U(j, i)
          + U(j, i + 1) + U(j - 1, i)

          + alpha * X(j, i) + bndW[j] + bndN[i]
          + dxs * U(j, i) * (1.0 - U(j, i));
      }
      {
        const int i = i_east;
        const int j = j_north;
        S(j, i) = -(4. + alpha) * U(j, i)
          + U(j, i - 1) + U(j - 1, i)
          + alpha * X(j, i) + bndE[j] + bndN[i]
          + dxs * U(j, i) * (1.0 - U(j, i));
      }
      {
        const int i = i_west;
        const int j = j_south;
        S(j, i) = -(4. + alpha) * U(j, i)
          + U(j, i + 1) + U(j + 1, i)
          + alpha * X(j, i) + bndW[j] + bndS[i]
          + dxs * U(j, i) * (1.0 - U(j, i));
      }
      {
        const int i = i_east;
        const int j = j_south;
        S(j, i) = -(4. + alpha) * U(j, i)
          + U(j, i - 1) + U(j + 1, i)
          + alpha * X(j, i) + bndE[j] + bndS[i]
          + dxs * U(j, i) * (1.0 - U(j, i));
      }
    }
  }
}

inline void diffusion(const double* const __restrict__ up, double* __restrict__ sp)
{
	{
		using namespace gpu;

		// Launch kernel for parallel processing of interior points.
		{
			using namespace diffusion_interior_grid_points_kernel;
			CUDA_LAUNCH_ERR_CHECK(kernel<1, gpu::double1><<<config.grid, config.block>>>(up, sp));
		}

		// Launch kernels for parallel processing of boundary points.
		{
			using namespace diffusion_east_west_boundary_points_kernel;
			CUDA_LAUNCH_ERR_CHECK(kernel<1, gpu::double1><<<config.grid, config.block>>>(up, sp));
		}
		{
			using namespace diffusion_north_south_boundary_points_kernel;
			CUDA_LAUNCH_ERR_CHECK(kernel<1, gpu::double1><<<config.grid, config.block>>>(up, sp));
		}

		// Finally, single-threaded processing of corner points.
		{
			using namespace diffusion_corner_points_kernel;
			CUDA_LAUNCH_ERR_CHECK(kernel<<<1, 1>>>(up, sp));
		}
	}

	{
		using namespace cpu;
		
		// Accumulate the flop counts
		// 8 ops total per point
		flops_diff +=
			+ 12 * (options.nx - 2) * (options.ny - 2) // interior points
			+ 11 * (options.nx - 2  +  options.ny - 2) // NESW boundary points
			+ 11 * 4;                                  // corner points}
	}
}

#endif // OPERATORS_H


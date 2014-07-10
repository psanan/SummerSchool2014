#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cstdlib>

#include <cuda.h>


int main(int argc, char* argv[])
{
  size_t N = 10000;   // Default value
  cudaEvent_t start;
  cudaEvent_t end;
  float elapsed_time;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // generate 32M random numbers serially
  if (argc > 1) {
     N = atoi(argv[1]);
     std::cout << "Using number of elements = " << N << std::endl;
  }

  thrust::host_vector<int> h_vec(N); 
  std::generate(h_vec.begin(), h_vec.end(), rand);

  cudaEventRecord(start,0);

  // A new device vector initialized to same values
  thrust::device_vector<int> d_vec(h_vec.begin(),h_vec.end());

  thrust::sort(d_vec.begin(),d_vec.end());

  // Copy back 
  thrust::copy(d_vec.begin(),d_vec.end(),h_vec.begin());

  cudaEventSynchronize(end);
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, start, end);

  std::cout << "device sort took " << elapsed_time << " milliseconds" << std::endl;

  // output smallest/largest value
  std::cout << "Smallest value is\n" << h_vec[0] << std::endl;
  std::cout << "Largest value is\n" << h_vec[h_vec.size()-1] << std::endl;

  return 0;
}

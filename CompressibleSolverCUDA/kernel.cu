#define PI 3.14159265359
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <chrono>
#include <fstream>

const int nx = 4097, ny = nx, nt = 100, N = nx * ny;
const double oneOverN = 1. / (double)N;

// Kernel launch constants for x-derivative
const unsigned int dx_threads_x = 1024;
const unsigned int dx_threads_y = 1;
const unsigned int dx_blocks_x = (unsigned int)ceil((double)nx / dx_threads_x);
const unsigned int dx_blocks_y = (unsigned int)ceil((double)ny / dx_threads_y);
const dim3 dx_blockGrid(dx_blocks_x, dx_blocks_y);
const dim3 dx_threadGrid(dx_threads_x, dx_threads_y);

// Kernel launch constants for y-derivative
const int ptsPerThrd = 4;
const unsigned int dy_threads_x = 32;
const unsigned int dy_threads_y = 32;
const unsigned int dy_blocks_x = (unsigned int)ceil((double)nx / dy_threads_x);
const unsigned int dy_blocks_y = (unsigned int)ceil((double)ny / ptsPerThrd / dy_threads_y);
const dim3 dy_blockGrid(dy_blocks_x, dy_blocks_y);
const dim3 dy_threadGrid(dy_threads_x, dy_threads_y);

__device__ __constant__ double deriv_consts[4];
__device__ __constant__ double adams_consts[2];
__device__ __constant__ double etatt_consts[2];
__device__ __constant__ int dev_nx;
__device__ __constant__ int dev_ny;
__device__ __constant__ int dev_N;
__device__ __constant__ int pointsPerThread;
__device__ __constant__ double deltaX;
__device__ __constant__ double deltaY;
__device__ __constant__ double xLength;
__device__ __constant__ double yLength;
__device__ __constant__ double tempInf;
__device__ __constant__ double pressInf;
__device__ __constant__ double uInf;
__device__ __constant__ double rhoInf;
__device__ __constant__ double heatCapacityV;
__device__ __constant__ double oneOverEta;
__device__ __constant__ double cylinderRadSquared;
__device__ __constant__ double dynViscosity;
__device__ __constant__ double xkt;
__device__ __constant__ double lambda;

void HandleError(cudaError);
double InitialiseDeviceConstants();
void PrintAverages(const int, const double*, const double*, const double*);
void WriteToFile(const int length, long long runtimes[]);
void Fluxx(double* cylinderMask, double* uVelocity, double* vVelocity, double* temp, double* energy,
    double* rho, double* pressure, double* rou, double* rov, double* roe, double* scp,
    double* tb1, double* tb2, double* tb3, double* tb4, double* tb5, double* tb6, double* tb7, double* tb8, double* tb9,
    double* tba, double* tbb, double* fro, double* fru, double* frv, double* fre, double* ftp);

__global__ void InitialiseArrays(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
template<int>
__global__ void Derix(const double*, double*);
template<int>
__global__ void Deriy(const double*, double*);
__global__ void SubFluxx1(const double*, const double*, const double*, double*, double*, double*);
__global__ void SubFluxx2(const double* tb3, const double* tb4, const double* tb5, const double* tb6, const double* tb7, const double* tb9,
    const double* cylinderMask, const double* uVelocity, const double* vVelocity, const double* rou, const double* rov,
    double* tb1, double* tb2, double* tba, double* fru);
__global__ void SubFluxx3(const double* tb3, const double* tb4, const double* tb5, const double* tb6, const double* tb7, const double* tb9,
    const double* cylinderMask, const double* vVelocity,
    double* tbb, double* frv);
__global__ void SubFluxx4(const double* tb1, const double* tb2, const double* tb3, const double* tb4,
    const double* uVelocity, const double* vVelocity, const double* scp, const double* cylinderMask, double* ftp);
__global__ void SubFluxx5(const double* uVelocity, const double* vVelocity,
    const double* tba, const double* tbb, const double* pressure, const double* roe,
    double* tb1, double* tb2, double* tb3, double* tb4, double* fre);
__global__ void SubFluxx6(const double* tb5, const double* tb6, const double* tb7, const double* tb8,
    const double* tb9, const double* tba, double* fre);
__global__ void Adams(const double* phi_current, double* phi_previous, double* phi_integral);
__global__ void Etatt(const double* rho, const double* rou, const double* rov, const double* roe,
    double* uVelocity, double* vVelocity, double* pressure, double* temp);

int main()
{
    double* cylinderMask, * uVelocity, * vVelocity, * temp, * energy, * rho, * pressure, * rou, * rov, * roe, * scp;
    double* tb1, * tb2, * tb3, * tb4, * tb5, * tb6, * tb7, * tb8, * tb9, * tba, * tbb;
    double* fro, * fru, * frv, * fre, * ftp;
    double* prev_fro, * prev_fru, * prev_frv, * prev_fre, * prev_ftp;

    double** N_sizedArrays[] = {
        &tb1, &tb2, &tb3, &tb4, &tb5, &tb6, &tb7, &tb8, &tb9, &tba, &tbb,
        &fro, &fru, &frv, &fre, &ftp,
        &prev_fro, &prev_fru, &prev_frv, &prev_fre, &prev_ftp,
        &cylinderMask, &uVelocity, &vVelocity, &temp, &energy, &rho, &pressure, &rou, &rov, &roe, &scp
    };

    double* deviceArrays;
    const unsigned long int bytes = N * sizeof(double);
    const unsigned long long numOfArrays = sizeof(N_sizedArrays) / sizeof(*N_sizedArrays);

    // Allocate one large chunk of memory, then divide into the separate arrays
    HandleError(cudaMalloc(&deviceArrays, numOfArrays * bytes));

    for (int i = 0; i < numOfArrays; i++) {
        *N_sizedArrays[i] = deviceArrays + i * N;
    }

    // Initilaise all 5 prev_f arrays to 0
    HandleError(cudaMemsetAsync(prev_fro, 0.0, 5 * bytes));
    HandleError(cudaMemsetAsync(cylinderMask, 0.0, bytes));

    double deltaT = InitialiseDeviceConstants();
    InitialiseArrays << < dx_blockGrid, dx_threadGrid >> > (cylinderMask, uVelocity, vVelocity, temp, energy, rho, pressure, rou, rov, roe, scp);

    printf("The time step of the simulation is %.9E \n", deltaT);
    printf("Average values at t=");
    std::cout.flush();
    PrintAverages(0, uVelocity, vVelocity, scp);

    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStream_t stream3;
    cudaStream_t stream4;
    cudaStream_t stream5;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream5);

    long long runtimes[nt];

    for (int i = 1; i <= nt; i++) {
        //auto start = std::chrono::high_resolution_clock::now();

        Fluxx(cylinderMask, uVelocity, vVelocity, temp, energy, rho, pressure, rou, rov, roe, scp,
            tb1, tb2, tb3, tb4, tb5, tb6, tb7, tb8, tb9, tba, tbb, fro, fru, frv, fre, ftp);
  
        Adams << < dx_blockGrid, dx_threadGrid, 0, stream1 >> > (fro, prev_fro, rho);
        cudaMemcpyAsync(prev_fro, fro, bytes, cudaMemcpyDeviceToDevice, stream1);
        Adams << < dx_blockGrid, dx_threadGrid, 0, stream2 >> > (fru, prev_fru, rou);
        cudaMemcpyAsync(prev_fru, fru, bytes, cudaMemcpyDeviceToDevice, stream2);
        Adams << < dx_blockGrid, dx_threadGrid, 0, stream3 >> > (frv, prev_frv, rov);
        cudaMemcpyAsync(prev_frv, frv, bytes, cudaMemcpyDeviceToDevice, stream3);
        Adams << < dx_blockGrid, dx_threadGrid, 0, stream4 >> > (fre, prev_fre, roe);
        cudaMemcpyAsync(prev_fre, fre, bytes, cudaMemcpyDeviceToDevice, stream4);
        Adams << < dx_blockGrid, dx_threadGrid, 0, stream5 >> > (ftp, prev_ftp, scp);
        cudaMemcpyAsync(prev_ftp, ftp, bytes, cudaMemcpyDeviceToDevice, stream5);

        Etatt << < ceil(nx * ny / 256) + 1, 256 >> > (rho, rou, rov, roe, uVelocity, vVelocity, pressure, temp);

        //cudaDeviceSynchronize();
        //auto stop = std::chrono::high_resolution_clock::now();
        //runtimes[i - 1] = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

        PrintAverages(i, uVelocity, vVelocity, scp);
    }

    // Free memory on host memory
    HandleError(cudaFree(deviceArrays));

    //WriteToFile(nt, runtimes);

    return 0;
}

double InitialiseDeviceConstants() {

    const double reynolds = 200., mach = 0.2, prandtl = 0.7;
    const double host_rhoInf = 1., cInf = 1., cylinderD = 1., heatCapacityP = 1., heatRatio = 1.4;
    const double host_cylinderRadSquared = cylinderD * cylinderD / 4.;

    const double host_heatCapacityV = heatCapacityP / heatRatio;
    const double host_uInf = mach * cInf;
    const double host_dynViscosity = host_rhoInf * host_uInf * cylinderD / reynolds;
    // thermal conductivity
    const double host_lambda = host_dynViscosity * heatCapacityP / prandtl;
    const double host_tempInf = cInf * cInf / (heatCapacityP * (heatRatio - 1.));
    const double eta = 0.1 / 2.;
    const double host_oneOverEta = 1. / eta;
    const double host_xkt = host_lambda / (heatCapacityP * host_rhoInf);
    const double host_pressInf = host_rhoInf * host_tempInf * heatCapacityP * (heatRatio - 1.) / heatRatio;

    const double host_xLength = 4. * cylinderD;
    const double host_yLength = 4. * cylinderD;
    const double host_deltaX = host_xLength / (double)nx;
    const double host_deltaY = host_yLength / (double)ny;
    const double CFL = 0.025;
    const double deltaT = CFL * host_deltaX;

    // Derivative stencil constants
    const double d_consts[] = { 1. / (2. * host_deltaX) , 1. / (2. * host_deltaY), 1. / host_deltaX / host_deltaX, 1. / host_deltaY / host_deltaY };
    const double a_consts[] = { 1.5 * deltaT, 0.5 * deltaT };
    const double e_consts[] = { heatRatio - 1., heatRatio / (heatRatio - 1.) / heatCapacityP };

    // Allocate constant gpu memory
    HandleError(cudaMemcpyToSymbolAsync(deriv_consts, d_consts, 4 * sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(adams_consts, a_consts, 2 * sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(etatt_consts, e_consts, 2 * sizeof(double)));
    
    HandleError(cudaMemcpyToSymbolAsync(dev_nx, &nx, sizeof(int)));
    HandleError(cudaMemcpyToSymbolAsync(dev_ny, &ny, sizeof(int)));
    HandleError(cudaMemcpyToSymbolAsync(dev_N, &N, sizeof(int)));
    HandleError(cudaMemcpyToSymbolAsync(pointsPerThread, &ptsPerThrd, sizeof(int)));
    HandleError(cudaMemcpyToSymbolAsync(deltaX, &host_deltaX, sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(deltaY, &host_deltaY, sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(xLength, &host_xLength, sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(yLength, &host_yLength, sizeof(double)));
    
    HandleError(cudaMemcpyToSymbolAsync(tempInf, &host_tempInf, sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(pressInf, &host_pressInf, sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(uInf, &host_uInf, sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(rhoInf, &host_rhoInf, sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(heatCapacityV, &host_heatCapacityV, sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(oneOverEta, &host_oneOverEta, sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(cylinderRadSquared, &host_cylinderRadSquared, sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(dynViscosity, &host_dynViscosity, sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(xkt, &host_xkt, sizeof(double)));
    HandleError(cudaMemcpyToSymbolAsync(lambda, &host_lambda, sizeof(double)));

    return deltaT;
}

void HandleError(cudaError error) {

    if (error != cudaSuccess) {
        printf("An error occured: %i: %s", error, cudaGetErrorString(error));
        printf("\nExiting...");
        exit(EXIT_FAILURE);
    }
}

void PrintAverages(const int timestep, const double* uVelocity, const double* vVelocity, const double* scp) {

    double mean[3];
    thrust::device_vector<double> arrayVect;
    const double* arrays[] = { uVelocity, vVelocity, scp };

    for (int i = 0; i < 3; i++) {
        arrayVect = thrust::device_vector<double>(arrays[i], arrays[i] + N);
        mean[i] = thrust::reduce(arrayVect.begin(), arrayVect.end(), 0., thrust::plus<double>()) * oneOverN;
    }   

    printf("%i %.9G %.9E %.9G \n", timestep, mean[0], mean[1], mean[2]);
}

void WriteToFile(const int length, long long runtimes[]) {

    using namespace std; {
        string filename = "Runtimes/cuda_runtimes_" + to_string(nx) + ".txt";
        ofstream outputFile;

        outputFile.open(filename, ios::trunc);

        for (int i = 0; i < nt; i++) {
            outputFile << runtimes[i] << "\n";
        }

        outputFile.close();
    }
}

void Fluxx(double* cylinderMask, double* uVelocity, double* vVelocity, double* temp, double* energy, 
    double* rho, double* pressure, double* rou, double* rov, double* roe, double* scp,
    double* tb1, double* tb2, double* tb3, double* tb4, double* tb5, double* tb6, double* tb7, double* tb8, double* tb9,
    double* tba, double* tbb, double* fro, double* fru, double* frv, double* fre, double* ftp) {

    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (rou, tb1);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (rov, tb2);

    SubFluxx1 << < ceil(nx * ny / 256) + 1, 256 >> > (uVelocity, vVelocity, rou, fro, tb1, tb2);

    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (pressure, tb3);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (tb1, tb4);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (tb2, tb5);
    Derix<2> << < dx_blockGrid, dx_threadGrid >> > (uVelocity, tb6);
    Deriy<2> << < dy_blockGrid, dy_threadGrid >> > (uVelocity, tb7);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (vVelocity, tb8);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (tb8, tb9);

    SubFluxx2 << < ceil(nx * ny / 256) + 1, 256 >> > (tb3, tb4, tb5, tb6, tb7, tb9,
        cylinderMask, uVelocity, vVelocity, rou, rov, tb1, tb2, tba, fru);

    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (pressure, tb3);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (tb1, tb4);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (tb2, tb5);
    Derix<2> << < dx_blockGrid, dx_threadGrid >> > (vVelocity, tb6);
    Deriy<2> << < dy_blockGrid, dy_threadGrid >> > (vVelocity, tb7);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (uVelocity, tb8);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (tb8, tb9);

    SubFluxx3 << < ceil(nx * ny / 256) + 1, 256 >> > (tb3, tb4, tb5, tb6, tb7, tb9,
        cylinderMask, vVelocity, tbb, frv);

    // Equation for the tempature
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (scp, tb1);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (scp, tb2);
    Derix<2> << < dx_blockGrid, dx_threadGrid >> > (scp, tb3);
    Deriy<2> << < dy_blockGrid, dy_threadGrid >> > (scp, tb4);

    SubFluxx4 << < ceil(nx * ny / 256) + 1, 256 >> > (tb1, tb2, tb3, tb4,
        uVelocity, vVelocity, scp, cylinderMask, ftp);

    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (uVelocity, tb1);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (vVelocity, tb2);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (uVelocity, tb3);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (vVelocity, tb4);

    SubFluxx5 << < ceil(nx * ny / 256) + 1, 256 >> > (uVelocity, vVelocity, tba, tbb, pressure, roe, tb1, tb2, tb3, tb4, fre);

    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (tb1, tb5);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (tb2, tb6);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (tb3, tb7);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (tb4, tb8);
    Derix<2> << < dx_blockGrid, dx_threadGrid >> > (temp, tb9);
    Deriy<2> << < dy_blockGrid, dy_threadGrid >> > (temp, tba);

    SubFluxx6 << < ceil(nx * ny / 256) + 1, 256 >> > (tb5, tb6, tb7, tb8, tb9, tba, fre);
}

void __global__ InitialiseArrays(double* cylinderMask, double* uVelocity, double* vVelocity, double* temp,
    double* energy, double* rho, double* pressure, double* rou, double* rov, double* roe, double* scp)
{
    // Local and global arrays use row-major storage
    int global_i = blockDim.x * blockIdx.x + threadIdx.x;
    int global_j = blockDim.y * blockIdx.y + threadIdx.y;
    int global_idx = dev_nx * global_j + global_i;

    if (global_i >= dev_nx || global_j >= dev_ny) { return; }

    // Distances from centre of grid
    double dx = (double)global_i * deltaX - xLength / 2.;
    double dy = (double)global_j * deltaY - yLength / 2.;

    // Masks covers points inside circle
    if (dx * dx + dy * dy < cylinderRadSquared) {
        cylinderMask[global_idx] = 1.;
    }

    uVelocity[global_idx] = uInf;
    // Add small velocity perturbation
    double vVel = 0.01 * (sin(4. * PI * (double)global_i * deltaX / xLength)
        + sin(7. * PI * (double)global_i * deltaX / xLength))
        * exp(-((double)global_j * deltaY - yLength / 2.) * ((double)global_j * deltaY - yLength / 2.));

    vVelocity[global_idx] = vVel;
    temp[global_idx] = tempInf;
    pressure[global_idx] = pressInf;

    double energyVal = heatCapacityV * tempInf + 0.5 * (uInf * uInf + vVel * vVel);
    energy[global_idx] = energyVal;
    rho[global_idx] = rhoInf;
    rou[global_idx] = rhoInf * uInf;
    rov[global_idx] = rhoInf * vVel;
    roe[global_idx] = rhoInf * energyVal;
    scp[global_idx] = 1.;
}

template<int derivative>
__global__ void Derix(const double* f, double* deriv_f) {

    // Local and global arrays use row-major storage
    int global_i = blockDim.x * blockIdx.x + threadIdx.x;
    int global_j = blockDim.y * blockIdx.y + threadIdx.y;
    int global_idx = dev_nx * global_j + global_i;
    int i = threadIdx.x + 1;
    int j = threadIdx.y;
    int local_idx = (blockDim.x + 2) * j + i;

    if (global_i > dev_nx || global_j >= dev_ny) { return; }
    if (global_i == dev_nx) { global_idx -= dev_nx; }

    __shared__ double tile_f[(1024 + 2) * 1];

    // Copy from global to shared memory
    tile_f[local_idx] = f[global_idx];
    if (global_i == dev_nx) { return; }


    // Apply periodic boundary conditions
    if (threadIdx.x == 0) {
        if (global_i == 0) {
            tile_f[local_idx - 1] = f[global_idx + dev_nx - 1];
        }
        else {
            tile_f[local_idx - 1] = f[global_idx - 1];
        }
    }
 
    if (threadIdx.x == blockDim.x - 1) {
        if (global_i == dev_nx - 1) {
            tile_f[local_idx + 1] = f[dev_nx * global_j];
        }
        else {
            tile_f[local_idx + 1] = f[global_idx + 1];
        }
    }

    __syncthreads();

    switch (derivative) {
    case 1:
        // Case of 1st x derivative
        deriv_f[global_idx] = deriv_consts[0] * (tile_f[local_idx + 1] - tile_f[local_idx - 1]);
        break;

    case 2:
        // Case of 2nd x derivative
        deriv_f[global_idx] = deriv_consts[2] * (tile_f[local_idx + 1] - 2 * tile_f[local_idx] + tile_f[local_idx - 1]);
        break;
    }
}

template<int derivative>
__global__ void Deriy(const double* f, double* deriv_f) {

    // Local and global arrays use row-major storage
    int global_i = blockDim.x * blockIdx.x + threadIdx.x;
    int global_j = blockDim.y * blockIdx.y * pointsPerThread + threadIdx.y;
    int global_idx = dev_nx * global_j + global_i;
    int i = threadIdx.x;
    int j = threadIdx.y + 1;
    int local_idx = (blockDim.y * pointsPerThread + 2) * i + j;

    if (global_i >= dev_nx || global_j > dev_ny) { return; }
    if (global_j == dev_ny) { global_idx = global_i; }

    __shared__ double tile_f[(32 * ptsPerThrd + 2) * 32];

    // Apply periodic boundary conditions
    if (threadIdx.y == 0) {
        if (global_j == 0) {
            tile_f[local_idx - 1] = f[global_idx + dev_nx * (dev_ny - 1)];
        }
        else {
            tile_f[local_idx - 1] = f[global_idx - dev_nx];
        }
    }

    int count = 0;
    while (count < pointsPerThread && global_idx < dev_nx * dev_ny) {
        tile_f[local_idx] = f[global_idx];

        global_idx += dev_nx * blockDim.y;
        local_idx += blockDim.y;
        count++;
    }

    if (global_j == dev_ny) { return; }

    if (threadIdx.y == blockDim.y - 1) {
        global_idx -= dev_nx * blockDim.y;
        local_idx -= blockDim.y;

        if (global_idx == dev_nx * (dev_ny - 1) + global_i) {
            tile_f[local_idx + 1] = f[global_i];
        }
        else {
            tile_f[local_idx + 1] = f[global_idx + dev_nx];
        }
    }

    global_idx = dev_nx * global_j + global_i;
    local_idx = (blockDim.y * pointsPerThread + 2) * i + j;

    __syncthreads();

    switch (derivative) {
    case 1:
        count = 0;
        while (count < pointsPerThread && global_idx < dev_nx * dev_ny) {
            // Case of 1st x derivative
            deriv_f[global_idx] = deriv_consts[1] * (tile_f[local_idx + 1] - tile_f[local_idx - 1]);

            global_idx += dev_nx * blockDim.y;
            local_idx += blockDim.y;
            count++;
        }
        
        break;

    case 2:
        count = 0;
        while (count < pointsPerThread && global_idx < dev_nx * dev_ny) {
            // Case of 2nd x derivative
            deriv_f[global_idx] = deriv_consts[3] * (tile_f[local_idx + 1] - 2 * tile_f[local_idx] + tile_f[local_idx - 1]);

            global_idx += dev_nx * blockDim.y;
            local_idx += blockDim.y;
            count++;
        }

        break;
    }

}

__global__ void SubFluxx1(const double* uVelocity, const double* vVelocity, const double* rou,
    double* fro, double* tb1, double* tb2) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dev_nx * dev_ny) { return; }
    
    fro[idx] = -tb1[idx] - tb2[idx];

    double local_rou = rou[idx];
    tb1[idx] = local_rou * uVelocity[idx];
    tb2[idx] = local_rou * vVelocity[idx];
    
}

__global__ void SubFluxx2(const double* tb3, const double* tb4, const double* tb5, const double* tb6, const double* tb7, const double* tb9,
    const double* cylinderMask, const double* uVelocity, const double* vVelocity, const double* rou, const double* rov,
    double* tb1, double* tb2, double* tba, double* fru) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dev_nx * dev_ny) { return; }

    const double temp = 1. / 3., temp2 = 4. * temp;
    double cache;

    cache = dynViscosity * (temp2 * tb6[idx] + tb7[idx] + temp * tb9[idx]);
    tba[idx] = cache;
    fru[idx] = -tb3[idx] - tb4[idx] - tb5[idx] + cache - oneOverEta * cylinderMask[idx] * uVelocity[idx];
    
    cache = vVelocity[idx];
    tb1[idx] = rou[idx] * cache;
    tb2[idx] = rov[idx] * cache;
}

__global__ void SubFluxx3(const double* tb3, const double* tb4, const double* tb5, const double* tb6, const double* tb7, const double* tb9,
    const double* cylinderMask, const double* vVelocity,
    double* tbb, double* frv) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dev_nx * dev_ny) { return; }
    const double temp = 1. / 3., temp2 = 4. * temp;
    double cache;

    cache = dynViscosity * (tb6[idx] + temp2 * tb7[idx] + temp * tb9[idx]);
    frv[idx] = -tb3[idx] - tb4[idx] - tb5[idx] + cache - oneOverEta * cylinderMask[idx] * vVelocity[idx];
    tbb[idx] = cache;
}

__global__ void SubFluxx4(const double* tb1, const double* tb2, const double* tb3, const double* tb4,
    const double* uVelocity, const double* vVelocity, const double* scp, const double* cylinderMask, double* ftp) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dev_nx * dev_ny) { return; }

    ftp[idx] = -uVelocity[idx] * tb1[idx] - vVelocity[idx] * tb2[idx]
        + xkt * (tb3[idx] + tb4[idx]) - oneOverEta * cylinderMask[idx] * scp[idx];
}


__global__ void SubFluxx5(const double* uVelocity, const double* vVelocity,
    const double* tba, const double* tbb, const double* pressure, const double* roe,
    double* tb1, double* tb2, double* tb3, double* tb4, double* fre) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dev_nx * dev_ny) { return; }

    double tb[] = { tb1[idx], tb2[idx], tb3[idx], tb4[idx] };
    double velCache[] = { uVelocity[idx], vVelocity[idx] };
    double locRoe = roe[idx], locPres = pressure[idx];

    fre[idx] = dynViscosity * ((velCache[0] * tba[idx] + velCache[1] * tbb[idx])
        + 2. * (tb[0] * tb[0] + tb[1] * tb[1])
        - 2. / 3. * (tb[0] + tb[1]) * (tb[0] + tb[1])
        + (tb[2] + tb[3]) * (tb[2] + tb[3]));

    tb1[idx] = locRoe * velCache[0];
    tb2[idx] = locPres * velCache[0];
    tb3[idx] = locRoe * velCache[1];
    tb4[idx] = locPres * velCache[1];
}

__global__ void SubFluxx6(const double* tb5, const double* tb6, const double* tb7, const double* tb8,
    const double* tb9, const double* tba, double* fre) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dev_nx * dev_ny) { return; }

    fre[idx] = fre[idx] - tb5[idx] - tb6[idx] - tb7[idx] - tb8[idx] + lambda * (tb9[idx] + tba[idx]);
}

__global__ void Adams(const double* phi_current, double* phi_previous, double* phi_integral) {

    // Local and global arrays use row-major storage
    int global_i = blockDim.x * blockIdx.x + threadIdx.x;
    int global_j = blockDim.y * blockIdx.y + threadIdx.y;
    int global_idx = dev_nx * global_j + global_i;

    if (global_i >= dev_nx || global_j >= dev_ny) { return; }

    phi_integral[global_idx] += adams_consts[0] * phi_current[global_idx] - adams_consts[1] * phi_previous[global_idx];

}

__global__ void Etatt(const double* rho, const double* rou, const double* rov, const double* roe,
    double* uVelocity, double* vVelocity, double* pressure, double* temp) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dev_nx * dev_ny) { return; }

    double oneOverRho = 1. / rho[idx];
    double roVel[] = { rou[idx], rov[idx] };
    double velocity[] = { oneOverRho * roVel[0], oneOverRho * roVel[1] };
    double pressCache = etatt_consts[0] * (roe[idx] - 0.5 * (roVel[0] * velocity[0] + roVel[1] * velocity[1]));

    temp[idx] = etatt_consts[1] * oneOverRho * pressCache;
    pressure[idx] = pressCache;
    uVelocity[idx] = velocity[0];
    vVelocity[idx] = velocity[1];
}
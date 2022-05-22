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
#include<fstream>

const int nx = 4097, ny = nx, nt = 100, ns = 3, nf = 3, N = nx * ny;
const int mx = ns * nx, my = nf * ny;

const double reynolds = 200., mach = 0.2, prandtl = 0.7;
const double rhoInf = 1., cInf = 1., cylinderD = 1., heatCapacityP = 1., heatRatio = 1.4;

const double heatCapacityV = heatCapacityP / heatRatio;
const double uInf = mach * cInf;
const double dynViscosity = rhoInf * uInf * cylinderD / reynolds;
// thermal conductivity
const double lambda = dynViscosity * heatCapacityP / prandtl;
const double tempInf = cInf * cInf / (heatCapacityP * (heatRatio - 1.));
const double eta = 0.1 / 2.;
const double oneOverEta = 1. / eta;
const double xkt = lambda / (heatCapacityP * rhoInf);

const double xLength = 4. * cylinderD;
const double yLength = 4. * cylinderD;
const double deltaX = xLength / (double)nx;
const double deltaY = yLength / (double)ny;
const double oneOverN = 1. / (double) N;
const double CFL = 0.025;
const double deltaT = CFL * deltaX;

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

// Derivative stencil constants
const double d_consts[] = { 1. / (2. * deltaX) , 1. / (2. * deltaY), 1. / deltaX / deltaX, 1. / deltaY / deltaY };
const double a_consts[] = { 1.5 * deltaT, 0.5 * deltaT };
const double e_consts[] = { heatRatio - 1., heatRatio / (heatRatio - 1.) / heatCapacityP };

__constant__ double deriv_consts[4];
__constant__ double adams_consts[2];
__constant__ double etatt_consts[2];
__constant__ int dev_nx;
__constant__ int dev_ny;
__constant__ int pointsPerThread;

void InitialiseArrays(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
void HandleError(cudaError);
void AllocateGpuMemory(double* [], double** [], const int, const unsigned long int);
void PrintAverages(const int, const double*, const double*, const double*);
void WriteToFile(const int length, long long runtimes[]);
void Fluxx(double* gpu_cylinderMask, double* gpu_uVelocity, double* gpu_vVelocity, double* gpu_temp, double* gpu_energy,
    double* gpu_rho, double* gpu_pressure, double* gpu_rou, double* gpu_rov, double* gpu_roe, double* gpu_scp,
    double* tb1, double* tb2, double* tb3, double* tb4, double* tb5, double* tb6, double* tb7, double* tb8, double* tb9,
    double* tba, double* tbb, double* fro, double* fru, double* frv, double* fre, double* ftp);

template<int>
__global__ void Derix(const double*, double*);
template<int>
__global__ void Deriy(const double*, double*);
__global__ void SubFluxx1(const double*, const double*, const double*, double*, double*, double*);
__global__ void SubFluxx2(const double dynViscosity, const double oneOverEta, const double* tb3, const double* tb4, const double* tb5, const double* tb6, const double* tb7, const double* tb9,
    const double* cylinderMask, const double* uVelocity, const double* vVelocity, const double* rou, const double* rov,
    double* tb1, double* tb2, double* tba, double* fru);
__global__ void SubFluxx3(const double dynViscosity, const double oneOverEta,
    const double* tb3, const double* tb4, const double* tb5, const double* tb6, const double* tb7, const double* tb9,
    const double* cylinderMask, const double* vVelocity,
    double* tbb, double* frv);
__global__ void SubFluxx4(const double oneOverEta, const double xkt, const double* tb1, const double* tb2, const double* tb3, const double* tb4,
    const double* uVelocity, const double* vVelocity, const double* scp, const double* cylinderMask, double* ftp);
__global__ void SubFluxx5(const double dynViscosity, const double* uVelocity, const double* vVelocity,
    const double* tba, const double* tbb, const double* pressure, const double* roe,
    double* tb1, double* tb2, double* tb3, double* tb4, double* fre);
__global__ void SubFluxx6(const double lambda, const double* tb5, const double* tb6, const double* tb7, const double* tb8,
    const double* tb9, const double* tba, double* fre);
__global__ void Adams(const double* phi_current, double* phi_previous, double* phi_integral);
__global__ void Etatt(const double* rho, const double* rou, const double* rov, const double* roe,
    double* uVelocity, double* vVelocity, double* pressure, double* temp);

int main()
{
    const int numOfVariables = 11;
    double* cylinderMask = new double[nx * ny]();
    double* uVelocity = new double[nx * ny];
    double* vVelocity = new double[nx * ny];
    double* temp = new double[nx * ny];
    double* energy = new double[nx * ny];
    double* rho = new double[nx * ny];
    double* pressure = new double[nx * ny];
    double* rou = new double[nx * ny];
    double* rov = new double[nx * ny];
    double* roe = new double[nx * ny];
    double* scp = new double[nx * ny];
    double averages[3], *gpu_averages;

    double* gpu_cylinderMask, * gpu_uVelocity, * gpu_vVelocity, * gpu_temp, * gpu_energy, * gpu_rho, * gpu_pressure, * gpu_rou, * gpu_rov, * gpu_roe, * gpu_scp;

    double* hostVariables[numOfVariables] = { cylinderMask, uVelocity, vVelocity, temp, energy, rho, pressure, rou, rov, roe, scp };
    double** gpuVariables[numOfVariables] = { &gpu_cylinderMask, &gpu_uVelocity, &gpu_vVelocity, &gpu_temp, &gpu_energy, &gpu_rho, &gpu_pressure, &gpu_rou, &gpu_rov, &gpu_roe, &gpu_scp };

    double* tb1, * tb2, * tb3, * tb4, * tb5, * tb6, * tb7, * tb8, * tb9;
    double* tba, * tbb, * fro, * fru, * frv, * fre, * ftp;
    double* prev_fro, * prev_fru, * prev_frv, * prev_fre, * prev_ftp;

    const unsigned long int bytes = nx * ny * sizeof(double);
    HandleError(cudaMalloc(&tb1, bytes));
    HandleError(cudaMalloc(&tb2, bytes));
    HandleError(cudaMalloc(&tb3, bytes));
    HandleError(cudaMalloc(&tb4, bytes));
    HandleError(cudaMalloc(&tb5, bytes));
    HandleError(cudaMalloc(&tb6, bytes));
    HandleError(cudaMalloc(&tb7, bytes));
    HandleError(cudaMalloc(&tb8, bytes));
    HandleError(cudaMalloc(&tb9, bytes));
    HandleError(cudaMalloc(&tba, bytes));
    HandleError(cudaMalloc(&tbb, bytes));
    HandleError(cudaMalloc(&fro, bytes));
    HandleError(cudaMalloc(&fru, bytes));
    HandleError(cudaMalloc(&frv, bytes));
    HandleError(cudaMalloc(&fre, bytes));
    HandleError(cudaMalloc(&ftp, bytes));
    HandleError(cudaMalloc(&prev_fro, bytes));
    HandleError(cudaMalloc(&prev_fru, bytes));
    HandleError(cudaMalloc(&prev_frv, bytes));
    HandleError(cudaMalloc(&prev_fre, bytes));
    HandleError(cudaMalloc(&prev_ftp, bytes));
    HandleError(cudaMemsetAsync(prev_fro, 0.0, bytes));
    HandleError(cudaMemsetAsync(prev_fru, 0.0, bytes));
    HandleError(cudaMemsetAsync(prev_frv, 0.0, bytes));
    HandleError(cudaMemsetAsync(prev_fre, 0.0, bytes));
    HandleError(cudaMemsetAsync(prev_ftp, 0.0, bytes));

    HandleError(cudaMalloc(&gpu_averages, 3 * sizeof(double)));

    InitialiseArrays(cylinderMask, uVelocity, vVelocity, temp, energy, rho, pressure, rou, rov, roe, scp);
    AllocateGpuMemory(hostVariables, gpuVariables, numOfVariables, bytes);

    printf("The time step of the simulation is %.9E \n", deltaT);
    printf("Average values at t=");
    std::cout.flush();
    PrintAverages(0, gpu_uVelocity, gpu_vVelocity, gpu_scp);

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
        auto start = std::chrono::high_resolution_clock::now();

        Fluxx(gpu_cylinderMask, gpu_uVelocity, gpu_vVelocity, gpu_temp, gpu_energy, gpu_rho, gpu_pressure, gpu_rou, gpu_rov, gpu_roe, gpu_scp,
            tb1, tb2, tb3, tb4, tb5, tb6, tb7, tb8, tb9, tba, tbb, fro, fru, frv, fre, ftp);
  
        Adams << < dx_blockGrid, dx_threadGrid, 0, stream1 >> > (fro, prev_fro, gpu_rho);
        cudaMemcpyAsync(prev_fro, fro, bytes, cudaMemcpyDeviceToDevice, stream1);
        Adams << < dx_blockGrid, dx_threadGrid, 0, stream2 >> > (fru, prev_fru, gpu_rou);
        cudaMemcpyAsync(prev_fru, fru, bytes, cudaMemcpyDeviceToDevice, stream2);
        Adams << < dx_blockGrid, dx_threadGrid, 0, stream3 >> > (frv, prev_frv, gpu_rov);
        cudaMemcpyAsync(prev_frv, frv, bytes, cudaMemcpyDeviceToDevice, stream3);
        Adams << < dx_blockGrid, dx_threadGrid, 0, stream4 >> > (fre, prev_fre, gpu_roe);
        cudaMemcpyAsync(prev_fre, fre, bytes, cudaMemcpyDeviceToDevice, stream4);
        Adams << < dx_blockGrid, dx_threadGrid, 0, stream5 >> > (ftp, prev_ftp, gpu_scp);
        cudaMemcpyAsync(prev_ftp, ftp, bytes, cudaMemcpyDeviceToDevice, stream5);

        Etatt << < ceil(nx * ny / 256) + 1, 256 >> > (gpu_rho, gpu_rou, gpu_rov, gpu_roe, gpu_uVelocity, gpu_vVelocity, gpu_pressure, gpu_temp);

        auto stop = std::chrono::high_resolution_clock::now();
        runtimes[i] = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        PrintAverages(i, gpu_uVelocity, gpu_vVelocity, gpu_scp);
    }

    // Free memory on both host and device
    for (int i = 0; i < numOfVariables; i++) {
        HandleError(cudaFree(*gpuVariables[i]) );
        delete[] hostVariables[i];
    }

    HandleError(cudaFree(tb1));
    HandleError(cudaFree(tb2));
    HandleError(cudaFree(tb3));
    HandleError(cudaFree(tb4));
    HandleError(cudaFree(tb5));
    HandleError(cudaFree(tb6));
    HandleError(cudaFree(tb7));
    HandleError(cudaFree(tb8));
    HandleError(cudaFree(tb9));
    HandleError(cudaFree(tba));
    HandleError(cudaFree(tbb));
    HandleError(cudaFree(fro));
    HandleError(cudaFree(fru));
    HandleError(cudaFree(frv));
    HandleError(cudaFree(fre));
    HandleError(cudaFree(ftp));
    HandleError(cudaFree(prev_fro));
    HandleError(cudaFree(prev_fru));
    HandleError(cudaFree(prev_frv));
    HandleError(cudaFree(prev_fre));
    HandleError(cudaFree(prev_ftp));
    HandleError(cudaFree(gpu_averages));

    WriteToFile(nt, runtimes);

    return 0;
}

void InitialiseArrays(double* cylinderMask, double* uVelocity, double* vVelocity, double* temp,
    double* energy, double* rho, double* pressure, double* rou, double* rov, double* roe, double* scp)
{

    int idx;
    double dx, dy;
    const double radSquared = cylinderD * cylinderD / 4.;
    const double pressInf = rhoInf * tempInf * heatCapacityP * (heatRatio - 1.) / heatRatio;

    for (int j = 0; j < nx; j++) {
        for (int i = 0; i < ny; i++) {
            idx = j * ny + i;

            // Masks covers points inside circle
            dx = (double)i * deltaY - yLength / 2.;
            dy = (double)j * deltaX - xLength / 2.;
            if (dx * dx + dy * dy < radSquared) {
                cylinderMask[idx] = 1.;
            }

            uVelocity[idx] = uInf;
            // Add small velocity perturbation
            vVelocity[idx] = 0.01 * (sin(4. * PI * (double)i * deltaX / xLength)
                + sin(7. * PI * (double)i * deltaX / xLength))
                * exp(-((double)j * deltaY - yLength / 2.) * ((double)j * deltaY - yLength / 2.));
            temp[idx] = tempInf;
            pressure[idx] = pressInf;
            energy[idx] = heatCapacityV * tempInf + 0.5 * (uInf * uInf + vVelocity[idx] * vVelocity[idx]);
            rho[idx] = rhoInf;
            rou[idx] = rhoInf * uInf;
            rov[idx] = rhoInf * vVelocity[idx];
            roe[idx] = rhoInf * energy[idx];
            scp[idx] = 1.;
        }
    }
}

void AllocateGpuMemory(double* hostVariableList[], double** gpuVariableList[], const int length, const unsigned long int variableBytes) {

    // Allocate gpu memory and copy data from host arrays
    for (int i = 0; i < length; i++) {
        HandleError(cudaMalloc((void**)gpuVariableList[i], variableBytes));
        HandleError(cudaMemcpyAsync(*gpuVariableList[i], hostVariableList[i], variableBytes, cudaMemcpyHostToDevice));
    }

    // Allocate constant gpu memory
    HandleError(cudaMemcpyToSymbolAsync(deriv_consts, d_consts, 4 * sizeof(double), 0, cudaMemcpyHostToDevice));
    HandleError(cudaMemcpyToSymbolAsync(adams_consts, a_consts, 2 * sizeof(double), 0, cudaMemcpyHostToDevice));
    HandleError(cudaMemcpyToSymbolAsync(etatt_consts, e_consts, 2 * sizeof(double), 0, cudaMemcpyHostToDevice));
    HandleError(cudaMemcpyToSymbolAsync(dev_nx, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
    HandleError(cudaMemcpyToSymbolAsync(dev_ny, &ny, sizeof(int), 0, cudaMemcpyHostToDevice));
    HandleError(cudaMemcpyToSymbolAsync(pointsPerThread, &ptsPerThrd, sizeof(int), 0, cudaMemcpyHostToDevice));
}

void HandleError(cudaError error) {

    if (error != cudaSuccess) {
        printf("An error occured: %i: %s", error, cudaGetErrorString(error));
        printf("\nExiting...");
        exit(EXIT_FAILURE);
    }
}

void PrintAverages(const int timestep, const double* gpu_uVelocity, const double* gpu_vVelocity, const double* gpu_scp) {

    double mean[3];
    thrust::device_vector<double> arrayVect;
    const double* arrays[] = { gpu_uVelocity, gpu_vVelocity, gpu_scp };

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

void Fluxx(double* gpu_cylinderMask, double* gpu_uVelocity, double* gpu_vVelocity, double* gpu_temp, double* gpu_energy, 
    double* gpu_rho, double* gpu_pressure, double* gpu_rou, double* gpu_rov, double* gpu_roe, double* gpu_scp,
    double* tb1, double* tb2, double* tb3, double* tb4, double* tb5, double* tb6, double* tb7, double* tb8, double* tb9,
    double* tba, double* tbb, double* fro, double* fru, double* frv, double* fre, double* ftp) {

    //Derix<1> << < dim3(1, ny), 256 >> > (gpu_rou, tb1);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (gpu_rou, tb1);
    //Deriy<1> << < dim3(nx, 1), 256 >> > (gpu_rov, tb2);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (gpu_rov, tb2);

    SubFluxx1 << < ceil(nx * ny / 256) + 1, 256 >> > (gpu_uVelocity, gpu_vVelocity, gpu_rou, fro, tb1, tb2);

    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (gpu_pressure, tb3);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (tb1, tb4);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (tb2, tb5);
    Derix<2> << < dx_blockGrid, dx_threadGrid >> > (gpu_uVelocity, tb6);
    Deriy<2> << < dy_blockGrid, dy_threadGrid >> > (gpu_uVelocity, tb7);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (gpu_vVelocity, tb8);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (tb8, tb9);

    SubFluxx2 << < ceil(nx * ny / 256) + 1, 256 >> > (dynViscosity, oneOverEta, tb3, tb4, tb5, tb6, tb7, tb9,
        gpu_cylinderMask, gpu_uVelocity, gpu_vVelocity, gpu_rou, gpu_rov, tb1, tb2, tba, fru);

    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (gpu_pressure, tb3);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (tb1, tb4);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (tb2, tb5);
    Derix<2> << < dx_blockGrid, dx_threadGrid >> > (gpu_vVelocity, tb6);
    Deriy<2> << < dy_blockGrid, dy_threadGrid >> > (gpu_vVelocity, tb7);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (gpu_uVelocity, tb8);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (tb8, tb9);

    SubFluxx3 << < ceil(nx * ny / 256) + 1, 256 >> > (dynViscosity, oneOverEta, tb3, tb4, tb5, tb6, tb7, tb9,
        gpu_cylinderMask, gpu_vVelocity, tbb, frv);

    // Equation for the tempature
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (gpu_scp, tb1);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (gpu_scp, tb2);
    Derix<2> << < dx_blockGrid, dx_threadGrid >> > (gpu_scp, tb3);
    Deriy<2> << < dy_blockGrid, dy_threadGrid >> > (gpu_scp, tb4);

    SubFluxx4 << < ceil(nx * ny / 256) + 1, 256 >> > (oneOverEta, xkt, tb1, tb2, tb3, tb4,
        gpu_uVelocity, gpu_vVelocity, gpu_scp, gpu_cylinderMask, ftp);

    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (gpu_uVelocity, tb1);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (gpu_vVelocity, tb2);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (gpu_uVelocity, tb3);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (gpu_vVelocity, tb4);

    SubFluxx5 << < ceil(nx * ny / 256) + 1, 256 >> > (dynViscosity, gpu_uVelocity, gpu_vVelocity, tba, tbb, gpu_pressure, gpu_roe, tb1, tb2, tb3, tb4, fre);

    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (tb1, tb5);
    Derix<1> << < dx_blockGrid, dx_threadGrid >> > (tb2, tb6);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (tb3, tb7);
    Deriy<1> << < dy_blockGrid, dy_threadGrid >> > (tb4, tb8);
    Derix<2> << < dx_blockGrid, dx_threadGrid >> > (gpu_temp, tb9);
    Deriy<2> << < dy_blockGrid, dy_threadGrid >> > (gpu_temp, tba);

    SubFluxx6 << < ceil(nx * ny / 256) + 1, 256 >> > (lambda, tb5, tb6, tb7, tb8, tb9, tba, fre);
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

//template<int derivative>
//__global__ void Derix(const double* f, double* deriv_f) {
//
//    __shared__ double row_f[nx + 2];
//
//    int thrdsPerBlock = blockDim.x;
//    int global_tid, shrd_mem_idx;
//
//    // Copy row of f into shared memory
//    for (int i = threadIdx.x; i < nx; i += thrdsPerBlock) {
//        global_tid = ny * blockIdx.y + i;
//        shrd_mem_idx = i + 1;
//        row_f[shrd_mem_idx] = f[global_tid];
//    }
//
//    __syncthreads();
//
//    // Apply periodic boundary conditions
//    if (threadIdx.x == 0) {
//        row_f[0] = row_f[nx];
//        row_f[nx + 1] = row_f[1];
//    }
//
//    __syncthreads();
//
//    // Calculate derivative using finite difference stencil
//    for (int i = threadIdx.x; i < nx; i += thrdsPerBlock) {
//        global_tid = ny * blockIdx.y + i;
//        shrd_mem_idx = i + 1;
//
//        switch (derivative) {
//        case 1:
//            // Case of 1st x derivative
//            deriv_f[global_tid] = deriv_consts[0] * (row_f[shrd_mem_idx + 1] - row_f[shrd_mem_idx - 1]);
//            break;
//
//        case 2:
//            // Case of 2nd x derivative
//            deriv_f[global_tid] = deriv_consts[2] * (row_f[shrd_mem_idx + 1] - 2 * row_f[shrd_mem_idx] + row_f[shrd_mem_idx - 1]);
//            break;
//        }
//    }
//}

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

//template<int derivative>
//__global__ void Deriy(const double* f, double* deriv_f) {
//
//    __shared__ double col_f[ny + 2];
//
//    int thrdsPerBlock = blockDim.x;
//    int global_tid, shrd_mem_idx;
//
//    // Copy column of f into shared memory
//    for (int i = threadIdx.x; i < ny; i += thrdsPerBlock) {
//        global_tid = ny * i + blockIdx.x;
//        shrd_mem_idx = i + 1;
//        col_f[shrd_mem_idx] = f[global_tid];
//    }
//
//    __syncthreads();
//
//    // Apply periodic boundary conditions
//    if (threadIdx.x == 0) {
//        col_f[0] = col_f[ny];
//        col_f[ny + 1] = col_f[1];
//    }
//
//    __syncthreads();
//
//    // Calculate derivative using finite difference stencil
//    for (int i = threadIdx.x; i < ny; i += thrdsPerBlock) {
//        global_tid = ny * i + blockIdx.x;
//        shrd_mem_idx = i + 1;
//
//        switch (derivative) {
//        case 1:
//            // Case of 1st y derivative
//            deriv_f[global_tid] = deriv_consts[1] * (col_f[shrd_mem_idx + 1] - col_f[shrd_mem_idx - 1]);
//            break;
//
//        case 2:
//            // Case of 2nd y derivative
//            deriv_f[global_tid] = deriv_consts[3] * (col_f[shrd_mem_idx + 1] - 2 * col_f[shrd_mem_idx] + col_f[shrd_mem_idx - 1]);
//            break;
//        }
//    }
//}

__global__ void SubFluxx1(const double* uVelocity, const double* vVelocity, const double* rou,
    double* fro, double* tb1, double* tb2) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dev_nx * dev_ny) { return; }
    
    fro[idx] = -tb1[idx] - tb2[idx];

    double local_rou = rou[idx];
    tb1[idx] = local_rou * uVelocity[idx];
    tb2[idx] = local_rou * vVelocity[idx];
    
}

__global__ void SubFluxx2(const double dynViscosity, const double oneOverEta, const double* tb3, const double* tb4, const double* tb5, const double* tb6, const double* tb7, const double* tb9,
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

__global__ void SubFluxx3(const double dynViscosity, const double oneOverEta,
    const double* tb3, const double* tb4, const double* tb5, const double* tb6, const double* tb7, const double* tb9,
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

__global__ void SubFluxx4(const double oneOverEta, const double xkt, const double* tb1, const double* tb2, const double* tb3, const double* tb4,
    const double* uVelocity, const double* vVelocity, const double* scp, const double* cylinderMask, double* ftp) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= dev_nx * dev_ny) { return; }

    ftp[idx] = -uVelocity[idx] * tb1[idx] - vVelocity[idx] * tb2[idx]
        + xkt * (tb3[idx] + tb4[idx]) - oneOverEta * cylinderMask[idx] * scp[idx];
}


__global__ void SubFluxx5(const double dynViscosity, const double* uVelocity, const double* vVelocity,
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

__global__ void SubFluxx6(const double lambda, const double* tb5, const double* tb6, const double* tb7, const double* tb8,
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
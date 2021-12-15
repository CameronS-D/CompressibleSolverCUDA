#define PI 3.14159265359
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

const int nx = 1025, ny = 1025, nt = 100, ns = 3, nf = 3;
const int mx = nf * nx, my = nf * ny;

const double reynolds = 200., mach = 0.2, prandtl = 0.7;
const double rhoInf = 1., cInf = 1., cylinderD = 1., heatCapacityP = 1., gamma = 1.4;

const double heatCapacityV = heatCapacityP / gamma;
const double uInf = mach * cInf;
const double dynViscosity = rhoInf * uInf * cylinderD / reynolds;
// thermal conductivity
const double lambda = dynViscosity * heatCapacityP / prandtl;
const double tempInf = cInf * cInf / (heatCapacityP * (gamma - 1));
const double eta = 0.1 / 2;

const double xLength = 4. * cylinderD;
const double yLength = 4. * cylinderD;
const double deltaX = xLength / nx;
const double deltaY = yLength / ny;
const double CFL = 0.25;
const double deltaT = CFL * deltaX;

// Derivative stencil constants
const double d_consts[] = { 1 / (2 * deltaX) , 1 / (2 * deltaY), 1 / deltaX / deltaX, 1 / deltaY / deltaY };
__constant__ double deriv_consts[4];

void InitialiseArrays(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
void HandleError(cudaError);
void AllocateGpuMemory(double* [], double** [], const int);

template<int>
__global__ void Derix(const double*, double*);
template<int>
__global__ void Deriy(const double*, double*);
__global__ void Average(const double*, double*);

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

    double* gpu_cylinderMask, * gpu_uVelocity, * gpu_vVelocity, * gpu_temp, * gpu_energy, * gpu_rho, * gpu_pressure, * gpu_rou, * gpu_rov, * gpu_roe, * gpu_scp;

    double* hostVariables[numOfVariables] = { cylinderMask, uVelocity, vVelocity, temp, energy, rho, pressure, rou, rov, roe, scp };
    double** gpuVariables[numOfVariables] = { &gpu_cylinderMask, &gpu_uVelocity, &gpu_vVelocity, &gpu_temp, &gpu_energy, &gpu_rho, &gpu_pressure, &gpu_rou, &gpu_rov, &gpu_roe, &gpu_scp };

    InitialiseArrays(cylinderMask, uVelocity, vVelocity, temp, energy, rho, pressure, rou, rov, roe, scp);

    double* sol = new double[nx * ny];
    for (int j = 0; j < nx; j++) {
        for (int i = 0; i < ny; i++) {
            int idx = j * ny + i;
            uVelocity[idx] = sin(2 * i * deltaX + j * deltaY);
            sol[idx] = -1 * sin(2 * i * deltaX + j * deltaY);
        }
    }

    AllocateGpuMemory(hostVariables, gpuVariables, numOfVariables);

    //Derix <2> << < dim3(1, ny), 256 >> > (gpu_uVelocity, gpu_temp);
    //Deriy <2> << < dim3(nx, 1), 256 >> > (gpu_uVelocity, gpu_temp);
    //Average << < 1, 128 >> > (gpu_uVelocity, gpu_avg);
    
    //HandleError(cudaMemcpy(temp, gpu_temp, nx * ny * sizeof(double), cudaMemcpyDeviceToHost));

    /*for (int j = 0; j < nx; j++) {
        for (int i = 0; i < ny; i++) {
            int idx = j * ny + i;
            printf("Exact: %f \t Approx: %f \n", sol[idx], temp[idx]);

        }
    }*/

    // Free memory on both host and device
    for (int i = 0; i < numOfVariables; i++) {
        HandleError(cudaFree(*gpuVariables[i]) );
        delete[] hostVariables[i];
    }

    return 0;
}

void InitialiseArrays(double* cylinderMask, double* uVelocity, double* vVelocity, double* temp,
    double* energy, double* rho, double* pressure, double* rou, double* rov, double* roe, double* scp)
{

    int idx;
    double dx, dy, radSquared = cylinderD * cylinderD / 4;
    double pressInf = rhoInf * tempInf * heatCapacityP * (gamma - 1) / gamma;

    for (int j = 0; j < nx; j++) {
        for (int i = 0; i < ny; i++) {
            idx = j * ny + i;

            // Masks covers points inside circle
            dx = i * deltaX - xLength / 2;
            dy = j * deltaY - yLength / 2;
            if (dx * dx + dy * dy < radSquared) {
                cylinderMask[idx] = 1.;
            }

            uVelocity[idx] = uInf;
            // Add small velocity perturbation
            vVelocity[idx] = 0.01 * (sin(4 * PI * i * deltaX / xLength)
                + sin(7 * PI * i * deltaX / xLength))
                * exp(-(j * deltaY - yLength / 2) * (j * deltaY - yLength / 2));
            temp[idx] = tempInf;
            pressure[idx] = pressInf;
            energy[idx] = heatCapacityV * tempInf
                + 0.5 * (uInf * uInf + vVelocity[i * nx + j] * vVelocity[i * nx + j]);
            rho[idx] = rhoInf;
            rou[idx] = rhoInf * uInf;
            rov[idx] = rhoInf * vVelocity[idx];
            roe[idx] = rhoInf * energy[idx];
            scp[idx] = 1.;
        }
    }
}

void AllocateGpuMemory(double* hostVariableList[], double** gpuVariableList[], const int length) {

    // Allocate gpu memory and copy data from host arrays
    int bytes = nx * ny * sizeof(double);
    for (int i = 0; i < length; i++) {
        HandleError(cudaMalloc((void**)gpuVariableList[i], bytes));
        HandleError(cudaMemcpy(*gpuVariableList[i], hostVariableList[i], bytes, cudaMemcpyHostToDevice));
    }

    // Allocate constant gpu memory
    HandleError(cudaMemcpyToSymbol(deriv_consts, d_consts, 4 * sizeof(double)));
}

void HandleError(cudaError error) {

    if (error != cudaSuccess) {
        printf("An error occured: %i: %s", error, cudaGetErrorString(error));
        printf("\nExiting...");
        exit(EXIT_FAILURE);
    }
}

__global__ void Average(const double* array, double* output) {

    __shared__ double shrd_temp[nx];

    double col_sum = 0.0;

    // Store sum of each column in shared memory
    for (int thrdNum = threadIdx.x; thrdNum < nx; thrdNum += blockDim.x) {
        for (int i = ny * thrdNum; i < ny * (thrdNum + 1); i++) {
            col_sum += array[i];
        }

        shrd_temp[thrdNum] = col_sum;
        col_sum = 0.0;
    }

    __syncthreads();
    
    if (threadIdx.x == 0) {
        *output = 0.0;
        for (int i = 0; i < nx; i++) {
            *output += shrd_temp[i];
        }

        *output /= (nx * ny);
    }


}

template<int derivative>
__global__ void Derix(const double* f, double* deriv_f) {

    __shared__ double row_f[nx + 2];

    int thrdsPerBlock = blockDim.x;
    int global_tid, shrd_mem_idx;

    // Copy row of f into shared memory
    for (int i = threadIdx.x; i < nx; i += thrdsPerBlock) {
        global_tid = ny * blockIdx.y + i;
        shrd_mem_idx = i + 1;
        row_f[shrd_mem_idx] = f[global_tid];
    }

    __syncthreads();

    // Apply periodic boundary conditions
    if (threadIdx.x == 0) {
        row_f[0] = row_f[nx];
        row_f[nx + 1] = row_f[1];
    }

    __syncthreads();

    // Calculate derivative using finite difference stencil
    for (int i = threadIdx.x; i < nx; i += thrdsPerBlock) {
        global_tid = ny * blockIdx.y + i;
        shrd_mem_idx = i + 1;

        switch (derivative) {
        case 1:
            // Case of 1st x derivative
            deriv_f[global_tid] = deriv_consts[0] * (row_f[shrd_mem_idx + 1] - row_f[shrd_mem_idx - 1]);
            break;

        case 2:
            // Case of 2nd x derivative
            deriv_f[global_tid] = deriv_consts[2] * (row_f[shrd_mem_idx + 1] - 2 * row_f[shrd_mem_idx] + row_f[shrd_mem_idx - 1]);
            break;
        }
    }
}

template<int derivative>
__global__ void Deriy(const double* f, double* deriv_f) {

    __shared__ double col_f[ny + 2];

    int thrdsPerBlock = blockDim.x;
    int global_tid, shrd_mem_idx;

    // Copy column of f into shared memory
    for (int i = threadIdx.x; i < ny; i += thrdsPerBlock) {
        global_tid = ny * i + blockIdx.x;
        shrd_mem_idx = i + 1;
        col_f[shrd_mem_idx] = f[global_tid];
    }

    __syncthreads();

    // Apply periodic boundary conditions
    if (threadIdx.x == 0) {
        col_f[0] = col_f[ny];
        col_f[ny + 1] = col_f[1];
    }

    __syncthreads();

    // Calculate derivative using finite difference stencil
    for (int i = threadIdx.x; i < ny; i += thrdsPerBlock) {
        global_tid = ny * i + blockIdx.x;
        shrd_mem_idx = i + 1;

        switch (derivative) {
        case 1:
            // Case of 1st y derivative
            deriv_f[global_tid] = deriv_consts[1] * (col_f[shrd_mem_idx + 1] - col_f[shrd_mem_idx - 1]);
            break;

        case 2:
            // Case of 2nd y derivative
            deriv_f[global_tid] = deriv_consts[3] * (col_f[shrd_mem_idx + 1] - 2 * col_f[shrd_mem_idx] + col_f[shrd_mem_idx - 1]);
            break;
        }
    }
}

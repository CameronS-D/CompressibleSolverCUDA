#define PI 3.14159265359
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

const int nx = 65, ny = 65, nt = 100, ns = 3, nf = 3;
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

const double dx_const = 1 / (2 * deltaX);
__constant__ double derix_const;

void InitialiseArrays(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);

void HandleError(cudaError);

__global__ void Derix(double*, double*);

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

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            uVelocity[i * ny + j] = sin(2 * i * deltaX + j * deltaY);
            sol[i * ny + j] = 2 * cos(2 * i * deltaX + j * deltaY);
        }
    }


    // Allocate gpu memory and copy data from host arrays
    int bytes = nx * ny * sizeof(double);
    for (int i = 0; i < numOfVariables; i++) {
        HandleError(cudaMalloc((void**)gpuVariables[i], bytes) );
        HandleError(cudaMemcpy(*gpuVariables[i], hostVariables[i], bytes, cudaMemcpyHostToDevice) );
    }

    HandleError(cudaMemcpyToSymbol(derix_const, &dx_const, sizeof(double)));

    dim3 grid(1, ny);

    Derix << < grid, 256 >> > (gpu_uVelocity, gpu_temp);

    HandleError(cudaMemcpy(temp, gpu_temp, bytes, cudaMemcpyDeviceToHost));


    double error = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double s = sol[i * ny + j];
            double f = temp[i * ny + j];
            printf("%f \t %f \n", s, f);
            error += (s - f) * (s - f);
        }
        break;
    }

    error = sqrt(error / nx / ny);

    printf("RMS error: %e\n", error);

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

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            idx = i * ny + j;

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

void HandleError(cudaError error) {

    if (error != cudaSuccess) {
        printf("An error occured: %i: %s", error, cudaGetErrorString(error));
        printf("\nExiting...");
        exit(EXIT_FAILURE);
    }
}

__global__ void Derix(double* f, double* deriv_f) {

    __shared__ double row_f[nx + 2];

    int thrdsPerBlock = blockDim.x;
    int global_tid = ny * threadIdx.x + blockIdx.y;
    int local_tid = threadIdx.x + 1;

    // Copy row of f into shared memory
    int offset = 0;
    while (local_tid + offset - 1 < nx) {
        row_f[local_tid + offset] = f[global_tid + offset * ny];
        offset += thrdsPerBlock;
    }

    __syncthreads();

    // Apply periodic boundary conditions
    if (local_tid == 1) {
        row_f[0] = row_f[nx];
        row_f[nx + 1] = row_f[1];
    }

    __syncthreads();

    offset = 0;
    while (local_tid + offset - 1 < nx) {
        deriv_f[global_tid + offset * ny] = derix_const * (row_f[local_tid + offset + 1] - row_f[local_tid + offset - 1]);
        offset += thrdsPerBlock;
    }
}

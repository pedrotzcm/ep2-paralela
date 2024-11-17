
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define WALL_TEMP 20.0
#define FIREPLACE_TEMP 100.0

#define FIREPLACE_START 3
#define FIREPLACE_END 7
#define ROOM_SIZE 10

float calcula_segundos(float ms) {
    return ms / 1000.0f;
}

void initialize(double *h, int n) {
    int fireplace_start = (FIREPLACE_START * n) / ROOM_SIZE;
    int fireplace_end = (FIREPLACE_END * n) / ROOM_SIZE;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
                h[i * n + j] = (i == n - 1 && j >= fireplace_start && j <= fireplace_end) ? FIREPLACE_TEMP : WALL_TEMP;
            } else {
                h[i * n + j] = 0.0;
            }
        }
    }
}



__global__ void jacobi_iteration(double *d_h, double *d_g, int n) { //função kernel do CUDA
    int i = blockIdx.y * blockDim.y + threadIdx.y; //declaração de variáveis para os índices i e j
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // evitar cálculo nas bordas
    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
    d_g[i * n + j] = 0.25 * (d_h[(i - 1) * n + j] + d_h[(i + 1) * n + j] +
                             d_h[i * n + (j - 1)] + d_h[i * n + (j + 1)]);
    } 
    
    else {
        d_g[i * n + j] = d_h[i * n + j]; // tive que adicionar esse else porque estava tendo problemas com as borrdas
    }
    }

void jacobi_iteration_cpu(double *h, double *g, int n, int iter_limit) {
    for (int iter = 0; iter < iter_limit; iter++) {
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                g[i * n + j] = 0.25 * (h[(i - 1) * n + j] + h[(i + 1) * n + j] + 
                                       h[i * n + (j - 1)] + h[i * n + (j + 1)]);
            }
        }

        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                h[i * n + j] = g[i * n + j];
            }
        }
    }
}


double compara_matriz(double *mat_cpu, double *mat_gpu, int n) {
    double error = 0.0;

    // Comparar elemento por elemento
    for (int i = 0; i < n * n; i++) {
        error = error +  fabs(mat_cpu[i] - mat_gpu[i])/(n*n);  // erro absoluto

    }

    return error;
}

double calculate_elapsed_time(struct timespec start, struct timespec end)
{
    double start_sec = (double)start.tv_sec * 1e9 + (double)start.tv_nsec;
    double end_sec = (double)end.tv_sec * 1e9 + (double)end .tv_nsec;
    return (end_sec - start_sec) / 1e9;
}

void save_to_file(double *h, int n) {
    FILE *file = fopen("roomcuda.txt", "w");
    

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(file, "%lf ", h[i * n + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void save_to_file_CPU(double *h, int n) {
    FILE *file = fopen("roomcuda_CPU.txt", "w");


    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(file, "%lf ", h[i * n + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int iter_limit = atoi(argv[2]);
    int block_size_x = atoi(argv[3]);  //numero de threads nas respectivas dimensões
    int block_size_y = atoi(argv[4]);

    double *h_h = (double *)malloc(n * n * sizeof(double));
    double *h_g = (double *)malloc(n * n * sizeof(double));

    // Allocate device memory
    double *d_h, *d_g;
    cudaMalloc(&d_h, n * n * sizeof(double));
    cudaMalloc(&d_g, n * n * sizeof(double));

    // inicializa as matrizes no host e copia para o device
    initialize(h_h, n);
    cudaMemcpy(d_h, h_h, n * n * sizeof(double), cudaMemcpyHostToDevice);

    // declaração de eventos CUDA
    cudaEvent_t startEvent, stopEvent, startMemcpyEvent, stopMemcpyEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&startMemcpyEvent);
    cudaEventCreate(&stopMemcpyEvent);

    // código só no host
    struct timespec start_cpu, end_cpu;
    clock_gettime(CLOCK_MONOTONIC, &start_cpu);
    jacobi_iteration_cpu(h_h, h_g, n, iter_limit);
    clock_gettime(CLOCK_MONOTONIC, &end_cpu);
    printf("CPU Elapsed time: %.6f seconds\n", calculate_elapsed_time(start_cpu, end_cpu));
    save_to_file_CPU(h_h, n);  // Save the CPU result

    // Compare CPU and GPU results
    // cudaMemcpy(h_h, d_h, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    // double error = compara_matriz(h_g, h_h, n);
    // printf("Error between CPU and GPU: %.6f\n", error);

    // configura o kernel
    dim3 threadsPerBlock(block_size_x, block_size_y); //definição do número de threads
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Measure data transfer (Host -> Device)
    cudaEventRecord(startMemcpyEvent, 0);
    cudaMemcpy(d_h, h_h, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stopMemcpyEvent, 0);
    cudaEventSynchronize(stopMemcpyEvent);
    float memcpyTime;
    cudaEventElapsedTime(&memcpyTime, startMemcpyEvent, stopMemcpyEvent);
    printf(" Memcpy host ->para o device: %.6f segundos\n", calcula_segundos(memcpyTime));

    // executa o kernel e mede o tempo
    cudaEventRecord(startEvent, 0);
    for (int iter = 0; iter < iter_limit; iter++) {
        jacobi_iteration<<<numBlocks, threadsPerBlock>>>(d_h, d_g, n);
        cudaDeviceSynchronize();
        double *temp = d_h;
        d_h = d_g;
        d_g = temp;
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, startEvent, stopEvent);
    printf("execução do kernel: %.6f segundos\n", calcula_segundos(kernelTime));

    // Measure data transfer (Device -> Host)
    cudaEventRecord(startMemcpyEvent, 0);
    cudaMemcpy(h_h, d_h, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stopMemcpyEvent, 0);
    cudaEventSynchronize(stopMemcpyEvent);
    cudaEventElapsedTime(&memcpyTime, startMemcpyEvent, stopMemcpyEvent);
    printf("Memcpy device para o host: %.6f segundos\n", calcula_segundos(memcpyTime));
    save_to_file(h_h, n);

    double erro = compara_matriz(h_g, h_h, n);  // comparando as matrizes
    printf("erro absoulto médio por casa da matriz: %.6f\n", erro);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startMemcpyEvent);
    cudaEventDestroy(stopMemcpyEvent);

    free(h_h);
    free(h_g);
    cudaFree(d_h);
    cudaFree(d_g);

    return 0;
}

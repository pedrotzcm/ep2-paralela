
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

void initialize(double *h, int n)
{
    int fireplace_start = (FIREPLACE_START * n) / ROOM_SIZE;
    int fireplace_end = (FIREPLACE_END * n) / ROOM_SIZE;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == 0 || i == n - 1 || j == 0 || j == n - 1)
            {
                h[i * n + j] = (i == n - 1 && j >= fireplace_start && j <= fireplace_end) ? FIREPLACE_TEMP : WALL_TEMP;
            }
            else
            {
                h[i * n + j] = 0.0;
            }
        }
    }
}

__global__ void jacobi_iteration_cuda(double *d_h, double *d_g, int n)
{
    extern __shared__ double shared_mem[];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int shared_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Carregar dados da matriz global para memória compartilhada
    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        shared_mem[shared_idx] = d_h[i * n + j];
    }

    __syncthreads();  // Sincronização dos threads no bloco

    if (i > 0 && i < n - 1 && j > 0 && j < n - 1)
    {
        double left = (i > 0) ? shared_mem[shared_idx - blockDim.x] : 0.0;
        double right = (i < n - 2) ? shared_mem[shared_idx + blockDim.x] : 0.0;
        double top = (j > 0) ? shared_mem[shared_idx - 1] : 0.0;
        double bottom = (j < n - 2) ? shared_mem[shared_idx + 1] : 0.0;

        d_g[i * n + j] = 0.25 * (left + right + top + bottom);
    }
    __syncthreads();
}

double calculate_elapsed_time(struct timespec start, struct timespec end)
{
    double start_sec = (double)start.tv_sec * 1e9 + (double)start.tv_nsec;
    double end_sec = (double)end.tv_sec * 1e9 + (double)end.tv_nsec;
    return (end_sec - start_sec) / 1e9;
}

void save_to_file(double *h, int n)
{
    FILE *file = fopen("roomcuda.txt", "w");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            fprintf(file, "%lf ", h[i * n + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Uso: %s <número de pontos> <limite de iterações> <tamanho do bloco X> <tamanho do bloco Y>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int iter_limit = atoi(argv[2]);
    int block_size_x = atoi(argv[3]);  // Tamanho do bloco na direção X
    int block_size_y = atoi(argv[4]);  // Tamanho do bloco na direção Y

    double *h_h = (double *)malloc(n * n * sizeof(double));
    double *h_g = (double *)malloc(n * n * sizeof(double));
    double *d_h, *d_g;

    if (h_h == NULL || h_g == NULL) {
        fprintf(stderr, "Erro ao alocar memória no host\n");
        exit(EXIT_FAILURE);
    }

    // Alocação de memória na GPU
    cudaError_t err;
    err = cudaMalloc((void **)&d_h, n * n * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro ao alocar memória na GPU para d_h: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_g, n * n * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro ao alocar memória na GPU para d_g: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Inicializando o vetor h_h no host
    initialize(h_h, n);

    // Copiando a matriz de h_h (host) para d_h (device)
    err = cudaMemcpy(d_h, h_h, n * n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro ao copiar dados de h_h para d_h: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Configurando o número de threads por bloco e o número de blocos na grade
    dim3 threadsPerBlock(block_size_x, block_size_y);  // Definindo threads por bloco
    dim3 numBlocks((n + block_size_x - 1) / block_size_x, (n + block_size_y - 1) / block_size_y);  // Calculando número de blocos

    // Medindo o tempo de execução com clock_gettime
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int iter = 0; iter < iter_limit; iter++) {
        // Lançamento do kernel
        jacobi_iteration_cuda<<<numBlocks, threadsPerBlock>>>(d_h, d_g, n);
        
        // Verificando por erros no lançamento do kernel
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Erro no kernel: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Sincronização da GPU
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Erro na sincronização da GPU: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Troca dos ponteiros d_h e d_g
        double *temp = d_h;
        d_h = d_g;
        d_g = temp;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculando o tempo de execução
    double elapsed_time = calculate_elapsed_time(start, end);
    printf("Tempo de execução GPU: %.9f segundos\n", elapsed_time);

    // Copiando o resultado de volta para a memória do host
    err = cudaMemcpy(h_h, d_h, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro ao copiar dados de d_h para h_h: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Salvando o resultado
    save_to_file(h_h, n);

    // Liberando a memória
    free(h_h);
    free(h_g);
    cudaFree(d_h);
    cudaFree(d_g);

    return 0;
}
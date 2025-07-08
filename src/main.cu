#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addArrays(int* a, int* b, int* result, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 10;
    int h_a[N], h_b[N], h_result[N];

    // Заполняем массивы
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Выделяем память на GPU
    int *d_a, *d_b, *d_result;
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_result, N * sizeof(int));

    // Копируем данные на GPU
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Запускаем ядро
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, N);

    // Копируем результат обратно в host-память
    cudaMemcpy(h_result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Выводим результат
    printf("Result of a + b:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_result[i]);
    }

    // Освобождаем память
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}

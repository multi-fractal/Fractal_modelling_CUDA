#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <curand_kernel.h>

#define MAX_N 3

// GPU структура узла дерева
struct NodeGPU {
    int depth;
    int parent_id;
    int value;
};

// Проверка неотрицательности вектора
bool allNonNegative(const std::vector<double>& v) {
    for (double x : v) if (x < 0) return false;
    return true;
}

// Моделирование вероятностей (CPU)
std::vector<double> Probabilities_1d(float D2) {
    std::vector<double> P(2);
    double R = pow(0.5, D2);
    P[0] = 0.5 * (1 + sqrt(1 - 2 * R * R));
    P[1] = 1.0 - P[0];
    return P;
}

std::vector<double> Probabilities_2d(float D2) {
    std::vector<double> P(4), F(4);
    double sum, RR = pow(0.5, D2), r = sqrt(RR - 0.25);
    bool flag = true;
    while (flag) {
        sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            F[i] = (double)rand() / RAND_MAX;
            sum += F[i];
        }
        for (int i = 0; i < 4; ++i) F[i] /= sum;

        sum = 0.0;
        for (int i = 0; i < 4; ++i) sum += pow(F[i] - 0.25, 2);
        sum = sqrt(sum);
        for (int i = 0; i < 4; ++i)
            P[i] = 0.25 + (F[i] - 0.25) * r / sum;

        if (allNonNegative(P)) flag = false;
    }
    return P;
}

std::vector<double> Probabilities_3d(float D2) {
    std::vector<double> P(8), F(8);
    double sum, RR = pow(0.5, D2), r = sqrt(RR - 0.125);
    bool flag = true;
    while (flag) {
        sum = 0.0;
        for (int i = 0; i < 8; ++i) {
            F[i] = (double)rand() / RAND_MAX;
            sum += F[i];
        }
        for (int i = 0; i < 8; ++i) F[i] /= sum;

        sum = 0.0;
        for (int i = 0; i < 8; ++i) sum += pow(F[i] - 0.125, 2);
        sum = sqrt(sum);
        for (int i = 0; i < 8; ++i)
            P[i] = 0.125 + (F[i] - 0.125) * r / sum;

        if (allNonNegative(P)) flag = false;
    }
    return P;
}

// CUDA: построение дерева
__global__ void build_tree_kernel(NodeGPU* nodes, int num_children, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_nodes = (int)((powf(num_children, depth + 1) - 1) / (num_children - 1));
    if (idx >= total_nodes) return;

    int level = 0, temp = idx, current_level_nodes = 1;
    while (temp >= current_level_nodes) {
        temp -= current_level_nodes;
        current_level_nodes *= num_children;
        level++;
    }

    int parent_id = (idx - 1) / num_children;
    nodes[idx].depth = level;
    nodes[idx].parent_id = parent_id;
    nodes[idx].value = idx % num_children;
}

// CUDA: выбор потомка по вероятностям
__device__ int sample_discrete(const float* probs, int count, curandState* state) {
    float r = curand_uniform(state);
    float cum = 0.0f;
    for (int i = 0; i < count; ++i) {
        cum += probs[i];
        if (r <= cum) return i;
    }
    return count - 1;
}

// CUDA: блуждание по дереву
__global__ void random_walk_kernel(NodeGPU* nodes, int total_nodes, int num_children, int depth,
                                   float* output, int n, int m, const float* d_P, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m) return;

    curandState state;
    curand_init(seed, tid, 0, &state);

    int current_id = 0;
    float X1[MAX_N] = {0.0f}, X2[MAX_N] = {1.0f, 1.0f, 1.0f};

    for (int level = 0; level < depth; ++level) {
        int idx = sample_discrete(d_P, num_children, &state);
        int child_id = current_id * num_children + 1 + idx;
        if (child_id >= total_nodes) break;
        current_id = child_id;

        for (int d = 0; d < n; ++d) {
            int bit = (nodes[current_id].value >> d) & 1;
            float mid = 0.5f * (X1[d] + X2[d]);
            if (bit == 0) X2[d] = mid;
            else X1[d] = mid;
        }
    }

    for (int d = 0; d < n; ++d) {
        float u = curand_uniform(&state);
        output[tid * n + d] = X1[d] + u * (X2[d] - X1[d]);
    }
}

int main() {
    srand(time(NULL));

    int K = 4;         // глубина дерева
    int n = 2;         // размерность
    int num_children = 1 << n;
    int m = 10000;     // количество блужданий
    float D2 = 1.75;

    std::vector<double> P;
    if (n == 1) P = Probabilities_1d(D2);
    else if (n == 2) P = Probabilities_2d(D2);
    else if (n == 3) P = Probabilities_3d(D2);
    else {
        std::cerr << "n must be 1, 2 or 3\n";
        return 1;
    }

    std::vector<float> P_float(P.begin(), P.end());
cout<<n<<" "<<P[0]<<" "<<P[1]<<P[2]<<" "<<P[3]<<'\n';
    int total_nodes = (int)((pow(num_children, K + 1) - 1) / (num_children - 1));

    NodeGPU* d_nodes;
    cudaMalloc(&d_nodes, total_nodes * sizeof(NodeGPU));
    build_tree_kernel<<<(total_nodes + 255) / 256, 256>>>(d_nodes, num_children, K);
    cudaDeviceSynchronize();

    float* d_output;
    cudaMalloc(&d_output, m * n * sizeof(float));

    float* d_P;
    cudaMalloc(&d_P, num_children * sizeof(float));
    cudaMemcpy(d_P, P_float.data(), num_children * sizeof(float), cudaMemcpyHostToDevice);

    random_walk_kernel<<<(m + 255) / 256, 256>>>(d_nodes, total_nodes, num_children, K,
                                                 d_output, n, m, d_P, time(NULL));
    cudaDeviceSynchronize();

    std::vector<float> h_output(m * n);
    cudaMemcpy(h_output.data(), d_output, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream fout("gpu_walk_with_probs.csv");
    fout << "walk_id";
    for (int i = 0; i < n; ++i) fout << ", x" << i;
    fout << "\n";
    for (int i = 0; i < m; ++i) {
        fout << i;
        for (int j = 0; j < n; ++j)
            fout << ", " << h_output[i * n + j];
        fout << "\n";
    }
    fout.close();
    std::cout << "✅ Результаты сохранены в gpu_walk_with_probs.csv\n";

    cudaFree(d_nodes);
    cudaFree(d_output);
    cudaFree(d_P);

    return 0;
}


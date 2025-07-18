#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <curand_kernel.h>

#define MAX_N 3

// GPU Tree node structure
struct NodeGPU {
    int depth;
    int parent_id;
    int value;
};

// Checking vector non-negativity
bool allNonNegative(const std::vector<double>& v) {
    for (double x : v) if (x < 0) return false;
    return true;
}

// Probability modeling (CPU)
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

// CUDA: building a tree
__global__ void build_tree_kernel(NodeGPU* nodes, int num_children, int depth, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_nodes = (int)((powf(num_children, depth + 1) - 1) / (num_children - 1));
    if (idx >= total_nodes) return;

    // Computing the node's level (depth)
    int level = 0, temp = idx, current_level_nodes = 1;
    while (temp >= current_level_nodes) {
        temp -= current_level_nodes;
        current_level_nodes *= num_children;
        level++;
    }

    // Computing the parent
    int parent_id = (idx - 1) / num_children;

    nodes[idx].depth = level;
    nodes[idx].parent_id = parent_id;

    if (idx == 0) {
        nodes[idx].value = 0;  // root
        return;
    }

    // Initialization of the random number generator
    curandState state;
    curand_init(seed + parent_id, 0, 0, &state); // one seed per parent

    // random permutation for the children of a given parent
    const int MAX_CHILDREN = 8;  // should be >= num_children
    int perm[MAX_CHILDREN];
    for (int i = 0; i < num_children; ++i) perm[i] = i;

    // Shuffling (Fisher-Yates)
    for (int i = num_children - 1; i > 0; --i) {
        int j = curand(&state) % (i + 1);
        int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }

    // Determining the node's position among its parent's children
    int child_local_index = (idx - 1) % num_children;
    nodes[idx].value = perm[child_local_index];
}

// CUDA: choosing a child node according to probabilities
__device__ int sample_discrete(const float* probs, int count, curandState* state) {
    float r = curand_uniform(state);
    float cum = 0.0f;
    for (int i = 0; i < count; ++i) {
        cum += probs[i];
        if (r <= cum) return i;
    }
    return count - 1;
}

// CUDA: one random walk on the tree
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
    cudaSetDevice(0);
    srand(time(NULL));

    int K = 7;         // tree depth
    int n = 2;         // dimensionality
    int num_children = 1 << n;
    int m = 1000;     // number of walks
    float D2 = 1.75;

    int total_nodes = (int)((pow(num_children, K + 1) - 1) / (num_children - 1));

    int threads = 256;
    int blocks = (total_nodes + threads - 1) / threads;

    std::vector<double> P;
    if (n == 1) P = Probabilities_1d(D2);
    else if (n == 2) P = Probabilities_2d(D2);
    else if (n == 3) P = Probabilities_3d(D2);
    else {
        std::cerr << "n must be 1, 2 or 3\n";
        return 1;
    }

    std::vector<float> P_float(P.begin(), P.end());

    NodeGPU* d_nodes;
    cudaMalloc(&d_nodes, total_nodes * sizeof(NodeGPU));

    unsigned int seed = static_cast<unsigned int>(time(NULL));
    build_tree_kernel<<<blocks, threads>>>(d_nodes, num_children, K,seed);
    cudaDeviceSynchronize();

    float* d_output;
    cudaMalloc(&d_output, m * n * sizeof(float));

    float* d_P;
    cudaMalloc(&d_P, num_children * sizeof(float));
    cudaMemcpy(d_P, P_float.data(), num_children * sizeof(float), cudaMemcpyHostToDevice);

    random_walk_kernel<<<(m + threads-1) / threads, threads>>>(d_nodes, total_nodes, num_children, K,
                                                 d_output, n, m, d_P, time(NULL));
    cudaDeviceSynchronize();

    std::vector<float> h_output(m * n);
    cudaMemcpy(h_output.data(), d_output, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream fout("output.csv");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j){
            fout << h_output[i * n + j];
            if(j<n-1) fout << ", " ;
        }
        fout << "\n";
    }
    fout.close();
    std::cout << "Results saved in output.csv\n";

    cudaFree(d_nodes);
    cudaFree(d_output);
    cudaFree(d_P);

    return 0;
}

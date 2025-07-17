# Fractal_modelling_CUDA

Solving the problem of modeling point fractal sets with a specified correlation fractal dimension. Refactoring the code developed at https://github.com/multi-fractal/Fractal_modelling for GPU computation using CUDA.


**Compilation and Execution**

nvcc -o main main.cu

./main  


**Compilation and Execution on Google Colab**

!nvcc main.cu -o main -arch=sm_75

!./main


**References**

Darcel, C., Bour, O., Davy, P., de Dreuzy, J., 2003. Connectivity properties of two-dimensional fracture network with stochastic fractal correlation.Water. Resour. Res. 39, 1272.

Kolyukhin D. (2020). Statistical modeling of three-dimensional fractal point sets with a given spatial probability distribution. Monte Carlo Methods and Applications. V. 26, pp. 245-252.

Kolyukhin D. (2021). Study the accuracy of the correlation fractal dimension estimation. Communication in Statistics. Simulation and Computation, 53(1), 219–233. doi.org/10.1080/03610918.2021.2014888

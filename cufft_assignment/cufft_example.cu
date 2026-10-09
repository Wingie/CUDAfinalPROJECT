#include "cufft_example.h"
#include <iostream>
#include <iomanip>
#include <tuple>
#include <string>
#include <cstdlib>
#include <cstdio>

using namespace std;

// Based on example found at http://techqa.info/programming/question/36889333/cuda-cufft-2d-example

// Every CUDA / cuFFT call is checked: a failure prints where it happened and exits
// immediately instead of carrying on with invalid memory or hanging.
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                           \
                    cudaGetErrorString(err__), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define CUFFT_CHECK(call)                                                         \
    do {                                                                          \
        cufftResult res__ = (call);                                               \
        if (res__ != CUFFT_SUCCESS) {                                             \
            fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)res__,              \
                    __FILE__, __LINE__);                                          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

__device__ Complex complexScaleMult(Complex a, Complex b, int scalar)
{
    // Create a variable of type Complex named c
    Complex c;

    // Calculate the x value for c by scalar * (a.x * b.x)
    c.x = scalar * (a.x * b.x);

    // Calculate the y value for c by scalar * (a.y * b.y)
    c.y = scalar * (a.y * b.y);

    return c;
}

__global__ void complexProcess(Complex *a, Complex *b, Complex *c, int size, int scalar)
{
    // calculate threadId variable
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // process complexScalarMult on values in a and b at index threadID and the passed scalar, place the result in c[threadId]
    if (threadId < size) {
        c[threadId] = complexScaleMult(a[threadId], b[threadId], scalar);
    }
}

__host__ std::tuple<int, int> parseCommandLineArguments(int argc, char** argv)
{
    // parse command line input for argument -n and place in variable N
    // Accepts both "-n=16" and "-n 16". Defaults to 16 (a 16 x 16 matrix).
    int N = 16;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        try {
            if (arg.rfind("-n=", 0) == 0) {
                N = stoi(arg.substr(3));
            } else if (arg == "-n" && i + 1 < argc) {
                N = stoi(argv[i + 1]);
                i++;
            }
        } catch (...) {
            fprintf(stderr, "Invalid value for -n, using default 16\n");
            N = 16;
        }
    }
    if (N < 1) {
        N = 16;
    }

    // Set variable SIZE equal to N squared
    int SIZE = N * N;

    return {N, SIZE};
}

__host__ Complex *generateComplexPointer(int SIZE)
{
    // Allocate an array of SIZE Complex values
    Complex *complex = new Complex[SIZE];

    // populate properties x and y of variable complex at index i to 2 and 3 respectively
    for (int i = 0; i < SIZE; i++) {
        complex[i].x = 2.0f;
        complex[i].y = 3.0f;
    }

    return complex;
}

__host__ void printComplexPointer(Complex *complex, int N)
{
    // '\n' instead of endl inside the loop: endl flushes on every row, which is slow for large N
    for (int i = 0; i < N * N; i = i + N)
    {
        for (int j = 0; j < N; j++) {
            cout << complex[i + j].x << " ";
        }
        cout << '\n';
    }
    cout << "----------------" << endl;
}

__host__ cufftComplex *generateCuFFTComplexPointerFromHostComplex(size_t mem_size, Complex *hostComplex)
{
    cufftComplex *d_complex;

    // Allocate device memory of mem_size bytes
    CUDA_CHECK(cudaMalloc((void**)&d_complex, mem_size));

    // Copy the host Complex data into the device cufftComplex memory
    CUDA_CHECK(cudaMemcpy(d_complex, hostComplex, mem_size, cudaMemcpyHostToDevice));

    return d_complex;
}

__host__ cufftHandle transformFromTimeToSignalDomain(int N, cufftComplex *d_a, cufftComplex *d_b, cufftComplex *d_c)
{
    // create a cufftHandle of size N*N and from Complex input to Complex output (2D transform)
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, N, N, CUFFT_C2C));

    // execute Complex 2 Complex Forward Transformation based on the cufftHandle for d_a, d_b, d_c
    // No newline here: the expected output has this text and the next message on one line
    printf("Performing Forward Transformation of a, b, and c ");
    CUFFT_CHECK(cufftExecC2C(plan, d_a, d_a, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan, d_b, d_b, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan, d_c, d_c, CUFFT_FORWARD));

    // return cufftHandle for later use
    return plan;
}

__host__ Complex *transformFromSignalToTimeDomain(cufftHandle plan, int SIZE, cufftComplex *d_c)
{
    // Initialize a Complex pointer with name results of size SIZE
    Complex *results = new Complex[SIZE];

    // Perform Complex to Complex INVERSE transformation of cufftComplex using the passed in plan and d_c
    printf("Transforming signal back cufftExecC2C\n");
    CUFFT_CHECK(cufftExecC2C(plan, d_c, d_c, CUFFT_INVERSE));

    // Perform memory copy from d_c into Complex variable results
    CUDA_CHECK(cudaMemcpy(results, d_c, sizeof(Complex) * (size_t)SIZE, cudaMemcpyDeviceToHost));

    return results;
}

int main(int argc, char** argv)
{
    auto [N, SIZE] = parseCommandLineArguments(argc, argv);

    // Print matrix values as fixed-point with one decimal, e.g. 2.0, matching the expected output format
    cout << fixed << setprecision(1);

    Complex *a = generateComplexPointer(SIZE);
    Complex *b = generateComplexPointer(SIZE);
    Complex *c = generateComplexPointer(SIZE);

    cout << "Input random data a:" << endl;
    printComplexPointer(a, N);
    cout << "Input random data b:" << endl;
    printComplexPointer(b, N);

    size_t mem_size = sizeof(Complex) * (size_t)SIZE;

    cufftComplex *d_a = generateCuFFTComplexPointerFromHostComplex(mem_size, a);
    cufftComplex *d_b = generateCuFFTComplexPointerFromHostComplex(mem_size, b);
    cufftComplex *d_c = generateCuFFTComplexPointerFromHostComplex(mem_size, c);

    cufftHandle plan = transformFromTimeToSignalDomain(N, d_a, d_b, d_c);

    printf("Launching Complex Division and Subtraction\n");
    int scalar = (rand() % 5) + 1;
    cout << "Scalar value: " << scalar << endl;

    // One thread per element with 256 threads per block. This works for any N,
    // unlike <<<N, N>>>, which cannot launch once N > 1024 (max threads per block).
    int threadsPerBlock = 256;
    int blocks = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    complexProcess<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, SIZE, scalar);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel is finished

    Complex *results = transformFromSignalToTimeDomain(plan, SIZE, d_c);
    cout << "Output data c: " << endl;
    printComplexPointer(results, N);

    // Free host memory arrays
    delete[] results;
    delete[] a;
    delete[] b;
    delete[] c;

    // Free cuFFT plan and device memory
    cufftDestroy(plan);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

#include <stdio.h>
#include <stdlib.h> 
#include <string.h> 
#include <math.h>
#include <CL/cl.h>
#include <time.h>
#define VECTOR_SIZE 100
#define SIZE 10
#define RAND
#define MAX_SOURCE_SIZE 4096
float det(int* A, int size);
// 10000 3.17
// 6400 1.66
// 2500 0.834
// 1600 0.785
// 1225 0.965
// 900 7.135
// 400 1.371
// 100 0.626
// 64 0.612
// 25 0.568
// 16 0.54
// 9  0.505
int main(void) {
    int len = sqrt(VECTOR_SIZE);
    int size_small = (SIZE - 1) * (SIZE - 1);
    int i, j;

    FILE* file = fopen("matr.txt", "r");
    char* code = (char*)malloc(5120);
    int len_file = 0;
    int c = fgetc(file);
    while (c != EOF)
    {
        code[len_file] = c;
        len_file++;
        c = fgetc(file);
    }
    fclose(file);
    code[len_file] = '\0';
    int* A = (int*)malloc(sizeof(int) * VECTOR_SIZE);
    int* B = (int*)malloc(sizeof(int) * VECTOR_SIZE);

    srand(time(0));
    for (i = 0; i < VECTOR_SIZE; i++)
    {
#ifdef RAND
        A[i] = 1 + rand() % 10;
#else
        scanf("%d", &A[i]);
#endif
    }
    clock_t start = clock();

    // Get platform and device information
    cl_platform_id* platforms = NULL;
    cl_uint     num_platforms;

    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
    char buf[1024];

    //clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
    //printf("Platform: %s\n", buf);

    //Get the devices list and choose the device
    cl_device_id* device_list = NULL;
    cl_uint num_devices;

    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    device_list = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);

    //clGetDeviceInfo(device_list[0], CL_DEVICE_NAME, sizeof(buf), buf, 0);
    //printf("Device: %s\n", buf);
    
    // Create one OpenCL context for each device in the platform
    cl_context context;
    context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_list[0], 0, &clStatus);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&code, NULL, &clStatus);

    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "del_extra", &clStatus);

    // Create memory buffers on the device for each vector
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(int), NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, VECTOR_SIZE * sizeof(int), NULL, &clStatus);

    // Copy the Buffer A to the device
    clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(int), A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(int), B, 0, NULL, NULL);

    // Set the arguments of the kernel
    clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&A_clmem);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&B_clmem);

    // Execute the OpenCL kernel on the list
    size_t global_size = VECTOR_SIZE; // Process the entire lists
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, 0, 0, NULL, NULL);

    // Read the cl memory B_clmem on device to the host variable B
    clStatus = clEnqueueReadBuffer(command_queue, B_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(int), B, 0, NULL, NULL);
    double timeGPUTotal = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("%lf\n", timeGPUTotal);

    
    // Display the result to the screen
   /* printf("initial matrix:\n");
    for (i = 0; i < len; i++)
    {
        for (j = 0; j < len; j++)
            printf("%3ld ", a[i * len + j]);
        printf("\n");
    }*/
    long long DET = det(A, SIZE);
    /*printf("%lld\n", DET);
    printf("Inverse matrix:\n");
    for (i = 0; i < len; i++)
    {
        for (j = 0; j < len; j++)
        {
            printf("%f ", (float)B[i * SIZE + j]/DET);
        }
        printf("\n");
    }*/

    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(B_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(B);
    free(A);
    free(device_list);
    free(platforms);

    return 0;
}

float det(int* A, int size)
{
    float tmp[VECTOR_SIZE];
    float koef;
    for (int i = 0; i < VECTOR_SIZE; i++)
        tmp[i] = A[i];
    for (int i = 0; i < size - 1; i++)
    {
        for (int j = i + 1; j < size; j++)
        {
            koef = (float)tmp[j * size + i] / tmp[i * size + i];
            for (int ind = 0; ind < size; ind++)
                tmp[j * size + ind] -= tmp[i * size + ind] * koef;
        }
    }

    float res = 1;
    for (int i = size - 1; i >= 0; i--)
        res *= tmp[(size + 1) * i];
    return res;
}

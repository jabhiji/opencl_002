#include <iostream>
#include <cmath>
#include <OpenCL/cl.h>

int main(void)
{
    // Identify a platform

    cl_platform_id platform;

    cl_uint num_entries = 1;     // maximum number of platforms we are interested in detecting
    cl_uint num_platforms = 0;   // number of platforms actually found during run-time

    cl_int err = clGetPlatformIDs(num_entries, &platform, &num_platforms);

    // error check
    if(err < 0) {
        std::cout << "Couldn't find any platforms\n";
        exit(1);
    }

    std::cout << "Detected " << num_platforms << " platforms\n";
    std::cout << "Using OpenCL to get more information about the platform:\n\n";

    char pform_name  [40];        // platform name
    char pform_vendor[40];        // platform vendor
    char pform_version[40];       // platform version
    char pform_profile[40];       // platform profile
    char pform_extensions[4096];  // platform extensions

    clGetPlatformInfo(platform, CL_PLATFORM_NAME,       sizeof(pform_name  ),     &pform_name,       NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR,     sizeof(pform_vendor),     &pform_vendor,     NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION,    sizeof(pform_version),    &pform_version,    NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_PROFILE,    sizeof(pform_profile),    &pform_profile,    NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, sizeof(pform_extensions), &pform_extensions, NULL);

    std::cout << "CL_PLATFORM_NAME       --- " << pform_name       << std::endl;
    std::cout << "CL_PLATFORM_VENDOR     --- " << pform_vendor     << std::endl;
    std::cout << "CL_PLATFORM_VERSION    --- " << pform_version    << std::endl;
    std::cout << "CL_PLATFORM_VERSION    --- " << pform_profile    << std::endl;
    std::cout << "CL_PLATFORM_EXTENSIONS --- " << pform_extensions << std::endl;

    // Determine number of GPUs connected to this platform

    cl_uint       numOfDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, NULL, &numOfDevices);

    // error check

    if(err < 0) {
        std::cout << "Couldn't find any devices\n";
        exit(1);
    }

    std::cout << "\nNumber of connected devices found = " << numOfDevices << std::endl;

    // allocate memory to store devices

    cl_device_id* devices = (cl_device_id*) malloc(sizeof(cl_device_id) * numOfDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numOfDevices, devices, NULL);

    // Extension data

    char ext_data[4096];
    char name_data[48];

    cl_bool  deviceAvl;
    cl_uint  deviceAdd;
    cl_uint  frequency;
    cl_ulong globalMem;
    cl_ulong blockSize;
    cl_ulong sharedMem;
    cl_uint  witemDims;
    cl_ulong xyzDims[3];

    // Obtain specifications for each connected GPU

    std::cout << std::endl;
    std::cout << "Running OpenCL code to get GPU specifications:\n\n";

    for(unsigned int i=0; i<numOfDevices; i++) {

        clGetDeviceInfo (devices[i], CL_DEVICE_NAME                      , sizeof(name_data), name_data  , NULL);
        clGetDeviceInfo (devices[i], CL_DEVICE_AVAILABLE                 , sizeof(ext_data) , &deviceAvl , NULL);
        clGetDeviceInfo (devices[i], CL_DEVICE_ADDRESS_BITS              , sizeof(ext_data) , &deviceAdd , NULL);
        clGetDeviceInfo (devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY       , sizeof(ext_data) , &frequency , NULL);
        clGetDeviceInfo (devices[i], CL_DEVICE_GLOBAL_MEM_SIZE           , sizeof(ext_data) , &globalMem , NULL);
        clGetDeviceInfo (devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE       , sizeof(ext_data) , &blockSize , NULL);
        clGetDeviceInfo (devices[i], CL_DEVICE_LOCAL_MEM_SIZE            , sizeof(ext_data) , &sharedMem , NULL);
        clGetDeviceInfo (devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS  , sizeof(ext_data) , &witemDims , NULL);
        clGetDeviceInfo (devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES       , sizeof(ext_data) , &xyzDims[0], NULL);

        std::cout << "CL_DEVICE_NAME                     " << name_data << std::endl;
        std::cout << "CL_DEVICE_AVAILABLE                " << deviceAvl << std::endl;
        std::cout << "CL_DEVICE_ADDRESS_BITS             " << deviceAdd               << " bits"       << std::endl;
        std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY      " << (float)  frequency/1000 << " MHz"        << std::endl;
        std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE          " << (double) globalMem/1E9  << " GB "        << std::endl;
        std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE      " << blockSize               << " work items" << std::endl;
        std::cout << "CL_DEVICE_LOCAL_MEM_SIZE           " << sharedMem               << " Bytes"      << std::endl;
        std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS " << witemDims                                << std::endl;
        std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES : X  " << xyzDims[0]              << " work items" << std::endl;
        std::cout << "                              : Y  " << xyzDims[1]              << " work items" << std::endl;
        std::cout << "                              : Z  " << xyzDims[2]              << " work items" << std::endl;

        std::cout << std::endl;
    }

    // create a context that uses some or all available GPU devices (detected above)

    std::cout << "\nCreating a context (grouping devices on which kernels will be run):\n\n";

    // STEP 1 of 2: set context specifications

    cl_context_properties* setContextProps          = NULL;           // need to understand this better
    cl_uint                setNumDevicesInContext   = 1;              // specify number of devices to be used in the context
    const cl_device_id     listOfDevicesInContext[] = {devices[0]};   // specify list of devices used (based on their ID)

    // STEP 2 of 2: create the context

    cl_context context = clCreateContext(setContextProps, setNumDevicesInContext, listOfDevicesInContext, NULL, NULL, &err);

    // How to query the context created above and get information about it

    // how to get the number of devices used in a context

    cl_uint getNumDevicesInContext;
    clGetContextInfo (context, CL_CONTEXT_NUM_DEVICES,     sizeof(cl_uint), &getNumDevicesInContext, NULL);

    // allocate an appropriately sized device list and get the IDs of the
    // devices used in the context

    cl_device_id* getContextDevices = (cl_device_id*) malloc(sizeof(cl_device_id) * getNumDevicesInContext);
    clGetContextInfo (context, CL_CONTEXT_DEVICES, sizeof(cl_device_id) * getNumDevicesInContext, getContextDevices, NULL);

    // figuring out how many times this context structure is accessed

    cl_uint       ref_count;
    clGetContextInfo (context, CL_CONTEXT_REFERENCE_COUNT, sizeof(ref_count), &ref_count,              NULL);

    // printing information about the context

    std::cout << "CL_CONTEXT_NUM_DEVICES             " << getNumDevicesInContext << std::endl;
    std::cout << "CL_CONTEXT_DEVICES                 ";

    for(unsigned int i=0; i<getNumDevicesInContext; i++) {
        clGetDeviceInfo (getContextDevices[i],      CL_DEVICE_NAME, sizeof(name_data), name_data  , NULL);
        std::cout << name_data << "   ";
    }
                                                                    std::cout << std::endl;
    std::cout << "CL_CONTEXT_REFERENCE_COUNT         " << ref_count           << std::endl;

    // Read kernel file into a buffer

    FILE *file_handle = fopen("kernel_add.cl", "r");  // create pointer to a file object
    fseek(file_handle, 0, SEEK_END);                  // move position to end of file
    size_t program_size = ftell(file_handle);         // get size of the kernel in bytes
    rewind(file_handle);

    char *program_buffer = (char*) malloc(program_size + 1);
    program_buffer[program_size] = '\0';

    fread(program_buffer, sizeof(char), program_size, file_handle);
    fclose(file_handle);

    // build the program

    cl_program program = clCreateProgramWithSource(context, 1, (const char**) &program_buffer, &program_size, &err);
    free(program_buffer);

    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // create kernel

    cl_kernel kernel = clCreateKernel(program, "add", &err);

    // create a command queue

    cl_command_queue command_queue = clCreateCommandQueue(context,                   // specify context
                                                          getContextDevices[0],      // specify device
                                                          0,                         //
                                                          &err);                     //

    // allocate memory buffers on the host

    const int N = 2048;

    int *h_a = new int[N];   // vector "a"
    int *h_b = new int[N];   // vector "b"
    int *h_c = new int[N];   // c = a + b

    // fill input buffers on the host

    for(int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i+1;
    }

    // allocate memory buffers on the device

    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY,  N * sizeof(int), NULL, &err);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY,  N * sizeof(int), NULL, &err);
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(int), NULL, &err);

    // copy input buffers from host memory to device memory

    clEnqueueWriteBuffer(command_queue, d_a, CL_TRUE, 0, N * sizeof(int), h_a, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, d_b, CL_TRUE, 0, N * sizeof(int), h_b, 0, NULL, NULL);

    // set kernel function parameters

    clSetKernelArg(kernel, 0, sizeof(cl_int), &N  );    // parameter 0
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);    // parameter 1
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);    // parameter 2
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_c);    // parameter 3

    // kernel launch parameters (work distribution)

    // Number of work items in each local work group (CUDA threads per block)

    size_t localSize = 128;

    // Number of total work groups (CUDA blocks)

    size_t globalSize = ceil(N/(float)localSize)*localSize;

    // execute OpenCL kernel

    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    // copy results from device memory to host memory

    clEnqueueReadBuffer(command_queue, d_c, CL_TRUE, 0, N * sizeof(int), h_c, 0, NULL, NULL);

    // display result using the buffer on the host

    std::cout << std::endl;

    for(int i = 0; i < N; i++) {
        std::cout << "a = " << h_a[i] << " b = " << h_b[i]  << " c = a + b = " << h_c[i] << std::endl;
    }

    // Clean up

    err = clFlush(command_queue);
    err = clFinish(command_queue);

    err = clReleaseKernel(kernel);
    err = clReleaseProgram(program);

    err = clReleaseMemObject(d_a);
    err = clReleaseMemObject(d_b);
    err = clReleaseMemObject(d_c);

    err = clReleaseCommandQueue(command_queue);

    free(getContextDevices);
    free(devices);

    delete [] h_a;
    delete [] h_b;
    delete [] h_c;

    // main program ends

    return 0;
}

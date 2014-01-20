// OpenCL kernel for vector addition

__kernel void add(   const int N,      // number of elements in the vector
                  __global int *a,     // input vector 1
                  __global int *b,     // input vector 2
                  __global int *c)     // output vector
{
    // get global location of this work-item

    int index = get_global_id(0);

    // perform addition with bound checking

    if (index < N)
        c[index] = a[index] + b[index];
}


#include <stdlib.h>
#include <iostream>
#include "HalideBuffer.h"

#include "demo_1_512_512_512.h"
using namespace std;

#define ZERO 0
#define ONE 1
#define RAND 2

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef FABS
#define FABS(a) (((a) > 0) ? (a) : (-a))
#endif

// For x86-64-linux:
//g++ demo_eval.cpp demo.s -I $HALIDE_HOME/include  -ldl -lpthread -o demo_eval

// For arm-64-linux:
//aarch64-linux-gnu-g++ demo_eval.cpp demo.s -I $HALIDE_HOME/include  -ldl -lpthread -o demo_eval

const int B=1;
const int N=512;
const int M=512;
const int K=512;

void ref_func(int8_t*data_a,int8_t*data_b,int8_t*data_c)
{
    for(int b=0;b<B;b++)
    {
        for(int i=0;i<M;i++)
        {
            for(int j=0;j<N;j++)
            {
                data_c[b*M*N+i*N+j]=0;
                for(int k=0;k<K;k++)
                {
                    //data_c[b][i][j] +=data_a[b][i][k]*data_b[b][k][j]
                    data_c[b*M*N+i*N+j]+=data_a[b*M*K+i*K+k]*data_b[b*N*K+k*N+j];
                }
            }
        }
    }
}
void init(int8_t* data,int size, int mode)
{
    srand(0); //set rand_seed
    int i;
    for (i = 0; i < size; ++i) {
        if (mode == ZERO)
            data[i] = 0;
        else if (mode == ONE)
            data[i] = 1;
        else
            data[i] = (int)rand();
    }
}

float maxerr(int8_t* pred, int8_t* gt, int size)
{
    float maxError = 0.f;
    for(int i=0; i< size; i++){
            maxError = MAX(FABS(gt[i] - pred[i]), maxError);
    }
    printf("maxerr %.6f\t", maxError);
    return maxError;
}

int main()
{
    int8_t a[M*K*B];
    int8_t b[N*K*B];
    int halide_c[M*N*B];
    int8_t ref_c[M*N*B];

    //input data random init 
    init(a,M*K*B,RAND);
    init(b,N*K*B,RAND);
    // output data zero init
    init(halide_c,M*N*B,ZERO);
    init(ref_c, M*N*B,ZERO);

    Halide::Runtime::Buffer<int8_t> Halide_A((int8_t*)a, K,M,B);
    Halide::Runtime::Buffer<int8_t> Halide_B((int8_t*)b, N,K,B);
    Halide::Runtime::Buffer<int8_t> Halide_C((int*)halide_c, N,M,B);

    matmul(Halide_A,Halide_B,Halide_C);
    ref_func(a,b,ref_c);

    if (maxerr(ref_c,halide_c,M*N*B)<0.001)
    {
        cout<<"Correctness check passed!"<<endl;
    }else
    {
        cout<<"Correctness check failed"<<endl;
    }
    return 0;
}

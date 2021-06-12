#include <stdio.h>
#include <stdlib.h>

int main(){
    //1D
    /*int N = 32; //for 1-based, act = 32
    for(int n = 0; n < N-1; n++){
        printf("%d %d 1\n", n, n+1);
    }*/

    int N = 25;
    //旁邊
    for(int i = 0; i < N*N; i += N){
        for(int j = i; j < (i+N-1); j++){
            printf("%d %d %d\n", j, j+1, 1);
        }
    }
    //下面
    for(int i = 0; i < N*(N-1); i += N){
        for(int j = i; j < (i+N); j++){
            printf("%d %d %d\n", j, j+N, 1);
        }
    }

    return 0;
}
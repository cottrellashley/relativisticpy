// main.c
#include <stdio.h>
#include <string.h>
#include "iterator.h"

int main() {
    const char *testStr = "G_{mu}_{nu} := [[-A_{a:1}_{r:2},0,0,0], [0,B(r),0,0], [0,0,r**2,0], [0,0,0,r**2*sin(theta)**2]]";
    Iterator iter;
    
    Iterator_Init(&iter, testStr);
    Iterator_Advance(&iter);

    while (Iterator_Current(&iter) != '\0') {
        Iterator_Advance(&iter);
    }

    Iterator_Cleanup(&iter);

    return 0;
}

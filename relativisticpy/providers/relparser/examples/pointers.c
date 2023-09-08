// main.c
#include <stdio.h>

// What is a pointer? (they are not that complicated once you understand Memory)
// Memory = { address : value }
// address = where the memory lives
// value = what is sotred in that address


// Integer named x is set to value 4.
int x = 4; // Let's store the value of integer 4 in the address x.

// Integer pointer named pX is set to the address of x.
int * pX = &x; // Let's tore the value of ( memory address x ) in the address pX. 

// *pX indicates the C compiler that this is a pointer i.e. its value is an address to another memory {address : value}

// Integer named y is set to pointer named pX
int y = * pX;

// Whatever is store within pX we know is a memory address. The ones and zeros within the address called pX are themselves one and zeros which represent another address.
// This means if we simply do something like 4 + pX we would not get "8"
// Think for a moment, the ones and zeros in pX are an address, so if we do not de-reference (*) the pointer, then it will read the ones and zeros in that address as an integer not an address
// We therefore write *pX to dereference meaning the * will tell the C computer to go to the value which the ones and zeros stored in pX are storing.
// * is called dereferencing the pointer.


// Decleration and Dereference have the same syntax. -> *

// 1: Decleration tells me how to use.

int n; // English: To get an integer, just use n.

int *n; // English: to get an integer just dereference n.

int n[3]; // English: To get an integer, just index the array n with a value (3 in this case).

int foo(int n, float n1); // English: To get an integer, just call the function foo and pass in the related inputs.

// (TL;DR) Declaration tells us how to use X to get Y.

int n; int *pn; int **pn2; // All these declerations will yield a type of integer when used in this way.

// Derefernces:

int n; // integer store in n ; to get an int use n
int *pn; // declared a 
int **ppn;
int ***pppn;

n = 42;
pn = &n;
ppn = &pn;
pppn = &ppn;

// Above we simply keept on passing the references
Certainly, let's delve deep into the intricacies of the C programming language, specifically in the realm of memory management and function invocation.

## **Memory in C**

In the C programming language, when your program runs, it uses two primary sections of memory:

1. **Stack** - This is used for static memory allocation. Local variables to a function reside here.
2. **Heap** - This is used for dynamic memory allocation, and variables need to be manually managed.

The distinction between these two is important because the stack is managed for you, while the heap is more manual.

## **Function Execution and Stack Frames**

When you invoke a function in C, the system creates a **stack frame** for that function. This frame contains all the local variables of that function, and other information to ensure that once the function finishes its execution, the program can continue where it left off. The most recent function that's been called is at the top of the stack. So, the sequence of function calls determines the stack's layout.

To demonstrate this visually:

```
Empty Stack
-----------
```

When you execute `foo`:

```
foo's stack frame
-----------
Empty Stack
-----------
```

If `foo` then calls `moo`:

```
moo's stack frame
-----------
foo's stack frame
-----------
Empty Stack
-----------
```

## **Variable Scope and Behavior**

### **Pass-by-value**

In C, by default, function parameters are passed by value. When you pass a variable to a function, you are essentially giving that function a copy of the original variable. 

Using our previous example, if `foo` had a local variable `x`, and it passed `x` to `moo`, it would look something like this:

```
moo's stack frame (has its own copy of x)
-----------
foo's stack frame (with the original x)
-----------
Empty Stack
-----------
```

Here, changes made to `x` inside `moo` won't affect the `x` inside `foo` because `moo` is working with its own copy.

### **Pass-by-reference**

If you want to allow `moo` to modify `foo`'s local variables, you need to pass the memory address of those variables to `moo`, which means you're using pointers.

This can be achieved by passing the address of `foo`'s `x` using the `&` (address-of) operator. Inside `moo`, you'd use the `*` (dereference) operator to access and modify the value.

In this case:

```
moo's stack frame (has a pointer to x)
-----------
foo's stack frame (with the original x)
-----------
Empty Stack
-----------
```

When `moo` dereferences its pointer to modify `x`, it's directly changing the `x` in `foo`'s frame. This means that any modification done by `moo` to this variable will be reflected in `foo`.

Let's illustrate how passing by reference using pointers works in C using the stack notations.

### Setting up `foo`

Let's start by assuming you have a function `foo` that initializes an integer variable `x` and then passes a pointer to `x` to another function `moo`.

In `foo`:

```c
int x = 10;
moo(&x);
```

The stack after declaring `x` in `foo` would look like this:

```
foo's stack frame (x = 10)
-----------
Empty Stack
-----------
```

### Calling `moo`

Now, when `foo` calls `moo` and passes the address of `x` to it:

```c
void moo(int *ptr_to_x) {
    *ptr_to_x = 20;
}
```

The stack just after entering `moo` looks like this:

```
moo's stack frame (ptr_to_x points to x)
-----------
foo's stack frame (x = 10)
-----------
Empty Stack
-----------
```

### Modifying `x` through the pointer

When `moo` dereferences `ptr_to_x` and modifies the value, it's modifying `x` in `foo`'s stack frame directly:

```
moo's stack frame (ptr_to_x points to x)
-----------
foo's stack frame (x = 20) <- Notice the value change!
-----------
Empty Stack
-----------
```

### Returning from `moo`

When `moo` completes its execution and returns, its stack frame is removed, and you're left with:

```
foo's stack frame (x = 20) <- Value remains modified.
-----------
Empty Stack
-----------
```

The change in `x` persists in `foo`'s frame because `moo` was given a direct reference (pointer) to the variable and modified it through that reference. This is the essence of "pass-by-reference" behavior achieved using pointers in C.

## **Summary**

In C:

1. The **stack** is where function frames (with their local variables) are pushed and popped. The layout depends on the sequence of function calls.
2. By default, C uses **pass-by-value**. Called functions get copies of variables.
3. To allow a function to modify another function's variables, use pointers and **pass-by-reference**.

The intricacies of C's memory management, combined with the powerful (but potentially hazardous) feature of pointers, underscore the importance of understanding these concepts deeply to write effective C code.
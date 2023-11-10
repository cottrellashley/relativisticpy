In C, the `->` arrow is used to access members of a structure through a pointer to that structure.

To understand this better, let's break down what's happening:

1. **Structures and Dot Operator (`.`):** If you have a structure (let's say named `StructType`) and an instance of that structure (let's say named `instance`), you access its members using the dot operator (`.`).

    ```c
    StructType instance;
    instance.member = value;
    ```

2. **Pointers and Arrow Operator (`->`):** If instead of an instance, you have a pointer to an instance of the structure (let's say named `ptr_instance`), you access its members using the arrow operator (`->`).

    ```c
    StructType* ptr_instance;
    ptr_instance->member = value;
    ```

    The arrow operator `->` is a shorthand for dereferencing the pointer and then using the dot operator. The line of code above is equivalent to:

    ```c
    (*ptr_instance).member = value;
    ```

In the provided code:

```c
void Iterator_Init(Iterator* iter, int* object, size_t length) {
    iter->object = object;
    iter->length = length;
    iter->current_item = NULL;
    iter->location = -1;
}
```

The function `Iterator_Init` takes a pointer `iter` to a structure of type `Iterator`. To set or access the members of the structure that `iter` points to, the arrow operator (`->`) is used.
### Concept Explanation

#### Simple Explanation:
Imagine you have a box (function) where you can put things (variables and operations). If you put a label named 'x' on an apple (assign a value to 'x') outside any box, everyone can see and use this apple labeled 'x'. But if you put a label named 'v' on an orange inside a box, only things inside this box can see and use the orange labeled 'v'. If you try to find the orange 'v' outside the box, it's like it doesn't exist.

#### Medium Explanation:
In programming, especially in languages that support functions and scopes like Python and your symbolic programming language, variables can have different "visibility" depending on where they are defined. If you define a variable outside of any function (globally), it's like putting it on a public display; any function can "see" and use it. However, if you define a variable inside a function, it's like keeping it inside a private room; only the operations and functions inside that specific function can access it. This concept ensures that you can have temporary, function-specific variables that don't interfere with variables elsewhere in your code.

#### Advanced Explanation:
In more technical terms, this is known as lexical scoping and variable shadowing. Lexical scoping refers to the accessibility of variables based on their location within the source code (lexical environment), and their visibility is controlled by the block structure (functions, conditional blocks, etc.). Variable shadowing occurs when a variable declared within a certain scope (a function, for example) has the same name as a variable declared in an outer scope, effectively "hiding" the outer variable within the inner scope. This mechanism is implemented under the hood using data structures like stacks and dictionaries, where each function call pushes a new "frame" onto the stack, containing mappings of variable names to their values, and popping it off when the function returns.

### Examples for Testing

#### Example 1: Global vs Local Scope
Global `x` is used inside the function.

```python
x := 10

f : (y) -> {
  result := (x**2 - 4) / (x - 2)
  result
}

f(5)  # Should use x = 10 from the global scope
```

#### Example 2: Local Variable Shadowing Global
Local `x` shadows global `x`.

```python
x := 10

g : (x) -> {
  result := (x**2 - 4) / (x - 2)
  result
}

g(5)  # Should use x = 5 passed to the function, not the global x
```

#### Example 3: Nested Functions with Scopes
Demonstrating nested functions and scope visibility.

```python
x := 10

outer : (y) -> {
  inner : (z) -> {
    result := (x + y + z)**2
    result
  }

  inner_result := inner(3)
  inner_result + y
}

outer(2)  # Uses global x = 10, y = 2 from outer, and z = 3 from inner
```

#### Example 4: Local Scope Not Accessible Outside
Variable defined in a function not accessible outside.

```python
f : (y) -> {
  local_var := y**2
  local_var
}

f(4)  # Returns 16
local_var  # Should result in an error or undefined variable response
```

#### Example 5: Using Function Scope for Temporary Calculations
Temporary variable within a function for intermediate calculation.

```python
temp_calc : (a, b) -> {
  temp := a + b
  result := temp**2
  result
}

temp_calc(2, 3)  # Uses temp within the function for calculation
```

These examples illustrate the use of global and local scopes, variable shadowing, and the concept of function-specific temporary variables. They should help both in understanding the concept and in testing the implementation of scopes in your symbolic programming language.
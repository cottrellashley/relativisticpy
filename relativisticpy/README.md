Workbook module.

What is the workbook compute flow:

1. User enters string.
2. We descerialize the string into a tree of compute-nodes as determined by the grammar of the RelParser compute module.
3. We traverse the tree node with the NodeTraverser class (which contains all of the methods which implement the nodes in the tree.)
4. After all tree nodes are computed we return the final result.

Notes:
1. If the string is a variable definition -> we define a pointer and add the value to the dictionary.
2. If the variable is being called -> we get and replace the pointer with the value within the dict tree.
3. If we are defining something -> we perform a specific task (We set a property in the class and re-call it latter within the workbook whenever we use it.)


```

Condition1
|
|------No
|       |
yes     |-- Action1
|
Condition2
|
|-- EinsteinArray
|
|-- Metric
|
|------ gr
|       |
|       |-- GeometricObject -> Metric Dependent
|       |   |
|       |   |-- Connection
|       |   |
|       |   |-- Riemann
|       |   |
|       |   |-- Ricci
|       |   |
|       |   |-- CovariantDerivative
|       |
|       |-- PhysicalObject -> User/State Defined
|           |
|           |-- StressEnergyTensor
|           |
|           |-- ElectromagneticTensor
|    
|------ em (TODO: Electromagnetism Module)
|
|------ ft (TODO: Field Theory Module)

```
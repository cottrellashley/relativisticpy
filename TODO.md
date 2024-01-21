## Prioritized Features: 

0. Branch off into a new branch called 'feature/semantic_analyzer'

1. The assignment Context right now is being added by the relativity_parser.py
    a. def equation(self) method should only have one Node creation not three different types -> this is decided by the Semantic_analyzer.py

2. completly remove/rename the type 'object' -> this should either be 'function', 'symbol', 'tensor'
    2.b. add an extra property to the node called 'type' - as this is important to keep consistent for the Semantic_analyzer.py layer - needs 'type (used to be node) - args - handler' 
3. hard code some of the symbolic methods 'simplify', 'derivative', etc ...
4. once these naming things are complete -> implement a working Semantic_analyzer.py which splits and adds all the context we need - first without a assignment_table
5. fix and re-write the ast_traverser since now we have better control over the method names and the context.
6. once we have t=more

- **Einstein Tensor**
- Integration of a setter function for defining tensor expressions such as T_{a}_{b} = [custom tensor expression with matching indices].
- Integration of an intelligent well defined rules for a dynamic derivative, which uses defined coordinates or user defined.
- Metric defined via line element: this will require big edit to the parser grammar => multiple different grammars.
- **Covariant Derivative**
- **Energy Momentum Tensor**
- **Ricci Scalar**
- **Weyl Tensor**
- **Symetric/Ant-Symmetric Tensors:** Define a T[_{a}_{b}] = 1/2*(T_{a}_{b} - T_{b}_{a}) and a T(_{a}_{b}) = 1/2*(T_{a}_{b} + T_{b}_{a})  
- Clean up Index and Indices properties handling/lifecycle -> (basis, shape, dimention, values, etc...)

## Grammar Syntax for RelPy

- **Context-Dependent Grammers:** Currectly I only have one Lexer and one Parser (Grammar) -> I would like to have the ability to trigger different lexers via contex driven letters i.e. "G^" or "G_" trigger the tensor grammar such that we decode the tensor and can use characters which are different meaning in the context of a tensor such as G_{a:[0,1,2]}_{b:0} will grab the tensor indices which the user specified and not create an array [0, 1, 2] (which is what the current grammar would interpret this as and break.)

### Additional Tensor Objects to Implement

- **Einstein Tensor**
- **Covariant Derivative**
- **Weyl Tensor**
- **Energy Momentum Tensor**
- **Ricci Scalar**

### Additional Equations to Implement
- **Geodesic Equation**: For general metric geodesic solutions + numerical methods with it.
- **Kerr-Newman Solution**: For charged, rotating black holes.
- **Reissner-Nordström Solution**: For charged, non-rotating black holes.
- **Oppenheimer-Snyder Model**: Describes gravitational collapse of a spherically symmetric body and formation of black holes.
- **Lemaître-Tolman-Bondi Models**: Solutions for dust-filled universe models, useful in cosmology.

### Additional Instances to Add
- **Kerr Metric**: A solution of the Einstein field equations that describes the spacetime geometry in the region surrounding a rotating mass. Essential for understanding rotating black holes.
- **Schwarzschild Metric**: Describes spacetime around a spherically symmetric, non-rotating mass. Fundamental for understanding black holes and gravitational time dilation.
- **Friedmann-Lemaître-Robertson-Walker (FLRW) Metric**: Used in cosmology, it describes a homogeneous, isotropic expanding or contracting universe.

## Additional Features (no implementation plan yet)
- **Penrose-Carter Diagrams**: Useful for visualizing and understanding the causal structure of spacetime, especially around black holes.
- **ADM Formalism**: Decomposes spacetime into space and time, useful in numerical relativity.
- **Penrose Process**: Describes energy extraction from a rotating black hole, which is an interesting aspect of black hole thermodynamics.
- **Hawking Radiation**: Code modules to model or simulate this quantum effect could be intriguing.

### New Manipulations or Features
- **Tensor Visualization Tools**: Implement ways to visually represent tensors, perhaps in a 3D space, to aid in understanding their properties.
- **Spacetime Diagrams and Simulations**: Create modules that allow users to simulate and visualize the effects of massive bodies on spacetime, like black hole warping.
- **Interactive Learning Modules**: Interactive tutorials or problems that guide users through complex concepts using your toolkit.
- **Symbolic Computation**: Incorporate symbolic computation for tensor algebra, making it easier for users to perform calculations without numerical values.
- **GR in Different Coordinate Systems**: Allow users to convert tensors between different coordinate systems, like from Cartesian to spherical coordinates.
- **Numerical Solvers for Einstein's Field Equations**: Implement or integrate solvers that can numerically solve Einstein's field equations under different conditions.
- **Integration with Astrophysical Data**: Provide functionality to import and use real astrophysical data (like from LIGO or other observatories) for analysis with your package.
- **Community Contributions**: Create a platform or system where users can contribute their own solutions or modules, fostering a community around your package.
- **Performance Optimization Tools**: Tools for users to optimize their calculations, particularly for complex simulations.
- **Quantum Gravity Extensions**: Although speculative, modules exploring the interplay between quantum mechanics and general relativity could be fascinating.

### General Suggestions
- **Documentation and Examples**: Comprehensive documentation and real-world examples are vital for educational tools. This includes both simple introductory examples and more complex case studies.
- **Modular Design**: Ensure that the package is modular so users can easily use only the parts they need.
- **User Feedback Loop**: Implement a system for receiving user feedback to continuously improve the package based on actual user experience and needs.

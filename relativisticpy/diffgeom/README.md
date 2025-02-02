General Relativity Module

Functionality and Object types:

Definition:

List = collection of python objects (any type).

GrArray

General Relativity = Spacetime + Matter

Spacetime = Differential Geometry Objects i.e. Manifold, Patch, Coordinate Patch, Coordinate Transformation, Metric, Riemann, Ricci, Weyl, etc...
Matter = Physical Matter Objects i.e. EnergyMomentum Tensor, any other vector/tensor representing a physical quantity.
General Relativity Module = Logic of combining Differential Geometric Objects and Physical Matter Objects.


== Logic for Differential Geometry ==
Manifold
Patch
Point
Coordinate Patch
Coordinate Transformation

== Logic for Multi-Components Object Products ==
Idx
Indices
EinsumArray
GrArray


LeviCivitaConnection
GrTensor
MetricIndices
Metric
Ricci
Riemann

The above objects will be more abstract and base-like. They will not have much logic 

Manifold Object
Patch Object

Array Object.
Logic for multiplying two arrays.
Logic for operation of array components within itself -> i.e. outputting another array.

EinsumArray -> The base class, contains logic for multiplying two array-like object 
Split into two parts:
    a) Geometric tensors
    b) Physical tensors



Quora Answers:

======
Differential geometry is about the shape of space and tensors are about things in that space. They are related at least through the metric tensor that can describe the shape of space in terms of things that are in that space.

My practical work with tensors is in fluid flow mostly in a way that has nothing to do with relativity - except that a lot of the mathematics is the same, as I have to study the way in which energy and momentum flow through space. And that is the core of general relativity. General relativity can be seen to be the study of fluid flow in special relativity.

Other related applications are classical electromagnetic theory (the electric field tensor is a better way of describing classical electromagnetic theory than the separate electric and magnetic fields), which is used for most work in wave guides and other electromagnetic technology. There are very few cases at that level where relativity is used directly. But, again, the mathematics is very similar.

Problems in the flow of fluids with electricity in them require quite heavy use of this type of mathematics. Multiphase fluid flow and flow through porous media - as well as describing the stresses in crystalline substances - these also can use tensor analysis. I have used tensor analysis in a variety of control engineering problems.

So, there is much much more to the use of tensors than general relativity.
=======

Differential geometry and tensors are closely related. Tensors are mathematical objects that can be used to represent geometric and physical quantities in a way that is independent of the choice of coordinates. In differential geometry, tensors are used to study the curvature of spaces and manifolds, and they play a fundamental role in the formulation of the theory of general relativity.

The difference between differential geometry and tensors lies in their focus and application. Differential geometry is a branch of mathematics that deals with the study of curves, surfaces, and manifolds using techniques from calculus and linear algebra. Tensors, on the other hand, are mathematical objects that can be used to represent geometric and physical quantities in a coordinate-independent manner.

Tensors have applications outside of relativity, such as in fluid dynamics, elasticity, and electromagnetism. In these areas, tensors are used to describe physical quantities that have direction and magnitude, such as stress, strain, and electromagnetic fields.

Similarly, differential geometry has applications outside of relativity. It is used in various fields such as computer graphics, robotics, and computer-aided design, where the study of curves and surfaces is essential for modeling and simulation. Differential geometry also plays a role in optimization, control theory, and mathematical physics, where the understanding of geometric structures is crucial for solving complex problems.
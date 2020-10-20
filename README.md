# Undergraduate Python Assignments

This repository contains three Python assignments I completed while in the third year of my physics degree.

# Assignment 1: Matrices and Linear Algebra

The manipulation of matrices is a core topic in numerical analysis. Matrix methods can be used to solve many classes of physical problems, although care has to be taken in the choice and implementation of appropriate algorithms.

This assignment explores the advantages and disadvantages of different methods for solving systems of linear simultaneous equations. Evidence has been found to support the theoretical computational cost associated with each method. This was achieved by collecting data of method speed against matrix dimensions \textit{n} and then performing a chi-square test of this data against that predicted by the theory. The LU Decomposition routine was then used to solve a physics problem.

# Assignment 2: Partial Differential Equations

Solving partial differential equations is crucial to a huge variety of physical problems encountered in science and engineering. There are many different techniques available, all with their own advantages and disadvantages, and often specific problems are best solved with very specific algorithms.

This assignment explores two methods for solving partial differential equations; Jacobi and Gauss-Seidel. Evidence has been found to support theoretical models that predict the number of iterations that these two methods require, for a given grid density and convergence condition. The Gauss-Seidel method is then used to find the potential and electric fields of a parallel plate capacitor. These results are compared to the infinite plate solution and its properties. Finally, the diffusion equation is solved for an iron poker using the implicit difference method, and by solving a set of linear equations to evolve the system in time.

# Assignment 3: Random Numbers and Monte Carlo

This assignment addresses the use of “random” numbers in Monte Carlo techniques. These are often the fastest or most straightforward way of tackling complicated problems in computational analysis. Monte Carlo techniques rely on the availability of reliable number generators. These are not truly random, but calculate a pseudo-random number based on their internal state. The state is changed with every call, so the next call will produce a different number. A good algorithm should produce a sequence of numbers that are close to random by any statistical test, and does not repeat until the state (typically an array of integers) returns to its starting value. Historically, some frequently used random number generators produced sequences that were rather short and easily predicted. Modern random number algorithms, such as Mersenne Twister, used within numpy.random, are better tested and avoid problem seen with earlier generators.

Two methods have been successfully used to produce random sinusoidal distributions to a confidence level of $100\%$. The differences between these two methods has been explored. Then Monte Carlo was used to simulate gamma ray detection from nuclear decay, taking into account the uncertainty brought about by the resolution of the detector. The distribution of the gamma rays detected has been supported by theory. Finally, pseudo-experiments were used to successfully determine the cross section at which the null hypothesis - a hypothetical particle existing - is supported at a $95\%$ confidence level.

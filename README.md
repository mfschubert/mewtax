# Differentiable minimization in jax using Newton's method
`v0.0.1`

This project essentially repackages code from the [implicit layers tutorial](https://implicit-layers-tutorial.org/implicit_functions/) to provide a `minimize_newton` function.

Given a function `fn(params, z)`, it finds the `z_star` which minimizes `fn` for given `params`. Further, the gradient of the solution with respect to `params` can be computed; this is done using a custom vjp rule, as shown in the tutorial.

## Installation

mewtax can be installed via pip:
```
pip install mewtax
```

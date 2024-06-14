"""Differentiable minimization using Newton's method.

The code in this module is adapted from the implicit layers tutorial by Kolter,
Duvenaud, and Johnson. https://implicit-layers-tutorial.org/implicit_functions/
"""

import functools
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import flatten_util

PyTree = Any


def minimize_newton(
    fn: Callable[[PyTree, PyTree], jnp.ndarray],
    params: PyTree,
    z_init: PyTree,
    tol: float = 1e-5,
    max_iter: int = 20,
    eps: float = 1e-8,
) -> PyTree:
    """Minimizes `fn` using the Newton method.

    Here, `fn` is a function parameterized by `params`, and `z_init` is the inital
    solution. Both `params` and `z_init` may be pytrees with arbitrary structure.
    Differentiation of the solution with respect to `params` is supported and
    implemented by a custom vjp rule; gradients with respect to `z_init` are zero.

    Note that a suitable `z_init` must be provided. For example, it should be complex
    if complex solutions are sought.

    Args:
        fn: The function to be minimized, with signature `fn(a, z) -> y`. Each of the
            arguments may be an arbitrary pytree, while the output must be a scalar.
        params: Pytree parameterizing the function.
        z_init: Pytree giving the initial guess for the argmin of `fn`.
        tol: Tolerance for convergence. Default value is 1e-5.
        max_iter: Maximum number of iterations for the solver. No more than this
            number of iterations is performed, regardless of whether the minimization
            has converged. Default value is 20.
        eps: Small positive value used to regularize the minimization problem. Default
            value is 1e-8.

    Returns:
        The pytree argmin of the function.
    """
    z_init_flat, unflatten_z_fn = flatten_util.ravel_pytree(
        z_init
    )  # type: ignore[no-untyped-call]

    def flat_fn(params: PyTree, z_flat: jnp.ndarray) -> jnp.ndarray:
        regularization: jnp.ndarray = eps * _linalg_norm_safe(z_flat) ** 2
        return fn(params, unflatten_z_fn(z_flat)) + regularization

    z_flat = _minimize_newton(
        flat_fn,
        a=params,
        z_init=z_init_flat,
        tol=tol,
        max_iter=max_iter,
    )
    return unflatten_z_fn(z_flat)


def _linalg_norm_safe(z: jnp.ndarray) -> jnp.ndarray:
    """Computes the norm of `z`, with special treatment to avoid `nan` gradients."""
    is_all_zeros = jnp.allclose(z, 0.0)
    z_safe = jnp.where(is_all_zeros, jnp.ones_like(z), z)
    norm = jnp.linalg.norm(z_safe)
    return jnp.where(is_all_zeros, jnp.zeros_like(norm), norm)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def _minimize_newton(
    fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    a: PyTree,
    z_init: jnp.ndarray,
    tol: float,
    max_iter: int,
) -> jnp.ndarray:
    """Minimizes `fn` using the Newton method.

    Args:
        fn: The function to be minimized, with signature `fn(a, z) -> y`, where `a` is
            a pytree, `z` is a rank-1 array, and `y` is a real-valued scalar.
        a: Pytree parameterizing the function.
        z_init: Rank-1 array giving the initial guess for the argmin of `fn`.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations for the solver. No more than this
            number of iterations is performed, regardless of whether the minimization
            has converged.

    Returns:
        The array argmin of the function.
    """

    def fixed_point_fn(z: jnp.ndarray) -> jnp.ndarray:
        grad: jnp.ndarray = jax.grad(fn, argnums=1)(a, z)
        return z - grad.conj()

    return _newton_solve_fixed_point(
        fixed_point_fn,
        z_init=z_init,
        tol=tol,
        max_iter=max_iter,
    )


def _minimize_newton_fwd(
    fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    a: PyTree,
    z_init: jnp.ndarray,
    tol: float,
    max_iter: int,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Compute the forward Newton minimization."""
    z_star = minimize_newton(fn, a, z_init, tol, max_iter)
    return z_star, (a, z_star)


def _minimize_newton_bwd(
    fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    tol: float,
    max_iter: int,
    res: Tuple[jnp.ndarray, jnp.ndarray],
    z_star_bar: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the backward Newton minimization."""
    a, z_star = res

    def fixed_point_fn(a: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        grad: jnp.ndarray = jax.grad(fn, argnums=1)(a, z)
        return z - grad.conj()

    # See https://implicit-layers-tutorial.org/implicit_functions/
    _, vjp_a = jax.vjp(lambda a: fixed_point_fn(a, z_star), a)
    _, vjp_z = jax.vjp(lambda z: fixed_point_fn(a, z), z_star)
    return (
        vjp_a(
            _newton_solve_fixed_point(
                fn=lambda u: vjp_z(u)[0] + z_star_bar,
                z_init=jnp.zeros_like(z_star),
                tol=tol,
                max_iter=max_iter,
            )
        )[0],
        jnp.zeros_like(z_star),
    )


_minimize_newton.defvjp(_minimize_newton_fwd, _minimize_newton_bwd)


def _fwd_solve_fixed_point(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    z_init: jnp.ndarray,
    tol: float,
    max_iter: int,
) -> jnp.ndarray:
    """Solve for a fixed point of `fn`."""

    def cond_fn(carry: Tuple[int, jnp.ndarray, jnp.ndarray]) -> bool:
        i, z_prev, z = carry
        cond: bool = (jnp.linalg.norm(z_prev - z) > tol) & (i < max_iter)
        return cond

    def body_fn(
        carry: Tuple[int, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
        i, _, z = carry
        return i + 1, z, fn(z)

    init_carry = (0, z_init, fn(z_init))
    i, z_star, z_star_next = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return jnp.where(
        jnp.asarray(i == 0) | ~jnp.any(jnp.isnan(z_star_next)),
        z_star_next,
        z_star,
    )


def _newton_solve_fixed_point(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    z_init: jnp.ndarray,
    tol: float,
    max_iter: int,
) -> jnp.ndarray:
    """Solves for a fixed point of `fn` using the Newton method."""
    holomorphic = jnp.iscomplexobj(z_init)

    def root_fn(z: jnp.ndarray) -> jnp.ndarray:
        return fn(z) - z

    def g_fn(z: jnp.ndarray) -> jnp.ndarray:
        jac = jax.jacfwd(root_fn, holomorphic=holomorphic)(z)
        z_next: jnp.ndarray = z - jnp.linalg.solve(jac, root_fn(z))
        return z_next

    return _fwd_solve_fixed_point(g_fn, z_init, tol, max_iter)

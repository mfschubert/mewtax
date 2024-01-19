"""Tests for mewtax."""

import functools
import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from jax import flatten_util, tree_util
from parameterized import parameterized

import mewtax


def minimize_newton_naive(fn, params, z_init, tol=1e-5, max_iter=20):
    """A naive implementation of the Newton method."""
    z, unflatten_z_fn = flatten_util.ravel_pytree(z_init)
    params, _ = flatten_util.ravel_pytree(params)
    for _ in range(max_iter):
        grad = jax.grad(fn, argnums=1)(params, z)
        holomorphic = jnp.iscomplexobj(grad)
        hessian = jax.jacfwd(
            jax.jacrev(fn, argnums=1), argnums=1, holomorphic=holomorphic
        )(params, z)
        delta = jnp.linalg.solve(hessian, grad.conj())
        if not (jnp.linalg.norm(delta) > tol):
            break
        z = z - delta
    return unflatten_z_fn(z)


def jacfwd_fd(fn, delta=1e-6):
    """Forward mode jacobian by finite differences."""
    # This function is taken from the fmmax `test_utils` module.

    def _jac_fn(x):
        f0 = fn(x)
        jac = jnp.zeros(f0.shape + x.shape, dtype=f0.dtype)
        for inds in itertools.product(*[range(dim) for dim in x.shape]):
            offset = jnp.zeros_like(x).at[inds].set(delta)
            grad = (fn(x + offset / 2) - fn(x - offset / 2)) / delta
            jac_inds = tuple([slice(0, d) for d in f0.shape]) + inds
            jac = jac.at[jac_inds].set(grad)
        return jac

    return _jac_fn


def quadratic_loss_fn(params, z):
    loss_tree = tree_util.tree_map(lambda a, b: jnp.sum(jnp.abs(a - b) ** 2), params, z)
    return jnp.sum(jnp.asarray(tree_util.tree_leaves(loss_tree)))


def quartic_loss_fn(params, z):
    loss_tree = tree_util.tree_map(lambda a, b: jnp.sum(jnp.abs(a - b) ** 4), params, z)
    return jnp.sum(jnp.asarray(tree_util.tree_leaves(loss_tree)))


class MinimizeNewtonTest(unittest.TestCase):
    @parameterized.expand(
        [
            [quadratic_loss_fn, jnp.arange(10).astype(float)],
            [quadratic_loss_fn, jnp.arange(30).reshape(2, 5, 3).astype(float)],
            [
                quadratic_loss_fn,
                {
                    "a": jnp.arange(5).astype(float),
                    "b": jnp.arange(5, 10).astype(float),
                    "c": jnp.arange(10, 20).reshape(2, 5).astype(float),
                },
            ],
            [quadratic_loss_fn, jnp.arange(10) * (1 + 0j)],
            [quadratic_loss_fn, jnp.arange(10) * 1j],
            # Quartic loss functions have more error, and we use looser test tolerances.
            [quartic_loss_fn, jnp.arange(10) * (1 + 1j), 5e-3, 5e-3],
            [quartic_loss_fn, jnp.arange(10).astype(float), 5e-3, 5e-3],
            [
                quartic_loss_fn,
                jnp.arange(30).reshape(2, 5, 3).astype(float),
                5e-3,
                5e-3,
            ],
            [
                quartic_loss_fn,
                {
                    "a": jnp.arange(5).astype(float),
                    "b": jnp.arange(5, 10).astype(float),
                    "c": jnp.arange(10, 20).reshape(2, 5).astype(float),
                },
                5e-3,
                5e-3,
            ],
            [quartic_loss_fn, jnp.arange(10) * (1 + 0j), 5e-3, 5e-3],
            [quartic_loss_fn, jnp.arange(10) * 1j, 5e-3, 5e-3],
            [quartic_loss_fn, jnp.arange(10) * (1 + 1j), 5e-3, 5e-3],
        ]
    )
    def test_solution_matches_expected(self, loss_fn, params, rtol=1e-6, atol=1e-6):
        keys = tree_util.tree_unflatten(
            tree_util.tree_structure(params),
            jax.random.split(
                jax.random.PRNGKey(0), num=len(tree_util.tree_leaves(params))
            ),
        )
        z_init = tree_util.tree_map(
            lambda k, p: jax.random.normal(k, p.shape).astype(p.dtype),
            keys,
            params,
        )
        z_star = mewtax.minimize_newton(fn=loss_fn, params=params, z_init=z_init)

        for a, b in zip(tree_util.tree_leaves(z_star), tree_util.tree_leaves(params)):
            onp.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

    @parameterized.expand(
        [
            [quadratic_loss_fn, jnp.arange(4).astype(float)],
            [quadratic_loss_fn, jnp.asarray([0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])],
            [quadratic_loss_fn, jnp.arange(30).reshape(2, 5, 3).astype(float)],
            [
                quadratic_loss_fn,
                {
                    "a": jnp.arange(5).astype(float),
                    "b": jnp.arange(5, 10).astype(float),
                    "c": jnp.arange(10, 20).reshape(2, 5).astype(float),
                },
            ],
            [quadratic_loss_fn, jnp.arange(4) * (1 + 0j)],
            [quadratic_loss_fn, jnp.arange(10) * 1j],
            [quartic_loss_fn, jnp.arange(10) * (1 + 1j)],
            [quartic_loss_fn, jnp.arange(10).astype(float)],
            [quartic_loss_fn, jnp.arange(30).reshape(2, 5, 3).astype(float)],
            [
                quartic_loss_fn,
                {
                    "a": jnp.arange(5).astype(float),
                    "b": jnp.arange(5, 10).astype(float),
                    "c": jnp.arange(10, 20).reshape(2, 5).astype(float),
                },
            ],
            [quartic_loss_fn, jnp.arange(10) * (1 + 0j)],
            [quartic_loss_fn, jnp.arange(10) * 1j],
            [quartic_loss_fn, jnp.arange(10) * (1 + 1j)],
        ]
    )
    def test_gradient_matches_naive(self, loss_fn, params, rtol=5e-3, atol=5e-3):
        keys = tree_util.tree_unflatten(
            tree_util.tree_structure(params),
            jax.random.split(
                jax.random.PRNGKey(0), num=len(tree_util.tree_leaves(params))
            ),
        )
        z_init = tree_util.tree_map(
            lambda k, p: jax.random.normal(k, p.shape).astype(p.dtype),
            keys,
            params,
        )
        z_init = tree_util.tree_map(jnp.zeros_like, params)

        def meta_loss_fn(params):
            z_star = mewtax.minimize_newton(fn=loss_fn, params=params, z_init=z_init)
            z_star, _ = flatten_util.ravel_pytree(z_star)
            return jnp.sum(jnp.abs(z_star) ** 2)

        def naive_meta_loss_fn(params):
            z_star = minimize_newton_naive(fn=loss_fn, params=params, z_init=z_init)
            z_star, _ = flatten_util.ravel_pytree(z_star)
            return jnp.sum(jnp.abs(z_star) ** 2)

        value, grad = jax.value_and_grad(meta_loss_fn)(params)
        naive_value, naive_grad = jax.value_and_grad(naive_meta_loss_fn)(params)

        params_flat, _ = flatten_util.ravel_pytree(params)
        expected_value = jnp.sum(jnp.abs(params_flat) ** 2)
        onp.testing.assert_allclose(value, expected_value, rtol=1e-3)
        onp.testing.assert_allclose(naive_value, expected_value, rtol=1e-3)

        for a, b in zip(tree_util.tree_leaves(grad), tree_util.tree_leaves(naive_grad)):
            onp.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

    @parameterized.expand(
        [
            [(jnp.arange(4) - 1.5) * 0.1],
            [(jnp.arange(4) - 1.5) * (0.1 + 0j)],
            [(jnp.arange(4) - 1.5) * (0 + 0.1j)],
            [(jnp.arange(4) - 1.5) * (0.1 + 0.1j)],
        ]
    )
    def test_gradient_matches_finite_difference(self, params):
        # Verifies that the Newton gradients match finite difference gradients.
        def loss_fn(params, x):
            return jnp.sum(jnp.abs(x - params) ** 2)

        def fn(params):
            return mewtax.minimize_newton(
                fn=loss_fn,
                params=params,
                z_init=jnp.zeros_like(params),
            )

        fd_grad = jacfwd_fd(fn, delta=1e-4)(params)
        grad = jax.jacrev(fn, holomorphic=jnp.iscomplexobj(params))(params)
        onp.testing.assert_allclose(grad, fd_grad, rtol=1e-2)

    @parameterized.expand(
        [
            [(jnp.arange(4) - 1.5) * 0.1],
            [(jnp.arange(4) - 1.5) * (0.1 + 0j)],
            [(jnp.arange(4) - 1.5) * (0 + 0.1j)],
            [(jnp.arange(4) - 1.5) * (0.1 + 0.1j)],
        ]
    )
    def test_gradient_matches_finite_difference_naive(self, params):
        # Verifies that the naive Newton gradients match finite difference gradients.
        def loss_fn(params, x):
            return jnp.sum(jnp.abs(x - params) ** 2)

        def fn(params):
            return minimize_newton_naive(
                fn=loss_fn,
                params=params,
                z_init=jnp.zeros_like(params),
            )

        fd_grad = jacfwd_fd(fn, delta=1e-4)(params)
        grad = jax.jacrev(fn, holomorphic=jnp.iscomplexobj(params))(params)
        onp.testing.assert_allclose(grad, fd_grad, rtol=1e-2)

    def test_gradient_with_respect_to_z_init_is_zero(self):
        def loss_fn(params, x):
            return jnp.sum(jnp.abs(x - params) ** 2)

        def fn(z_init):
            return minimize_newton_naive(
                fn=loss_fn,
                params=jnp.arange(4).astype(float),
                z_init=z_init,
            )

        grad = jax.jacrev(fn)(jnp.ones((4,)))
        onp.testing.assert_array_equal(grad, jnp.zeros((4, 4)))

# Optimization algorithms

## Gradient Descent

One of the simplest optimization algorithms, gradient descent works by constructing a first-order
approximation to a given cost function at a specific point and stepping opposite the direction of the 
gradient to find a lower value of the cost function. The guess updates iteratively until a specified number of
iterations are reached, the gradient is of sufficiently small magnitudes, or the change in the cost function
between steps is sufficiently small. 

We define our cost function such that it maps an $n$-dimensional vector input $\mathbf{x}_n$
to a scalar cost we aim to minimize:

$$
F : \mathbb{R}^n \to \mathbb{R}
$$

Not yet written

## Newton's Method

Gradient descent is useful and relatively straightforward, yet by virtue of only being a first-order approximation,
it misses out on compensating for the curvature of a function. To get more information about a cost function's local
behavior, the next step is to take the second derivatives of the function, thereby encoding more information about the
cost function. 

For a one-dimensional function, this approximation is just the Taylor polynomial:

$$
f(x) = f(a) + f'(a)(x - a) + 0.5 f''(a)(x - a)^2
$$

Generalized for higher-dimensional inputs, the same approximation becomes:

$$
f(\mathbf{x} + \mathbf{d}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{d} + 0.5\, \mathbf{d}^\top \mathbf{H} \mathbf{d}
$$

for a step in direction $\mathbf{d}$ from $\mathbf{x}$. Here, the matrix $\mathbf{H}$ represents the Hessian matrix,
containing all the second derivatives of the cost function at our given point. 

Just as with gradient descent, we want to choose the right step $\mathbf{d}$ that will help us
minimize the right side of the equation. To make the notation easier, it also helps to let $f(\mathbf{x} + \mathbf{d})$
be equivalent to the value of a new function $m$ at $\mathbf{d}$.

Setting the derivative of the right side with respect to $\mathbf{d}$ equal to zero, we get:

$$
\nabla f(\mathbf{x}) + \mathbf{H}\mathbf{d} = 0 \quad \Rightarrow \quad \mathbf{d} = -\mathbf{H}^{-1} \nabla f(\mathbf{x})
$$

Unlike with first-order gradient descent, there's no step size parameter to set here! However, in practice,
adding a step size factor makes the algorithm more stable, so we'll add a step size factor $\gamma$. Now, our optimization step looks
as follows:

$$
x_{k+1} = x_k - \gamma \mathbf{H}^{-1} \nabla f(\mathbf{x})
$$

In theory, that's all there is to using Newton's method for optimization! However, it turns out that going at this calculation directly
is a pain, and as such, Newton's method is rarely used *exactly*. Instead, there are more practical algorithms
that offer many of the same advantages.

## Quasi-Newton Methods

Not yet written

## Broyden-Fletcher-Goldfarb-Shanno (BFGS) Algorithm

Not yet written
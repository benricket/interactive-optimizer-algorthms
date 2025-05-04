---
layout: default
title: Home
---

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

The most straightforward approach is to approximate our cost function with a first-order approximation (i.e. a plane).
To do this, we compute the gradient of the cost function $f$ at a specific "guess" for our optimization variables
$\mathbf{x}$ for iteration $k$. Our approximation at a point $\mathbf{x}_k + \mathbf{d}$ is then given by:

$$
f(\mathbf{x}_k + \mathbf{d}) = f(\mathbf{x}_k) + \nabla f(\mathbf{x}_k)^\top \mathbf{d}
$$

We want to choose a step $\mathbf{d}$ that minimizes $f(\mathbf{x}_k + \mathbf{d})$. As the rightmost term describes
the dot product of the gradient of $f$ with $\mathbf{d}$, we know the dot product is minimized when the two
vectors have opposite directions --- thus, $\mathbf{d}$ should step directly away from 
$\nabla f(\mathbf{x}_k)$, and can therefore be expressed as a scalar $\alpha$, referred to as the step size,
multiplied by the negative of the cost function gradient, $-\nabla f(\mathbf{x}_k)$. As such, our next iteration guess 
can be expressed as:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)
$$

This still leaves us choice in selecting a value for $\alpha$. Small values let us update our approximation more frequently,
but will take longer to converge; large values can converge more quickly but also run the risk of diverging.

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
f(\mathbf{x} + \mathbf{d}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{d} + 0.5\ \mathbf{d}^\top \mathbf{H} \mathbf{d}
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

Newton's method may sound great, but the issue is with the matrix inversion step. Straightforward approaches have a time complexity 
of $O(n^3)$, and while there do exist methods for performing the inversion a bit more efficiently, it's still a very intensive
process. For a large matrix, it isn't particularly feasible. Therefore, quasi-Newton methods are built around
using an approximation to the inverse Hessian that can be more efficiently computed.

A reasonable goal for these methods is that the objective function approximation match the gradient of the objective function both
at the initial guess $f(\mathbf{x}\_k+\mathbf{d})$, as well as the next guess in the algorithm progress, 
$f(\mathbf{x}\_{k+1} + \mathbf{d})$. This is equivalent to asking that the approximation $m(\mathbf{d})$ have a gradient
that matches the objective function gradient for $\mathbf{d}\_{k+1} = 0$ as well as $\mathbf{d}_{k+1} = - \gamma \mathbf{d}_k$. In other words, in
our second-order approximation, we want to make sure the first-order approximation (the gradient) is accurate at our current step
and our previous step --- this way, we can assume it isn't that far off along the distance we step. Let's look back at our approximation, 
and let $\mathbf{B}$ be our Hessian.

$$
m_{k+1}(\mathbf{d}) = f(\mathbf{x}\_{k+1}) + \nabla f(\mathbf{x}\_{k+1})^\top \mathbf{d} + 0.5 \mathbf{d}^\top \mathbf{B}_{k+1} \mathbf{d}
$$

Take the gradient with respect to $\mathbf{d}$, and we get:

$$
\nabla m_{k+1}(\mathbf{d}) = \nabla f(\mathbf{x}\_{k+1}) + \mathbf{B}_{k+1} \mathbf{d}
$$

Substitute in our first condition, that our approximation match the gradient at the current step, we substitute $\mathbf{d} = 0$ and find:

$$
\nabla m_{k+1}(\mathbf{0}) = \nabla f(\mathbf{x}_{k+1})
$$

In other words, because $\mathbf{B}$ doesn't appear in this equation, we know
our approximation will match the gradient of $f$ at the current point no matter how we approximate $\mathbf{B}$. 

What about the second equation? Substituting in $\mathbf{d} = - \gamma \mathbf{d}_k$, 
and setting it equal to $\nabla f(\mathbf{x}_k)$, we get:

$$
\nabla m(-\gamma \mathbf{d}\_{k}) = \nabla f(\mathbf{x}\_k) - \gamma \mathbf{B}\_{k+1} \mathbf{d}_{k} = \nabla f(\mathbf{x}_k)
$$

We can rearrange this equation, and substitute $\gamma \mathbf{d}$ (our step) for $(\mathbf{x}_{k+1} - \mathbf{x}_k)$, to find:

$$
\mathbf{B}\_{k+1} (\mathbf{x}\_{k+1} - \mathbf{x}\_k) = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)
$$

That is, multiplying our step (the difference in position between iterations) by our approximate Hessian should give the difference in
objective function gradients between iterations. If this condition is satisfied, we know that our approximation matches the objective function
gradient at both our current step as well as our previous step. For ease of notation, we can express our step as $\mathbf{s}_k$ and our difference
in objective function gradient as $\mathbf{y}_k$, simplifying the previous expression to:

$$
\mathbf{B}_{k+1} \mathbf{s}_k = \mathbf{y}_k
$$

In the context of optimization, this is called the **secant equation**, and is what quasi-Newton methods aim to satisfy.
Together with the requirement that $\mathbf{B}$ be symmetric and thus $\mathbf{B} = \mathbf{B}^\top$, we have two equality constraints
for our choice of the next matrix $\mathbf{B}\_{k+1}$. To narrow our choices further, choosing a specific matrix, we'll introduce the condition
that $\mathbf{B}$ not change too rapidly --- therefore, we'll choose $\mathbf{B}\_{k+1}$ such that 
$||\mathbf{B}_{k+1} - \mathbf{B}_k||$ is minimized. Which norm we choose to use for this minimization determines the actual quasi-Newton method
we'll end up using.

## Broyden-Fletcher-Goldfarb-Shanno (BFGS) Algorithm


To reiterate the point from above, our main goal with this algorithm is to avoid the costly matrix inversion step that would have us 
invert $\mathbf{B}$. If we go from one approximate-Hessian to another, we'll still have to invert these Hessians, and we won't save 
ourselves much time at all. Instead, it makes sense to try to sidestep this inversion altogether, and instead move from one
approximate-inverse-Hessian to the next, given that the inverted Hessian is all we care about. 

Towards that point, we'll let $\mathbf{H}\_k$ represent the approximate inverse Hessian; i.e., $\mathbf{B}\_{k}^{-1}$. Most of our previous 
constraints still carry over --- $\mathbf{H}\_{k+1}$ should be close to $\mathbf{H}\_k$, and if $\mathbf{H}$ is 
symmetric, we know $\mathbf{B}$ is symmetric as well. The one thing we need to change is our secant equation 
$\mathbf{B}\_{k+1} \mathbf{s}\_k = \mathbf{y}\_k$. Multiplying each side by $\mathbf{B}\_{k+1}^\top = \mathbf{H}\_{k+1}$, our secant equation turns
into $\mathbf{H}_{k+1} \mathbf{y}_k = \mathbf{s}_k$.

Calculating a value for $\mathbf{H}_{k+1}$ therefore boils down to choosing a useful norm for the minimization. For BFGS,
the norm chosen is called a weighted Frobenius norm. Whereas a Frobenius norm is analogous to a Euclidean norm for a matrix, and
simply consists of summing all the squared values of each element in the matrix, the weighted Frobenius norm 
left and right multiplies the matrix by a consciously chosen matrix of weights before computing the Frobenius norm.

Carefully choosing this weight yields an update step that depends only on $\mathbf{y}_k$, $\mathbf{s}_k$, and $\mathbf{H}_k$:

$$
\mathbf{H}_{k+1} = (\mathrm{I} - \rho_k \mathbf{s}_k \mathbf{y}_k^\top)\mathbf{H}_k(\mathrm{I} - \rho_k \mathbf{y}_k \mathbf{s}_k^\top) + \rho_k \mathbf{s}_k \mathbf{s}_k^\top
$$

$$
\rho_k = \frac{1}{\mathbf{y}_k^\top \mathbf{s}_k}
$$

This can be shown to satisfy the secant equation --- multiplying by $\mathbf{y}_k$ reduces the right term to $\mathbf{s}_k$
once the dot products cancel, and reduces the second pair of parentheses to $(\mathrm{I}\mathbf{y} - \mathbf{y})$, canceling all but the 
last terms. Given this update equation, we can now iteratively update our approximate inverse Hessian matrix without any
matrix inversion, and the resulting function approximation is guaranteed to match the objective function gradient at the two most recent
guesses. There's a bit of math that goes into implementing a backtracking line search to select a good value for 
$\gamma$ at every iteration, but other than that, this is the BFGS algorithm, and it, along with some further extensions, is 
implemented in many standard optimization packages. 

### Limited memory BFGS

As a side note, the implementation of BFGS as described above can still be expensive in terms of memory.
The matrix $\mathbf{H}$, which needs to be stored, grows in space on the order of $n^2$ for an $n$-dimensional input space,
which, while better than growing cubically, can still be undesirable for problems with very large numbers of dimensions.

The limited-memory version of BFGS relies on the fact that only two vectors are used to update $\mathbf{H}$ every iteration,
and that, from a given starting point, the state of $\mathbf{H}$ after $k$ iterations can be stored as an initial starting point plus
$k$ occurrences of each of the update vectors. For instance, suppose our input space has 1000 dimensions. While the matrix $\mathbf{H}$
has $1000^2$ entries, we can instead start from an easy to store guess (perhaps a diagonal $\mathbf{H}_0$) and update it using a number of
1000-element vectors. If we choose to keep only a certain amount of previous steps in history, we can significantly reduce the 
memory usage of the algorithm. 

## Symmetric Rank 1 Update (SR1)

Another quasi-Newton method that builds off of the secant equation is SR1, named for the fact that the update step
always adds a symmetric matrix of rank 1 to the previous approximate Hessian or approximate inverse Hessian. (BFGS,
in contrast, has a rank 2 update.) For an inverse Hessian approximation, the update step of the algorithm is given as:

$$
\mathbf{H}_{k+1} = \mathbf{H}_k + \frac{(\mathbf{s}_k - \mathbf{H}_k \mathbf{y}_k)(\mathbf{s}_k - \mathbf{H}_k \mathbf{y}_k
)^\top}{(\mathbf{s}_k - \mathbf{H}_k \mathbf{y}_k)^\top \mathbf{y}_k}
$$

In this equation, the added term is a rank 1 matrix (vector times its transposed) divided by a scalar. The scalar term
in the denominator can make the update less stable --- there are no direct safeguards that 
$(\mathbf{s}_k - \mathbf{H}_k \mathbf{y}_k)^\top \mathbf{y}_k$ be non-zero, and if this approaches zero, the update 
step can explode unreasonably large. As such, in actual use, the update is generally only performed if the denominator is 
sufficiently large, lest a small denominator introduce a lot of instability into the optimizer. 

This denominator, however, is part of what makes this algoritm useful. With BFGS, the approximate Hessian is constructed
in a way that ensures the Hessian is always positive-definite --- in other words, it only has positive eigenvalues. This works
so long as functions are convex, but if all eigenvalues of the Hessian are positive, it can't describe any function
with negative curvature or saddle points. Locally, it may still perform fine, but this limitation restricts how useful
BFGS can be in optimizing over non-convex cost functions.

SR1, however, can have negative eigenvalues because of how the denominator in the update step can be negative. This allows 
it to more closely approximate Hessians with negative curvature, and therefore helps the optimizer perform better on these 
sorts of cost functions.

Despite its instability, SR1 can be used as an update step in optimization algorithms like trust-region, which enforce a 
radius in which the next update must stay (the trust region). Based on how accurate the prediction is found to be, the radius
of this trust region can be increased or decreased. With an update like SR1, the instability is mitigated by the trust region
imposing a bound on each update step, helping avoid the potential drawbacks. 



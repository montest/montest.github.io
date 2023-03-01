---
title: 'Optimal Quantization with PyTorch - Part 1: Implementation of Stochastic Lloyd Method'
collection: blog posts
excerpt: "




TODO



All explanations are accompanied by some code examples in Python and is available in the following Github repository: [montest/stochastic-methods-optimal-quantization](https://github.com/montest/stochastic-methods-optimal-quantization)."
date: 2023-03-01
permalink:  /:year/:month/:day/:title/
bibliography: bibli.bib  
tags:
  - PyTorch
  - Numerical Probability
  - Optimization
  - Optimal Quantization
---


Table of contents
======
{:.no_toc}


* TOC
{:toc}

Introduction
======

In this post, I present a PyTorch implementation of the stochastic version of the Lloyd algorithm in order to build Optimal Quantizers of $X$, a random variable of dimension one. The use of PyTorch allows me perform all the numerical computations on GPU and drastically increase the speed of the algorithm. I compare the mono-threaded implementation I made in numpy in a [previous blog post][blog_post_stochastic_methods] with the PyTorch version and study how it scales.

All the code presented in this blog post is available in the following Github repository: [montest/stochastic-methods-optimal-quantization](https://github.com/montest/stochastic-methods-optimal-quantization)

Short Reminder
======

In this part, I quickly remind how to build an optimal quantizer using the Monte-Carlo simulation-based Lloyd procedure with a focus on the $1$-dimensional case. To get more background on the notations and the theory, do not hesitate to check-out my previous blog articles on Stochastic and Deterministic methods for building optimal quantizers.

### Voronoï Quantization

A Voronoï tesselation is a way, given a set of points called centroids in $\mathbb{R}$, to divide the real line into regions in such a way that: for each cell, all the points in it are closer to the centroid associated to the cell than any other centroid. 

In particular, given a quantizer of size N: $\Gamma_N = \big\{ x_{1}^{N}, \dots , x_{N}^{N} \big\}$ where $x_i^{N}$ are the centroids. If we consider that the centroids $(x_i^{N})_i$ are ordered: $$x_1^{N} < x_2^{N} < \cdots < x_{N-1}^{N} < x_{N}^{N} $$, then the Voronoï cells $C_i (\Gamma_N)$ are intervals in $\mathbb{R}$ and are defined by

$$
	C_i ( \Gamma_N ) =
    \left\{ \begin{aligned}
        & \big( x_{i - 1/2}^N , x_{i + 1/2}^N \big] &\qquad i = 1, \dots, N-1 \\
        & \big( x_{i - 1/2}^N , x_{i + 1/2}^N \big) & i = N
    \end{aligned} \right.
$$

where the vertices $x_{i-1/2}^N$ are defined by

$$
    \forall i = 2, \dots, N, \qquad x_{i-1/2}^N := \frac{x_{i-1}^N + x_i^N}{2}
$$

and

$$
    x_{1/2}^N := \textrm{inf} (\textrm{supp} (\mathbb{P}_{_{X}})) \, \textrm{ and } \, x_{N+1/2}^N := \textrm{sup} (\textrm{supp} (\mathbb{P}_{_{X}})).
$$


Now, going back to our initial problem: let $X$ be a random variable. In simple terms, an optimal quantization of a random vector $X$ is the best approximation of $X$ by a discrete random vector $\widehat X^N$ with cardinality at most $N$.

More precisely, a **Voronoï quantization** of $X$ by $\Gamma_{N}$, $\widehat X^N$, is defined as nearest neighbor projection of $X$ onto $\Gamma_{N}$ associated to a Voronoï partition $\big( C_{i} (\Gamma_{N}) \big)_{i =1, \dots, N}$ for the euclidean norm

$$
	\widehat X^N := \textrm{Proj}_{\Gamma_{N}} (X) = \sum_{i = 1}^N x_i^N \mathbb{1}_{X \in C_{i} (\Gamma_N) }
$$

and its associated **probabilities**, also called weights, are given by

$$
	\mathbb{P} \big( \widehat X^N = x_i^N \big) = \mathbb{P}_{_{X}} \big( C_{i} (\Gamma_N) \big) = \mathbb{P} \big( X \in C_{i} (\Gamma_N) \big).
$$


For example, for a list of centroid `centroids` ( $\Gamma_N$) and a given point `p`, the closest centroid to `p` can be find using the following method that returns the index `i` of the closest centroid and the distance between this centroid `x_i` and `p`.

````python
from typing import List
Point = np.ndarray

def find_closest_centroid(centroids: List[Point], p: Point):
    index_closest_centroid = -1
    min_dist = sys.float_info.max
    for i, x_i in enumerate(centroids):
        dist = np.linalg.norm(x_i - p)
        if dist < min_dist:
            index_closest_centroid = i
            min_dist = dist
    return index_closest_centroid, min_dist
````


### Optimal quantization


Now, we can define what an optimal quantization of $X$ is: we are looking for the best approximation of $X$ in the sense that we want to minimize the distance between $X$ and $\widehat X^N$. This distance is measured by the standard $L^2$ norm, denoted $\Vert X - \widehat X^N \Vert_{_2}$, and is called the mean quantization error. But, more often, the quadratic distortion defined as half of the square of the mean quantization error is used.

#### Definition
{:.no_toc}

The quadratic distortion function at level $N$ induced by an $N$-tuple $x := (x_1^N, \dots, x_N^N)$ is given by

$$
	\mathcal{Q}_{2,N} : x \longmapsto \frac{1}{2} \mathbb{E} \Big[ \min_{i = 1, \dots, N} \vert X - x_i^N \vert^2 \Big] = \frac{1}{2} \mathbb{E} \Big[ \textrm{dist} (X, \Gamma_N )^2 \Big] = \frac{1}{2} \Vert X - \widehat X^N \Vert_{_2}^2 .
$$

Of course, the above result can be extended to the $L^p$ case by considering the $L^p$-mean quantization error in place of the quadratic one.



Thus, we are looking for quantizers $\widehat X^N$ taking value in grids $\Gamma_N$ of size $N$ which minimize the quadratic distortion

$$
	\min_{\Gamma_N \subset \mathbb{R}^d, \vert \Gamma_N \vert \leq N } \Vert X - \widehat X^N \Vert_{_2}^2.
$$


Classical theoretical results on optimal quantizer can be found in {% cite graf2000foundations pages2018numerical %}. Check those books if you are interested in results on existence and uniqueness of optimal quantizers or if you want further details on the asymptotic behavior of the distortion (such as Zador's Theorem).

### How to build an optimal quantizer?

In this part, I will focus on how to build an optimal quadratic quantizer or, equivalently, find a solution to the following minimization problem

$$
	\textrm{arg min}_{(\mathbb{R}^d)^N} \mathcal{Q}_{2,N}.
$$

For that, let's differentiate the distortion function $\mathcal{Q}_{2,N}$. The gradient $\nabla \mathcal{Q}_{2,N}$ is given by

$$
\nabla \mathcal{Q}_{2,N} (x) = \bigg[ \int_{C_i (\Gamma_N)} (x_i^N - \xi ) \mathbb{P}_{_{X}} (d \xi) \bigg]_{i = 1, \dots, N } = \Big[ \mathbb{E}\big[ \mathbb{1}_{X \in C_i (\Gamma_N)} ( x_i^N - X ) \big] \Big]_{i = 1, \dots, N }.
$$

The latter expression is useful for numerical methods based on deterministic procedures while the former featuring a local gradient is handy when we work with stochastic algorithms, which is the case in this post.

Two main stochastic algorithms exist for building an optimal quantizer in $\mathbb{R}^d$. The first is a fixed-point search, called Lloyd method, see {% cite lloyd1982least pages2003optimal%} or **K-means** in the case of unsupervised learning and the second is a stochastic gradient descent, called Competitive Learning Vector Quantization (CLVQ) or also Kohonen algorithm see {% cite pages1998space fort1995convergence %}.



Lloyd method
------

Starting from the previous equation, when we search a zero of the gradient, we derive a fixed-point problem. Let $\Lambda_i : \mathbb{R}^N \mapsto \mathbb{R}$ defined by

$$
\Lambda_i (x) = \frac{\mathbb{E}\big[ X \mathbb{1}_{ X \in C_i (\Gamma_N)} \big]}{\mathbb{P} \big( X \in C_i (\Gamma_N) \big)}
$$

then

$$
\nabla \mathcal{Q}_{2,N} (x) = 0 \quad \iff \quad \forall i = 1, \dots, N  \qquad x_i = \Lambda_i ( x ).
$$

Hence, from this equality, we deduce a fixed-point search algorithm. This method, known as the **Lloyd method**, was first devised by Lloyd in {% cite lloyd1982least %}. Let $x^{[n]}$ be the quantizer of size $N$ obtained after $n$ iterations, the Lloyd method with initial condition $x^0$ is defined as follows

$$
x^{[n+1]} = \Lambda \big( x^{[n]} \big).
$$

In our setup, in absence of deterministic methods for computing the expectations, they will be approximated using Monte-Carlo simulation. Let $\xi_1, \dots, \xi_M$ be independent copies of $X$, the stochastic version of $\Lambda_i$  is defined by

$$
\Lambda_i^M ( x ) = \frac{\displaystyle \sum_{m=1}^M \xi_m \mathbb{1}_{ \big\{ \textrm{Proj}_{\Gamma_N} (\xi_m) = x_i^N \big\} } }{\displaystyle \sum_{m=1}^M \mathbb{1}_{ \big\{ \textrm{Proj}_{\Gamma_N} (\xi_m) = x_i^N \big\} } }. % \qquad \mbox{with} \qquad \Gamma_N = \{ x_1^N, \dots, x_N^N \}
$$

Hence, the $n+1$ iteration of the Randomized Lloyd method is given by

$$
x^{[n+1]} = \Lambda^M \big( x^{[n]} \big).
$$

During the optimization of the quantizer it is possible to compute the weight $p_i^N$ and the local distortion $q_i^N$ associated to a centroid defined by

$$
p_i^N = \mathbb{P} \big( X \in C_i (\Gamma_N) \big) \quad \mbox{ and } \quad q_i^N = \mathbb{E}\big[ (X - x_i^N)^2 \mathbb{1}_{X \in C_i (\Gamma_N)} \big].
$$

I give below a Python code example for the Randomized Lloyd method that takes as input the quantizer $x^{[n]}$ and $M$ samples $(\xi_m)_{m = 1, \dots, M}$ of $X$ and returns $x^{[n+1]}$, the weights and the local-distortion approximated using Monte-Carlo.

```python
import numpy as np

def fixed_point_iteration(centroids, xs: List[Point]):
    N = len(centroids)  # Size of the quantizer
    M = len(xs)  # Number of samples

    # Initialization step
    local_mean = np.zeros((N, 2))
    local_count = np.zeros(N)
    local_dist = 0.

    for x in xs:
        # find the centroid which is the closest to sample x
        index, l2_dist = find_closest_centroid(centroids, x)

        # Compute local mean, proba and distortion
        local_mean[index] = local_mean[index] + x
        local_dist += l2_dist ** 2  # Computing distortion
        local_count[index] += 1  # Count number of samples falling in cell 'index'

    for i in range(N):
        centroids[i] = local_mean[i] / local_count[i] if local_count[i] > 0 else centroids[i]

    probas = local_count / float(M)
    distortion = local_dist / float(2*M)

    return centroids, probas, distortion
```

Then, using `fixed_point_iteration` and starting from a initial guess $x^0$ of size $N$, we can build an optimal quantizer of a random vector $X$ as long as we have access to a random generator of $X$.

Here is a small code example for building an optimal quantizer of a gaussian random vector in dimension 2 where you can select `N` the size of the optimal quantizer, `M` the number of sample you want to generate and `nbr_iter` the number of fixed-point iterations you want to do using `M` samples each time.


```python
from tqdm import trange

def lloyd_method(N: int, M: int, nbr_iter: int):
    centroids = np.random.normal(0, 1, size=[N, 2])  # Initialize the Voronoi Quantizer

    with trange(nbr_iter, desc='Lloyd method') as t:
        for step in t:

            xs = np.random.normal(0, 1, size=[M, 2])  # Draw M samples of gaussian vectors

            centroids, probas, distortion = fixed_point_iteration(centroids, xs)  # Apply fixed-point search iteration
            t.set_postfix(distortion=distortion)

            # This is only useful when plotting the results
            save_results(centroids, probas, distortion, step, M, method='lloyd')

    make_gif(get_directory(N, M, method='lloyd'))
    return centroids, probas, distortion
```



[blog_post_stochastic_methods]: {% post_url 2022-02-13-StochasticMethodsForOptimQuantif %}
[blog_post_deterministic_methods]: {% post_url 2022-06-21-DeterministicdMethodsForOptimQuantifUnivariates %}

# References

{% bibliography --cited %}

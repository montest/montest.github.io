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

### Voronoï Quantization in dimension 1

Given a quantizer of size N: $$\Gamma_N = \big\{ x_{1}^{N}, \dots , x_{N}^{N} \big\}$$ where $x_i^{N}$ are the centroids. Keeping in mind that we are in the $1$-dimensional, if we consider that the centroids $(x_i^{N})_i$ are ordered: $$x_1^{N} < x_2^{N} < \cdots < x_{N-1}^{N} < x_{N}^{N}$$, then the Voronoï cells $C_i (\Gamma_N)$ are intervals in $\mathbb{R}$ and are defined by

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

Now, let $X$ be a random variable, a **Voronoï quantization** of $X$ by $\Gamma_{N}$, $\widehat X^N$, is defined as nearest neighbor projection of $X$ onto $\Gamma_{N}$ associated to a Voronoï partition $\big( C_{i} (\Gamma_{N}) \big)_{i =1, \dots, N}$ for the euclidean norm

$$
	\widehat X^N := \textrm{Proj}_{\Gamma_{N}} (X) = \sum_{i = 1}^N x_i^N \mathbb{1}_{X \in C_{i} (\Gamma_N) }
$$

and its associated **probabilities**, also called weights, are given by

$$
	\mathbb{P} \big( \widehat X^N = x_i^N \big) = \mathbb{P}_{_{X}} \big( C_{i} (\Gamma_N) \big) = \mathbb{P} \big( X \in C_{i} (\Gamma_N) \big).
$$


### Optimal quantization


In order to build an optimal quantizer of $X$, we are looking for the best approximation of $X$ in the sense that we want to find the quantizer $\widehat X^N$ which is the $$\arg \min$$ of the quadratic distortion function at level $N$ induced by an $N$-tuple $x := (x_1^N, \dots, x_N^N)$ given by

$$
	\mathcal{Q}_{2,N} : x \longmapsto \frac{1}{2} \Vert X - \widehat X^N \Vert_{_2}^2 .
$$


### Randomized Lloyd algorithm

One of the first method deployed in order to build optimal quantizers was the Lloyd method, which is a fixed-point search algorithm. Let $x^{[n]}$ be the quantizer of size $N$ obtained after $n$ iterations, the Randomized Lloyd method with initial condition $x^0$ is defined as follows

$$
x^{[n+1]} = \Lambda^M \big( x^{[n]} \big).
$$

where

$$
\Lambda_i^M ( x ) = \frac{\displaystyle \sum_{m=1}^M \xi_m \mathbb{1}_{ \big\{ \textrm{Proj}_{\Gamma_N} (\xi_m) = x_i^N \big\} } }{\displaystyle \sum_{m=1}^M \mathbb{1}_{ \big\{ \textrm{Proj}_{\Gamma_N} (\xi_m) = x_i^N \big\} } }. % \qquad \mbox{with} \qquad \Gamma_N = \{ x_1^N, \dots, x_N^N \}
$$

with $\xi_1, \dots, \xi_M$ be independent copies of $X$.

I detail below a Python code example, which is a optimized version of the code I detailed in my [previous blog post][blog_post_stochastic_methods] of the randomized Lloyd method using numpy.

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

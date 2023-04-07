---
title: 'Optimal Quantization with PyTorch - Part 2: Implementation of Stochastic Gradient Descent'
collection: blog posts
excerpt: "
<img align='left' src='/images/posts/quantization/pytorch/1d/stochastic_clvq_1d_method_comparison_M_1000000.svg' width='250' >
In this post, I present several PyTorch implementations of the Competitive Learning Vector Quantization algorithm (CLVQ) in order to build Optimal Quantizers of $X$, a random variable of dimension one. As seen in [my previous blog post][blog_post_pytorch_lloyd_stochastic], the use of PyTorch allows me perform all the numerical computations on GPU and drastically increase the speed of the algorithm. Moreover, in this article, I also take advantage of the autograd implementation in PyTorch allowing me to make use of all the optimizers in `torch.optim`.


All explanations are accompanied by some code examples in Python and is available in the following Github repository: [montest/stochastic-methods-optimal-quantization](https://github.com/montest/stochastic-methods-optimal-quantization)."
date: 2023-03-16
permalink:  /:year/:month/:day/:title/
bibliography: bibli.bib  
tags:
  - PyTorch
  - Numerical Probability
  - Optimization
  - Stochastic Gradient Descent
  - Optimal Quantization
---


Table of contents
======
{:.no_toc}


* TOC
{:toc}

Introduction
======
In this post, I present several PyTorch implementations of the Competitive Learning Vector Quantization algorithm (CLVQ) in order to build Optimal Quantizers of $X$, a random variable of dimension one. As seen in [my previous blog post][blog_post_pytorch_lloyd_stochastic], the use of PyTorch allows me perform all the numerical computations on GPU and drastically increase the speed of the algorithm. Moreover, in this article, I also take advantage of the autograd implementation in PyTorch allowing me to make use of all the optimizers in `torch.optim`. I compare the implementation I made in numpy in a [previous blog post][blog_post_stochastic_methods] with the PyTorch version and study how it scales.

All the codes presented in this blog post are available in the following Github repository: [montest/stochastic-methods-optimal-quantization](https://github.com/montest/stochastic-methods-optimal-quantization)

Short Reminder
======

Using the notations I used in my previous articles ([1][blog_post_pytorch_lloyd_stochastic], [2][blog_post_stochastic_methods] and [3][blog_post_deterministic_methods]). First, I remind the expression of the minimization problem we want to solve in order to build an optimal quantizer. Then, I detail the expression of the gradient of the distortion and, finally, the stochastic version of CLVQ algorithm used to build optimal quantizers.

### Distortion function

In order to build an optimal quantizer of $X$, we are looking for the best approximation of $X$ in the sense that we want to find the quantizer $\widehat X^N$ which is the $$\arg \min$$ of the quadratic distortion function at level $N$ induced by an $N$-tuple $x := (x_1^N, \dots, x_N^N)$ 

$$
	\textrm{arg min}_{(\mathbb{R})^N} \mathcal{Q}_{2,N},
$$

where $$\mathcal{Q}_{2,N}$$ given by

$$
	\mathcal{Q}_{2,N} : x \longmapsto \frac{1}{2} \Vert X - \widehat X^N \Vert_{_2}^2 .
$$

### Gradient of the distortion function

In order to find the $$\arg \min$$, we differentiate the distortion function $$\mathcal{Q}_{2,N}$$. The gradient $$\nabla \mathcal{Q}_{2,N}$$ is given by

$$
    \nabla \mathcal{Q}_{2,N} (x) = \bigg[ \int_{C_i (\Gamma_N)} (x_i^N - \xi ) \mathbb{P}_{_{X}} (d \xi) \bigg]_{i = 1, \dots, N } = \Big[ \mathbb{E}\big[ \mathbb{1}_{X \in C_i (\Gamma_N)} ( x_i^N - X ) \big] \Big]_{i = 1, \dots, N }.
$$

### Stochastic Competitive Learning Vector Quantization algorithm

In order to build an optimal, one can use a Stochastic Gradient Descent, also called the CLVQ algorithm, in order to build an optimal quantizer. Let $x^{[n]}$ be the quantizer of size $N$ obtained after $n$ iterations, the Stochastic CLVQ method with initial condition $x^0$ is defined as follows

$$
	x^{[n+1]} = x^{[n]} - \gamma_{n+1} \nabla \textit{q}_{2,N} (x^{[n]}, \xi_{n+1})
$$

with $\xi_1, \dots, \xi_n, \dots$ a sequence of independent copies of $X$ and

$$
    \nabla \textit{q}_{2,N} (x^{[n]}, \xi_{n+1}) = \Big( \mathbb{1}_{\xi_{n+1} \in C_i (\Gamma_N)} ( x_i^{[n]} - \xi_{n+1} ) \Big)_{1 \leq i \leq N}.
$$


Numpy Implementation
=======


I detail below a Python code example using numpy, which is an optimized version of the code I detailed in my [previous blog post][blog_post_stochastic_methods] of the stochastic CLVQ. It samples `M` samples of the distribution you want to quantize. Then it applies `num_epochs` times `M` gradient descent steps in order to build an optimal quantizer of size `N`.


```python
import numpy as np
from tqdm import trange
from utils import get_probabilities_and_distortion

def lr(N: int, n: int):
    a = 4.0 * N
    b = np.pi ** 2 / float(N * N)
    return a / float(a + b * (n+1.))

def clvq_method_dim_1(N: int, M: int, num_epochs: int, seed: int = 0):
    """
    Apply `nbr_iter` iterations of the Competitive Learning Vector Quantization algorithm in order to build an optimal
     quantizer of size `N` for a Gaussian random variable. This implementation is done using numpy.

    N: number of centroids
    M: number of samples to generate
    num_epochs: number of epochs of fixed point search
    seed: numpy seed for reproducibility

    Returns: centroids, probabilities associated to each centroid and distortion
    """
    ## To uncomment if you want to compute the probability and the distortion inline
    # probabilities = np.zeros(N)
    # distortion = 0.
    np.random.seed(seed)  # Set seed in order to be able to reproduce the results

    # Draw M samples of gaussian variable
    xs = np.random.normal(0, 1, size=M)

    # Initialize the Voronoi Quantizer randomly and sort it
    centroids = np.random.normal(0, 1, size=N)
    centroids.sort(axis=0)

    with trange(num_epochs, desc=f'CLVQ method - N: {N} - M: {M} - seed: {seed} (numpy)') as epochs:
        for epoch in epochs:
            for step in range(M):
                # Compute the vertices that separate the centroids
                vertices = 0.5 * (centroids[:-1] + centroids[1:])

                # Find the index of the centroid that is closest to each sample
                index_closest_centroid = np.sum(xs[step, None] >= vertices[None, :])
                ## To uncomment if you want to compute the probability and the distortion inline
                # l2_dist = np.linalg.norm(centroids[index_closest_centroid] - xs[step])

                gamma_n = lr(N, epoch*M + step)
                # gamma_n = 0.01

                # Update the closest centroid using the local gradient
                centroids[index_closest_centroid] = centroids[index_closest_centroid] - gamma_n * (centroids[index_closest_centroid] - xs[step])

            ## To uncomment if you want to compute the probability and the distortion inline
            #     # Update the distortion using gamma_n
            #     distortion = (1 - gamma_n) * distortion + 0.5 * gamma_n * l2_dist ** 2
            #
            #     # Update probabilities
            #     probabilities = (1 - gamma_n) * probabilities
            #     probabilities[index_closest_centroid] += gamma_n
            #
            #     if any(np.isnan(centroids)):
            #         break
            # epochs.set_postfix(distortion=distortion)

    probabilities, distortion = get_probabilities_and_distortion(centroids, xs)
    return centroids, probabilities, distortion
```

Compared to the version presented in a previous blog post, I do not resample for each epoch, I reuse the same `M` samples each time. Moreover, the probabilities and the distortion are computed at the end using the following method `get_probabilities_and_distortion` where `centroids` are the centroids and `xs` are the `M` samples.

```python
import torch
import numpy as np
from typing import Union

def get_probabilities_and_distortion(centroids: Union[np.ndarray, torch.tensor], xs: Union[np.ndarray, torch.tensor]):
    """
    Compute the probabilities and the distortion associated to `centroids` using the samples `xs`
    centroids: centroids of size `N`
    xs: `M` samples to use in order to compute the probabilities and the distortion

    Returns: probabilities associated to each centroid and distortion
    """
    centroids_ = centroids.clone().detach() if torch.is_tensor(centroids) else torch.tensor(centroids)
    xs_ = xs.clone().detach() if torch.is_tensor(xs) else torch.tensor(xs)
    M = len(xs_)
    vertices = 0.5 * (centroids_[:-1] + centroids_[1:])
    index_closest_centroid = torch.sum(xs_[:, None] >= vertices[None, :], dim=1).long()
    # Compute the probability of each centroid
    probabilities = torch.bincount(index_closest_centroid).to('cpu').numpy() / float(M)
    # Compute the final distortion between the samples and the quantizer
    distortion = torch.sum(torch.pow(xs_ - centroids_[index_closest_centroid], 2)).item() / float(2 * M)
    return probabilities, distortion
```


<!-- 

The advantage of this optimized version is twofold. First, it drastically reduces the computation time in order to build an optimal quantizer. Second, this new version is written is a more pythonic way compared to the one detailed in my [previous article][blog_post_stochastic_methods]. This simplifies greatly the conversion of this code to PyTorch, as you can see in the next section. -->


PyTorch Implementation
=======

<!-- Using the numpy code version written above, we can easily implement the Lloyd algorithm in PyTorch. The main difference is the usage of `torch.no_grad()` in order to make sure we don't accumulate the gradients in the tensors and before applying the fixed point iterator, we send the centroids and the samples to the chosen device: `cpu` or `cuda`.

As above, `lloyd_method_dim_1_pytorch` applies `nbr_iter` iterations of the fixed point function in order to build an optimal quantizer of a gaussian random variable where you can select `N` the size of the optimal quantizer, `M` the number of sample you want to generate. 

```python
import torch
from tqdm import trange

def lloyd_method_dim_1_pytorch(N: int, M: int, nbr_iter: int, device: str, seed: int = 0):
    """
    Apply `nbr_iter` iterations of the Randomized Lloyd algorithm in order to build an optimal quantizer of size `N`
    for a Gaussian random variable. This implementation is done using torch.

    N: number of centroids
    M: number of samples to generate
    nbr_iter: number of iterations of fixed point search
    device: device on which perform the computations: "cuda" or "cpu"
    seed: torch seed for reproducibility

    Returns: centroids, probabilities associated to each centroid and distortion
    """
    torch.manual_seed(seed=seed)  # Set seed in order to be able to reproduce the results

    with torch.no_grad():
        # Draw M samples of gaussian variable
        xs = torch.randn(M)
        # xs = torch.tensor(torch.randn(M), dtype=torch.float32)
        xs = xs.to(device)  # send samples to correct device

        # Initialize the Voronoi Quantizer randomly
        centroids = torch.randn(N)
        centroids, index = centroids.sort()
        centroids = centroids.to(device)  # send centroids to correct device

        for step in trange(nbr_iter, desc=f'Lloyd method - N: {N} - M: {M} - seed: {seed} (pytorch: {device})'):
            # Compute the vertices that separate the centroids
            vertices = 0.5 * (centroids[:-1] + centroids[1:])

            # Find the index of the centroid that is closest to each sample
            index_closest_centroid = torch.sum(xs[:, None] >= vertices[None, :], dim=1).long()

            # Compute the new quantization levels as the mean of the samples assigned to each level
            centroids = torch.tensor([torch.mean(xs[index_closest_centroid == i]) for i in range(N)]).to(device)

            if torch.isnan(centroids).any():
                break

        # Find the index of the centroid that is closest to each sample
        vertices = 0.5 * (centroids[:-1] + centroids[1:])
        index_closest_centroid = torch.sum(xs[:, None] >= vertices[None, :], dim=1).long()
        # Compute the probability of each centroid
        probabilities = torch.bincount(index_closest_centroid).to('cpu').numpy()/float(M)
        # Compute the final distortion between the samples and the quantizer
        distortion = torch.sum(torch.pow(xs - centroids[index_closest_centroid], 2)).item() / float(2 * M)
        return centroids.to('cpu').numpy(), probabilities, distortion
``` -->


Numerical experiments
=======

<!-- Now, I compare the average elapsed time of a fixed-point search iteration of the previous two algorithms. I analyze the computation time of the algorithms for different sample size (`M`), grid size (`N`) and devices for the PyTorch implementation.
All the tests were conducted on Google Cloud Platform on an instance `n1-standard-4` with 4 cores, 16 Go of RAM and a `NVIDIA T4` GPU. 

In order to reproduce those results, you can run the script `benchmark/run.py` in the GitHub repository [montest/stochastic-methods-optimal-quantization](https://github.com/montest/stochastic-methods-optimal-quantization).


In the left graph, I display, for each method, the average time of an iteration for several values of `N`.
In the right graph, I plot, for each `N`, the ratio between the average time for each method and the average time spent by PyTorch implementation using `cuda`. 


We can notice that when it comes to cpu-only computations, numpy is a better choice than PyTorch. However, when using the GPU, we notice that the PyTorch + cuda version is up to 20 times faster than the numpy implementation. And this is even more noticeable when we increase the sample size.


<center>
    <figcaption><font size=4>Methods comparison for M=200000</font></figcaption>
    <img alt="method_comparison_M_200000" src="/images/posts/quantization/pytorch/1d/stochastic_lloyd_1d_method_comparison_M_200000.svg" width=370 />
    <img alt="ratio_comparison_M_200000" src="/images/posts/quantization/pytorch/1d/stochastic_lloyd_1d_ratio_comparison_M_200000.svg" width=370 />
</center>
<br/><br/>
 
<center>
    <figcaption><font size=4>Methods comparison for M=500000</font></figcaption>
    <img alt="method_comparison_M_500000" src="/images/posts/quantization/pytorch/1d/stochastic_lloyd_1d_method_comparison_M_500000.svg" width=370 />
    <img alt="ratio_comparison_M_500000" src="/images/posts/quantization/pytorch/1d/stochastic_lloyd_1d_ratio_comparison_M_500000.svg" width=370 />
</center>
<br/><br/>

<center>
    <figcaption><font size=4>Methods comparison for M=1000000</font></figcaption>
    <img alt="method_comparison_M_1000000" src="/images/posts/quantization/pytorch/1d/stochastic_lloyd_1d_method_comparison_M_1000000.svg" width=370 />
    <img alt="ratio_comparison_M_1000000" src="/images/posts/quantization/pytorch/1d/stochastic_lloyd_1d_ratio_comparison_M_1000000.svg" width=370 />
</center> -->


[blog_post_pytorch_lloyd_stochastic]: {% post_url 2023-03-16-StochasticMethodsForOptimQuantifWithPyTorchPart1 %}
[blog_post_stochastic_methods]: {% post_url 2022-02-13-StochasticMethodsForOptimQuantif %}
[blog_post_deterministic_methods]: {% post_url 2022-06-21-DeterministicdMethodsForOptimQuantifUnivariates %}

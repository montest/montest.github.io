---
layout: archive
title: "Research Projects"
permalink: /research/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

{% for post in site.research %}
  {% include archive-single.html %}
{% endfor %}
<!-- ## Optimal Quantization -->

<!-- ### New error bounds for optimal quantization based cubature formula and weak error development (Paper in progress).
### Build a hybrid quantization tree for a Randomized Heston Model using Product Recursive Quantization (Paper in progress).
### Optimize existing methods in order to build optimal quantizers: Fixed Point Research Acceleration.

## Multilevel Monte-Carlo

### Optimizing xVA's risk (counterparty risk) computation using Multilevel Monte-Carlo that allows us to kill the bias while reducing the variance of the estimator.  
 -->

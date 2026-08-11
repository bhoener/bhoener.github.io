---
title: Flow Matching and Image Generation
date: 2026-08-11 12:00:00 +/-0000
categories: [ml, deep_learning, image]
tags: [ml, deep_learning, image, ai, pytorch]     
math: true
---

# Flow Matching and Image Generation

## [[Link to full code]](https://github.com/bhoener/diffusion-transformer)

## Introduction

Images are hard. Very hard.

I have spent the last ~2 years working almost exclusively on text-based models and NLP. Too often, I take for granted the simplicity of the GPT-style architecture and training setup.

I assumed diffusion would be relatively easy. Perhaps a few conv layers stacked on top of one another, maybe even with some residuals mixed in.

I watched [videos](https://share.google/IKD0tTmeB7RfxDpNk) and felt like I understood the math. How hard could it be just to predict the noise?

But nothing could have prepared me for the rabbit hole into which I was about to descend.

---

## Background on Diffusion/Score Matching

Score matching is a technique for image generation in which noise is progressively added to an image in a forward diffusion process and a generative model is used to approximate the reverse process (going from noise to a clean image).

Say we have a datapoint $x \sim p(x)$, where $p(x)$ is the underlying data distribution.

We can try to model $p(x)$ as the pdf 
$$
p_\theta(x) = \frac{e^{-f_\theta(x)}}{\underbrace{Z_\theta}_\text{normalize}}
$$

Where $Z_\theta$ is some normalizing constant we divide by that ensures everything integrates to 1.

It is hard to find $Z_\theta$ in practice since it would require integrating over all possible $x$, so we must take a slightly different approach.

Instead of looking at $p_\theta(x)$, we can look at the *score*, or the gradient of the log of $p_\theta(x)$

$$p_\theta(x) = \frac{e^{-f_\theta(x)}}{Z_\theta}$$

$$\nabla_x \log p_\theta(x) = \nabla_x \log \frac{e^{-f_\theta(x)}}{Z_\theta}$$

$$
 = \nabla_x (\log e^{-f_\theta(x)} - 
\log Z_\theta)
$$

$$
 = \nabla_x (-f_\theta(x)) - \log Z_\theta) = \nabla_x (-f_\theta(x)) - \cancel{\nabla_x \log Z_\theta}
$$

$$
 = \nabla_x (-f_\theta(x)) =: s_\theta(x)
$$

The score will point in the direction of data. 

Next, we need to add noise to our datapoints to reach a suitable objective.

Let $\tilde{x} = x + \epsilon; \epsilon \sim \mathcal{N}(0, \sigma^2)$ be a noised datapoint with noise $\epsilon$ of variance $\sigma^2$. 

This gives us a new corresponding data distribution $p(x) \longrightarrow p_\sigma(\tilde{x})$.

We want to minimize the difference between our model's predicted score $s_\theta(\tilde{x})$ and the real score $\nabla_\tilde{x} \log p_\sigma(\tilde{x})$.

We minimize the following objective

$$
\frac{1}{2} \mathbb{E}_{\tilde{x} \sim p_\sigma(\tilde{x})}[||\nabla_\tilde{x} \log p_\sigma(\tilde{x}) - s_\theta(\tilde{x})||_2^2]
$$

Doing some more math, this objective reduces to:

$$
\frac{1}{2} \mathbb{E}_{\tilde{x} \sim p_\sigma(\tilde{x})}[||s_\theta(\tilde{x}) - \frac{\epsilon}{\sigma^2}||_2^2]
$$

Which essentially just involves predicting the noise that was added to each training datapoint divided by the variance. 

I'm omitting the full derivation in order to keep this post short, but I highly recommend checking out the Outlier video for a more in-depth explanation.

---


After watching the [Outlier video on score matching](https://share.google/IKD0tTmeB7RfxDpNk), I took a shot at making my own diffusion model. I used a [u-net architecture](https://en.wikipedia.org/wiki/U-Net) with conv layers and overfit on a single test image.

Here is a sample generation from the model:

<img src="assets/Noise.png" alt="Generation from the model">

Just kidding! That is random gaussian noise. 

But the model didn't do much better. The loss went down (slightly), though the generations were entirely random. I tried numerous different fixes, but nothing seemed to really work.

Discouraged, I recognized my immense skill issue and decided to do more research before implementing my own model. During this time, I came across [Flow Matching](https://arxiv.org/abs/2210.02747), an alternative to diffusion. 

Flow Matching is very cool.

## Background on Flow Matching

Similar to Score Matching, imagine you have a dataset of datapoints $x \sim p(x)$, where $p(x)$ is some unknown underlying data distribution that you want to model.

We can train a model to turn random noise into datapoints by predicting a trajectory.

Let $p_0(x_0)$ be some random distribution, like $\mathcal{N}(0, I)$, and $p_1(x_1) = p(x)$. We can interpolate between these two and train a model to predict the path.

The simplest way to interpolate (transition) between two things is linearly.

If we have a noisy data sample $x_0 \sim p_0(x_0)$ and clean datapoint $x_1 \sim p_1(x_1)$, we can interpolate between them:

$$
x_t = t x_1 + (1-t) x_0
$$

Ideally, a model would predict how this interpolation changes with time, or the trajectory. With linear interpolation, this is easy to find:

$$
\frac{d}{dt}x_t = \frac{d}{dt} (t x_1 + (1-t)x_0)
$$

$$
= \frac{d}{dt}(t x_1) + \frac{d}{dt}(x_0 - tx_0)
$$

$$
 = x_1 - x_0
$$


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

### Slightly Longer Derivation

---

The authors of Flow Matching formalize this by defining the flow:

$$
\psi_t(x) = x_t
$$

Such that $\psi_0(x) = x_0$ and $\psi_1(x) = x_1$

They make the assumpion that the time derivative (velocity) of the flow can be modeled by some function of the intermediate datapoints $x_t$ and time.

$$
\frac{d}{dt}\psi_t(x) = u_t(\psi_t(x))
$$

Where $u_t$ is the velocity vector pointing from $x_0$ to $x_1$. 

We do not know $u_t(\cdot)$, so we want to approximate it with a neural network $v_t^\theta(\cdot)$.

This can be done by minimizing the objective:

$$
\mathbb{E}_{x_t \sim p_t(x_t)}[||u_t(x_t) - v_t^\theta(x_t)||_2^2]
$$

Doing some tricks, we can expand to get

$$
\mathbb{E}_{x_t \sim p_t(x_t)}[||u_t(x_t)||_2^2 -2u_t(x_t)v_t^\theta(x_t) + ||v_t^\theta(x_t)||_2^2]
$$

Focusing on the middle term, we can rewrite as an expectation

$$
\mathbb{E}_{x_t \sim p_t(x_t)}[2u_t(x_t)v_t^\theta(x_t)] = \int 2u_t(x_t)v_t^\theta(x_t) p_t(x_t) dx_t 
$$

$$
\mathbb{E}_{x_t \sim p_t(x_t)}[2u_t(x_t)v_t^\theta(x_t)] = \int 2u_t(x_t)v_t^\theta(x_t) p_t(x_t) dx_t 
$$

Focusing on $u_t(x_t)$, we can marginalize:

$$
u_t(x_t) = \int u_t(x_t|x_1) \frac{p_t(x_t|x_1)q(x_1)}{p_t(x_t)}dx_1
$$

Since $p(x_1 \vert x_t) = \frac{p_t(x_t \vert x_1)q(x_1)}{p_t(x_t)}$ and $u_t(x_t) = \int u_t(x_t \vert x_1)p(x_1 \vert x_t)dx_1$

But what does $ u_t (x_t \vert x_1) $ mean?

Realistically, we will always have more than one datapoint in the dataset. At any given "location", there will be multiple valid paths to data, depending on which datapoint you want to go to. $u_t(x_t \vert x_1)$ is the velocity vector pointing to the particular datapoint $x_1$. 

Then, we can rewrite the full expectation as

$$
\int 2u_t(x_t)v_t^\theta(x_t) p_t(x_t) dx_t  = \int 2\left(\int u_t(x_t|x_1) \frac{p_t(x_t|x_1)q(x_1)}{p_t(x_t)}dx_1\right)v_t^\theta(x_t) p_t(x_t) dx_t 
$$

There is a [very convenient theorem](https://en.wikipedia.org/wiki/Fubini's_theorem) that allows us to rewrite the integral of an integral as a double integal under certain conditions:

$$
\int_Y \left(\int_X f(x, y) dx\right)dy = \iint_{X \times Y} f(x, y) dx dy
$$

This gives us:


$$
\int 2\left(\int u_t(x_t|x_1) \frac{p_t(x_t|x_1)q(x_1)}{p_t(x_t)}dx_1\right)v_t^\theta(x_t) p_t(x_t) dx_t  = 2 \iint u_t(x_t|x_1)\frac{p_t(x_t|x_1)q(x_1)}{\cancel{p_t(x_t)}}v_t^\theta(x_t)\cancel{p_t(x_t)}dx_t dx_1
$$


$$
 = 2 \iint u_t(x_t|x_1)p_t(x_t|x_1)q(x_1)v_t^\theta(x_t)dx_t dx_1
$$

We can rewrite this as an expectation

$$
 = 2 \mathbb{E}_{x_t \sim p_t(x_t), x_t \sim p_1(x_1)} [u_t(x_t|x_1) v_t^\theta(x_t)]
$$

Plugging this back into the original expression, we get

$$
\mathbb{E}_{x_t \sim p_t(x_t), x_1 \sim p_1(x_1)}[||u_t(x_t)||_2^2 -2u_t(x_t|x_1)v_t^\theta(x_t) + ||v_t^\theta(x_t)||_2^2]
$$

We can add and subtract $\vert \vert u_t(x_t \vert x_1)\vert \vert^2$

$$
 = \mathbb{E}_{x_t \sim p_t(x_t), x_1 \sim p_1(x_1)}[||u_t(x_t)||_2^2 -2u_t(x_t|x_1)v_t^\theta(x_t) + ||v_t^\theta(x_t)||_2^2 + ||u_t(x_t|x_1)||^2 - ||u_t(x_t|x_1)||^2]
$$

And since
$$
||v_t^\theta(x_t)||_2^2 -2u_t(x_t|x_1)v_t^\theta(x_t) + ||u_t(x_t|x_1)||^2 = ||v_t^\theta(x_t) - u_t(x_t|x_1)||^2
$$

We get

$$
 ... = \mathbb{E}_{x_t \sim p_t(x_t), x_1 \sim p_1(x_1)}[||v_t^\theta(x_t) - u_t(x_t|x_1)||^2 + ||u_t(x_t)||_2^2 + ||u_t(x_t|x_1)||^2]
$$


$$
 = \mathbb{E}_{x_t \sim p_t(x_t), x_1 \sim p_1(x_1)}[||v_t^\theta(x_t) - u_t(x_t|x_1)||^2] + \underbrace{\cancel{\mathbb{E}_{x_t \sim p_t(x_t), x_1 \sim p_1(x_1)}[||u_t(x_t)||_2^2] + \mathbb{E}_{x_t \sim p_t(x_t), x_1 \sim p_1(x_1)}[||u_t(x_t|x_1)||^2]}}_{\text{constant wrt. } \theta}
$$

And we are left with

$$
\mathbb{E}_{x_t \sim p_t(x_t), x_1 \sim p_1(x_1)}[||v_t^\theta(x_t) - u_t(x_t|x_1)||^2]
$$

We still need to define $u_t(x_t\vert x_1)$

The authors choose to define the flow using linear interpolation as

$$
\psi_t(x_0; x_1) = (1-t)x_0 + tx_1 
$$

We can sample some noise $\epsilon \sim \mathcal{N}(0, I)$ to be $x_0$, a timestep $t \in [0, 1]$ and a clean image $x$.

This gives us a final objective of

$$
\mathbb{E}_{x, t, \epsilon}[||v_t^\theta((1 - t)\epsilon + t x) - u_t((1 - t)\epsilon + t x)||^2]
$$

$$
= \boxed{\mathbb{E}_{x, t, \epsilon}[||v_t^\theta((1 - t)\epsilon + t x) - (x - \epsilon)||^2]}
$$

If you want to learn more, I highly recommend both the [Outlier video on Flow Matching](https://www.youtube.com/watch?v=7cMzfkWFWhI) and the [MIT lecture series by Peter Holderrieth](https://www.youtube.com/playlist?list=PL57nT7tSGAAXwjhDYcxEycx5W7YoSrZyt)

## Implementation

As it turns out, the difficult part about image models is not the math or theory behind them, but rather the architecture. Flow matching is incredibly simple in concept, simply predicting $x_1 - x_0$ given $x_t$. Yet it requires an incredible amount of architectural duct-tape and magic tricks to get even a mildly decent result.

After the failure of my U-net, I decided to look into [DiTs](https://arxiv.org/abs/2212.09748). The DiT architecture itself isn't too different from that of a normal text-based transformer.
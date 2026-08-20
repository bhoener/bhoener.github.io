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

After the failure of my U-net, I decided to look into [DiTs](https://arxiv.org/abs/2212.09748). The DiT architecture itself isn't too different from that of a normal text-based transformer. Here is a diagram of the architecture I settled with (svg so can be zoomed):

![Diagram of the DiT architecture used](/assets/DiT-Excalidraw.svg){: width="200" }


I originally tried using cross-attention instead of the double stream block, where image token queries attend to text keys. This worked, but would have been less effective for [classifier-free guidance (CFG)](https://arxiv.org/pdf/2207.12598) since the attention would have to be skipped entirely for unconditional training, so I decided to switch to a [Flux/MM-DiT-style double stream block](https://arxiv.org/pdf/2403.03206) (fig. 2b) instead later on.

As a little test, once the DiT architecture was implmented, I decided to overfit it on a single image. The model was able to learn and recreate the image, but just barely. It took nearly 10,000 steps and the loss kept plateauing. I was clearly doing something spectacularly wrong.

A bit of digging later, I found that image models (at least the underlying DiTs) don't operate on actual images. Having a transformer take in raw pixels as an input is very, very costly. Even if you were to use $256 \times 256$ images with a standard patch size of 2, you would end up with a sequence length of $\frac{256^2}{2^2} = 16384$ which is generally far too much to deal with on average hardware if you don't use some form of modified attention or an extremly tiny model. So, we [operate inside of a VAE's latent space instead](https://arxiv.org/pdf/2112.10752).

### VAEs

A [VAE (Variational Autoencoder)](https://arxiv.org/pdf/1312.6114) is a type of [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) that tries to minimize reconstruction error while keeping a "smooth" latent space. It consists of an encoder model $q_\phi(z \vert x)$ that takes in some input $x$ (eg. an image) and outputs mean ($\mu$) and log variance ($\log \sigma^2$) tensors from which a latent (tensor) or hidden state $z$ is sampled. This latent is then sent through the decoder model $p_\theta(x \vert z)$ that tries to predict the original input $x$. Normally, the latent representation, $z$, is smaller than the input $x$, so the VAE must learn to perform some sort of compression.


$$
x \overset{q_\phi(z|x)}{\longrightarrow} z \overset{p_\theta(x|z)}{\longrightarrow} \hat{x}
$$

For me, the hardest part about VAEs to understand was the sampling. It seems kind of weird to have a model that doesn't have full control over its outputs. In VAEs, the latent is literally just noise that was sampled with some given variances and means.

```python
mean, logvar = encoder(x)
z = sample_noise(mean, logvar.exp())
reconstruction = decoder(z)
```

However, it is important to note that `mean` and `logvar` are not scalar values. If $z \in \mathbb{R}^d$, then $\mu, \log \sigma^2$ (`mean`, `logvar`) are $\in \mathbb{R}^d$ as well. 

This means each value in the latent tensor is sampled from a distribution with given $\mu_i, \log \sigma^2_i$ from the encoder. So, if the model wanted to, it could set all the variances to near zero and have $z \approx \mu$. This is because (somewhat obviously)

$$x \sim \mathcal{N}(\mu, 0) \implies x = \mu$$

So why not just do that? The model could learn to drive its output bias for `logvar` down to negative infinity ($\sigma^2 \rightarrow 0$) and take complete control over the latent, getting rid of the pesky noise from sampling and turning itself essentially into a "normal" autoencoder. 

One of the goals of VAEs is to have a smooth latent space, so that a small change in the latent will still give an output that could reasonably be within the data distribution.

![Diagram of VAE latent space](assets/VAEDiagram.png){: width="300" }
_From [https://arxiv.org/abs/1906.02691](https://arxiv.org/pdf/1906.02691), Fig. 2.1_

The authors enforce this by adding a KL-divergence term to the loss.

A normal reconstruction loss would be:

$$
\mathcal{L}_\text{recon} = \frac{1}{N} \sum_{i=1}^{N} ||x_i - \hat{x}_i||_2^2; \hspace{1cm} \hat{x} \sim p_\theta(x|z), z \sim q_\phi(z|x)
$$

In order to keep the latent space from being too sensitive to small changes, the model is punished for having a latent distribution different from a standard normal distribution $N(0, I)$ with the KL divergence loss term:

$$
\mathcal{L}_{KL} = - \sum \left[1  + \log \sigma^2 - \mu^2 - \exp (\log \sigma^2)\right]
$$

In practice this is normally weighted by a small amount, like $10^{-4}$. Otherwise, the VAE prioritizes the latent space so much that it is almost entirely unable to learn to reconstruct images.

So, we end up with

$$
\mathcal{L}_\text{total} = \mathcal{L}_\text{recon} + \lambda \mathcal{L}_{KL}
$$

Where $\lambda$ is the KL loss weight.

But there is a problem. When we try to find the gradients for the encoder, we get stuck.

Since we cannot backward through $z \sim \mathcal{N}(\mu, \sigma^ 2I)$ (how do you take the derivative of a random value with respect to mean and variance? What is $\frac{\partial z}{\partial \mu}$?), we can't easily get the gradients. Because of this, we have to use the **reparameterization trick**.

![VAE Reparameterization Trick](assets/VAE-Reparam-Excalidraw.svg){: width="400"}
_Reparameterization trick. Instead of sampling $z \sim \mathcal{N}(\mu, \sigma^2I)$, we sample $\epsilon \sim \mathcal{N}(0, I)$ and let $z = \sigma * \epsilon + \mu$._


The reparameterization trick is simple but clever. Normally, as before, we would do something like:

```python
mean, logvar = encoder(x)
latent = sample_noise(mu, logvar.exp()) # cannot backward through. encoder gets no grad!
reconstruction = decoder(latent)
```

Which is bad because `sample_noise` is not differentiable.

To solve this, we sample $\epsilon \sim \mathcal{N}(0, I)$. Then,

$$
z = \sigma * \epsilon + \mu
$$

Which is basically the same thing as sampling $z$ directly. This can easily be differentiated since $z$ is now truly a function of $\mu$ and $\sigma$, also recalling that $\mu = \mu(x; \phi), \sigma = \sigma(x; \phi)$. Now we can do backprop!

```python
mean, logvar = encoder(x)
noise = torch.randn_like(mean) # N(0, I)
latent = (logvar.exp() ** 0.5) * noise + mean # now latent is a function of logvar & mean!
reconstruction = decoder(latent)
```

For a much better explanation of VAEs, I recommend [this paper by the original VAE author](https://arxiv.org/abs/1906.02691). Apparently bro also invented the Adam optimizer. Very much the goat.

---

Traditionally, diffusion model training is done with a pretrained VAE. Images are fed into the encoder, which produces a latent. The latent is then corrupted as an image would be in flow matching. This corrupted image is sent to the DiT which predicts the velocity $z - \epsilon$. At inference, generation starts with random gaussian noise which is repeatedly fed into the model and updated as

$$
\hat{z}_{t+1} = \hat{z} + \Delta t \cdot  v_\theta(\hat{z}_t)
$$

![Diagram of VAE training setup](assets/VAE-Training-Excalidraw.svg)

Okay, cool. Instead of doing flow matching on images, we have the VAE encode the images first and do flow matching on the latent.

This is great, but it's a little too easy for my tastes. It simply wouldn't involve enough suffering.

### REPA-E

What if I told you there was a magical way to speed up diffusion model training by 45x? Just look at this graph.

![REPA-E training FID graph](assets/REPA-E-Graph.png){: width="300"}
_Graph taken from [https://arxiv.org/abs/2504.10483](https://arxiv.org/abs/2504.10483), Fig. 1d_

REPA-E proposes to train the diffusion/flow matching model and VAE at the same time, while using a pretrained encoder model to speed up and stabilize training.

During training, the VAE encoder encodes images as normal into a sampled latent (with reparamerization trick). One copy of the latent is sent to the VAE decoder, and similarity losses are calculated with the input image (along with KL divergence). 

Another copy is sent through a batchnorm layer before being detached, corrupted and sent to the DiT. Without the detach, the VAE encoder would recieve gradients from the diffusion loss, causing it to learn a simplified latent representation in order to make the diffusion process easier, rather than learning to reconstruct images well. The DiT then does velocity prediction on the normalized, corrupted latent (with additional text inputs).

However, the hidden state from one of the DiT layers is kept for later. The image is also sent to a pretrained encoder model, such as [DINOv3](https://arxiv.org/abs/2508.10104), which outputs a latent representation. The DiT hidden state is sent through a projection layer (a conv, as in [iREPA](https://arxiv.org/pdf/2512.10794)) and then compared with the pretrained encoder latent using a cosine similarity loss. This incentivizes the model to learn a meaningful and structured representation, speeding up training by taking advantage of the pretrained encoder. 

![Diagram of the REPA-E training setup](assets/REPA-Ediagram.svg)
_Diagram of the REPA-E training setup_

The gradients from the alignment loss also flow into the encoder model. The latent we used as an input to the DiT was detached to prevent the diffusion loss from having an impact, so it might seem difficult to get gradients for the encoder without running the DiT once again on a non-detached latent.

And that seems to be [pretty much what the authors did (lines 372-424)](https://github.com/End2End-Diffusion/REPA-E/blob/main/train_repae.py). However, that wasn't good enough for me. Without having seen the source code, I decided to ask Gemini if the model needed to be run twice, and it suggested to do something like the following:

```python
grad_latent_repa = torch.autograd.grad(loss_alignment, latent_detached, retain_graph=True)[0]
loss_encoder_repa = (latent * grad_latent_repa.detach()).sum() # this gives latent the grad calculated with only one forward
loss_encoder_repa.backward()
```

Which I did, and it seemed to work fine.

However, my initial REPA-E implementation was far from correct. For example, I forgot to pass the noised latent into the dit and passed the clean latent instead, effectively having it predict noise.

I spent about a month and a half trying to get the loss to go down as much as possible. The biggest change was `patch_size: 8 -> 2`, which is what splits the two big groups of runs in the image below.

![Loss graph for all my runs](assets/FlowMatchingLossGraphs.png)
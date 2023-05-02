# Introduction

Denoising diffusion probabilistic models
[@sohl2015deep; @ho2020denoising] and closely-related score-based
generative models through stochastic differential equations
[@song2020score] have recently been demonstrated as powerful machine
learning based tools for conditional and unconditional image generation.
These diffusion models are based on a stochastic process in which the
true images are degraded over time through additive white noise and, in
some cases, a deterministic scaling down of the signal to zero. Then, a
neural network can be trained to estimate the time-dependent score
function which allows one to run the reverse-time stochastic process
starting with a known prior distribution (pure noise in many cases),
iteratively running reverse-time update steps, and eventually ending on
an approximate sample from the same distribution as the training images.
Compared to another popular method, generative adversarial neural
networks, diffusion models can achieve higher image quality as measured
with standard benchmarks while avoiding the difficulties of adversarial
training [@dhariwal2021diffusion].

In this article, we present a new method called *Fourier Diffusion
Models,* which allow for control of the modulation transfer function
(MTF) and noise power spectrum (NPS) at each time step of the forward
and reverse stochastic process. Our approach is to model the forward
process as a cascade of linear shift-invariant (LSI) systems with
additive stationary Gaussian noise (ASGN). Then, we train a neural
network to approximate the time-dependent score function for iterative
sharpening and denoising of the images to generate high-quality
posterior samples given measured images with spatial blur and stationary
correlated noise. One new feature of Fourier diffusion models compared
to conventional scalar diffusion models is the capability to model
continuous probability flow from ground truth images to measured images
with a certain MTF and NPS. The initial results presented in this work
show that Fourier diffusion models require fewer time steps for
conditional image generation relative to scalar diffusion models. We
believe this improvement is due to the fact that the true images are
more similar to measured images than they are to pure white noise used
to initialize the reverse process for most scalar diffusion models.

In the sections to follow, we provide detailed mathematical descriptions
of Fourier diffusion models including the forward stochastic process,
the training loss function for score-matching neural networks, and
instructions on how to sample the reverse process for conditional or
unconditional image generation. Finally, we present experimental methods
and results for image restoration of low-radiation-dose CT measurements
to demonstrate one practical application of the proposed method.

# Methods

## Linear Shift-Invariant Systems with Stationary Gaussian Noise

In this section, we describe the theoretical background for LSI systems
with ASGN. This is a standard mathematical model used to evaluate the
spatial resolution and noise covariance of medical imaging systems in
the spatial frequency domain.

A two-dimensional LSI system is mathematically defined by convolution
with the impulse response function, also known as the point spread
function (PSF), of the system. Fourier convolution theorem states that
it is equivalent to multiply the two-dimensional Fourier transform of
the input by the Fourier transform of the PSF, referred to as the
modulation transfer function (MTF) of the system, followed by the
inverse Fourier transform to produce the convolved output. For discrete
systems, the voxelized values of a medical image can be represented as a
flattened column vector and the convolution operation can be represented
by a circulant matrix operator,
$\mathbf{H} = \mathbf{U}^*_\text{DFT} \boldsymbol{\Lambda}_\text{MTF} \mathbf{U}_\text{DFT}$,
where $\mathbf{U}_\text{DFT}$ is the unitary discrete two-dimensional
Fourier transform, $\mathbf{U}^*_\text{DFT}$ is the unitary discrete
two-dimensional inverse Fourier transform, and
$\boldsymbol{\Lambda}_\text{MTF}$ is a diagonal matrix representing
element-wise multiplication by the MTF in the spatial frequency domain.

If we assume the noise covariance between two voxels in an image does
not depend on position, only on relative displacement between two
positions, then we say the noise is spatially stationary, and the
covariance can be fully defined by the noise power spectrum (NPS) in the
spatial frequency domain. For the discrete case, stationary noise can be
modeled by a circulant covariance matrix,
$\boldsymbol{\Sigma} = \mathbf{U}^*_\text{DFT} \boldsymbol{\Lambda}_\text{NPS} \mathbf{U}_\text{DFT}$,
where $\boldsymbol{\Lambda}_\text{NPS}$ is a diagonal matrix of
spatial-frequency-dependent noise power spectral densities. In
probabilistic terms, the output of an LSI system with ASGN is a
multivariate Gaussian conditional probability density function
parameterized by the MTF and NPS as follows:

$$p(\mathbf{x}_\text{out}|\mathbf{x}_\text{in}) =  \mathcal{N}(\mathbf{x}_\text{out};  \mathbf{H} \hspace{1mm} \mathbf{x}_\text{in}, \boldsymbol{\Sigma} ).$$

where $\mathbf{x}_\text{in}$ is the ground truth image and
$\mathbf{x}_\text{out}$ is the degraded image. The notation,
$\mathcal{N}(\mathbf{v};\boldsymbol{\mu}_\mathbf{v},\boldsymbol{\Sigma}_\mathbf{v})$
represents a multivariate Gaussian probability density function with
argument, $\mathbf{v}$, parameterized by the mean vector
$\boldsymbol{\mu}_\mathbf{v}$ and covariance matrix
$\boldsymbol{\Sigma}_\mathbf{v}$.

## Discrete-Time Stochastic Process with MTF and NPS Control

![A probabilistic graphical model for the stochastic process, consisting
of linear shift invariant systems and additive stationary Gaussian
noise. The measurement-conditioned diffusion model, in light purple, is
trained to approximate the reverse process, in dark
purple.](figures/ddpm_cartoon.png){#fig:network_diagram
width="\\textwidth"}

Consider a sequence of LSI systems with ASGN resulting in a
discrete-time forward stochastic process as shown in Figure
[1](#fig:network_diagram){reference-type="ref"
reference="fig:network_diagram"}. The update rule for a forward time
step is defined as

$$\mathbf{x}^{[n+1]} = \mathbf{H}_\Delta^{[n]} \mathbf{x}^{[n]} + {\boldsymbol{\Sigma}_\Delta^{[n]}}^{ \hspace{0mm}1/2}\boldsymbol{\eta}^{[n]}  .
    \label{eq:discrete_forward_update}$$

where $\mathbf{H}_\Delta^{[n]}$ is a circulant matrix representing an
LSI system, $\boldsymbol{\Sigma}_\Delta$ is a circulant matrix
representing the noise covariance of the ASGN, and
$\boldsymbol{\eta}^{[n]}$ is zero-mean identity-covariance Gaussian
noise. Also, we assume the noise at a given time step is independent of
the noise at all other time steps; that is,
$\boldsymbol{\eta}^{[n]} \perp \!\!\! \perp\boldsymbol{\eta}^{[m]} \hspace{1mm} \forall \hspace{1mm} n \neq m$.
Stated differently, the conditional probability density function for a
forward step is

$$\text{p}(\mathbf{x}^{[n+1]}|\mathbf{x}^{[n]}) = \mathcal{N}(\mathbf{x}^{[n+1]} ; \mathbf{H}_\Delta^{[n]} \hspace{1mm} \mathbf{x}^{[n]},  \boldsymbol{\Sigma}_\Delta) .
    \label{eq:forward_step}$$

Figure [1](#fig:network_diagram){reference-type="ref"
reference="fig:network_diagram"} shows that the full process can be
represented by a directed acyclic probabilistic graphical model, which
means the random vector at a given time step, $\mathbf{x}^{[n]}$, is
defined as conditionally independent of random vectors at earlier time
steps given the previous image, $\mathbf{x}^{[n-1]}$. Therefore, the
joint distribution of the full forward process can be written as

$$\text{p}(\mathbf{x}^{[0]}, \mathbf{x}^{[1]},\mathbf{x}^{[2]},\ldots,\mathbf{x}^{[N]}) = \text{p}(\mathbf{x}^{[0]})\text{p}(\mathbf{x}^{[1]}| \mathbf{x}^{[0]})\text{p}(\mathbf{x}^{[2]}| \mathbf{x}^{[1]})\ldots\text{p}(\mathbf{x}^{[N]}| \mathbf{x}^{[N-1]}) \enspace ,$$

where $N$ is the last time step.

One example of this stochastic process would be the case where
$\mathbf{H}_\Delta^{[n]}$ is convolutional blur and
${\boldsymbol{\Sigma}_\Delta^{[n]}}$ is correlated stationary noise. In
that case, the image will become more blurry and noisy as time passes in
the forward process. Then, as we will describe in a later section, a
neural network can be trained to run the reverse-time stochastic
process, which should result in a series of sharpening and
noise-reducing steps to restore image quality.

The cascade of LSI systems with ASGN leading up to a certain time point,
$n$, can be described by an equivalent LSI system,
$\mathbf{H}^{[n]} =  \mathbf{U}^*_\text{DFT} \boldsymbol{\Lambda}_{\text{MTF}}^{[n]} \mathbf{U}_\text{DFT}$,
and ASGN with covariance,
$\boldsymbol{\Sigma^{[n]}}=\mathbf{U}^*_\text{DFT}\boldsymbol{\Lambda}_{\text{NPS}}^{[n]}\mathbf{U}_\text{DFT}$,
applied to the original image, $\mathbf{x^{[0]}}$ as shown below:

$$\begin{gathered}
    \mathbf{x}^{[n]} = \mathbf{H}^{[n]} \mathbf{x}^{[0]} + {\boldsymbol{\Sigma}^{[n]}}^{ \hspace{0mm} 1/2} \boldsymbol{\epsilon}^{[n]}\\
    \text{p}(\mathbf{x}^{[n]}|\mathbf{x}^{[0]}) = \mathcal{N}(\mathbf{x}^{[n]} ;   \mathbf{H}^{[n]} \hspace{1mm} \mathbf{x}^{[0]},  \boldsymbol{\Sigma^{[n]}} ) .
    \label{eq:effective_MTF_NPS}
\end{gathered}$$

where $\boldsymbol{\epsilon}^{[n]}$ is identity-covariance zero-mean
Gaussian random process defined such that non-overlapping time intervals
are independent. Our goal is to prescribe the effective MTF and NPS at
every time step, and then define the forward process parameters
accordingly. To that end, we can combine
[\[eq:forward_step\]](#eq:forward_step){reference-type="eqref"
reference="eq:forward_step"} and
[\[eq:effective_MTF_NPS\]](#eq:effective_MTF_NPS){reference-type="eqref"
reference="eq:effective_MTF_NPS"} to define $\mathbf{H}_\Delta^{[n]}$ as
the inverse MTF for the current time step matrix multiplied by the MTF
for the next time step as follows

$$\begin{gathered}
    \mathbf{H}_\Delta^{[n]}  = \mathbf{H}^{[n+1]} {\mathbf{H}^{[n]}}^{-1} = \mathbf{U}^*_\text{DFT} \boldsymbol{\Lambda}_{\text{MTF}}^{[n+1]} {\boldsymbol{\Lambda}_{\text{MTF}}^{[n]}}^{\hspace{-3mm}-1} \mathbf{U}_\text{DFT} . \label{eq:LSI_in_terms_of_MTF_}
\end{gathered}$$

When this LSI system, $\mathbf{H}_\Delta^{[n]}$, is applied to the
Gaussian random vector at time step $n$ which has mean vector,
$\mathbf{H}^{[n]} \mathbf{x}^{[0]}$, and covariance matrix,
$\boldsymbol{\Sigma}^{[n]}$, the result is a new Gaussian random vector
with mean vector $\mathbf{H}^{[n+1]} \mathbf{x}^{[0]}$ and covariance
matrix,
$\mathbf{H}_\Delta^{[n]} \boldsymbol{\Sigma}^{[n]}  \mathbf{H}_\Delta^{[n]}$.
Therefore, we can define the ASGN covariance,
$\boldsymbol{\Sigma}_\Delta^{[n]}$, as follows:

$$\begin{gathered}
    \boldsymbol{\Sigma}_\Delta^{[n]} = \boldsymbol{\Sigma}^{[n+1]} - \mathbf{H}_\Delta^{[n]} \boldsymbol{\Sigma}^{[n]}  \mathbf{H}_\Delta^{[n]} =  \mathbf{U}^*_\text{DFT}[ \boldsymbol{\Lambda}_{\text{NPS}}^{[n+1]} - \boldsymbol{\Lambda}_{\text{MTF}}^{2 \hspace{0.5mm} [n+1]} \boldsymbol{\Lambda}_{\text{MTF}}^{-2 \hspace{0.5mm} [n]}
    \boldsymbol{\Lambda}_{\text{NPS}}^{[n]} ] \mathbf{U}_\text{DFT} \label{eq:noise_in_terms_of_NPS}
\end{gathered}$$

So the output of the LSI system and zero-mean ASGN applied to the
Gaussian random vector at time step, $n$, is a new Gaussian random
vector with mean vector, $\mathbf{H}^{[n+1]} \mathbf{x}^{[0]}$, and
covariance matrix, $\boldsymbol{\Sigma}^{[n+1]}$. Note, this relies on
the assumption that all eigenvalues (spatial-frequency-dependent
variances) of $\boldsymbol{\Sigma}^{[n+1]}$ are greater than or equal to
the corresponding eigenvalues of
$\mathbf{H}_\Delta^{[n]} \boldsymbol{\Sigma}^{[n]}  \mathbf{H}_\Delta^{[n]}$.

We can substitute
[\[eq:LSI_in_terms_of_MTF\_\]](#eq:LSI_in_terms_of_MTF_){reference-type="eqref"
reference="eq:LSI_in_terms_of_MTF_"} and
[\[eq:noise_in_terms_of_NPS\]](#eq:noise_in_terms_of_NPS){reference-type="eqref"
reference="eq:noise_in_terms_of_NPS"} into
[\[eq:discrete_forward_update\]](#eq:discrete_forward_update){reference-type="eqref"
reference="eq:discrete_forward_update"} to arrive at the discrete
forward update rule in terms of the prescribed MTF and NPS:

$$\begin{gathered}
    \mathbf{x}^{[n + 1]} = \mathbf{H}^{[n + 1]} {\mathbf{H}^{[n]}}^{\hspace{0mm}-1} \mathbf{x}^{[n]} + (\boldsymbol{\Sigma}^{[n+1]} -  {\mathbf{H}^{[n+1]}}^{\hspace{-0mm} 2} {\mathbf{H}^{[n]}}^{\hspace{0mm}-2} \boldsymbol{\Sigma}^{[n]} )^{1/2} \boldsymbol{\eta}^{[n]}
    \label{eq:discrete_forward_update_MTF_NPS}
\end{gathered}$$

## Continuous-Time Process and Stochastic Differential Equations

Consider a continuous-time stochastic process, $\mathbf{x}^{(t)}$, given
by:

$$\begin{gathered}
\mathbf{x}^{(t)} = \mathbf{H}^{(t)} \mathbf{x}^{(0)} + {\boldsymbol{\Sigma}^{(t)}}^{\hspace{0mm}  1/2} \boldsymbol{\epsilon}^{(t)}  
\label{eq:x_t} \\
\text{p}(\mathbf{x}^{(t)}|\mathbf{x}^{(0)}) = \mathcal{N}(\mathbf{x}^{(t)}; \mathbf{H}^{(t)} \mathbf{x}^{(0)}, {\boldsymbol{\Sigma}^{(t)}}) \label{eq:x_t_dist}
\end{gathered}$$

where $\boldsymbol{\epsilon}^{(t)}$ is a zero-mean identity-covariance
Gaussian process where the updates for non-overlapping time intervals
are independent (i.e., a Lévy process). We define $\mathbf{H}^{(t)}$ and
${\boldsymbol{\Sigma}^{(t)}}$ as continuously differentiable
time-dependent spatially-circulant matrices, which control the MTF and
NPS, respectively, over time in the stochastic process.

An instance of the discrete-time forward stochastic process in
[\[eq:effective_MTF_NPS\]](#eq:effective_MTF_NPS){reference-type="eqref"
reference="eq:effective_MTF_NPS"} can be defined by sampling the
continuous-time forward stochastic process in
[\[eq:x_t_dist\]](#eq:x_t_dist){reference-type="eqref"
reference="eq:x_t_dist"} with $N+1$ time points evenly spaced on the
interval $t \in (0,T)$ with sample time $\Delta t = T/N$. Note the
discrete-time process, $\mathbf{x}^{[n]}$, and the continuous-time
process, $\mathbf{x}^{(t)}$ are distinct variables related by this
sampling procedure shown below:

$$\begin{gathered}
    \mathbf{x}^{[n]} = \mathbf{H}^{[n]} \mathbf{x}^{[0]} + {\boldsymbol{\Sigma}^{[n]}}^{\hspace{-0mm}  1/2} \boldsymbol{\epsilon}^{[n]} \\
    = \mathbf{x}^{(n \Delta t)} = \mathbf{H}^{(n \Delta t)} \mathbf{x}^{(0)} + {\boldsymbol{\Sigma}^{(n \Delta t)}}^{\hspace{-0mm}  1/2} \boldsymbol{\epsilon}^{(n \Delta t)} 
    \label{eq:sample_discrete}
\end{gathered}$$

In Appendix A, we use the discrete forward update defined in
[\[eq:discrete_forward_update_MTF_NPS\]](#eq:discrete_forward_update_MTF_NPS){reference-type="eqref"
reference="eq:discrete_forward_update_MTF_NPS"} and take the limit as
$\Delta t$ approaches zero to show that the continuous-time process,
$\mathbf{x}^{(t)}$, can be described by the following stochastic
differential equation:

$$\mathbf{dx} = \mathbf{H^{'}}^{(t)}{\mathbf{H}^{(t)}}^{\hspace{0mm}-1} \mathbf{x}^{(t)} \text{dt} + (\boldsymbol{\Sigma^{'}}^{(t)}  -  2 {\mathbf{H^{'}}^{(t)}} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1}  \boldsymbol{\Sigma}^{(t)} )^{1/2} \mathbf{dw} \label{eq:SDE}$$

where
$\mathbf{H^{'}}^{(t)} = \frac{\text{d}}{\text{dt}} \mathbf{H}^{(t)}$,
$\boldsymbol{\Sigma^{'}}^{(t)} = \frac{\text{d}}{\text{dt}} \boldsymbol{\Sigma}^{(t)}$,
and $\mathbf{dw}$ is infinitesimal white Gaussian noise with covariance,
$\text{dt} \mathbf{I}$. The stochastic differential equation in
[\[eq:SDE\]](#eq:SDE){reference-type="eqref" reference="eq:SDE"} is one
of the main results of this work. It enables Fourier diffusion models
with prescriptive control of MTF and NPS as a function of time in the
forward and reverse stochastic processes.

If we compare $\eqref{eq:SDE}$ to the standard form,

$$\mathbf{dx} = \mathbf{f}(\mathbf{x}, t) \text{dt} + \mathbf{G}(t) \mathbf{dw},$$

then we can identify,

$$\begin{gathered}
    \mathbf{f}(\mathbf{x}, t) = \mathbf{H^{'}}^{(t)}{\mathbf{H}^{(t)}}^{\hspace{0mm}-1} \mathbf{x}^{(t)} \hspace{2mm},\hspace{4mm} 
    \mathbf{G}(t) = (\boldsymbol{\Sigma^{'}}^{(t)}  -  2 {\mathbf{H^{'}}^{(t)}} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1}  \boldsymbol{\Sigma}^{(t)} )^{1/2} .
    \label{eq:f_g}
\end{gathered}$$

It has previously been shown there is an exact solution for the
time-reversed stochastic differential equation
[@anderson1982reverse][@song2020score] The formula is shown below:

$$\mathbf{dx} = [\mathbf{f}(\mathbf{x}, t) - \mathbf{G}(t) {\mathbf{G}(t)}^T \nabla \log{\text{p} (\mathbf{x}^{(t)})}] \text{dt} + \mathbf{G}(t) \mathbf{dw}.
    \label{eq:reverse_sde_standard_form}$$

The inclusion of the score function,
$\nabla \log{\text{p} (\mathbf{x}^{(t)})}$, results in a deterministic
drift towards higher probability values of $\mathbf{x}^{(t)}$.
Substituting the values in [\[eq:f_g\]](#eq:f_g){reference-type="eqref"
reference="eq:f_g"} into
[\[eq:reverse_sde_standard_form\]](#eq:reverse_sde_standard_form){reference-type="eqref"
reference="eq:reverse_sde_standard_form"} results in the following
formula for the reverse stochastic differential equation for Fourier
diffusion models:

$$\begin{gathered}
     \mathbf{dx} \hspace{-1mm} = \hspace{-1mm} [\mathbf{H^{'}}^{(t)} \hspace{-.5mm} {\mathbf{H}^{(t)}}^{\hspace{-.5mm}-1} \hspace{-2mm}\mathbf{x}^{(t)} \hspace{-1mm} - \hspace{-1mm}(\boldsymbol{\Sigma^{'}}^{(t)}  \hspace{-3mm} -  2 {\mathbf{H^{'}}^{(t)}} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1}  \boldsymbol{\Sigma}^{(t)} )  \nabla \log{\text{p} (\mathbf{x}^{(t)})}] \text{dt} + (\boldsymbol{\Sigma^{'}}^{(t)}   \hspace{-3mm} -  2 {\mathbf{H^{'}}^{(t)}} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1}  \boldsymbol{\Sigma}^{(t)} )^{1/2} \mathbf{dw} .
    \label{eq:reverse_SDE_unconditional}
\end{gathered}$$

Most existing diffusion models for image generation use a forward
stochastic process defined by additive white Gaussian noise and if there
is any deterministic drift, it is almost always by scalar multiplication
of the image signal, usually causing it to decay towards zero. We refer
to these as scalar diffusion models, and they are a special case of
Fourier models, where $\mathbf{H}^{(t)}$ and $\boldsymbol{\Sigma}^{(t)}$
are set to scalar matrices (identity times a time-dependent scalar) as
shown below:

$$\begin{gathered}
    \mathbf{H}^{(t)} = e^{- \frac{1}{2}\int_0^t \beta(s)\text{ds}}\hspace{1mm} \mathbf{I} \label{eq:scalar_drift}\\
    \boldsymbol{\Sigma}^{(t)} = \sigma^2(t) \hspace{1mm} \mathbf{I} \label{eq:scalar_drift}
\end{gathered}$$

which results in the forward stochastic differential equation:

$$\begin{gathered}
    \mathbf{dx} = -\frac{1}{2} \beta(t) \mathbf{x}^{(t)}\text{dt} + \sqrt{\beta(t)\sigma^2(t) + \frac{\text{d}}{\text{dt}}\sigma^2(t)}\mathbf{dw} ,
    \label{eq:white_noise}
\end{gathered}$$

where $\sigma^2(t)$ controls the so-called variance-exploding (VE)
component and $\beta(t)$ controls the variance-preserving component (VP)
as defined in [@song2020score]. Note that $\beta(t)$ is a function of
time, not necessarily a constant exponential decay, so there is no loss
of generality and any differentiable magnitude function can be achieved
with this parameterization. For the special case of the original
denoising diffusion probabilistic models [@sohl2015deep]
[@ho2020denoising], there is the additional constraint,
$\sigma^2(t) = 1 - e^{-\int_0^{t}\beta(s)\text{ds}}$, meaning the
process is fully defined by $\beta(t)$.

One way to interpret Fourier diffusion models is to consider them as
conventional diffusion models in the spatial frequency domain with
spatial-frequency-dependent diffusion rates. For example, it would be
mathematically equivalent to take the two-dimensional Fourier transform
of the training images, and then train a diagonal diffusion model (using
diagonal matrices for spatial-frequency dependent diffusion rates) to
generate new samples of those Fourier coefficients. One practical
advantage of formulating the process with LSI systems in the image
domain is that the reverse time steps may be more suitable for
approximation with convolutional neural networks, which are also
composed of shift-invariant operations.

## Conditional Image Generation and Supervised Learning

The reverse-time stochastic differential equation in
[\[eq:reverse_SDE_unconditional\]](#eq:reverse_SDE_unconditional){reference-type="eqref"
reference="eq:reverse_SDE_unconditional"} applies to unconditional image
generation with Fourier diffusion models. Assuming we have access to the
score function, $\nabla \log{\text{p} (\mathbf{x}^{(t)})}$ or an
approximation thereof, and assuming the distribution of the endpoint,
$\text{p}(\mathbf{x}^{(T)})$, is a known prior, we can use
[\[eq:reverse_SDE_unconditional\]](#eq:reverse_SDE_unconditional){reference-type="eqref"
reference="eq:reverse_SDE_unconditional"} to run the reverse-time
process and generate new samples from $\text{p}(\mathbf{x}^{(0)})$ which
typically represents the training data distribution.

In this work, we consider the case where samples of both the target
images, $\mathbf{x}^{(0)}$, and some corresponding measurements,
$\mathbf{y}$, are available at training time. Our goal is to train a
deep learning model to sample from the posterior distribution
$\text{p}(\mathbf{x}^{(0)}|\mathbf{y})$. For this work, we will consider
the following forward model:

$$\begin{gathered}
    \mathbf{y} = \mathbf{H}_{\mathbf{y}|\mathbf{x}^{(0)}} \mathbf{x}^{(0)} + \boldsymbol{\Sigma}_{\mathbf{y}|\mathbf{x}^{(0)}}^{1/2} \boldsymbol{\varepsilon} \\
    \text{p}(\mathbf{y}|\mathbf{x}^{(0)}) = \mathcal{N}(\mathbf{y}; \mathbf{H}_{\mathbf{y}|\mathbf{x}^{(0)}} \hspace{0.5mm} \mathbf{x}^{(0)}, \boldsymbol{\Sigma}_{\mathbf{y}|\mathbf{x}^{(0)}})
    \label{eq:forward_model}
\end{gathered}$$

where $\boldsymbol{\varepsilon}$ is a zero-mean identity-covariance
Gaussian random vector, $\mathbf{H}_{\mathbf{y}|\mathbf{x}^{(0)}}$ is
circulant matrix representing the MTF of the measurements, and
$\boldsymbol{\Sigma}_{\mathbf{y}|\mathbf{x}^{(0)}}$ is a circulant
matrix representing the NPS of the measurements.

As shown in Figure [2](#fig:conditional_options){reference-type="ref"
reference="fig:conditional_options"}, there are at least two possible
causal relationships between the stochastic process, $\mathbf{x}^{(t)}$,
and the measurements, $\mathbf{y}$. The first option shows measurements
as outside information, where $\mathbf{y}$ is a stochastic function of
$\mathbf{x}^{(0)}$, but separate from the forward process. In that case,
we assume that $\text{p}(\mathbf{x}^{(T)}|\mathbf{y})$ is a known prior,
which we can use to initialize the reverse process, and we replace the
unconditional score function in
[\[eq:reverse_SDE_unconditional\]](#eq:reverse_SDE_unconditional){reference-type="eqref"
reference="eq:reverse_SDE_unconditional"} with the posterior score
function, $\nabla\log\text{p}(\mathbf{x}^{(t)}|\mathbf{y})$. The second
option in Figure [2](#fig:conditional_options){reference-type="ref"
reference="fig:conditional_options"} shows measurements as the final
time step. In that case, we can use the reverse process defined in
[\[eq:reverse_SDE_unconditional\]](#eq:reverse_SDE_unconditional){reference-type="eqref"
reference="eq:reverse_SDE_unconditional"} without modification. By
initializing with the measurements, $\mathbf{y}$, the final step of the
reverse process at $t=0$ will be a sample from
$\text{p}(\mathbf{x}^{(0)}|\mathbf{y})$.

In this work, we evaluate and compare scalar diffusion models with
Fourier diffusion models for conditional image generation. For the
forward model defined in
[\[eq:forward_model\]](#eq:forward_model){reference-type="eqref"
reference="eq:forward_model"}, scalar diffusion models cannot generally
be used for the second option, with measurements as the last time step.
This is one of the most important new capabilities of Fourier diffusion
models; the forward process can be crafted such that the MTF and NPS at
the final time step of the forward process are exactly equal to the
measurement MTF and NPS. That is,
$\mathbf{H}^{(T)} = \mathbf{H}_{\mathbf{y}|\mathbf{x}^{(0)}}$, and,
$\boldsymbol{\Sigma}^{(T)}=\boldsymbol{\Sigma}_{\mathbf{y}|\mathbf{x}^{(0)}}$.
In this way, Fourier diffusion models can describe continuous
probability flow from the ground truth images to measured images with
shift-invariant blur and stationary correlated noise.

![Two options for causal relationships between stochastic process and
measurements. The second option is only possible with Fourier diffusion
models. In this work, we evaluate and compare scalar diffusion models
using the first option and Fourier diffusion models using the second
option. ](figures/conditional_options.png){#fig:conditional_options
width="95%"}

## Score-Matching Loss Function for Neural Network Training

It is possible to train a neural network to estimate the score function
in order to run an approximation of the reverse process, allowing for
conditional or unconditional generative modeling. We assume the inputs
of the neural network are the image at a certain time step,
$\mathbf{x}^{(t)}$, the measurements, $\mathbf{y}$, and the time, $t$,
so we define the neural network by the function,
$\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}^{(t)}, \mathbf{y}, t)$,
parameterized by the network weights $\boldsymbol{\theta}$. During
training, we start with a sample from $\mathbf{x}^{(0)}$. Next, we
uniformly sample the time, $t\in (0,T)$, and we generate a corresponding
sample from $\mathbf{x}^{(t)}$ given $\mathbf{x}^{(0)}$ using
[\[eq:x_t\]](#eq:x_t){reference-type="eqref" reference="eq:x_t"} and the
pre-defined time-dependent matrices, $\mathbf{H}^{(t)}$ and
$\boldsymbol{\Sigma}^{(t)}$, which describe the effective MTF and NPS.
Finally, we generate a sample from $\mathbf{y}$ given $\mathbf{x}^{(0)}$
and $\mathbf{x}^{(t)}$. If using the first option, in Figure
[2](#fig:conditional_options){reference-type="ref"
reference="fig:conditional_options"}, where measurements are treated as
outside information, then $\mathbf{y}$ is conditionally independent of
$\mathbf{x}^{(t)}$ given $\mathbf{x}^{(0)}$, so we can generate
$\mathbf{y}$ as a stochastic function of $\mathbf{x}^{(0)}$ using
[\[eq:forward_model\]](#eq:forward_model){reference-type="eqref"
reference="eq:forward_model"}. If using the second option in Figure
[2](#fig:conditional_options){reference-type="ref"
reference="fig:conditional_options"}, where the measurements are the
last time step, we can view $\mathbf{y}$ as a stochastic function of
$\mathbf{x}^{(t)}$ and sample it using the following formula:

$$\begin{gathered}
    \mathbf{y} = \mathbf{H}^{(T)} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1} \mathbf{x}^{(t)} + (\boldsymbol{\Sigma}^{(T)} -  {\mathbf{H}^{(T)}}^{\hspace{-0mm} 2} {\mathbf{H}^{(t)}}^{\hspace{0mm}-2} \boldsymbol{\Sigma}^{[n]} )^{1/2} \boldsymbol{\zeta} \\
    \text{p}(\mathbf{y}|\mathbf{x}^{(t)})  = \mathcal{N}(\mathbf{y}; \mathbf{H}^{(T)} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1} \mathbf{x}^{(t)}, \boldsymbol{\Sigma}^{(T)} -  {\mathbf{H}^{(T)}}^{\hspace{-0mm} 2} {\mathbf{H}^{(t)}}^{\hspace{0mm}-2} \boldsymbol{\Sigma}^{(t)})
    \label{eq:option_2_sample_y}
\end{gathered}$$

where $\boldsymbol{\zeta}$ is a zero-mean identity-covariance Gaussian
random vector. The logic for this formula is very similar to the logic
in the derivation of
[\[eq:discrete_forward_update_MTF_NPS\]](#eq:discrete_forward_update_MTF_NPS){reference-type="eqref"
reference="eq:discrete_forward_update_MTF_NPS"}. Applying
$\mathbf{H}^{(T)}{\mathbf{H}^{(t)}}^{-1}$ to $\mathbf{x}^{(t)}$ and
adding
$(\boldsymbol{\Sigma}^{(T)} -  {\mathbf{H}^{(T)}}^{\hspace{-0mm} 2} {\mathbf{H}^{(t)}}^{\hspace{0mm}-2} \boldsymbol{\Sigma}^{(t)})$
will result in a new Gaussian random variable
$\mathbf{x}^{(T)}=\mathbf{y}$ with mean vector
$\mathbf{H}^{(T)} \mathbf{x}^{(0)} = \mathbf{H}_{\mathbf{y}|\mathbf{x}^{(0)}} \mathbf{x}^{(0)}$
and covariance,
$\boldsymbol{\Sigma}^{(T)} = \boldsymbol{\Sigma}_{\mathbf{y}|\mathbf{x}^{(0)}}$

We train the network with samples from the supervised training data,
$(\mathbf{x}^{(0)}, \mathbf{x}^{(t)}, \mathbf{y}, t)$, and minimize the
mean squared error between the predicted score,
$\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}^{(t)}, \mathbf{y}, t)$, and
the target score,
$\nabla \log{\text{p} (\mathbf{x}^{(t)}|\mathbf{x}^{(0)})}$, (known as
the Jensen-Fisher divergence [@sanchez2012jensen]) as shown below:

$$\begin{gathered}
    \underset{\mathbf{x}^{(0)}, t}{\mathbb{E}}[||\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}^{(t)}, \mathbf{y}, t) - \nabla \log{\text{p} (\mathbf{x}^{(t)}|\mathbf{x}^{(0)})}||^2] .
    \nonumber \\
    =\underset{\mathbf{x}^{(0)}, t}{\mathbb{E}}[||\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}^{(t)}, \mathbf{y}, t) - {\boldsymbol{\Sigma}^{(t)}}^{\hspace{0mm}-1} (\mathbf{x}^{(t)} - \mathbf{H}^{(t)} \mathbf{x}^{(0)})   ||^2].
    \label{eq:score_matching_loss}
\end{gathered}$$

Note that $\mathbf{x}^{(0)}$ is available at training time to evaluate
the score function,
$\nabla \log{\text{p} (\mathbf{x}^{(t)}|\mathbf{x}^{(0)})}$, but not at
test time. In this supervised learning approach, $\mathbf{y}$ is
available at both training and testing time, so it is used as a network
input. Since $\mathbf{y}$ is a stochastic function of
$\mathbf{x}^{(0)}$, it may contain information about the target image
that is useful for improving score prediction.

The training loss function in
[\[eq:score_matching_loss\]](#eq:score_matching_loss){reference-type="eqref"
reference="eq:score_matching_loss"} is valid for both options in Figure
[2](#fig:conditional_options){reference-type="ref"
reference="fig:conditional_options"}. For the first option, with
measurements as outside information, we can use the property that
$\mathbf{y}$ is conditionally independent of $\mathbf{x}^{(t)}$ given
$\mathbf{x}^{(0)}$. Since $\mathbf{x}^{(0)}$ is given at training time,
we can make the substitution:
$\nabla \log{\text{p} (\mathbf{x}^{(t)}|\mathbf{x}^{(0)}, \mathbf{y} )}$
= $\nabla \log{\text{p} (\mathbf{x}^{(t)}|\mathbf{x}^{(0)})}$.
Therefore,
[\[eq:score_matching_loss\]](#eq:score_matching_loss){reference-type="eqref"
reference="eq:score_matching_loss"} is valid to approximate
$\nabla \log \text{p}(\mathbf{x}^{(t)}| \mathbf{y})$ in the case where
measurements are considered to be outside information. For the second
option, with measurements as the final time step, we seek to approximate
$\nabla \log \text{p}(\mathbf{x}^{(t)})$; so,
[\[eq:score_matching_loss\]](#eq:score_matching_loss){reference-type="eqref"
reference="eq:score_matching_loss"} is also valid for that case.

After the score-matching neural network is trained, one can run a
discrete-time approximation of the reverse process using the
Euler-Maryuama method, as shown below:

$$\begin{gathered}
     \mathbf{x}^{[n-1]} = \mathbf{x}^{[n]} - [\mathbf{H^{'}}^{(n\Delta t)}{\mathbf{H}^{(n \Delta t)}}^{\hspace{0mm}-1} \mathbf{x}^{[n]} - (-  2 {\mathbf{H^{'}}^{(n \Delta t)}} {\mathbf{H}^{(n \Delta t)}}^{\hspace{0mm}-1}  \boldsymbol{\Sigma}^{(n \Delta t)} + \boldsymbol{\Sigma^{'}}^{(n \Delta t)} )  \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}^{[n]}, \mathbf{y}, n\Delta t)] \Delta t \nonumber \\
    \hspace{70mm} + (-  2 {\mathbf{H^{'}}^{(n \Delta t)}} {\mathbf{H}^{(n\Delta t)}}^{\hspace{0mm}-1}  \boldsymbol{\Sigma}^{(n \Delta t)} + \boldsymbol{\Sigma^{'}}^{(n\Delta t)}  )^{1/2} \sqrt{\Delta t} \boldsymbol{\zeta}^{[n]} \label{eq:euler-maryuama}
\end{gathered}$$

where $\boldsymbol{\zeta}^{[n]}$ is zero-mean identity-covariance
Gaussian noise with independent time steps.

## Experimental Methods: Low-Dose CT Image Restoration

In this section, we describe an implementation of our proposed method
for low-dose CT image restoration. We used the publicly available Lung
Image Database Consortium (LIDC) dataset, which consists of
three-dimensional image volumes reconstructed from thoracic CT scans for
lung imaging. The first 80% of image volumes were used for training data
and the last 20% were reserved for validation data. We randomly
extracted 8000 two-dimensional axial slices from the training volumes
and 2000 axial slices from the validation volumes. The slices were
registered to a common coordinate system using bilinear interpolation so
that all images are $512\times512$ with $1.0$ mm voxel spacing. The
image values were shifted and scaled such that 0.0 represents -1000 HU
and 10.0 represents 1000 HU. The reason we chose this scale was so that
we can use zero-mean identity-covariance Gaussian noise as the final
time step of scalar diffusion models and the noise standard deviation
will be comparable to the measurements as shown in Figure
[4](#fig:MTF_NPS_vs_time_scalar){reference-type="ref"
reference="fig:MTF_NPS_vs_time_scalar"}. These images were used as the
ground-truth for training and validation; however, it is important to
note that these images still contain errors such as noise, blur, and
artifacts. These errors are part of the target training distribution, so
they may impact the results for the trained diffusion models. Our
approach is to simulate lower-quality images that one may measure with a
low-radiation-dose CT scan by applying convolutional blur and adding
stationary noise. These low-quality CT images represent the output of a
low-dose CT scan, so they are treated as the measurements for the
purposes of this study. Our goal is to train conditional score-based
diffusion models to sample posterior estimate images given low-dose CT
measurements. If successful, the posterior estimate images sampled by
the trained model should have similar image quality to the normal-dose
training images in the LIDC dataset.

For this implementation, we chose to parameterize both MTF and NPS using
a set of band-pass filters. Let the matrix
$\boldsymbol{\mathcal{G}}(h) =  \mathbf{U}^*_\text{DFT} \boldsymbol{\Lambda}_{\boldsymbol{\mathcal{G}}}(h) \mathbf{U}_\text{DFT}$
represent an isotropic Gaussian low-pass filter in the two-dimensional
spatial frequency domain, where $h$ describes the PSF full width at half
maximum in units of $\text{mm}$. That is, the diagonal of
$\boldsymbol{\Lambda}_{\boldsymbol{\mathcal{G}}}(h)$ is the Fourier
transform of a convolutional Gaussian blur kernel proportional to
$\exp{(-\frac{1}{2} (\sqrt{x^2 + y^2})^2 / \sigma^2(h))}$ where
$\sigma(h)=(2\sqrt{2\log{2}})^{-1}h \approx (0.425) h$. We used the
following parameterization of low-pass, band-pass, and high-pass
filters:

$$\begin{gathered}
    \mathbf{H}_\text{LPF} = \boldsymbol{\mathcal{G}}(3.0~\text{mm}) \\
    \mathbf{H}_\text{BPF} = \boldsymbol{\mathcal{G}}(1.0~\text{mm}) - \boldsymbol{\mathcal{G}}(3.0~\text{mm}) \\
    \mathbf{H}_\text{HPF} = \mathbf{I} - \boldsymbol{\mathcal{G}}(1.0~\text{mm}) 
\end{gathered}$$

Our model of low-dose CT measurements, $\mathbf{y}$, given the ground
truth image, $\mathbf{x}$ is

$$\begin{gathered}
    \text{p}(\mathbf{y}|\mathbf{x}^{(0)}) = \mathcal{N}(\mathbf{y}; \mathbf{H}_{\mathbf{y}|\mathbf{x}^{(0)}} \hspace{0.5mm} \mathbf{x}^{(0)}, \boldsymbol{\Sigma}_{\mathbf{y}|\mathbf{x}^{(0)}})\\
    \mathbf{H}_{\mathbf{y}|\mathbf{x}^{(0)}} = (1.0) \enspace \mathbf{H}_\text{LPF}  + (0.5) \enspace \mathbf{H}_\text{BPF}  + (0.1) \enspace \mathbf{H}_\text{HPF} \\ 
    \boldsymbol{\Sigma}_{\mathbf{y}|\mathbf{x}^{(0)}} = (0.1) \enspace \mathbf{H}_\text{LPF}  + (1.0) \enspace \mathbf{H}_\text{BPF}  + (0.5) \enspace \mathbf{H}_\text{HPF}  
\end{gathered}$$

This particular choice of measured MTF and NPS is arbitrary but intended
to roughly match the typical patterns observed in low-dose CT images. In
general, one can substitute the MTF and NPS to match the calibrated
values for a medical imaging system.

In this experiment, we compare two cases: 1) scalar diffusion models
using multiplicative drift and additive white Gaussian noise and 2)
Fourier diffusion models using linear shift invariant systems and
additive stationary Gaussian noise. As shown in
[\[eq:scalar_drift\]](#eq:scalar_drift){reference-type="eqref"
reference="eq:scalar_drift"} and
[\[eq:white_noise\]](#eq:white_noise){reference-type="eqref"
reference="eq:white_noise"}, scalar diffusion models are a special case
of Fourier diffusion models that can be written using time-dependent
scalar matrices. We define the scalar diffusion model using the
following parameters:

$$\begin{gathered}
    \mathbf{H}^{(t)} = (e^{-  
 5t^2})\hspace{1mm} \mathbf{I} \label{eq:scalar_diffusion_model_drift}\\
    \boldsymbol{\Sigma}^{(t)} = (1 - e^{- 10 t^2}) \hspace{1mm} \mathbf{I} \label{eq:scalar_diffusion_model_noise}
\end{gathered}$$

This forward stochastic process begins with $\mathbf{x}^{(0)}$ having
the same distribution as the true images and converges to approximately
zero-mean identify-covariance Gaussian noise at the final time step,
$\mathbf{x}^{(T)}$. For the Fourier diffusion model case, we design the
forward stochastic process so that the final time step has the same
distribution as the low-dose CT measurements. This capability to model
continuous probability flow from true images to measured images is one
of the key benefits of Fourier diffusion models. For this case, we use
the formulae shown below:

$$\begin{gathered}
    \mathbf{H}^{(t)} = (1.0) \enspace \mathbf{H}_\text{LPF}  + (0.5 + 0.5e^{-5t^2}) \enspace \mathbf{H}_\text{BPF}  + (0.1 + 0.9e^{-5t^2}) \enspace \mathbf{H}_\text{HPF} \\ 
    \boldsymbol{\Sigma}^{(t)} = (0.1 - 0.1 e^{- 10 t^2}) \enspace \mathbf{H}_\text{LPF}  + (1.0 - 1.0 e^{- 10 t^2}) \enspace \mathbf{H}_\text{BPF}  + (0.5 - 0.5 e^{- 10 t^2}) \enspace \mathbf{H}_\text{HPF}  
\end{gathered}$$

![Diagram of the score-matching neural network. The inputs are the
forward process sample, low-dose CT measured image, and sample time. The
output is the predicted score.
](figures/score_matching_nn_diagram.png){#fig:unet_diagram width="82%"}

<figure id="fig:MTF_NPS_vs_time_scalar">
<p><img src="figures/MTF_vs_time_scalar.png" style="width:43.0%"
alt="image" /> <img src="figures/NPS_vs_time_scalar.png"
style="width:43.0%" alt="image" /></p>
<figcaption>MTF and NPS vs time for the scalar diffusion model.
</figcaption>
</figure>

<figure id="fig:MTF_NPS_vs_time_fourier">
<p><img src="figures/MTF_vs_time_fourier.png" style="width:43.0%"
alt="image" /> <img src="figures/NPS_vs_time_fourier.png"
style="width:43.0%" alt="image" /></p>
<figcaption>MTF and NPS vs time for the Fourier diffusion model.
</figcaption>
</figure>

For the score-matching neural network, we used the u-net architecture
shown in Figure [3](#fig:unet_diagram){reference-type="ref"
reference="fig:unet_diagram"}. The model inputs are the forward process
sample, the low-dose CT measurements, and the sample time. The model
output is an estimation of the score function. For time encoding, we
applied a multi-layer perceptron to the sample time and converted the
output to constant-valued images. The forward process sample image, the
low-dose CT measured image, and the time encoding images are
concatenated and passed to the score-matching u-net. Each convolutional
block consists of a convolutional layer, rectified linear units
activation, and batch normalization. The final output layer has no
activation function (linear) or batch normalization. Dropout layers were
also applied to each convolutional block with a drop out rate of 20%. We
used the Adam optimizer with a learning rate of $10^{-3}$
[@kingma2014adam]. All machine learning modules were implemented with in
Pytorch [@NEURIPS2019_9015]. We ran 10,000 training epochs, 32 images
per batch, and the training loss function in
[\[eq:score_matching_loss\]](#eq:score_matching_loss){reference-type="eqref"
reference="eq:score_matching_loss"}, where the expectation over
$\mathbf{x}^{(0)}$ is implemented via the sample mean over multiple
training images per batch and the expectation over time, $t$, is
implemented by sampling a different time step independently for each
image so that there are also 32 time samples per batch.

After training, we run the reverse process using
[\[eq:euler-maryuama\]](#eq:euler-maryuama){reference-type="eqref"
reference="eq:euler-maryuama"} for both diffusion models. We discretize
the reverse process with 1024, 512, 256, 128, 64, 32, 16, 8 and 4 time
steps uniformly spaced between $t=0$ and $T$, inclusively. That way, we
can analyze the error due to time discretization for scalar and Fourier
diffusion models. We ran the reverse process 32 times using the same
measurements. For the posterior estimates at $t=0$ of the reverse
process, we compute the mean squared error, mean squared bias, and mean
variance where the mean refers to a spatial average over the image,
error/bias are with respect to the ground truth and the variance refers
to the ensemble of samples from the reverse process.

![Forward and reverse stochastic process for the scalar diffusion model
with 1024 time steps applied to a full CT image.
](figures/process_scalar_1024.png){#fig:process_scalar_1024 width="99%"}

![Forward and reverse stochastic process for the Fourier diffusion model
with 1024 time steps applied to a full CT image.
](figures/process_fourier_1024.png){#fig:process_fourier_1024
width="99%"}

![Forward and reverse stochastic process for the scalar diffusion model
with 16 time steps applied to a full CT image.
](figures/process_scalar_16.png){#fig:process_scalar_16 width="99%"}

![Forward and reverse stochastic process for the Fourier diffusion model
with 16 time steps applied to a full CT image.
](figures/process_fourier_16.png){#fig:process_fourier_16 width="99%"}

![Forward and reverse stochastic process for the scalar diffusion model
with 1024 time steps applied to an image patch showing a lung nodule.
](figures/process_patch_scalar_1024.png){#fig:process_patch_scalar_1024
width="99%"}

![Forward and reverse stochastic process for the Fourier diffusion model
with 1024 time steps applied to an image patch showing a lung nodule.
](figures/process_patch_fourier_1024.png){#fig:process_patch_fourier_1024
width="99%"}

![Forward and reverse stochastic process for the scalar diffusion model
with 16 time steps applied to an image patch showing a lung nodule.
](figures/process_patch_scalar_16.png){#fig:process_patch_scalar_16
width="99%"}

![Forward and reverse stochastic process for the Fourier diffusion model
with 16 time steps applied to an image patch showing a lung nodule.
](figures/process_patch_fourier_16.png){#fig:process_patch_fourier_16
width="99%"}

# Results

An example of the forward and reverse process for scalar diffusion
models is displayed in Figure
[6](#fig:process_scalar_1024){reference-type="ref"
reference="fig:process_scalar_1024"} for a full CT image and Figure
[10](#fig:process_patch_scalar_1024){reference-type="ref"
reference="fig:process_patch_scalar_1024"} for a zoomed patch showing a
lung nodule. This shows the existing method for diffusion models, which
will be our reference to evaluate our new proposed method. The forward
process is initialized, at $t=0$, with the ground truth images. The
signal fades to zero over time in the forward process, and Gaussian
white noise is added at each time step. The final result is
approximately zero-mean identity-covariance Gaussian noise. The score
matching neural network is trained to run the reverse process, sampling
high quality images given low-radiation-dose CT measurements. For the
processes shown in Figure
[6](#fig:process_scalar_1024){reference-type="ref"
reference="fig:process_scalar_1024"} and Figure
[10](#fig:process_patch_scalar_1024){reference-type="ref"
reference="fig:process_patch_scalar_1024"}, we used 1024 time steps to
run the reverse process. Comparing the top row and the bottom row, the
samples from the reverse process appear to have similar image quality to
the forward process. The final result of the reverse process at $t=0$ is
a posterior estimate, or an approximation of the ground truth, given the
low-radiation-dose CT measurements. Examples of the Fourier diffusion
model with 1024 time steps are shown in Figure
[7](#fig:process_fourier_1024){reference-type="ref"
reference="fig:process_fourier_1024"} and
[9](#fig:process_fourier_16){reference-type="ref"
reference="fig:process_fourier_16"}. The forward process for this case
begins with the true images at $t=0$ and converges to the same
distribution as the measured images at $T$. Note the final column of the
Fourier diffusion model shows an example from the same distribution as
the measured images. All the reverse processes in these Figures are for
conditional image generation; so both the scalar and Fourier diffusion
models are guided by measurements with the same image quality shown at
$T$ in the Fourier diffusion models.

Figures [6](#fig:process_scalar_1024){reference-type="ref"
reference="fig:process_scalar_1024"},
[7](#fig:process_fourier_1024){reference-type="ref"
reference="fig:process_fourier_1024"},
[10](#fig:process_patch_scalar_1024){reference-type="ref"
reference="fig:process_patch_scalar_1024"}, and
[11](#fig:process_patch_fourier_1024){reference-type="ref"
reference="fig:process_patch_fourier_1024"} use 1024 time steps, which
means one reverse process sample requires 1024 passes of the
score-matching neural network. Corresponding examples using only 16 time
steps are shown in Figures
[8](#fig:process_scalar_16){reference-type="ref"
reference="fig:process_scalar_16"},
[9](#fig:process_fourier_16){reference-type="ref"
reference="fig:process_fourier_16"},
[12](#fig:process_patch_scalar_16){reference-type="ref"
reference="fig:process_patch_scalar_16"}, and
[13](#fig:process_patch_fourier_16){reference-type="ref"
reference="fig:process_patch_fourier_16"}, respectively. For the case of
the scalar diffusion model with fewer time steps shown in Figure
[8](#fig:process_scalar_16){reference-type="ref"
reference="fig:process_scalar_16"} and Figure
[12](#fig:process_patch_scalar_16){reference-type="ref"
reference="fig:process_patch_scalar_16"}, the image quality in the
reverse process is much worse than the forward process. Comparing the
1024 time step reverse process, shown in Figure
[6](#fig:process_scalar_1024){reference-type="ref"
reference="fig:process_scalar_1024"}, with the 16 time step reverse
process, shown in Figure
[9](#fig:process_fourier_16){reference-type="ref"
reference="fig:process_fourier_16"}, the increased error is most likely
due to time discretization. Figure
[13](#fig:process_patch_fourier_16){reference-type="ref"
reference="fig:process_patch_fourier_16"} shows an example of the
Fourier diffusion model using only 16 time steps for the reverse
process. Notice the improvement in image quality for the Fourier
diffusion model reverse process at $t=0$ in Figure
[9](#fig:process_fourier_16){reference-type="ref"
reference="fig:process_fourier_16"} relative to the Fourier diffusion
model reverse process at $t=0$ in Figure
[8](#fig:process_scalar_16){reference-type="ref"
reference="fig:process_scalar_16"}. The qualitative improvement in image
quality for these two cases shows a convincing visual example of
improved image quality for Fourier diffusion models when using a lower
number of time steps and merits further quantitative image quality
analysis.

Figure [14](#fig:image_quality_metrics){reference-type="ref"
reference="fig:image_quality_metrics"} shows the mean squared error,
mean squared bias, and mean variance for scalar diffusion models and
Fourier diffusion models. Here, the mean refers to spatial average over
the images. The line plot represents the sample mean for the population
of validation images and the shaded region represents one standard
deviation over the population. From these plots, we conclude that
Fourier diffusion models out-perform scalar diffusion models overall.
All three metrics show improved performance for the Fourier diffusion
models. In particular, we note the improved performance at a low number
of time steps. Fourier diffusion models with only 8 time steps achieve
similar mean squared error to scalar diffusion models using 128 or even
1024 time steps. The next section provides explanations and conclusions
for these results.

# Conclusion

The results of the experiments in the previous section show that Fourier
diffusion models achieve higher performance than scalar diffusion models
across multiple image quality metrics and number of time steps. The
improved performance may be related to the greater apparent similarity
between the initial images at $t=0$ and final images at $T$ for Fourier
diffusion models relative to scalar diffusion models. It follows that
the reverse process updates for the Fourier diffusion model are smaller
than those of the scalar diffusion model, which may result in improved
performance for a neural network with a fixed number of parameters.
Intuitively, some denoising problems are harder than others and harder
denoising problems require more computational power. The neural network
used for the scalar diffusion model reverse updates must dedicate some
of its computational power to inverting the imagined artificial process
of the image signal fading to zero; whereas the Fourier diffusion model
reverse updates are completely dedicated to moving the measured image
distribution towards the true image distribution. Another possible
explanation is the similarity between the LSI systems of the Fourier
diffusion model and the convolutional layers of the neural network. It
is possible that convolutional neural networks are better suited to
model local sharpening and denoising operations of the Fourier diffusion
model reverse updates, as opposed to the image-wide effects in the
scalar diffusion models.

While this work was originally motivated by the goal of controlling MTF
and NPS in the forward process, we note that the derivations have not
relied on any special properties of circulant matrices. Therefore, we
believe it should be possible to train score-based generative machine
learning models defined by any Gaussian stochastic process composed of
linear systems and additive Gaussian noise without specifying shift
invariant systems or stationary noise. So far, we have only tested the
machine learning implementation with Fourier diffusion models. In future
work, we hope to explore new applications of the more general model.

Our final conclusion is that Fourier diffusion models have the potential
to improve performance for conditional image generation relative to
conventional scalar diffusion models. Fourier diffusion models can apply
to medical imaging systems that are approximately shift invariant with
stationary Gaussian noise. For the low-radiation-dose CT image
restoration example, these improvements have the potential to improve
image quality, diagnostic accuracy and precision, and patient health
outcomes while keeping radiation dose at a suitable level for patient
screening applications. We look forward to exploring new medical imaging
applications of Fourier diffusion models in the future.

<figure id="fig:image_quality_metrics">
<p><img src="figures/discrete_time_root_mean_squared_error.png"
style="width:32.0%" alt="image" /> <img
src="figures/discrete_time_root_mean_squared_bias.png"
style="width:32.0%" alt="image" /> <img
src="figures/discrete_time_root_mean_variance.png" style="width:32.0%"
alt="image" /></p>
<figcaption>Mean squared bias vs number of time steps for the Fourier
diffusion model and the scalar diffusion model. The shaded region shows
one standard deviation of the metric over the population of validation
images. </figcaption>
</figure>

# Appendix A {#appendix-a .unnumbered}

Consider a stochastic process defined by the following

$$\mathbf{x}^{(t)} = \mathbf{H}^{(t)} \mathbf{x}^{(0)} + {\boldsymbol{\Sigma}^{(t)}}^{\hspace{0mm}  1/2} \boldsymbol{\epsilon}^{(t)} 
\label{eq:x_t_general}$$

where $\mathbf{H}^{(t)}$ and $\boldsymbol{\Sigma}^{(t)}$ are
time-dependent square matrices and $\boldsymbol{\epsilon}^{(t)}$ is a
zero-mean identity-covariance Gaussian random process with independent
non-overlapping time intervals. We assume
$\mathbf{H}^{(0)} = \mathbf{I}$ and
$\boldsymbol{\Sigma}^{(0)} = \mathbf{0}$, $\mathbf{H}^{(t)}$ is
invertible, and
$\boldsymbol{\Sigma}^{(t+\Delta t)}\geq \mathbf{H}^{(t+\Delta t)}{\mathbf{H}^{(t)}}^{-1}\boldsymbol{\Sigma}^{(t)}{{\mathbf{H}^T}^{(t)}}^{-1} {\mathbf{H}^T}^{(t+\Delta t)}$
for all elements.

For the continuous forward stochastic process, $\mathbf{x}^{(t)}$
defined in [\[eq:x_t\]](#eq:x_t){reference-type="eqref"
reference="eq:x_t"}, we can write the update for a time step $\Delta t$
by combining
[\[eq:discrete_forward_update_MTF_NPS\]](#eq:discrete_forward_update_MTF_NPS){reference-type="eqref"
reference="eq:discrete_forward_update_MTF_NPS"} and
[\[eq:sample_discrete\]](#eq:sample_discrete){reference-type="eqref"
reference="eq:sample_discrete"} as follows: $$\begin{gathered}
    \mathbf{x}^{(t+\Delta t)} = \mathbf{H}^{(t + \Delta t)} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1} \mathbf{x}^{(t)} + (\boldsymbol{\Sigma}^{(t + \Delta t)} -  {\mathbf{H}^{(t + \Delta t)}}^{\hspace{-0mm} 2} {\mathbf{H}^{(t)}}^{\hspace{0mm}-2} \boldsymbol{\Sigma}^{(t)} )^{1/2} \boldsymbol{\eta}^{(t)}
    \label{eq:x_t_plus_delta_t}
\end{gathered}$$

where $\boldsymbol{\eta}^{(t)}$ is a zero-mean identity-covariance
Gaussian process. Subtracting $\mathbf{x}^{(t)}$ yields

$$\begin{gathered}
    \mathbf{x}^{(t + \Delta t)} - \mathbf{x}^{(t)}  =  (\mathbf{H}^{(t + \Delta t)} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1} - \mathbf{I}) \mathbf{x}^{(t)} + (\boldsymbol{\Sigma}^{(t + \Delta t)} -  {\mathbf{H}^{(t+\Delta t)}}^{\hspace{-0mm} 2} {\mathbf{H}^{(t)}}^{\hspace{-3mm}-2} \boldsymbol{\Sigma}^{(t)} )^{1/2} \boldsymbol{\eta}^{(t)}
    \label{eq:x_t_difference}
\end{gathered}$$

The first term of
[\[eq:x_t_difference\]](#eq:x_t_difference){reference-type="eqref"
reference="eq:x_t_difference"} can be algebraically rearranged as
follows:

$$\begin{gathered}
    (\mathbf{H}^{(t + \Delta t)} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1} - \mathbf{I}) \mathbf{x}^{(t)} \\
    (\mathbf{H}^{(t + \Delta t)} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1} - \mathbf{H}^{(t)} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1}) \mathbf{x}^{(t)} \\
    (\mathbf{H}^{(t + \Delta t)}  - \mathbf{H}^{(t)}) {\mathbf{H}^{(t)}}^{\hspace{0mm}-1} \mathbf{x}^{(t)} \\
    \frac{\mathbf{H}^{(t + \Delta t)}  - \mathbf{H}^{(t)}}{\Delta t} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1} \mathbf{x}^{(t)} \Delta t \label{eq:first_term}
\end{gathered}$$

Taking the limit of
[\[eq:first_term\]](#eq:first_term){reference-type="eqref"
reference="eq:first_term"} as $\Delta t$ approaches zero yields

$$\begin{gathered}
    \lim_{\Delta t \rightarrow 0} \frac{\mathbf{H}^{(t + \Delta t)}  - \mathbf{H}^{(t)}}{\Delta t} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1} \mathbf{x}^{(t)} \Delta t = \mathbf{H^{'}}^{(t)}{\mathbf{H}^{(t)}}^{\hspace{0mm}-1} \mathbf{x}^{(t)} \text{dt}
\end{gathered}$$

where
$\mathbf{H^{'}}^{(t)} = \frac{\text{d}}{\text{dt}} \mathbf{H}^{(t)}$

The second term of
[\[eq:x_t_difference\]](#eq:x_t_difference){reference-type="eqref"
reference="eq:x_t_difference"} can also be algebraically rearranged as
follows:

$$\begin{gathered}
    (\boldsymbol{\Sigma}^{(t + \Delta t)} -  {\mathbf{H}^{(t+\Delta t)}}^{\hspace{-0mm} 2} {\mathbf{H}^{(t)}}^{\hspace{0mm}-2} \boldsymbol{\Sigma}^{(t)} )^{1/2} \boldsymbol{\eta}^{(t)}  \\
    (\boldsymbol{\Sigma}^{(t + \Delta t)} - ( {\mathbf{H}^{(t+\Delta t)}} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1})^2 \boldsymbol{\Sigma}^{(t)} )^{1/2} \boldsymbol{\eta}^{(t)}  \\
    (\boldsymbol{\Sigma}^{(t + \Delta t)} - (\mathbf{I} +  {\mathbf{H}^{(t+\Delta t)}}{\mathbf{H}^{(t)}}^{\hspace{0mm}-1} - \mathbf{I})^{2} \boldsymbol{\Sigma}^{(t)} )^{1/2} \boldsymbol{\eta}^{(t)}  \\
    (\boldsymbol{\Sigma}^{(t + \Delta t)} - (\mathbf{I} +  {\mathbf{H}^{(t+\Delta t)}}{\mathbf{H}^{(t)}}^{\hspace{0mm}-1} -  {\mathbf{H}^{(t)}}{\mathbf{H}^{(t)}}^{\hspace{0mm}-1})^{2} \boldsymbol{\Sigma}^{(t)} )^{1/2} \boldsymbol{\eta}^{(t)}  \\
    (\boldsymbol{\Sigma}^{(t + \Delta t)} - (\mathbf{I} +  ({\mathbf{H}^{(t+\Delta t)}} -  {\mathbf{H}^{(t)}}) {\mathbf{H}^{(t)}}^{\hspace{0mm}-1})^{2} \boldsymbol{\Sigma}^{(t)} )^{1/2} \boldsymbol{\eta}^{(t)}  \\
    (\boldsymbol{\Sigma}^{(t + \Delta t)} - (\mathbf{I} +  \frac{{\mathbf{H}^{(t+\Delta t)}} -  {\mathbf{H}^{(t)}}}{\Delta t} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1} \Delta t)^{2} \boldsymbol{\Sigma}^{(t)} )^{1/2} 
    \boldsymbol{\eta}^{(t)} \\
    (\frac{\boldsymbol{\Sigma}^{(t + \Delta t)} - (\mathbf{I} + 
    2\frac{\mathbf{H}^{(t+\Delta t)} -  {\mathbf{H}^{(t)}}}{\Delta t} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1} \Delta t + \mathcal{O}(\Delta t^2)) \boldsymbol{\Sigma}^{(t)}}{\Delta t} )^{1/2}
     \sqrt{\Delta t} \enspace \boldsymbol{\eta}^{(t)} \\
    (-  2 \frac{{\mathbf{H}^{(t+\Delta t)}} -  {\mathbf{H}^{(t)}}}{\Delta t} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1}  \boldsymbol{\Sigma}^{(t)} + \frac{\boldsymbol{\Sigma}^{(t + \Delta t)} - \boldsymbol{\Sigma}^{(t)}}{\Delta t}  - \frac{\mathcal{O}(\Delta t^2)}{\Delta t} \boldsymbol{\Sigma}^{(t)}  )^{1/2} \sqrt{\Delta t} \enspace \boldsymbol{\eta}^{(t)} \label{eq:second_term}
\end{gathered}$$

where $\mathcal{O}(\Delta t^2)$ indicates second order and higher terms
of the Taylor expansion. Note, we have made the approximation that
$\frac{\mathbf{H}^{(t+\Delta t)} -  {\mathbf{H}^{(t)}}}{\Delta t}$ can
be considered as a constant with respect to $\Delta t$ for the purposes
of the Taylor expansion, which is valid for continuously differentiable
functions of time in the limit as $\Delta t$ approaches zero. Taking the
limit of [\[eq:second_term\]](#eq:second_term){reference-type="eqref"
reference="eq:second_term"} as $\Delta t$ approaches zero yields

$$\begin{gathered}
    \lim_{\Delta t \rightarrow 0} (-  2 \frac{{\mathbf{H}^{(t+\Delta t)}} -  {\mathbf{H}^{(t)}}}{\Delta t} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1}  \boldsymbol{\Sigma}^{(t)} + \frac{\boldsymbol{\Sigma}^{(t + \Delta t)} - \boldsymbol{\Sigma}^{(t)}}{\Delta t} + \frac{\mathcal{O}(\Delta t^2)}{\Delta t} \boldsymbol{\Sigma}^{(t)})^{1/2} \sqrt{\Delta t} \enspace \boldsymbol{\eta}^{(t)} \\
    (-  2 {\mathbf{H^{'}}^{(t)}} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1}  \boldsymbol{\Sigma}^{(t)} + \boldsymbol{\Sigma^{'}}^{(t)}  )^{1/2} \mathbf{dw}
\end{gathered}$$

where
$\boldsymbol{\Sigma^{'}}^{(t)} = \frac{\text{d}}{\text{dt}} \boldsymbol{\Sigma}$
and $\mathbf{dw}$ is infinitesimal white Gaussian noise with covariance,
$\text{dt} \mathbf{I}$. Therefore, the limit of
[\[eq:x_t_difference\]](#eq:x_t_difference){reference-type="eqref"
reference="eq:x_t_difference"} as $\Delta t$ approaches zero is:

$$\mathbf{dx} = \mathbf{H^{'}}^{(t)}{\mathbf{H}^{(t)}}^{\hspace{0mm}-1} \mathbf{x}^{(t)} \text{dt} + (-  2 {\mathbf{H^{'}}^{(t)}} {\mathbf{H}^{(t)}}^{\hspace{0mm}-1}  \boldsymbol{\Sigma}^{(t)} + \boldsymbol{\Sigma^{'}}^{(t)}  )^{1/2} \mathbf{dw}$$

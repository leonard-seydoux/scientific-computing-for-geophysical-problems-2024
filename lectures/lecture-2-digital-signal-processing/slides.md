---
marp: true
theme: presentation
math: mathjax
paginate: true
---

<!-- _class: titlepage -->

# Scientific computing for geophysical problems

### 2. Digital signal processing

`Léonard Seydoux` 

September 2024 at the [institut de physique du globe de Paris](https://www.ipgp.fr). 

<div class="logo">

![height:100px](images/logo/ipgp-upcite.svg) 
![height:90px](https://avatars.githubusercontent.com/u/20685754?s=280&v=4) 
![height:90px](https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg)

</div>

[<img src="images/logo/logo-github.svg" width=25 align="left" style="margin-top:10px; margin-right: -23px;"/>  `leonard-seydoux/scientific-computing-for-geophysical-problems`](https://github.com/leonard-seydoux/scientific-computing-for-geophysical-problems) 

---

## Analog and digital signals

<div>

- Analog signals $x(t)$ are continuous in time and amplitude
- Digital signals $x[n]$ are discrete in time and amplitude (sampling rate, vertical resolution)

</div>

![width:600](images/signal/digital.svg)

---

## Digital signal processing in music


<div>

24 bits
<br>

<audio controls>
  <source src="data/jazz-piano-24-bits.wav" type="audio/wav">
</audio> 

correspond to 16,777,216 
levels and 1.6MB

</div>
<div>

8 bits 
<br>

<audio controls>
  <source src="data/jazz-piano-8-bits.wav" type="audio/wav">
</audio>

correspond to 256 levels, 
but only 95kB

</div>

---

## Signal processing as a system

A system $\mathcal{S}$ maps an input signal $x(t)$ to an output signal $y(t)$ 
<img src="images/system/system.svg" style="scale:1.5; margin-top:70px;">

---

## Linear systems


The system $\mathcal{S}$ is __linear__ if it satisfies the following property
<img src="images/system/linear-system.svg" style="scale:1.5; margin-top:70px;">

---

## Time-invariant systems

The system $\mathcal{S}$ is __time-invariant__ if it satisfies the following property
<img src="images/system/time-invariant-system.svg" style="scale:1.5; margin-top:70px;">

---

## Linear and time-invariant systems

$\mathcal{S}$ is linear and time-invariant (LTI) if it satisfies both properties:

- Linearity: $\mathcal{S}\{a x_1(t) + b x_2(t)\} = a y_1(t) + b y_2(t)$
- Time-invariance: $\mathcal{S}{x(t - \tau)} = y(t - \tau)$

> Linear and time-invariant systems are fully modeled by the __impulse response__

--- 

## Impulse response

The impulse response $h(t)$ of a system $\mathcal{S}$ is the output<br>of the system when the input is the Dirac delta function $\delta(t)$
<img src="images/system/impulse-response.svg" style="scale:1.5; margin-top:70px;">

--- 

## Impulse response

The impulse response $h(t)$ of a system $\mathcal{S}$ is the output<br>of the system when the input is the Dirac delta function $\delta(t)$
<img src="images/system/impulse-response-translated.svg" style="scale:1.5; margin-top:70px;">

--- 

## Impulse response

The impulse response $h(t)$ of a system $\mathcal{S}$ is the output<br>of the system when the input is the Dirac delta function $\delta(t)$
<img src="images/system/impulse-response-super.svg" style="scale:1.5; margin-top:70px;">


--- 

## Convolution

The impulse response $h(t)$ of a system $\mathcal{S}$ is the output<br>of the system when the input is the Dirac delta function $\delta(t)$
<img src="images/system/impulse-response-sum.svg" style="scale:1.5; margin-top:70px;">

---

## Convolution

<span>

The output signal $y(t)$ of a system $\mathcal{S}$ is the convolution 
of the input signal $x(t)$ with the impulse response $h(t)$ such as
<br>
$$y(t) = \int_{-\infty}^{+\infty} x(\tau) h(t - \tau) d\tau$$

</span>

<img src="images/system/convolution.svg" style="scale:1.5; margin-top:70px;">

---

## Example: adding reverb to a piano record


</div>

<div style="flex-basis:40%">

The impulse response 
$i(t)$ of a large hall

<br>

<audio controls>
  <source src="data/hall-impulse-response.wav" type="audio/wav">
</audio>

</div>

<div style="flex-basis:40%">

The original piano 
record $x(t)$ in a dry room

<br>

<audio controls>
  <source src="data/jazz-piano-24-bits.wav" type="audio/wav">
</audio>

</div>

<div>

Convolution theorem to get the piano record $y(t)$ as if it was played in the hall
<br>

$$y(t) = \frac{1}{T} \int_{0}^T x(\tau) i(t - \tau) d\tau$$


</div>

<div style="flex-basis:40%">

The piano record $y(t)$ as if 
it was played in the hall

<br>

<audio controls>
  <source src="data/jazz-piano-with-reverb.wav" type="audio/wav">
</audio>

</div>

--- 

## Fourier transform (continuous)

<span>

The Fourier transform of a signal $x(t)$ is defined as
<br>
$$\hat{x}(f) = \int_{-\infty}^{+\infty} x(t) e^{-i \omega t} dt$$
<br>

- The Fourier transform is linear: $\mathcal F\{a x_1(t) + b x_2(t)\} = a \hat{x}_1(\omega) + b \hat{x}_2(\omega)$
- Convolution in time is multiplication in frequency: $\mathcal F\{x(t) * h(t)\} = \hat{x}(\omega) \hat{h}(\omega)$

</span>

--- 

## Sampling 

<div>

The sampling of a continuous signal $x(t)$ at a every period $\Delta t$ is defined as

<br>

$$x[n] = x(t) \mathrm{Ш}\,_{\Delta t}(t)$$

<br>

where the sampling rate is defined as $f_s = \Delta t^{-1}$. 

</div>

![width:600](images/signal/digital.svg)

---

## Discrete-time Fourier transform

<div>

The Discrete-time Fourier transform (DTFT) of a discrete signal $x[n]$ is defined as

<br>

$$\hat{x}(\omega) = \mathcal{F}x[n] =\sum_{n=-\infty}^{+\infty} x[n] e^{-i \omega n}$$

<br>

- The DTFT is linear: $\mathcal F\{a x_1 + b x_2\}(\omega) = a \hat{x}_1(\omega) + b \hat{x}_2(\omega)$
- Convolution in time is multiplication in frequency: $\mathcal F\{x * h\}(\omega) = \hat{x}(\omega) \hat{h}(\omega)$

</div>

---

## Sampling and aliasing

<div>

In the spectral domain, sampling duplicates the spectrum at every multiple of the sampling rate $\omega_s$.

<br>

$$\hat{x}(\omega) = \sum_{n=-\infty}^{+\infty} x(\omega - n \omega_s)$$

<br>

__Sampling theorem__: a signal can be reconstructed if $\omega_s \ge 2\omega_{max}$

</div>

![width:600](images/signal/digital.svg)

---

<!-- _class: titlepage -->


![bg 20%](https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg)


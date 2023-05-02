# INTRODUCTION {#sec:intro}

Multi-contrast agent imaging is an active area of research. For example,
iodine and gadolinium have been used together in various applications
including multi-phase kidney and liver imaging [@symons2017photon],
colonography [@muenzel2016spectral], and post-operational imaging for
endovascular aneurysm repair [@dangelmaier2018experimental] among
others. Iodine, gold, and calcium phosphate have also been used as
target materials to study atherosclerotic plaque composition
[@cormode2010atherosclerotic; @baturin2012spectral]. New technology
which allows for decomposition into more material components or improved
low-concentration estimation will greatly benefit the field of
multi-contrast imaging.

![Spectral CT using moving spatial-spectral filters and
energy-integrating
detectors.](latex_source/tivnan2019physical/figures/system_figure_new){#fig:spectralCT width="50%"}

Developments have focused on incorporating different and varied spectral
sensitivities into measurements to enable *spectral CT*. Methods include
dual sources [@flohr2006first], kV-switching [@xu2009dual], split
filters [@rutt1980split], dual-layer-detectors [@carmi2005material], and
photon-counting detectors [@schlomka2008experimental]. With the
exception of photon-counting, these methods typically offer only two
spectral channels.

A new method to enable spectral CT with ordinary energy-integrating
detectors is shown in Figure [1](#fig:spectralCT){reference-type="ref"
reference="fig:spectralCT"}. Specifically, a "spatial-spectral" filter,
composed of a repeating pattern of K-edge filter materials, is placed in
front of the x-ray source dividing the full x-ray beam into spectrally
varied beamlets [@stayman2018model]. The filter is translated parallel
to the detector as the CT gantry rotates to provide spatially interlaced
projection data with different spectral channels. Since each spectral
channel is sparse, conventional reconstruction methods involving
material decomposition in the projection domain or the image domain are
ill-suited for data processing. In contrast, a model-based material
decomposition (MBMD) algorithm [@tilley2018model] permits simultaneous
processing of all data as well as sophisticated regularization schemes
(e.g. compressed sensing) to overcome traditional sampling limitations.
Advantages of the spatial-spectral filter include flexibility in
spectral shaping, scaling to include more spectral channels, and
possible low-cost integration into current CT systems. There is also the
potential to combine this approach with other approaches to extend
low-concentration performance.

Preliminary investigations [@stayman2018model] demonstrated the
feasibility of the spatial-spectral filtering approach under highly
idealized conditions. In this work, a more accurate physical model is
developed, taking into account focal spot effects and motion blur
associated with the moving filter. This more accurate model is used to
investigate potential performance limitations and to guide future
spatial-spectral CT system design. Performance is evaluated in a
multi-contrast digital CT phantom across a range of practical focal spot
sizes and filter speeds.

# METHODS

## General Physical Model for CT Acquisitions with Spatial-Spectral Filters

A general forward model for spectral CT with varied spectral
sensitivities and energy-integrating detectors is

$$\begin{gathered}
    y_i = y(u_i,\theta_i) = \int_E S(u_i,\theta_i,E) \exp{\Big(-\sum_j l_j(u_i,\theta_i)  \mu_j(E) \Big)} dE, \quad \quad \\
    \mu_j(E) = \sum_k  {\rho}_{j,k} q_k(E) 
\end{gathered}$$

where $y_i$ is the $i^{th}$ measurement, a sample of the projection
data, $y(u_i,\theta_i)$, at detector position, $u_i$, and rotation
angle, $\theta_i$. The system spectral response, $S$, is
measurement-dependent, $l_j$ are projection contributions of the
$j^{th}$ voxel, and $\mu_j(E)$ is the energy-dependent attenuation
coefficient of the $j^{th}$ voxel. The latter coefficient is modeled as
a weighted sum over material index $k$ of material basis functions
$q_k(E)$ weighted by material densities, $\rho_{j,k}$ for each voxel.
This model is extremely general. For example, for a kV-switching CT
system, $S(u,\theta, E)$ is equal to a high-kV spectrum, $s_H(E)$ for
$\theta_i$ with odd indexes and to a low-kV spectrum $s_L(E)$ for
$\theta_i$ with even views. One may define $S$ for an ideal
spatial-spectral CT system as $S(u,\theta,E) = S_0(u+f(\theta),E)$ where
$S_0(u,E)$ represents a spatial function of all beamlet energies across
the detector (found, e.g., by computing the polyenergetic spectrum that
exits each sub-filter). The filter is translated laterally according to
the function $f(\theta)$ with rotation angle. Unfortunately this model
excludes some important physical aspects of a real spatial-spectral
system.

First, realistic x-ray focal spots are extended resulting in blur of
objects in the imaging system. While such blur is relatively minor for
objects at the center of the field-of-view, filters placed near the
x-ray source will "see" significant blur due to magnification effects.
This has the effect of mixing the spectra of neighboring beamlets as
illustrated in Figure [2](#fig:focalSpotBlur){reference-type="ref"
reference="fig:focalSpotBlur"} and
[3](#fig:spectra){reference-type="ref" reference="fig:spectra"}. For a
thin filter and a flat detector, this blur is accurately modeled by a
convolution applied to the ideal spectrum $S_0$. We approximate the
shape of a realistic focal spot distribution using a rectangular kernel,
$h_{FS}(u)$, with width equal to the focal spot width magnified by the
ratio between the filter-to-detector distance and the source-to-filter
distance. The second important effect involves the moving filter. In
particular, for realis andtic CT gantry rotation rates, step-and-shoot
motion of the filter is impractical. We consider the more realistic case
where the filter moves continuously including during the detector
integration interval. For a fixed gantry rotational speed, the
spatial-spectral sampling profile is defined by
$f(\theta) = \alpha \enspace\theta$, where $\alpha$ is proportional to
filter speed. This motion changes both the spatial-spectral sampling
(Figure [5](#fig:spectralSampling){reference-type="ref"
reference="fig:spectralSampling"}) and imparts an additional spatial
blur of spectra (Figure [4](#fig:filterMotionBlur){reference-type="ref"
reference="fig:filterMotionBlur"}) which we model by a convolution with
a second kernel $h_M(u)$. We module this kernel as rectangular with
width equal to the distance the filter moves per view magnified by the
ratio of the source-to-detector distance and the source-to-filter
distance.

<figure id="fig:spectralBlur">
<figure id="fig:focalSpotBlur">
<img src="latex_source/tivnan2019physical/figures/focal_spot_blur" style="height:38mm" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figure id="fig:spectra">
<img src="latex_source/tivnan2019physical/figures/spectra" style="height:38mm" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figure id="fig:filterMotionBlur">
<img src="latex_source/tivnan2019physical/figures/filter_motion_blur" style="height:38mm" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figure id="fig:spectralSampling">
<img src="latex_source/tivnan2019physical/figures/sampling_figure" style="height:38mm" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figcaption>Focal spot (a), filter motion (c), spectral blur (b), and
the spectral response, <span
class="math inline"><em>S</em>(<em>u</em>,<em>θ</em>,<em>E</em>)</span>
(d).</figcaption>
</figure>

Thus, the overall spectral model with these physical effects modeled may
be written as

$$S(u,\theta,E) = h_M(u) * h_{FS}(u) * S_0 (u + f(\theta),E)$$

Any filter motion pattern can be chosen and modeled by $f(\theta)$, but
for the remainder of this work we will consider the constant-speed
linear filter motion given by $f(\theta) = \alpha \enspace\theta$. The
compounding effect of the operators $h_M(u) * (\cdot{})$ and
$h_{FS}(u) * (\cdot{})$ will be a blur of spectra in the projection
domain. Although the extended focal spot and filter motion can be
characterized separately, both effects occur together in physical
acquisitions. The experimental methods presented below include both
filter motion effects and focal spot blur despite the diagram in Figure
[6](#fig:spectralBlur){reference-type="ref"
reference="fig:spectralBlur"} which shows the two effects separately.

## Simulation Study on Spatial-Spectral Performance

In general, one would expect that the mixed spectral responses will
degrade the ability to separate different materials even when those
spectral are appropriately modeled. The overall impact of filter speed
on material decomposition performance is potentially more complex. If
the filter speed is zero, the spatial-spectral sampling is poor. (E.g.,
For a static filter, the central detector measurement will only be
probed by a single spectral channel.) However, as the filter speed
increases, the motion blur effects are more dramatic. For extremely fast
motion, all spectral channel can potentially blur together. Between
these two extremes and within the constraints of realistic filter speeds
there may be an optimum filter speed. Numerical experiments were
employed to characterize the impact of these physical effects on MBMD
estimation performance.

::: {#tab:CTgeom}
  source-filter distance      380 mm
  --------------------------- ------------------
  source-isocenter distance   890 mm
  source-detector distance    1040 mm
  gantry rotation speed       120 RPM
  views per rotation          360
  projections per view        512
  pixel size                  0.556 mm
  image space dimensions      128 $\times$ 128
  voxel size (square)         0.5 mm

  : Geometry and sampling.
:::

The geometry and sampling conditions for the studies are summarized in
Table [1](#tab:CTgeom){reference-type="ref" reference="tab:CTgeom"}. A
digital phantom (Figure [7](#fig:numericalPhantom){reference-type="ref"
reference="fig:numericalPhantom"}) of a 100 mm diameter water cylinder
and several 15mm diameter cylindrical inserts containing various
mixtures of iodine, gold, and gadolinium was employed. The outer ring
includes single-contrast inserts of 0.5-4.0 mg/mL concentrations. The
inner ring includes mixtures of 1.0 mg/mL and 2.0 mg/mL for all
combinations of two materials. The center of the phantom also includes
10.0 mg/mL single voxel impulses of each material for regularization
tuning.

<figure id="fig:numericalPhantom">
<p><img src="latex_source/tivnan2019physical/figures/recon_image_ground_truth" alt="image" /> <span
id="fig:images_ground_truth" label="fig:images_ground_truth"></span></p>
<figcaption>Ground truth of the numerical phantom. Magenta text
indicates the density in mg/mL of iodine, gold, or gadolinium
(corresponding to image subtitle) in cylindrical inserts.</figcaption>
</figure>

The filter materials were chosen based on a previous study
[@stayman2018model] that tested all three and four filter-material
combinations to maximize multi-contrast-agent concentration estimation
performance. Specifically, we select a filter comprised of
0.25 mm-thick, 1.46 mm-wide strips of bismuth, gold, lutetium, and
erbium. Thus, each spectral beamlet covers an area on the detector that
is 8 pixels wide. Incident fluence was uniform across the filter and the
level was adjusted such that the bare-beam fluence for the
bismuth-filtered beamlet was $10^5$ photons/pixel.

In the first numerical experiment, we simulated focal spot widths of
0.2-4.0 mm and held filter motion speed constant at 131.4 mm/s which
corresponds to one detector pixel per view after magnification. In the
second experiment, we simulated filter motion speeds between 50-450 mm/s
and held the focal spot width constant at 0.4 mm.

We used the MBMD algorithm with the new models for focal spot blur and
filter motion effects to reconstruct density distributions for the four
materials present in the phantom. All numerical experiments used 1000
iterations of the algorithm. We used a quadratic regularizer with
material-dependent regularization strengths which were tuned such that
the FWHM of the PSF corresponding to the 10.0 mg/mL voxel impulse was
1.8 mm $\pm$ 0.2 mm for all target materials, focal spot widths, and
filter motion speeds. Importantly, our current aim is not to analyze the
impact of a mis-match between the reconstruction model used by the MBMD
algorithm and the true acquisition parameters. Rather, we implement a
matched reconstruction model and aim to characterize the image
degradation when transition between spectra is blurred.

Root-mean-squared error (RMSE) was used for analysis and was computed by
first finding the RMSE within regions of interest (ROIs) inside each
cylindrical insert and then taking the mean across all ROIs. This was
done separately for each target material.

# RESULTS

In the imaging results for the focal spot experiment, the 0.2 mm and
1.0 mm focal spot width cases are very difficult to distinguish by eye.
Overall the reconstructed densities are a reasonable approximation of
the ground truth. The low-contrast 0.5 mg/mL insert is visible in both
cases which implies that the spatial-spectral filter has the potential
to improve sensitivity to lower concentrations. A focal spot size of
1.0mm is fairly standard, so the material decomposition appears to be
effective in the presence of realistic focal spot blur effects.

<figure id="fig:trendPlots">
<figure id="fig:trendFocalSpot">
<img src="latex_source/tivnan2019physical/figures/trend_FocalSpotWidth_vs_RMSE" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figure id="fig:trendFilterSpeed">
<img src="latex_source/tivnan2019physical/figures/trend_FilterSpeed_vs_RMSE" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figcaption>Error vs focal spot blur (a) and filter motion
(b).</figcaption>
</figure>

For focal spot widths between 0.2-1.0 mm, the final RMSE values were
less than 0.35 mg/mL for each material. In the case of gadolinium, the
RMSE was less than 0.18 mg/mL for this range. Larger-than-average focal
spot widths such as 2.0 mm and 4.0 mm resulted in RMSE values around
0.47 mg/mL. The overall trend shows that larger focal spot widths lead
to greater error. However, the change in RMSE is less than 15% between
the 0.2 mm and 1.0 mm cases for any individual contrast agent so the
impact is not severe. One notable error is the insert containing
4.0 mg/mL of iodine on the left side of the image. The reconstructed
density of iodine is underestimated at 2.75 mg/mL and around 1.25 mg/mL
is erroneously attributed to gold. This could indicate that for the
given combination of filter materials, iodine and gold are particularly
difficult to distinguish. This issue may also be improved with a more
sophisticated regularization scheme, more spectral channels, or higher
fluence.

In the filter speed experiment, RMSE consistently decreased for higher
filter speeds. For all materials, the RMSE of the 450 mm/s filter speed
was around 40% lower than the 50 mm/s case. This result would seem to
indicate that in the realistic range of filter motion speeds, the
benefits of improved spatial-spectral sampling outweigh the negatives of
filter motion spectral blur.

<figure id="fig:reconExample">
<p><img src="latex_source/tivnan2019physical/figures/recon_image_FSW_1000" alt="image" /> <span
id="fig:sub1" label="fig:sub1"></span></p>
<figcaption>Example of a material decomposition result for a focal spot
width of 1.0mm.</figcaption>
</figure>

# Conclusion

The focal spot blur experiment has shown that error increases as focal
spot size increases. However, for realistic focal spot sizes between
0.2 mm and 1.0 mm there is a relatively small change in performance.
This suggests that, as long as spectra are modeled for each measurement,
spatial-spectral filters are viable for use with a range of realistic
x-ray sources. The filter motion experiment demonstrates the importance
of the spatial-spectral sampling pattern on the MBMD algorithm's ability
to separate various target materials. Spectral blur effects from filter
motion were shown to be outweighed by the benefits of improved sampling
for the filter speed range in the study. Overall, error decreased as
filter speed increased - giving finer spectral sampling over projection
angles. Knowledge of this performance trade-off will be valuable for the
next stages of this work where choice of filter speed must be balanced
with the realistic range of speeds that can be precisely controlled in a
CT acquisition. For example linear motor have been investigated for CT
filter actuation with speeds up to 5000 mm/s but this is not necessarily
achievable with sufficient precision or within acceleration constraints.

In light of the results presented in this work, it would be prudent to
revisit the optimization of filter design with this improved physical
model. The order of filter materials may now have a greater impact since
the spectral blur occurs between neighboring beamlets. We will also need
to characterize the impact of reconstruction model mismatches and
develop calibration methods. As we build upon our understanding of this
new technology, we move closer to the physical implementation of a
spatial-spectral filter system with the ultimate goal of heightening
sensitivity to low concentrations and improving material discrimination
for multi-contrast-enhanced CT.

This work was supported, in part, by NIH grant R21EB026849.

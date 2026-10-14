# Optimal lattices as the sampling reference in 3D, 4D and 5D

Status: analysis only; nothing here is implemented yet. It follows [sampling-density.md](sampling-density.md).

## Question

The coverage and Nyquist numbers in this project use a hypercubic reference: Cartesian k-space cells of $1/\text{FOV}$, and the relaxation time $\tau$ and scan time $T$ binned separately, which is the product of 1D Nyquist limits. For a signal whose dual-domain support is a ball, the most efficient sampling lattice is the dual of the densest ball packing (D4 in 4D, D5 in 5D), which needs fewer samples. Does using that reference change the analysis?

## Short answer

It changes the **absolute** numbers, but not the derivations, the exponents, or the ranking of designs. Whether the 4D and 5D gains apply at all depends on the shape of the signal's support, which the analysis has not yet modelled.

## 1. What the lattices buy

Samples needed by the optimal sampling lattice for a ball-shaped support, relative to the hypercubic grid. In general this is $2^{n/2-1}$ for the $D_n$ family.

| Dimension | Densest packing (sampling lattice: its dual) | Fewer samples than hypercubic |
|---|---|---|
| 3 | FCC $= D_3$ (BCC sampling) | $\sqrt 2 \approx 1.41\times$ |
| 4 | $D_4$ | $2\times$ |
| 5 | $D_5$ | $2\sqrt 2 \approx 2.83\times$ |

Even in 3D the project's metrics use cubic cells, so they leave out the $\sqrt 2$ that BCC sampling allows for a spherical head support.

## 2. What does not change

Changing the lattice multiplies every Nyquist count by a constant that is the same at every scale.

- **Exponent model** ([sampling-density.md](sampling-density.md), section 2): "the Nyquist count of the ball $\propto \rho^3$" becomes $c\,\rho^3$. $W(\text{res})$ is rescaled, but $\alpha$ and the derived density $n \propto \rho^{2-\alpha}$ are not.
- **SNR model** (section 3): it never uses a Nyquist count; it minimizes $\int w^2/D$.
- **Comparison of designs**: relative, so the ranking holds.

What does change is every absolute statement: the undersampling factors quoted during design (e.g. about 15.7× at 1 mm per 2 s window, about 47× per 2 s frame), the absolute $W(\text{res})$ times, and what gap counts as acceptable. All of them would shrink by the lattice factor, *if* the gain applies.

## 3. Whether D4 and D5 apply depends on the support

A lattice gain holds only if the support is a ball in some scaled joint metric.

**4D, $(k, \tau)$.** The dual of $\tau$ is the spectral content along relaxation time (decay rates, off-resonance). Its extent does not obviously shrink with spatial frequency, so the joint support is roughly a product: a ball in space × an interval in that dual. For a product support, the optimal lattice is the product of the optimal lattices (BCC × uniform), and the gain is only the 3D $\sqrt 2$. D4 helps only if fine spatial structure has a narrower $\tau$-bandwidth than coarse structure. More fundamentally, the subspace reconstruction does not sample $\tau$ to a band limit: it needs $L$ coefficients per voxel, so the requirement is set by degrees of freedom and incoherence, not lattice packing.

**5D, adding scan time $T$.** Here the shape of the support is the whole question. The resolution–time trade-off is a statement about the $(k, \omega_T)$ support: $W(\text{res}) \propto \text{res}^{-\alpha}$ corresponds to a resolvable temporal bandwidth that falls with $|k|$ as $\rho^{-\alpha}$, a hyperbolic or cone-like region rather than a ball. The relevant theory is optimal spatio-temporal sampling for a given support (Willis & Bresler on time-varying tomography; the k-t lattices behind k-t BLAST, Tsao et al.): for a non-separable support the best lattice is sheared or non-separable, and the saving depends on the support's shape, not only its dimension. D5 ball packing is the right reference only if the 5D support is a ball in some scaling. The exponent model in effect chooses a support shape, and $\alpha$ is that choice.

## 4. Where it would change the measurements

- **Gaps are a covering measure, not a packing one.** The gap metric (`relaxed-rosettes coverage`) asks how far any point is from a sample. The optimal *covering* lattices are $A_3^* = $ BCC, $A_4^*$ and $A_5^*$, not $D_4$ and $D_5$. Their advantage over the hypercubic grid is larger: coverings about 1.9×, 2.8× and 4.3× thinner in 3D, 4D and 5D. A dimension-aware gap would compare the measured gap with the covering radius of $A_n^*$ at the same sample density.
- **The $(|k|, \tau)$ banding metric uses separable bins** (20 $|k|$ shells × 0.5 ms). A joint version would use the distance to the nearest sample in a scaled $(k, \tau)$ metric. That needs one physical input first: how many k-space units one millisecond of $\tau$ is worth, which is what would make the support a ball. That scale factor is the real input that D4-style reasoning requires.

## Proposed actions (not started)

1. **Absolute numbers:** normalize coverage and Nyquist counts with BCC instead of cubic cells. Cheap; changes no conclusions.
2. **Joint covering metrics:** add a $(k, \tau)$ covering metric with an explicit $\tau$-to-$k$ scale, normalized to $A_4^*$, and optionally $(k, \tau, T)$, normalized to $A_5^*$. Measurement only.
3. **Explicit 5D support:** write down the expected support, for example a temporal bandwidth $B(\rho)$ for each spatial scale. Then generalize the exponent model from a power law to "match this support", and take the packing or covering reference from it.

Actions 2 and 3 need physics input before they can be implemented: the $\tau$-to-$k$ scaling, and the expected shape of the 5D support.

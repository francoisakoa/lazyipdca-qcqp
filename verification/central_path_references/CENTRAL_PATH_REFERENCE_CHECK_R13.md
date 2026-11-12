# Primary-source check: redundant inequalities and central paths

The author name is **Eissa Nematollahi**, with a double `l` and final `i`.
Use `Nematollahi`, not `Nematolahi`, `Nematollah`, or `Nematollahy`. No claim
about a spelling error in an unseen bibliography is made here.

## Recommended two references

1. **Antoine Deza, Eissa Nematollahi and Tamás Terlaky (2008).**
   *How good are interior point methods? Klee–Minty cubes tighten
   iteration-complexity bounds.* Mathematical Programming 113(1), 1–14.
   DOI: <https://doi.org/10.1007/s10107-006-0044-x>.

   Publisher record:
   <https://link.springer.com/article/10.1007/s10107-006-0044-x>
   (`turn141view1`). Published online 21 October 2006; final issue year **2008**.
   Primary published full text on the author's university site:
   <https://www.cas.mcmaster.ca/~deza/mp2008.pdf> (`turn141view0`).

   Exact scope: **linear programming**, redundant Klee–Minty inequalities and
   central-path geometry. Section 2, printed pages 3–5: the feasible region is
   unchanged; Proposition 2.1 proves that the path enters a neighbourhood of
   every vertex. Corollary 2.2 gives a path-following iteration lower bound.
   Remark 2.5(iv) explicitly limits the result to path-following methods,
   excluding a blanket assertion about all interior-point algorithms.
   Targeted full-text retrievals are `turn142view2` and `turn142view3`.

2. **Eissa Nematollahi and Tamás Terlaky (2008).**
   *A redundant Klee–Minty construction with all the redundant constraints
   touching the feasible region.* Operations Research Letters 36(4), 414–418.
   DOI: <https://doi.org/10.1016/j.orl.2008.02.004>.

   Primary publisher abstract and metadata:
   <https://www.sciencedirect.com/science/article/abs/pii/S0167637708000278>
   (`turn140search0`, `turn141search0`; direct open returned HTTP 403, so the
   indexed publisher record was used for final metadata).
   Author-deposited primary full text:
   <https://optimization-online.org/wp-content/uploads/2008/01/1870.pdf>
   (`turn144view0`), linked by
   <https://optimization-online.org/2008/01/1870/> (`turn143view0`).

   Exact scope: **linear programming**. Repeated implied coordinate bounds
   leave the Klee–Minty feasible polytope unchanged. Section 2, equations (2)
   and (3), explicitly changes the logarithmic barrier through multiplicities
   of coordinate-log terms. Remark 2.1 distinguishes the same polytope from
   its different algebraic representations. Proposition 2.3 shows that the
   central path remains near the simplex edge path. This also holds when all
   added inequalities touch the feasible region. The deposited version is
   dated 5 January 2008; final publication is July 2008.

## Optional original antecedent, not needed in addition to both above

**Antoine Deza, Eissa Nematollahi, Reza Peyghami and Tamás Terlaky (2006).**
*The central path visits all the vertices of the Klee–Minty cube.*
Optimization Methods and Software 21(5), 851–865.
DOI <https://doi.org/10.1080/10556780500407725>.

Publisher metadata retrieved as `turn139search6` and `turn141search10`:
<https://www.tandfonline.com/doi/full/10.1080/10556780500407725>.
The 2008 Mathematical Programming paper cites this earlier four-author paper
and refines its redundant construction. Do not conflate this author list with
the three-author 2008 paper. Author-hosted file
<https://www.cas.mcmaster.ca/~deza/oms2006.pdf> was found by search
(`turn141search25`), but opening that PDF failed; no assertion here depends on
an unread part of that file.

## Interpretation for LazyQCQP and ASU

- These references rigorously establish **representation dependence of the
  logarithmic barrier/central trajectory**, even for identical feasible sets.
- Their curvature means geometric bending of a parameterized central path.
  It is not the relative tangent Hessian excess \(\kappa\), nor a measurement
  of the majorant correction used by LazyQCQP.
- The LP examples have the same optimal endpoint. They do **not** prove
  selection of a different local minimum in nonconvex QCQP, or improved
  stationary objective quality after retaining/removing redundant bounds.
- Consequently they do not explain the ASU objective difference near 18%,
  establish its cause, or isolate the contribution of preprocessing. Such an
  attribution needs a matched experiment varying only that factor.
- A precise positive use is to motivate recording/controlling the retained
  inequality representation alongside correction and globalization choices.

The supplied paragraph states this narrow role and keeps the main manuscript
free of an unsupported causal claim.

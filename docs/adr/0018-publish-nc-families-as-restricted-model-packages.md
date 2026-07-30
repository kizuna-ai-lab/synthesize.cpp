---
status: accepted
---

# Publish NC-licensed families as Restricted Model Packages

A Model Family whose weights carry a non-commercial or otherwise
redistribution-restricted grant can pass the same Port Validation Suite as any
other family, but its packages must not be called Published Model Packages:
that term carries docs/scope.md's promise of weights embeddable in other
people's programs. synthesize.cpp therefore defines a second publication
category, the Restricted Model Package: identical repository layout, model-card
workflow, validation evidence, and acquisition flow, plus the complete upstream
terms carried with the artifact.

The deciding argument, recorded from the OmniVoice fourth-family intake
(2026-07-30): a license restriction follows the weights regardless of who runs
the converter. A user converting locally holds the same restricted derivative,
so withholding publication does not enlarge the legal audience — it only adds
friction for the users the license does permit. When the upstream terms allow
non-commercial redistribution with attribution, the project may republish under
exactly those terms.

## Consequences

- A Restricted Model Package passes the full Port Validation Suite before
  publication, like any package. Validation Levels are unchanged.
- Its model card declares the restrictive license in frontmatter, quotes the
  upstream statement verbatim, and carries a prominent statement that the
  weights are not usable in commercial products. When upstream omits a license
  version, the card says so rather than inventing one.
- Every license in the artifact travels with it: for OmniVoice that is the
  CC-BY-NC statement for the LM weights and the Boson Higgs Audio 2 Community
  License text as a declared Sidecar Resource for the codec weights, with dual
  attribution.
- The C interface, runtime, and Adapters are unaffected: restriction is a
  publication and documentation fact, not a runtime capability.
- docs/scope.md's "Publishing validated, directly loadable Model Packages"
  line continues to mean Published Model Packages; Restricted Model Packages
  are an addition, not a reinterpretation.
- Publication of any package, restricted or not, remains a separate act
  requiring jiangzhuo's explicit per-act confirmation.

## Considered Options

- Supported Model Family only (no published weights) was rejected because it
  does not change who may legally use the weights; it only pushes a Python
  conversion environment onto every legitimate non-commercial user.
- Publishing under the ordinary Published Model Package label with a license
  footnote was rejected because that term's embedding promise would become
  conditionally false, and a term that is sometimes false is not a term.
- Waiting for upstream relicensing was rejected as indefinite: the restriction
  derives from training data (Emilia), which retraining alone could lift.

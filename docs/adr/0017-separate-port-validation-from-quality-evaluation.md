---
status: accepted
---

# Separate port validation from quality evaluation

synthesize.cpp publishes a Model Package after a compact, deterministic Port
Validation Suite proves conversion fidelity, reference parity, end-to-end
stability, and claimed Execution Backend behavior. Corpus-scale intelligibility,
naturalness, Voice similarity, quantization-quality comparison, and cross-model
comparison move to a later Quality Evaluation Suite so adding architecture support
does not require large training corpora and evaluator stacks first.

## Consequences

- Each Model Variant normally carries 12 to 32 port-validation cases, beginning
  with usable examples from its pinned upstream repository and adding cases only
  to cover remaining inference branches.
- A Published Model Package declares a Validation Level. `port_validated` is a
  functional correctness claim, while `quality_evaluated` requires the later
  versioned Quality Evaluation Suite.
- A port-validated model page reports `quality_evaluation: not_run` and makes no
  perceptual-quality, quantization-transparency, or comparative-ranking claim.
- F16 and other Quantization Profiles must each pass port validation before
  publication, but that result alone does not establish that quality loss is
  acceptable.
- Full datasets and learned evaluators are not downloaded, redistributed, or
  required by the initial model-porting workflow.
- This evidence distinction remains in manifests, reports, and model cards; it
  does not add quality concepts to the runtime C Interface.

## Considered Options

- Requiring the full Quality Evaluation Suite before every first model publication
  was rejected because it couples architecture work to large corpora, evaluator
  models, and thresholds that are more useful when several model families can be
  compared together.
- Treating load success or audible output as sufficient was rejected because it
  cannot detect tensor mapping, conditioning, control, stochastic-state, or
  backend-placement errors. Port validation retains pinned upstream and
  intermediate-tensor comparison.

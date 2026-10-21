# Model licensing review

No third-party pretrained model or model code is bundled, downloaded, or enabled by this change. The production adapter remains **unconfigured** until an operator provides a suitable, commercially authorized temporal model. The `test-identity` adapter is diagnostic only and never removes logos.

## ProPainter — checked and excluded

The actual upstream `LICENSE` was read at commit **`e870e79321c31b733e2031af5aa2fb1fe3ac7eec`** of `sczhou/ProPainter` during this implementation:

- Authoritative pinned source: <https://github.com/sczhou/ProPainter/blob/e870e79321c31b733e2031af5aa2fb1fe3ac7eec/LICENSE>
- License: **S-Lab License 1.0**.
- Git blob SHA: `a8f075b52308e65bc7d578d270e1a95a6255021b`.
- Downloaded license SHA-256: `322da691bf0272644565c84830a139d9f3c700c16cb415043a476d50bf7f2f90`.
- It permits “Redistribution and use for **non-commercial purpose**”. Clause 4 says commercial redistribution/use requires contacting the contributors. The file identifies Dr. Shangchen Zhou and Prof. Chen Change Loy for permission requests.

This is not a blanket commercial permission. We have **not** requested permission, integrated the repository, downloaded its checkpoints, or represented any checkpoint as commercially cleared. A separate written grant would need to cover the intended hosted processing, code modifications, weights, distribution, users and jurisdiction. Weight provenance and any separate checkpoint/data terms must also be checked; the repository license alone does not establish every checkpoint's rights.

## Gate before enabling any model

The operator must review the **exact code revision and exact weights**, record their authoritative license URLs, preserve required notices, and confirm commercial hosted inference is allowed. Use immutable versioned model names. Record the artifact's SHA-256, license name/reference and authorization decision in the operator-owned service configuration. Provision the artifact out of band on a read-only model volume. Do not put private commercial agreements or keys in Git.

`license_reviewed=true` and `commercial_use_authorized=true` are operator attestations, **not a license grant or automated legal verification**. The service fails closed if either is missing/false, if a production artifact is missing, or if its SHA-256 differs at model load. Review again when code/weights change; the job pins the full configuration revision and rejects execution after configuration drift. Operators can supply their own commercial model via the TorchScript tensor contract or an installed Python factory adapter.

The only bypass is the explicitly labeled `test-identity` diagnostic adapter with `LOGO_ALLOW_TEST_MODEL=1`. Do not enable that environment variable in production. It produces no AI inpainting and is surfaced as a test model in capabilities and desktop consent.

Also audit transitive model dependencies, PyTorch/CUDA redistribution terms, and FFmpeg/x264/x265/PySide6 obligations for the actual image you ship. The repository does not authorize redistribution of unrelated model weights.

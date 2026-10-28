# Publication provenance: 28 October 2026

This folder records the original delivered source snapshots, the public copies,
and available evidence included with the 27 October work.

- `original_v3_source_manifest.json`: manifest from the original v3 ZIP.
- `original_compact_source_manifest.json`: manifest from the original compact ZIP.
- `source_and_transformations.json`: original/public hashes for source files,
  local logs and console attachments; original ZIP sizes and hashes; exclusions.
- `publication_manifest.json`: exact public paths, byte sizes and Git blob hashes
  for the staged publication (excluding this self-referential manifest).
- `validation_report.json`: source syntax, package-manifest, metric consistency,
  link and credential-pattern checks performed for publication.

## Public transformations

Account/home paths and GPU/MIG UUIDs are generalized. The two hardware diagnostic
attachments omit the section listing other users' process paths and PIDs.
Numeric training/test/geometry-summary readings remain unchanged. Text is
published as UTF-8 with LF line endings. Original/private bytes are identified by
hash, not embedded in a public ZIP. Credentials, actual settings.env files,
download tokens, model weights, caches, raw probe arrays and trained checkpoint
binaries are not published.

Each source folder has a new SOURCE_MANIFEST.json for its public bytes and a
PUBLICATION.md explaining its status. The original manifests are preserved here.
Public copies must be customized and used to build a NEW plan; an existing
server plan's source identity will not match a changed checkout.

The source ZIPs are delivery snapshots. Their original hashes do not establish
which manually patched source files were executed on the H100 server. That
requires the native run/config/freeze receipts, which were not received.

## Evidence integrity

The GPU log is a supplied concatenated console attachment, not a recovered
complete server filesystem. Its 18 stage dictionaries were parsed into 180
task/stage/split rows. For every row, the confusion-matrix total equals n,
its diagonal equals correct predictions, and accuracy equals correct/n.
Final means and BWT are derived transparently from those rows.

The CPU analysis outcome is a labelled transcription of user-pasted terminal
evidence. Original hashes and detailed geometry outputs are unavailable.
Synthetic Gaussian tests are instrument diagnostics, not real-model H2 results.
Failed local fixture attempts remain archived alongside passing runs.

A SHA256 hash identifies bytes and detects changes. It does not recover missing
artifacts or constitute an independent experiment, a signature, or an external
backup. No full-scale primary campaign or new H100 run is claimed by publication.

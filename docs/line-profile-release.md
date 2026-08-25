# Line profile release and maintenance gates

This document records the release evidence and the operational boundaries for
the preview implementation.

## Baseline

On 2026-08-25, using Python 3.12.3 on a 24-vCPU Linux runner, the deterministic
synthetic benchmark measured a 1024×1024 grayscale image and a 864-pixel
centerline with 109 samples:

- automatic measurement: 7.35 ms per run, averaged over 20 warm runs;
- 128-sample boundary preview: 0.79 ms per run, averaged over 100 runs.

These numbers are a baseline, not a performance guarantee. Release checks should
repeat them on the supported CI runner and include a dense, curved centerline.
The preview target is 100 ms per repaint; measurement must remain off the GUI
thread.

## Safe logging and metrics

Profile-related diagnostics may record only event name, schema version,
measurement version, shape count, sample count, reason code, status, and elapsed
milliseconds. They must not record image bytes, complete image paths, labels,
or arbitrary profile payloads. The recommended event names are:

- `line_profile_load_failed`;
- `line_profile_measurement_failed`;
- `line_profile_measurement_canceled`;
- `line_profile_save_failed`.

Collection is opt-in through the host application's existing log/metrics
pipeline; no new network endpoint is required by this feature.

## Schema and rollback procedure

1. Run the format round-trip, invalid-profile preservation, migration, and
   ordinary-annotation regression suites.
2. Run the full test suite and the GUI smoke checklist with an isolated HOME.
3. If a schema migration fails, keep the raw profile in `other_data`, report the
   error without deleting the annotation, and disable profile editing for that
   Shape.
4. If a release must be rolled back, revert the application version. Older
   LabelMe versions continue to operate on the centerline and ignore the
   additive `line_profile` field; do not rewrite files merely to remove it.

Continuous-frame transfer now has an explicit compatibility check and a
difference preview and confirmation step. Large-directory batch review remains a separate follow-up
capability; it must not silently modify another frame or bypass the single-file
atomic writer.

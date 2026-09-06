# ADR 0008: Linestrip line-profile annotations

## Status

Accepted for the preview implementation.

## Context

Some `linestrip` annotations describe a centerline whose width and visibility
change along its path. Replacing that centerline with a polygon or mask would
lose the editable measurement source and would change the behavior of existing
tools. The format also needs to remain readable by older LabelMe versions.

## Decision

- Store the optional data in a `line_profile` object on a `linestrip` Shape.
- Parameterize anchors by normalized arc length (`0.0` through `1.0`), not by
  vertex index, so inserting, deleting, moving, splitting, and merging centerline
  vertices can remap the profile deterministically.
- Store width as the full diameter. The Canvas may display a circular handle
  using half that value, but serialization always uses `width`.
- Store one sorted `anchors` sequence. Each shared anchor has one `position`
  and optional `width` and `visibility` values; each value carries its own
  `source`, `confidence`, and `confirmed` metadata.
- Use explicit `schema_version=2` and `measurement_version` fields. The
  unified `anchors` format is strict; unsupported or malformed profiles remain
  as raw data with a load error and are not silently discarded.
- Do not create ordinary `circle`, `point`, polygon, or mask Shapes for profile
  handles or previews.

## Compatibility and recovery

Older tools that do not know `line_profile` continue to use the ordinary
centerline fields. LabelMe preserves unknown profile data while loading an
invalid profile, allowing a user to repair or remove it explicitly. Writes use
the existing single-file atomic save path.

## Consequences

The profile is an additive extension and does not alter ordinary Shape geometry.
Profile-aware geometry operations must use the Qt-free transformation helpers,
and user edits must go through the existing Shape backup/dirty/undo workflow.

# Line profile annotations

A line profile is optional metadata attached to a `linestrip`. It describes
width and visibility samples along the editable centerline; it is not a new
Shape type and does not replace the centerline with a polygon or mask.

## Workflow

1. Draw or select a `linestrip`.
2. Use **Measure Line Profile** to calculate automatic samples.
3. Review the circular width handles and square visibility handles. Automatic
   results remain unconfirmed until accepted.
4. Select an anchor to edit its position, value, source, confidence, or
   confirmation state. Insert and delete anchors from the Edit menu or the
   Canvas context menu.
5. Toggle **Show Line Profile Preview** when the boundary overlay is distracting.
6. Save normally. Clearing a profile removes only the profile metadata and
   preserves the centerline.

Automatic measurement runs in a worker and checks that the image, centerline,
and profile have not changed before a result can be accepted. Low-confidence
samples may carry a neighboring-width recommendation, but their confidence is
not raised automatically.

## File format

The optional `line_profile` object contains:

```json
{
  "schema_version": 1,
  "path_mode": "continuous",
  "parameterization": "normalized_arc_length",
  "width_anchors": [],
  "visibility_anchors": [],
  "min_width": null,
  "max_width": null,
  "measurement_version": "line-profile-measurement-v1",
  "reviewed": false
}
```

Width is the full diameter in image pixels. Anchor positions are normalized
arc-length values in `[0, 1]`; duplicate positions are invalid. `source` is
`auto` or `manual`, `confidence` is in `[0, 1]`, and `confirmed` is a boolean.
The fixed round-trip example is in
[`docs/line_profile_example.json`](line_profile_example.json).

Older LabelMe versions ignore unknown Shape fields and therefore retain their
ordinary centerline behavior. LabelMe keeps malformed profile data in memory
for explicit repair instead of silently deleting it.

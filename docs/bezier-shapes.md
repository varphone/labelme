# Bezier annotation shapes

Labelme supports two open Bezier annotation shapes:

- `bezier2` is a quadratic (second-order) curve with three points: start,
  control, and end.
- `bezier3` is a cubic (third-order) curve with four points: start, first
  control, second control, and end.

When drawing, click the start point first, the end point second, and then the
control point(s). The saved JSON uses the conventional start, control point(s),
end order described above.

The points are stored in the normal Labelme `points` field, in that order. The
`shape_type` field identifies the curve degree. For example:

```json
{
  "label": "road_edge",
  "points": [[20, 80], [50, 10], [90, 80]],
  "shape_type": "bezier2",
  "flags": {},
  "group_id": null,
  "description": ""
}
```

Choose **Quadratic Bezier** or **Cubic Bezier** from the Draw menu. Click the
points in the order shown above; the annotation is committed after the final
point. In Edit mode, all control points remain draggable. Curves are rendered
and hit-tested as Bezier paths, and raster export samples the path at a fixed
resolution for line-mask generation.

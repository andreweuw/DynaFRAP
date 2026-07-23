import vtk


def add_cylinder(viewer, height, radius, center, color, opacity=0.3):
    z, y, x = center
    sx, sy, sz = viewer.spacing
    cx = x * sx
    cy = y * sy
    cz = z * sz

    cyl = vtk.vtkCylinderSource()
    cyl.SetRadius(radius * sx)
    cyl.SetHeight(height * sz)
    cyl.SetResolution(48)
    cyl.SetCenter(0.0, 0.0, 0.0)
    cyl.SetCapping(True)

    transform = vtk.vtkTransform()
    transform.Translate(cx, cy, cz)
    transform.RotateX(90)

    tf_filter = vtk.vtkTransformPolyDataFilter()
    tf_filter.SetTransform(transform)
    tf_filter.SetInputConnection(cyl.GetOutputPort())
    tf_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tf_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    return actor


def rebuild_lut(viewer):
    if viewer.values is None or viewer.current_lut is None:
        return

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.Build()
    lut.SetRange(viewer.vmin, viewer.vmax)

    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(viewer.vmin, 0.0, 0.0, 1.0)
    ctf.AddRGBPoint((viewer.vmin + viewer.vmax) / 2, 0.0, 1.0, 0.0)
    ctf.AddRGBPoint(viewer.vmax, 1.0, 0.0, 0.0)

    for i in range(256):
        v = viewer.vmin + (i / 255.0) * (viewer.vmax - viewer.vmin)
        r, g, b = ctf.GetColor(v)
        alpha = (
            0.0
            if v < viewer.vmin or v > viewer.vmax
            else (v - viewer.vmin) / (viewer.vmax - viewer.vmin)
        )
        lut.SetTableValue(i, r, g, b, alpha)

    viewer.current_lut.DeepCopy(lut)
    viewer.render_window.Render()


def make_slider(min_val, max_val, x1, y1, x2, y2, callback, is_int=False):
    slider_rep = vtk.vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(min_val)
    slider_rep.SetMaximumValue(max_val)
    slider_rep.SetValue(min_val)
    slider_rep.SetTitleText("")
    slider_rep.SetSliderLength(0.01)
    slider_rep.SetSliderWidth(0.02)
    slider_rep.SetEndCapLength(0.005)
    slider_rep.SetEndCapWidth(0.02)
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(x1, y1)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(x2, y2)

    slider_widget = vtk.vtkSliderWidget()
    slider_widget.SetRepresentation(slider_rep)

    def on_slider(obj, event):  # noqa: ARG001
        val = obj.GetRepresentation().GetValue()
        if is_int:
            val = round(val)
            obj.GetRepresentation().SetValue(val)
        callback(val)

    slider_widget.AddObserver("InteractionEvent", on_slider)
    return slider_widget


def make_saliency_lut(values, n_colors=256, threshold=None):
    min_val, max_val = float(values.min()), float(values.max())
    if threshold is None:
        threshold = min_val

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n_colors)
    lut.Build()
    lut.SetRange(min_val, max_val)

    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(threshold, 0.0, 0.0, 1.0)
    ctf.AddRGBPoint((min_val + max_val) / 2, 0.0, 1.0, 0.0)
    ctf.AddRGBPoint(max_val, 1.0, 0.0, 0.0)

    for i in range(n_colors):
        v = threshold + (i / (n_colors - 1)) * (max_val - threshold)
        r, g, b = ctf.GetColor(v)
        alpha = 0.0 if v < threshold else (v - threshold) / (max_val - threshold + 1e-6)
        lut.SetTableValue(i, r, g, b, alpha)

    return lut

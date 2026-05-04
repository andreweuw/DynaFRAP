import vtk

from . import csv_export


def print_help():
    print("\n" + "=" * 60)
    print("VOXEL VIEWER - KEYBOARD CONTROLS")
    print("=" * 60)
    print("  r         - Reset target box")
    print("  R         - Reset control box (blue)")
    print("  Enter     - Find and propagate target peak (red)")
    print("  o         - Find and propagate control peak (blue)")
    print("  u         - Estimate radius at frame 0 (pre-image)")
    print("  f         - Compute averages and save to CSV")
    print("  i         - Toggle image-only view")
    print("  c         - Clear all cylinder visualizations")
    print("  Use mouse to rotate, zoom, and pan the view")
    print("  Use slider to navigate through time frames")
    print("=" * 60 + "\n")


def reset_control_selection_box(viewer):
    if viewer.control_box_widget:
        viewer.control_box_widget.Off()

    viewer.control_box_widget = vtk.vtkBoxWidget()
    viewer.control_box_widget.SetInteractor(viewer.render_window_interactor)
    viewer.control_box_widget.SetPlaceFactor(1.0)
    viewer.control_box_widget.TranslationEnabledOn()
    viewer.control_box_widget.RotationEnabledOff()
    viewer.control_box_widget.SetHandleSize(0.1)
    viewer.control_box_widget.GetOutlineProperty().SetColor(0, 0, 1)
    viewer.control_box_widget.GetOutlineProperty().SetLineWidth(3.0)

    viewer.control_box_widget.SetInputData(viewer.vtk_img)
    viewer.control_box_widget.PlaceWidget()
    viewer.control_box_widget.On()
    viewer.control_box_widget.AddObserver("InteractionEvent", lambda obj, event: on_control_box_interaction(viewer, obj, event))

    viewer.control_frame_boxes[viewer.current_t] = viewer.get_box_bounds(viewer.control_box_widget)
    viewer.render_window.Render()


def on_control_box_interaction(viewer, obj, event):
    viewer.control_frame_boxes[viewer.current_t] = viewer.get_box_bounds(obj)


def reset_selection_box(viewer):
    viewer.box_widget.Off()
    viewer.box_widget = vtk.vtkBoxWidget()
    viewer.box_widget.SetInteractor(viewer.render_window_interactor)
    viewer.box_widget.SetPlaceFactor(1.0)
    viewer.box_widget.TranslationEnabledOn()
    viewer.box_widget.RotationEnabledOff()
    viewer.box_widget.SetHandleSize(0.01)

    viewer.box_widget.SetInputData(viewer.vtk_img)
    viewer.box_widget.PlaceWidget()
    viewer.box_widget.On()
    viewer.box_widget.AddObserver("InteractionEvent", lambda obj, event: on_box_end_interaction(viewer, obj, event))

    viewer.frame_boxes[viewer.current_t] = viewer.get_box_bounds(viewer.box_widget)
    viewer.render_window.Render()


def place_box_widget_split(widget, full_bounds, side="left", gap_fraction=0.05):
    xmin, xmax, ymin, ymax, zmin, zmax = full_bounds
    x_width = xmax - xmin
    gap = x_width * gap_fraction / 2
    x_center = (xmin + xmax) / 2

    if side == "left":
        bounds = (xmin, x_center - gap, ymin, ymax, zmin, zmax)
    elif side == "right":
        bounds = (x_center + gap, xmax, ymin, ymax, zmin, zmax)
    else:
        raise ValueError(f"side must be 'left' or 'right', got '{side}'")

    widget.PlaceWidget(*bounds)


def on_box_end_interaction(viewer, obj, event):
    planes = vtk.vtkPlanes()
    obj.GetPlanes(planes)
    extract = vtk.vtkExtractGeometry()
    extract.SetInputData(viewer.vtk_img)
    extract.SetImplicitFunction(planes)
    extract.Update()
    selected_poly = extract.GetOutput()
    if selected_poly.GetNumberOfPoints() == 0:
        print("No voxels selected")
        return
    viewer.current_selection_voxels = selected_poly.GetPoints().GetData()
    viewer.frame_boxes[viewer.current_t] = viewer.get_box_bounds(obj)


def add_key_listener(viewer):
    def keypress_callback(obj, event):
        key = obj.GetKeySym()
        slider_rep = viewer.time_slider.GetRepresentation()
        value = slider_rep.GetValue()
        print(f"{key} pressed.")
        
        # Cancel default VTK key press behavior
        obj.ForceAbortCallback()
        
        match key:
            case "Right":
                new_val = min(value + 1, slider_rep.GetMaximumValue())
                slider_rep.SetValue(new_val)
                viewer.time_slider.InvokeEvent(vtk.vtkCommand.InteractionEvent)
            case "Left":
                new_val = max(value - 1, slider_rep.GetMinimumValue())
                slider_rep.SetValue(new_val)
                viewer.time_slider.InvokeEvent(vtk.vtkCommand.InteractionEvent)
            case "Return":
                viewer._find_peak_and_propagate(peak_type="target")
            case "o":
                viewer._find_peak_and_propagate(peak_type="control")
            case "u":
                viewer._estimate_radius_at_frame_zero()
            case "R":
                reset_control_selection_box(viewer)
            case "r":
                reset_selection_box(viewer)
            case "i":
                viewer.show_image = not viewer.show_image
                viewer.update_frame(viewer.current_t)
            case "f":
                csv_export.compute_averages_and_save_to_csv(viewer)

    viewer.render_window_interactor.AddObserver("KeyPressEvent", keypress_callback)
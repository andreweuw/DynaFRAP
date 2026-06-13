import numpy as np
import vtk
from vtkmodules.util import numpy_support

from dynafrap_andreweuw.process import detect_local_max

from . import analysis, interaction
from .analysis import propagate_peak_bounds
from .primitives import make_saliency_lut, make_slider, rebuild_lut


class VoxelViewer:
    def __init__(self, volumes, filenames, images, min, spacing=(1, 1, 1)):
        self.volumes = volumes
        self.filenames = filenames
        self.spacing = spacing
        self.show_image = False
        self.images = images
        self.min = min
        self.T, self.Z, self.Y, self.X = volumes.shape
        self.current_t = 0
        self.current_actor = None
        self.current_lut = None
        self.values = None
        self.frame_boxes = {}
        self.cylinder_actors = {}
        self.target_peaks = {}
        self.box_widgets_initialized = False
        self.target_peak_actor = None
        self.control_peaks = {}
        self.control_frame_boxes = {}
        self.control_peak_actor = None
        self.control_cylinder_actors = {}
        self.radius = 4.0

        self.global_min = None
        self.global_max = None
        self.vmin = 0.0
        self.vmax = 1.0
        self.first_enter = False
        self.shift = np.array([0, 0, 0])

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.7, 0.7, 0.7)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1600, 900)

        self.current_mapper = None

        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.render_window_interactor.SetRenderWindow(self.render_window)

        self.box_widget = vtk.vtkBoxWidget()
        self.box_widget.SetInteractor(self.render_window_interactor)
        self.box_widget.SetPlaceFactor(1.0)
        self.box_widget.TranslationEnabledOn()
        self.box_widget.RotationEnabledOff()
        self.box_widget.SetHandleSize(0.005)

        self.control_box_widget = vtk.vtkBoxWidget()
        self.control_box_widget.SetInteractor(self.render_window_interactor)
        self.control_box_widget.SetPlaceFactor(1.0)
        self.control_box_widget.TranslationEnabledOn()
        self.control_box_widget.RotationEnabledOff()
        self.control_box_widget.SetHandleSize(0.005)
        self.control_box_widget.GetOutlineProperty().SetColor(0, 0, 1)
        self.control_box_widget.GetOutlineProperty().SetLineWidth(3.0)

        self.text_actor = vtk.vtkTextActor()
        self.text_actor.GetTextProperty().SetFontSize(25)
        self.text_actor.GetTextProperty().SetColor(0, 0, 0)
        self.text_actor.SetDisplayPosition(50, 0)
        self.renderer.AddActor2D(self.text_actor)

        self.time_slider = make_slider(
            0, self.T - 1, 0.1, 0.05, 0.9, 0.05, self.update_frame, is_int=True
        )
        self.time_slider.SetInteractor(self.render_window_interactor)
        self.time_slider.EnabledOn()
        interaction.add_key_listener(self)

        self.min_slider = make_slider(0, 1, 0.1, 0.95, 0.45, 0.95, self.update_window_min)
        self.min_slider.SetInteractor(self.render_window_interactor)
        self.min_slider.EnabledOn()

        self.max_slider = make_slider(0, 1, 0.55, 0.95, 0.9, 0.95, self.update_window_max)
        self.max_slider.SetInteractor(self.render_window_interactor)
        self.max_slider.EnabledOn()
        self.vtk_img = None

        self.update_frame(self.current_t)
        interaction.print_help()

    def _find_peak_and_propagate(self, peak_type="target"):
        if peak_type == "target":
            box_widget = self.box_widget
            frame_boxes = self.frame_boxes
            peaks = self.target_peaks
            peak_actor_attr = "target_peak_actor"
            peak_color = (1, 0, 0)
        elif peak_type == "control":
            box_widget = self.control_box_widget
            frame_boxes = self.control_frame_boxes
            peaks = self.control_peaks
            peak_actor_attr = "control_peak_actor"
            peak_color = (0, 0, 1)
        else:
            raise ValueError(f"peak_type must be 'target' or 'control', got '{peak_type}'")

        t = self.current_t
        vol = self.volumes[t]
        bounds = self.get_box_bounds(box_widget)
        frame_boxes[t] = bounds

        mask = analysis.mask_from_bounds(bounds, vol.shape, self.spacing)
        peak_t = analysis.compute_target_peak(vol, mask=mask)
        peaks[t] = peak_t

        origin = np.array(self.vtk_img.GetOrigin())
        spacing = np.array(self.vtk_img.GetSpacing())

        for next_t in range(t + 1, self.T):
            next_vol = self.volumes[next_t]
            next_mask = analysis.mask_from_bounds(bounds, next_vol.shape, self.spacing)
            peak_next = analysis.compute_target_peak(next_vol, mask=next_mask)

            if peak_next is None:
                break

            bounds = propagate_peak_bounds(bounds, peak_next, origin, spacing)
            frame_boxes[next_t] = bounds
            peaks[next_t] = peak_next

        box_widget.PlaceWidget(frame_boxes[t])
        self._show_peak(peaks[t], peak_actor_attr, peak_color)

    def _estimate_radius_at_frame_zero(self):
        """
        At frame 0 (pre-image), find peak in target selection box and estimate
        the radius where probability of cell is at least 80%.
        """
        if self.current_t != 0:
            print("Radius estimation only available at frame 0 (pre-image)")
            return

        t = 0
        vol = self.volumes[t]
        bounds = self.get_box_bounds(self.box_widget)
        self.frame_boxes[t] = bounds

        image = self.images[t]

        peak_t, best_radius, _ = analysis.estimate_radius_at_frame_zero(
            vol, image, bounds, vol.shape, self.spacing, self.shift
        )

        if peak_t is None:
            return

        self.target_peaks[t] = peak_t
        self.radius = best_radius if best_radius is not None else 4.0
        if best_radius is not None:
            self._show_peak(peak_t, "target_peak_actor", (1, 0, 0), radius=best_radius)
        else:
            # Show peak with default radius anyway
            self._show_peak(peak_t, "target_peak_actor", (1, 0, 0), radius=best_radius)

    def _show_peak(self, peak_coords, actor_attr, color, radius=None):
        radius = 2
        existing_actor = getattr(self, actor_attr)
        if existing_actor:
            self.renderer.RemoveActor(existing_actor)

        z, y, x = peak_coords
        spacing = self.spacing

        cube = vtk.vtkCubeSource()
        cube.SetXLength(spacing[0] * radius)
        cube.SetYLength(spacing[1] * radius)
        cube.SetZLength(spacing[2] * radius)
        cube.SetCenter(x * spacing[0], y * spacing[1], z * spacing[2])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(1.0)

        self.renderer.AddActor(actor)
        setattr(self, actor_attr, actor)
        self.render_window.Render()

    def world_to_voxel(self, x, y, z):
        i = round((z - self.shift[2]) / self.spacing[2])
        j = round((y - self.shift[1]) / self.spacing[1])
        k = round((x - self.shift[0]) / self.spacing[0])
        i = np.clip(i, 0, self.Z - 1)
        j = np.clip(j, 0, self.Y - 1)
        k = np.clip(k, 0, self.X - 1)
        return np.array([i, j, k])

    def update_window_min(self, v):
        self.vmin = v
        rebuild_lut(self)

    def update_window_max(self, v):
        self.vmax = v
        rebuild_lut(self)

    def get_box_bounds(self, widget):
        poly = vtk.vtkPolyData()
        widget.GetPolyData(poly)
        pts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
        xmin, ymin, zmin = pts.min(axis=0)
        xmax, ymax, zmax = pts.max(axis=0)
        return [xmin, xmax, ymin, ymax, zmin, zmax]

    def on_box_end_interaction(self, obj, event):
        planes = vtk.vtkPlanes()
        obj.GetPlanes(planes)
        extract = vtk.vtkExtractGeometry()
        extract.SetInputData(self.vtk_img)
        extract.SetImplicitFunction(planes)
        extract.Update()
        selected_poly = extract.GetOutput()
        if selected_poly.GetNumberOfPoints() == 0:
            print("No voxels selected")
            return
        self.current_selection_voxels = selected_poly.GetPoints().GetData()
        self.frame_boxes[self.current_t] = self.get_box_bounds(obj)

    def collect_selection(self, world):
        voxels = np.array([self.world_to_voxel(x, y, z) for x, y, z in world])
        cur_vol = self.volumes[self.current_t]
        zi, yi, xi = voxels[:, 0], voxels[:, 1], voxels[:, 2]
        zi = np.clip(zi, 0, cur_vol.shape[0] - 1)
        yi = np.clip(yi, 0, cur_vol.shape[1] - 1)
        xi = np.clip(xi, 0, cur_vol.shape[2] - 1)
        zmin, zmax = zi.min(), zi.max()
        ymin, ymax = yi.min(), yi.max()
        xmin, xmax = xi.min(), xi.max()
        return cur_vol[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1]

    def add_image_volume(self, vol_data):
        vol_data = vol_data.astype(np.float32)
        flat = vol_data.ravel()
        vtk_data = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)

        img = vtk.vtkImageData()
        img.SetDimensions(vol_data.shape[2], vol_data.shape[1], vol_data.shape[0])
        img.GetPointData().SetScalars(vtk_data)
        img.SetSpacing(*self.spacing)

        p1, p99 = np.percentile(vol_data, [1, 99])
        if p99 <= p1:
            p1 = float(np.min(vol_data))
            p99 = float(np.max(vol_data))

        alpha = vtk.vtkPiecewiseFunction()
        alpha.AddPoint(p1, 0.0)
        alpha.AddPoint((p1 + p99) * 0.5, 0.25)
        alpha.AddPoint(p99, 0.9)

        color = vtk.vtkColorTransferFunction()
        color.AddRGBPoint(p1, 0.0, 0.0, 0.0)
        color.AddRGBPoint(p99, 1.0, 1.0, 1.0)

        vol_prop = vtk.vtkVolumeProperty()
        vol_prop.SetColor(color)
        vol_prop.SetScalarOpacity(alpha)
        vol_prop.ShadeOff()
        vol_prop.SetInterpolationTypeToLinear()

        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(img)

        actor = vtk.vtkVolume()
        actor.SetMapper(mapper)
        actor.SetProperty(vol_prop)

        dx, dy, dz = -self.min
        sx, sy, sz = self.spacing
        transform = vtk.vtkTransform()
        transform.Translate(dx * sx, dy * sy, dz * sz)
        actor.SetUserTransform(transform)

        self.renderer.AddActor(actor)
        return actor

    def add_logits_as_cubes(self, volume, shift=np.zeros(3)):
        spacing = self.spacing
        coords = np.argwhere(np.ones_like(volume))
        self.values = volume[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
        if self.global_min is None or self.global_max is None:
            self.global_min, self.global_max = float(self.values.min()), float(self.values.max())
            self.vmin, self.vmax = self.global_min, self.global_max

            rep_min = self.min_slider.GetRepresentation()
            rep_max = self.max_slider.GetRepresentation()
            rep_min.SetMinimumValue(self.global_min)
            rep_min.SetMaximumValue(self.global_max)
            rep_min.SetValue(self.vmin)
            rep_max.SetMinimumValue(self.global_min)
            rep_max.SetMaximumValue(self.global_max)
            rep_max.SetValue(self.vmax)

        vtk_points = vtk.vtkPoints()
        for p in coords:
            x = p[2] * spacing[0] + shift[0]
            y = p[1] * spacing[1] + shift[1]
            z = p[0] * spacing[2] + shift[2]
            vtk_points.InsertNextPoint(x, y, z)

        vtk_values = numpy_support.numpy_to_vtk(self.values, deep=True, array_type=vtk.VTK_FLOAT)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.GetPointData().SetScalars(vtk_values)

        cube_source = vtk.vtkCubeSource()
        cube_source.SetXLength(spacing[0])
        cube_source.SetYLength(spacing[1])
        cube_source.SetZLength(spacing[2])

        glyph_mapper = vtk.vtkGlyph3DMapper()
        glyph_mapper.SetInputData(polydata)
        glyph_mapper.SetSourceConnection(cube_source.GetOutputPort())
        glyph_mapper.SetScalarModeToUsePointData()
        glyph_mapper.SetColorModeToMapScalars()
        glyph_mapper.ScalarVisibilityOn()
        glyph_mapper.SetUseLookupTableScalarRange(True)

        if not self.current_lut:
            self.current_lut = make_saliency_lut(self.values, threshold=self.vmin)
        glyph_mapper.SetLookupTable(self.current_lut)
        glyph_mapper.ScalarVisibilityOn()

        self.current_mapper = glyph_mapper

        self.vtk_img = vtk.vtkImageData()
        self.vtk_img.SetDimensions(volume.shape[2], volume.shape[1], volume.shape[0])
        self.vtk_img.SetSpacing(self.spacing)
        vtk_array = numpy_support.numpy_to_vtk(volume.ravel(order="F"), deep=True)
        self.vtk_img.GetPointData().SetScalars(vtk_array)

        if not self.box_widgets_initialized:
            bounds = self.vtk_img.GetBounds()
            self.box_widget.SetInputData(self.vtk_img)
            interaction.place_box_widget_split(self.box_widget, bounds, side="left")
            self.box_widget.On()
            self.box_widget.AddObserver(
                "EndInteractionEvent",
                lambda obj, event: interaction.on_box_end_interaction(self, obj, event),
            )

            self.control_box_widget.SetInputData(self.vtk_img)
            interaction.place_box_widget_split(self.control_box_widget, bounds, side="right")
            self.control_box_widget.On()
            self.control_box_widget.AddObserver(
                "EndInteractionEvent",
                lambda obj, event: interaction.on_control_box_interaction(self, obj, event),
            )

            self.box_widgets_initialized = True
        else:
            self.box_widget.SetInputData(self.vtk_img)
            self.control_box_widget.SetInputData(self.vtk_img)

        actor = vtk.vtkActor()
        actor.SetMapper(glyph_mapper)
        self.renderer.AddActor(actor)
        return actor

    def add_peaks_mip(self, volume, radius=5.0, axis=0):
        mip = np.max(volume, axis=axis)
        coords = detect_local_max(mip)

        actors = []
        for c in coords:
            y, x = c
            z = np.argmax(volume[:, y, x])

            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(x * self.spacing[0], y * self.spacing[1], z * self.spacing[2])
            sphere.SetRadius(radius)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.5, 0.5, 0)
            actor.GetProperty().SetOpacity(1)
            self.renderer.AddActor(actor)
            actors.append(actor)

        return actors

    def add_peaks(self, volume, radius=5.0):
        coords = detect_local_max(volume)
        actors = []
        for c in coords:
            z, y, x = c
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(x * self.spacing[0], y * self.spacing[1], z * self.spacing[2])
            sphere.SetRadius(radius)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 0, 0)
            actor.GetProperty().SetOpacity(0.95)
            self.renderer.AddActor(actor)
            actors.append(actor)

        return actors

    def _ensure_box_widgets_for_image_mode(self, image_volume):
        if self.vtk_img is None:
            raise RuntimeError(
                "Box widgets require vtk_img to be initialized before image mode. Switch to segmentation mode first."
            )

        self.box_widget.SetInputData(self.vtk_img)
        self.control_box_widget.SetInputData(self.vtk_img)

    def _collect_old_frame_actors(self):
        old_actors = {
            "current_actor": self.current_actor if hasattr(self, "current_actor") else None,
            "image_actor": self.image_actor if hasattr(self, "image_actor") else None,
            "peak_actors": list(self.peak_actors) if hasattr(self, "peak_actors") else [],
            "cylinder_actors": [],
        }

        if (
            hasattr(self, "cylinder_actors")
            and hasattr(self, "current_t")
            and self.current_t in self.cylinder_actors
        ):
            old_actors["cylinder_actors"] = list(self.cylinder_actors[self.current_t])

        return old_actors

    def _build_frame_actors(self, t, volume, image_volume):
        new_state = {
            "current_actor": None,
            "image_actor": None,
            "peak_actors": [],
        }

        if t in self.control_peaks:
            self._show_peak(
                self.control_peaks[t],
                "control_peak_actor",
                (0, 0, 1),
                radius=self.radius,
            )

        if self.show_image and np.any(image_volume):
            self._ensure_box_widgets_for_image_mode(image_volume)
            new_state["image_actor"] = self.add_image_volume(image_volume)
            return new_state

        if np.any(volume):
            # This may be slow. Old image/current actors are still visible here.
            new_state["current_actor"] = self.add_logits_as_cubes(volume)
            new_state["peak_actors"] = self.add_peaks_mip(volume, radius=4)

        return new_state

    def _remove_old_frame_actors(self, old_actors):
        self._remove_actor_if_present(old_actors["current_actor"])
        self._remove_actor_if_present(old_actors["image_actor"])

        for actor in old_actors["peak_actors"]:
            self._remove_actor_if_present(actor)

        for actor in old_actors["cylinder_actors"]:
            self._remove_actor_if_present(actor)

    def _remove_actor_if_present(self, actor):
        if actor is not None:
            self.renderer.RemoveActor(actor)

    def _update_text_actor(self, t):
        self.text_actor.SetInput(f"{self.filenames[t]}")

    def _update_peak_markers(self, t, volume):
        if not self.show_image:
            if t not in self.target_peaks:
                self.target_peaks[t] = analysis.compute_target_peak(volume)

            self._show_peak(
                self.target_peaks[t],
                "target_peak_actor",
                (1, 0, 0),
                radius=self.radius,
            )
        elif self.target_peak_actor:
            self.renderer.RemoveActor(self.target_peak_actor)
            self.target_peak_actor = None

        if self.show_image and self.control_peak_actor:
            self.renderer.RemoveActor(self.control_peak_actor)
            self.control_peak_actor = None

    def _update_current_t(self, t):
        self.current_t = t

    def _update_box_widgets(self, t):
        if t in self.frame_boxes:
            self.box_widget.PlaceWidget(self.frame_boxes[t])
        else:
            if self.vtk_img is None:
                raise RuntimeError("Box widget placement requires vtk_img to be initialized.")

            interaction.place_box_widget_split(
                self.box_widget,
                self.vtk_img.GetBounds(),
                side="left",
            )

        if t in self.control_frame_boxes:
            self.control_box_widget.PlaceWidget(self.control_frame_boxes[t])
        else:
            if self.vtk_img is None:
                raise RuntimeError("Control box placement requires vtk_img to be initialized.")

            interaction.place_box_widget_split(
                self.control_box_widget,
                self.vtk_img.GetBounds(),
                side="right",
            )

        if self.box_widgets_initialized:
            self.box_widget.On()
            self.control_box_widget.On()

    def _update_cylinder_actors(self, t):
        if hasattr(self, "cylinder_actors") and t in self.cylinder_actors:
            for actor in self.cylinder_actors[t]:
                self.renderer.AddActor(actor)

    def update_frame(self, t):
        old_actors = self._collect_old_frame_actors()

        volume = self.volumes[t]
        image_volume = self.images[t]

        new_state = self._build_frame_actors(t, volume, image_volume)

        self._remove_old_frame_actors(old_actors)

        self.current_actor = new_state["current_actor"]
        self.image_actor = new_state["image_actor"]
        self.peak_actors = new_state["peak_actors"]

        self._update_text_actor(t)
        self._update_peak_markers(t, volume)
        self._update_current_t(t)
        self._update_box_widgets(t)
        self._update_cylinder_actors(t)

        self.render_window.Render()

    def start(self):
        self.render_window_interactor.Initialize()
        self.render_window.Render()
        self.render_window_interactor.Start()

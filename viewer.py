import vtk 
import numpy as np
from vtkmodules.util import numpy_support
from process import detect_local_max
import os, csv
from skimage.feature import peak_local_max

class VoxelViewer:

    @staticmethod
    def print_help():
        """Print keyboard controls help at startup"""
        print("\n" + "="*60)
        print("VOXEL VIEWER - KEYBOARD CONTROLS")
        print("="*60)
        print("  r         - Reset target box")
        print("  R         - Reset control box (blue)")
        print("  Enter     - Find and propagate target peak (red)")
        print("  o         - Find and propagate control peak (blue)")
        print("  f         - Compute averages and save to CSV")
        print("  i         - Toggle image-only view")
        print("  c         - Clear all cylinder visualizations")
        print("  Use mouse to rotate, zoom, and pan the view")
        print("  Use slider to navigate through time frames")
        print("="*60 + "\n")

    def _find_peak_and_propagate(self, peak_type='target'):
        """
        Unified function to find and propagate peaks for either target or control.
        
        Args:
            peak_type: 'target' for main target peak or 'control' for control peak
        """
        # Select appropriate widget and storage based on peak type
        if peak_type == 'target':
            box_widget = self.box_widget
            frame_boxes = self.frame_boxes
            peaks = self.target_peaks
            show_peak_func = self.show_target_peak
        elif peak_type == 'control':
            box_widget = self.other_box_widget
            frame_boxes = self.other_frame_boxes
            peaks = self.other_target_peaks
            show_peak_func = self.show_other_target_peak
        else:
            raise ValueError(f"peak_type must be 'target' or 'control', got '{peak_type}'")
        
        t = self.current_t
        vol = self.volumes[t]
        bounds = self.get_box_bounds(box_widget)
        frame_boxes[t] = bounds

        # Peak in current frame (inside current box)
        mask = self.mask_from_bounds(bounds, vol.shape)
        peak_t = self.compute_target_peak(vol, mask=mask)
        peaks[t] = peak_t
        
        # ---- Propagate forward ----
        O = np.array(self.vtk_img.GetOrigin())
        S = np.array(self.vtk_img.GetSpacing())
        
        prev_peak = peak_t
        for next_t in range(t + 1, self.T):
            next_vol = self.volumes[next_t]

            # Mask based on current bounds
            next_mask = self.mask_from_bounds(bounds, next_vol.shape)
            
            peak_next = self.compute_target_peak(next_vol, mask=next_mask)

            if peak_next is None:  # e.g. empty mask
                break

            # Compute current box center (world coords)
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)

            # Peak center in world coords
            px = O[0] + (peak_next[2] + 0.5) * S[0]
            py = O[1] + (peak_next[1] + 0.5) * S[1]

            dx, dy = px - cx, py - cy

            # Shift the bounds
            bounds = [
                xmin + dx, xmax + dx,
                ymin + dy, ymax + dy,
                zmin,      zmax
            ]

            # Store for this frame
            frame_boxes[next_t] = bounds
            peaks[next_t] = peak_next
            prev_peak = peak_next

        # Refresh the current box + peak
        box_widget.PlaceWidget(frame_boxes[t])
        show_peak_func(peaks[t])

    def reset_other_selection_box(self):
        if self.other_box_widget:
            self.other_box_widget.Off()

        self.other_box_widget = vtk.vtkBoxWidget()
        self.other_box_widget.SetInteractor(self.render_window_interactor)
        self.other_box_widget.SetPlaceFactor(1.0)
        self.other_box_widget.TranslationEnabledOn()
        self.other_box_widget.RotationEnabledOff()
        self.other_box_widget.SetHandleSize(0.1)
        self.other_box_widget.GetOutlineProperty().SetColor(0, 0, 1)  # Blue outline
        self.other_box_widget.GetOutlineProperty().SetLineWidth(3.0)
        
        self.other_box_widget.SetInputData(self.vtk_img)
        self.other_box_widget.PlaceWidget()
        self.other_box_widget.On()
        self.other_box_widget.AddObserver("InteractionEvent", self.on_other_box_interaction)

        self.other_frame_boxes[self.current_t] = self.get_box_bounds(self.other_box_widget)
        self.render_window.Render()


    def on_other_box_interaction(self, obj, event):
        self.other_frame_boxes[self.current_t] = self.get_box_bounds(obj)
        
    def reset_selection_box(self):
        self.box_widget.Off()
        # create new widget
        self.box_widget = vtk.vtkBoxWidget()
        self.box_widget.SetInteractor(self.render_window_interactor)
        self.box_widget.SetPlaceFactor(1.0)
        self.box_widget.TranslationEnabledOn()
        self.box_widget.RotationEnabledOff()
        self.box_widget.SetHandleSize(0.01)

        # place it at original bounds
        self.box_widget.SetInputData(self.vtk_img)
        self.box_widget.PlaceWidget()
        self.box_widget.On()
        self.box_widget.AddObserver("InteractionEvent", self.on_box_end_interaction)

        # store/update frame box if needed
        self.frame_boxes[self.current_t] = self.get_box_bounds(self.box_widget)

        self.render_window.Render()

    def _place_box_widget_split(self, widget, full_bounds, side='left', gap_fraction=0.05):
        """
        Place a box widget on either left or right half of the volume with a gap.
        
        Args:
            widget: vtkBoxWidget to position
            full_bounds: tuple (xmin, xmax, ymin, ymax, zmin, zmax) of full volume
            side: 'left' or 'right' 
            gap_fraction: fraction of width to use as gap between boxes (default 5%)
        """
        xmin, xmax, ymin, ymax, zmin, zmax = full_bounds
        x_width = xmax - xmin
        gap = x_width * gap_fraction / 2  # half gap on each side
        x_center = (xmin + xmax) / 2
        
        if side == 'left':
            # Left half: from xmin to center, minus gap
            bounds = (xmin, x_center - gap, ymin, ymax, zmin, zmax)
        elif side == 'right':
            # Right half: from center to xmax, minus gap
            bounds = (x_center + gap, xmax, ymin, ymax, zmin, zmax)
        else:
            raise ValueError(f"side must be 'left' or 'right', got '{side}'")
        
        widget.PlaceWidget(*bounds)

    # --- draw cylinders ---
    def add_cylinder(self, height, radius, center, color, opacity=0.3):
        """
        Create a cylinder actor centered at the same world coords as show_target_peak.
        Uses a transform: cylinder is created around origin, then translated to center.
        """
        z, y, x = center

        # Use same coordinate convention as show_target_peak (world coords = index * spacing)
        sx, sy, sz = self.spacing
        cx = x * sx
        cy = y * sy
        cz = z * sz

        # Create cylinder around origin (axis along Z by default)
        cyl = vtk.vtkCylinderSource()
        cyl.SetRadius(radius * sx)
        cyl.SetHeight(height * sz)
        cyl.SetResolution(48)
        cyl.SetCenter(0.0, 0.0, 0.0)  # center around origin
        cyl.SetCapping(True)

        # Transform: translate to desired world center (no rotation needed)
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
    

    @staticmethod
    def cylinder_mean(image, shift, center, radius, volume, only_segmentation=False):
        # Apply shift with correct axis order
        reordered = np.array([shift[2], shift[1], shift[0]])
        zc, yc, xc = center + reordered 
        zz, yy, xx = np.indices(image.shape)
        dist2d = np.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
        mask = (dist2d <= radius) & (zz == zc)
        vals = image[mask]
        return float(vals.mean()) if vals.size > 0 else 0.0
    
    def compute_averages_and_save_to_csv(self):
        s = self.filenames[0]
        csv_file = f"{s[:s.find('_Time')]}.csv"
        start_t = self.current_t
        t0 = self.current_t
        new_rows = []

        # process all frames from current time onward
        target_means = []
        other_means = []
        for t in range(start_t, self.T):
            if t in self.cylinder_actors:
                for a in self.cylinder_actors[t]:
                    self.renderer.RemoveActor(a)                
                self.cylinder_actors[t] = []

            if t not in self.target_peaks:
                print(f"No target peak selected for frame {t}.")
                continue

            volume = self.volumes[t]
            image = self.images[t]
            peak = self.target_peaks[t]
            other = self.other_target_peaks[t]
            radius = 4  # voxels
            height = self.spacing[2] * 3  # ~3 slices thick

            # --- Target mean ---
            target_mean = VoxelViewer.cylinder_mean(image, self.min, peak, radius, volume, only_segmentation=False)
            other_target_mean = VoxelViewer.cylinder_mean(image, self.min, other, radius, volume, only_segmentation=False)


            new_rows.append([t, target_mean, other_target_mean])
            print(f"Frame {t}: target={target_mean:.3f}, other_target_mean={other_target_mean:.3f}")
            target_means.append(target_mean)
            other_means.append()
            # --- Visualize --- 

            actors = []
            actors.append(self.add_cylinder(height, radius, peak, (1, 0, 0), opacity=0.4))  # target red
            actors.append(self.add_cylinder(height, radius, other, (0, 0, 1), opacity=0.4))  # target red
            self.cylinder_actors[t] = actors

        self.update_csv(csv_file, new_rows, t0)

        self.render_window.Render()
    
    def update_csv(self, csv_file, new_rows, t0):
        # --- Rewrite CSV properly ---
        if os.path.exists(csv_file):
            with open(csv_file, newline="") as f:
                reader = list(csv.reader(f))
            # Detect header and data
            header_line_idx = None
            for i, row in enumerate(reader):
                if len(row) >= 2 and row[0] == "time":
                    header_line_idx = i
                    break
            if header_line_idx is not None:
                header = reader[header_line_idx]
                old_rows = reader[header_line_idx + 1:]
            else:
                header = ["time", "target_mean", "other_target_mean"]
                old_rows = []
            # Preserve the first line if it is the special target_volume_pb
            if reader and reader[0][0] == "target_volume_pb":
                preserved_first_line = reader[0]
            else:
                preserved_first_line = None
        else:
            header = ["time", "target_mean", "other_target_mean"]
            old_rows = []
            preserved_first_line = None

        # keep rows < t0
        kept_rows = [row for row in old_rows if row[0] == 'target_volume_pb' or (row[0].isdigit() and int(row[0]) < t0)]

        # merge with new data
        all_rows = kept_rows + new_rows

        # write back the file
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            # Write preserved volume line or compute if missing
            if preserved_first_line:
                writer.writerow(preserved_first_line)
            elif not self.first_enter:
                self.first_enter = True
                volume_first_target = self.compute_volume_of_target_time_zero()
                writer.writerow(["target_volume_pb", volume_first_target, "", ""])
            # Write header and data
            writer.writerow(header)
            writer.writerows(all_rows)

        print(f"CSV updated from frame {t0} onwards → {csv_file}")

    def add_key_listener(self):
        def keypress_callback(obj, event):
            key = obj.GetKeySym()
            slider_rep = self.time_slider.GetRepresentation()
            value = slider_rep.GetValue()
            print(f'{key} pressed.')
            # Left / Right arrows move one integer step
            match key:
                case "Right":
                    new_val = min(value + 1, slider_rep.GetMaximumValue())
                    slider_rep.SetValue(new_val)
                    self.time_slider.InvokeEvent(vtk.vtkCommand.InteractionEvent)
                case "Left":
                    new_val = max(value - 1, slider_rep.GetMinimumValue())
                    slider_rep.SetValue(new_val)
                    self.time_slider.InvokeEvent(vtk.vtkCommand.InteractionEvent)
                case "Return":  # Enter key
                    if hasattr(self, "current_selection_voxels"):
                        self.find_local_peak_and_propagate_consequent_peaks()
                    # self.highlight_selection_spheres(radius=10, color=(1,0,0,1))
                case "Return":  # main target
                    self._find_peak_and_propagate(peak_type='target')
                case "o":  # blue other target
                    
                    self._find_peak_and_propagate(peak_type='control')
                case "R":  # capital R for other box
                    self.reset_other_selection_box()
                case "r":
                    self.reset_selection_box()
                case "i":
                    self.show_image = not self.show_image
                    self.update_frame(self.current_t)
                case 'f':
                    self.compute_averages_and_save_to_csv()

        self.render_window_interactor.AddObserver("KeyPressEvent", keypress_callback)

    def show_other_target_peak(self, peak_coords, radius=4):
        if self.other_target_peak_actor:
            self.renderer.RemoveActor(self.other_target_peak_actor)

        z, y, x = peak_coords
        spacing = self.spacing

        cube = vtk.vtkCubeSource()
        cube.SetXLength(spacing[0]*radius)
        cube.SetYLength(spacing[1]*radius)
        cube.SetZLength(spacing[2]*radius)
        cube.SetCenter(x * spacing[0], y * spacing[1], z * spacing[2])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0, 1)   # Blue cube
        actor.GetProperty().SetOpacity(1.0)

        self.renderer.AddActor(actor)
        self.other_target_peak_actor = actor
        self.render_window.Render()

    def compute_target_peak2(self, volume, mask=None, prev_peak=None,
                        min_distance=2, num_candidates=5,
                        intensity_weight=1.0, distance_penalty=0.2, jump_penalty=0.0):
        """
        Find the best peak among multiple local maxima.
        
        Parameters
        ----------
        volume : ndarray
            The (Z, Y, X) 3D array.
        mask : ndarray[bool], optional
            Mask restricting search area.
        prev_peak : tuple (z,y,x), optional
            Previous peak coordinates to favor continuity.
        min_distance : int
            Minimum spacing (voxels) between detected peaks.
        num_candidates : int
            Max number of candidate peaks to evaluate.
        intensity_weight : float
            Higher = prefer brighter peaks.
        distance_weight : float
            Higher = prefer peaks closer to prev_peak.
        jump_weight : float
            Penalize extreme intensity jumps relative to prev_peak.

        Returns
        -------
        (z,y,x) : tuple
            Selected peak coordinates.
        """

        if mask is not None:
            search_vol = np.where(mask, volume, -np.inf)
        else:
            search_vol = volume

        # Find local maxima
        candidates = peak_local_max(search_vol, 
                                    min_distance=min_distance,
                                    num_peaks=num_candidates)

        if len(candidates) == 0:
            return prev_peak if prev_peak is not None else np.unravel_index(np.argmax(search_vol), volume.shape)

        # Evaluate candidates
        scores = []
        for (z, y, x) in candidates:
            intensity = volume[z, y, x]

            # distance to previous peak (if given)
            if prev_peak is not None:
                dz, dy, dx = np.array([z, y, x]) - np.array(prev_peak)
                dist = np.sqrt(dz**2 + dy**2 + dx**2)
            else:
                dist = 0.0

            # intensity jump relative to previous peak
            if prev_peak is not None:
                prev_int = volume[tuple(prev_peak)]
                jump = abs(intensity - prev_int)
            else:
                jump = 0.0

            # scoring function (bigger is better, so subtract penalties)
            score = (intensity_weight * intensity
                    - distance_penalty * dist
                    - jump_penalty * jump)

            scores.append(score)

        # Pick best candidate
        best_idx = int(np.argmax(scores))
        return tuple(candidates[best_idx])
    
    def compute_target_peak(self, volume, mask=None):
        """
        Compute the local maximum in the volume or in a masked region.
        Returns (z, y, x).
        """
        if mask is not None:
            masked_volume = np.where(mask, volume, -np.inf)
        else:
            masked_volume = volume

        # Find coordinates of max
        flat_idx = np.argmax(masked_volume)
        return np.unravel_index(flat_idx, volume.shape)
    
    def show_target_peak(self, peak_coords, radius=4):
        """
        Show a bright red cube at the target peak.
        """
        if self.target_peak_actor:
            self.renderer.RemoveActor(self.target_peak_actor)

        z, y, x = peak_coords
        spacing = self.spacing

        cube = vtk.vtkCubeSource()
        cube.SetXLength(spacing[0]*radius)
        cube.SetYLength(spacing[1]*radius)
        cube.SetZLength(spacing[2]*radius)
        cube.SetCenter(x * spacing[0], y * spacing[1], z * spacing[2])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)   # bright red
        actor.GetProperty().SetOpacity(1.0)

        self.renderer.AddActor(actor)
        self.target_peak_actor = actor
        self.render_window.Render()

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
        self.current_lut = None   # <-- store LUT reference
        self.values = None        # store last saliency values
        self.frame_boxes = {} 
        self.cylinder_actors = {}
        self.target_peaks = {}   # dict: frame index -> (z, y, x)
        self.box_widgets_initialized = False  # Track if box widgets have been set up
        self.target_peak_actor = None
        self.other_target_peaks = {}
        self.other_frame_boxes = {}
        self.other_target_peak_actor = None
        self.other_cylinder_actors = {}

        self.global_min = None
        self.global_max = None
        self.vmin = 0.0
        self.vmax = 1.0
        self.first_enter = False
        self.shift = np.array([0,0,0])
        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.7, 0.7, 0.7)  # white background

        # Render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1600, 900)

        self.current_mapper = None

        # Interactor
        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.render_window_interactor.SetRenderWindow(self.render_window)

        self.box_widget = vtk.vtkBoxWidget()
        self.box_widget.SetInteractor(self.render_window_interactor)
        self.box_widget.SetPlaceFactor(1.0)
        self.box_widget.TranslationEnabledOn()
        self.box_widget.RotationEnabledOff()
        self.box_widget.SetHandleSize(0.005)

        self.other_box_widget = vtk.vtkBoxWidget()
        self.other_box_widget.SetInteractor(self.render_window_interactor)
        self.other_box_widget.SetPlaceFactor(1.0)
        self.other_box_widget.TranslationEnabledOn()
        self.other_box_widget.RotationEnabledOff()
        self.other_box_widget.SetHandleSize(0.005)
        self.other_box_widget.GetOutlineProperty().SetColor(0, 0, 1)  # Blue outline for control
        self.other_box_widget.GetOutlineProperty().SetLineWidth(3.0)

        # Text actor
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.GetTextProperty().SetFontSize(25)
        self.text_actor.GetTextProperty().SetColor(0, 0, 0)
        self.text_actor.SetDisplayPosition(50, 0)
        self.renderer.AddActor2D(self.text_actor)

        # Slider
        self.time_slider = self.make_slider(0, self.T - 1, 0.1, 0.05, 0.9, 0.05, self.update_frame, is_int=True)
        self.time_slider.SetInteractor(self.render_window_interactor)
        self.time_slider.EnabledOn()
        self.add_key_listener()

        self.min_slider = self.make_slider(0, 1, 0.1, 0.95, 0.45, 0.95, self.update_window_min)
        self.min_slider.SetInteractor(self.render_window_interactor)
        self.min_slider.EnabledOn()

        self.max_slider = self.make_slider(0, 1, 0.55, 0.95, 0.9, 0.95, self.update_window_max)
        self.max_slider.SetInteractor(self.render_window_interactor)
        self.max_slider.EnabledOn()
        self.vtk_img = None 
        # Draw first frame
        self.update_frame(self.current_t)
        
        # Print help controls
        self.print_help()

    def compute_volume_of_target_time_zero(self):
        """
        Compute the segmentation volume at t=0 inside the cylinder
        centered on the target peak. Uses logits > 0.5 as segmentation mask.
        Returns voxel count (no spacing scaling).
        """
        t0 = 0
        if t0 not in self.target_peaks:
            print("No target peak defined for time 0.")
            return 0

        volume = self.volumes[t0]       # (Z, Y, X) logit values
        peak = self.target_peaks[t0]    # (z, y, x)
        radius = 4                      # cylinder radius in voxels
        height = 3                      # cylinder height in slices

        # segmentation mask
        segmentation = volume > 0.5

        # center coordinates
        zc, yc, xc = peak

        # axial slices range for the cylinder
        half_h = height // 2
        zmin = max(0, zc - half_h)
        zmax = min(volume.shape[0], zc + half_h + 1)

        # build a cylindrical mask
        zz, yy, xx = np.indices(volume.shape)
        dist2d = np.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
        cyl_mask = (dist2d <= radius) & (zz >= zmin) & (zz <= zmax)

        # apply mask to segmentation
        seg_cyl = segmentation & cyl_mask

        # raw voxel count
        seg_volume = int(np.sum(seg_cyl))

        print(f"Segmentation voxel volume at t=0 in cylinder = {seg_volume}")
        return seg_volume

    def world_to_voxel(self, x,y,z):
        """
        Convert N x 3 world coordinates to voxel indices
        """
        i = int(round((z - self.shift[2]) / self.spacing[2]))  # Z → i
        j = int(round((y - self.shift[1]) / self.spacing[1]))  # Y → j
        k = int(round((x - self.shift[0]) / self.spacing[0]))  # X → k
        # Clip to volume bounds
        i = np.clip(i, 0, self.Z-1)
        j = np.clip(j, 0, self.Y-1)
        k = np.clip(k, 0, self.X-1)
        return np.array([i, j, k])

    def update_window_min(self, v):
        self.vmin = v
        self.rebuild_lut()

    def update_window_max(self, v):
        self.vmax = v
        self.rebuild_lut()
        
    def rebuild_lut(self):
        if self.values is None or self.current_lut is None:
            return

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.Build()
        lut.SetRange(self.vmin, self.vmax)

        # Color ramp: blue → green → yellow → red
        ctf = vtk.vtkColorTransferFunction()
        ctf.AddRGBPoint(self.vmin, 0.0, 0.0, 1.0)
        ctf.AddRGBPoint((self.vmin + self.vmax) / 2, 0.0, 1.0, 0.0)
        ctf.AddRGBPoint(self.vmax, 1.0, 0.0, 0.0)
        n_colors = 256
        for i in range(n_colors):

            v = self.vmin + (i / 255.0) * (self.vmax - self.vmin)
            r, g, b = ctf.GetColor(v)

            # Transparency mapping: below threshold = invisible
            if v < self.vmin or v > self.vmax:
                alpha = 0.0
            else:
                # Scale alpha gradually with logit
                alpha = (v - self.vmin) / (self.vmax - self.vmin)

            lut.SetTableValue(i, r, g, b, alpha)

        self.current_lut.DeepCopy(lut)
        self.render_window.Render()

    @staticmethod
    def make_saliency_lut(values, n_colors=256, threshold=None):
        """
        Create a LUT that maps logits to saliency-like colors.
        - values: numpy array of logits to extract min/max
        - threshold: optional lower cutoff; everything below is transparent
        """
        min_val, max_val = float(values.min()), float(values.max())

        if threshold is None:
            threshold = min_val

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(n_colors)
        lut.Build()
        lut.SetRange(min_val, max_val)

        # Color transfer function: blue → green → yellow → red
        ctf = vtk.vtkColorTransferFunction()
        ctf.AddRGBPoint(threshold, 0.0, 0.0, 1.0)   # blue
        ctf.AddRGBPoint((min_val + max_val) / 2, 0.0, 1.0, 0.0)  # green
        ctf.AddRGBPoint(max_val, 1.0, 0.0, 0.0)     # red

        for i in range(n_colors):
            v = threshold + (i / (n_colors - 1)) * (max_val - threshold)
            r, g, b = ctf.GetColor(v)

            # Transparency mapping: below threshold = invisible
            if v < threshold:
                alpha = 0.0
            else:
                # Scale alpha gradually with logit
                alpha = (v - threshold) / (max_val - threshold + 1e-6)

            lut.SetTableValue(i, r, g, b, alpha)

        return lut

    def mask_from_bounds(self, bounds, shape):
        """
        Create a boolean mask from vtkBox bounds: [xmin, xmax, ymin, ymax, zmin, zmax].
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        # Convert to voxel indices
        zi_min = int(np.clip(zmin / self.spacing[2], 0, shape[0]-1))
        zi_max = int(np.clip(zmax / self.spacing[2], 0, shape[0]-1))
        yi_min = int(np.clip(ymin / self.spacing[1], 0, shape[1]-1))
        yi_max = int(np.clip(ymax / self.spacing[1], 0, shape[1]-1))
        xi_min = int(np.clip(xmin / self.spacing[0], 0, shape[2]-1))
        xi_max = int(np.clip(xmax / self.spacing[0], 0, shape[2]-1))

        mask = np.zeros(shape, dtype=bool)
        mask[zi_min:zi_max+1, yi_min:yi_max+1, xi_min:xi_max+1] = True
        return mask

    def get_box_bounds(self, widget):
        # Ask widget to output its current geometry
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

        # Get current volume (use current frame, not self.T)
        cur_vol = self.volumes[self.current_t]   # shape (Z, Y, X)

        # Split into index arrays
        zi, yi, xi = voxels[:,0], voxels[:,1], voxels[:,2]

        # Make sure we stay inside the array bounds
        zi = np.clip(zi, 0, cur_vol.shape[0]-1)
        yi = np.clip(yi, 0, cur_vol.shape[1]-1)
        xi = np.clip(xi, 0, cur_vol.shape[2]-1)
        zmin, zmax = zi.min(), zi.max()
        ymin, ymax = yi.min(), yi.max()
        xmin, xmax = xi.min(), xi.max()
        sub_vol = cur_vol[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
        # Sample intensities
        return sub_vol

    def add_image_volume(self, vol_data):
        """
        Add raw microscopy volume (Z, Y, X) as a grayscale intensity rendering.
        """

        # Ensure float32
        vol_data = vol_data.astype(np.float32)

        # Convert numpy to VTK image
        flat = vol_data.ravel()
        vtk_data = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)

        img = vtk.vtkImageData()
        img.SetDimensions(vol_data.shape[2], vol_data.shape[1], vol_data.shape[0])
        img.GetPointData().SetScalars(vtk_data)
        img.SetSpacing(*self.spacing)

        # Robust windowing improves visibility even with outliers.
        p1, p99 = np.percentile(vol_data, [1, 99])
        if p99 <= p1:
            p1 = float(np.min(vol_data))
            p99 = float(np.max(vol_data))

        # Opacity mapping for image-only rendering.
        alpha = vtk.vtkPiecewiseFunction()
        alpha.AddPoint(p1, 0.0)
        alpha.AddPoint((p1 + p99) * 0.5, 0.25)
        alpha.AddPoint(p99, 0.9)

        # Grayscale color transfer
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
        sx, sy, sz = self.spacing  # spacing from your dataset
        transform = vtk.vtkTransform()
        transform.Translate(dx * sx, dy * sy, dz * sz)
        actor.SetUserTransform(transform)
        
        self.renderer.AddActor(actor)
        return actor

    def add_logits_as_cubes(self, volume, shift=np.zeros(3)):
        """Render every voxel as a cube with color based on logit value."""
        spacing = self.spacing

        # volume, offset = crop_to_nonzero_bbox(volume)


        # Get voxel coordinates and values
        coords = np.argwhere(np.ones_like(volume))   # all voxels
        self.values = volume[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
        if self.global_min is None or self.global_max is None:
            self.global_min, self.global_max = float(self.values.min()), float(self.values.max())
            self.vmin, self.vmax = self.global_min, self.global_max            
            
            rep_min = self.min_slider.GetRepresentation()
            rep_max = self.max_slider.GetRepresentation()

            rep_min.SetMinimumValue(self.global_min)
            rep_min.SetMaximumValue(self.global_max)
            rep_min.SetValue( self.vmin)

            rep_max.SetMinimumValue(self.global_min)
            rep_max.SetMaximumValue(self.global_max)
            rep_max.SetValue(self.vmax)
            

        # thresholded = volume[volume > self.vmin and volume < self.vmax]
        # Create vtkPoints
        vtk_points = vtk.vtkPoints()
        for p in coords:
            x = p[2] * spacing[0] + shift[0]
            y = p[1] * spacing[1] + shift[1]
            z = p[0] * spacing[2] + shift[2]
            vtk_points.InsertNextPoint(x, y, z)

        # Attach scalars (logits) to points
        vtk_values = numpy_support.numpy_to_vtk(self.values, deep=True, array_type=vtk.VTK_FLOAT)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.GetPointData().SetScalars(vtk_values)

        # Cube source
        cube_source = vtk.vtkCubeSource()
        cube_source.SetXLength(spacing[0])
        cube_source.SetYLength(spacing[1])
        cube_source.SetZLength(spacing[2])

        # Glyph mapper (cubes at points)
        glyph_mapper = vtk.vtkGlyph3DMapper()
        glyph_mapper.SetInputData(polydata)
        glyph_mapper.SetSourceConnection(cube_source.GetOutputPort())
        glyph_mapper.SetScalarModeToUsePointData()
        glyph_mapper.SetColorModeToMapScalars()
        glyph_mapper.ScalarVisibilityOn()

        # Make mapper respect LUT’s [vmin, vmax] instead of data range
        glyph_mapper.SetUseLookupTableScalarRange(True)

        # Create LUT and hook it up
        if not self.current_lut:
            self.current_lut = self.make_saliency_lut(self.values, threshold=self.vmin)
        glyph_mapper.SetLookupTable(self.current_lut)
        glyph_mapper.ScalarVisibilityOn()

        self.current_mapper = glyph_mapper

        self.vtk_img = vtk.vtkImageData()
        self.vtk_img.SetDimensions(volume.shape[2], volume.shape[1], volume.shape[0])  # X,Y,Z
        self.vtk_img.SetSpacing(self.spacing)
        vtk_array = numpy_support.numpy_to_vtk(volume.ravel(order='F'), deep=True)
        self.vtk_img.GetPointData().SetScalars(vtk_array)

        # Only initialize box widgets once with split placement
        if not self.box_widgets_initialized:
            # Get full image bounds
            bounds = self.vtk_img.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
            
            # configure box widget to operate on the image bounds
            self.box_widget.SetInputData(self.vtk_img)
            # Place target box on LEFT half (with small gap)
            self._place_box_widget_split(self.box_widget, bounds, side='left')
            self.box_widget.On()
            self.box_widget.AddObserver("EndInteractionEvent", self.on_box_end_interaction)
            
            self.other_box_widget.SetInputData(self.vtk_img)
            # Place control box on RIGHT half (with small gap)
            self._place_box_widget_split(self.other_box_widget, bounds, side='right')
            self.other_box_widget.On()
            self.other_box_widget.AddObserver("EndInteractionEvent", self.on_other_box_interaction)
            
            self.box_widgets_initialized = True
        else:
            # Just update the input data without resetting placement
            self.box_widget.SetInputData(self.vtk_img)
            self.other_box_widget.SetInputData(self.vtk_img)
        # Actor
        actor = vtk.vtkActor()
        actor.SetMapper(glyph_mapper)
        self.renderer.AddActor(actor)
        return actor

    def add_peaks_mip(self, volume, radius=5.0, axis=0):
        """
        Detect peaks in the 2D maximum intensity projection along a given axis
        and add spheres at each peak position (projected at Z of max intensity).
        """
        # Compute 2D maximum intensity projection
        mip = np.max(volume, axis=axis)  # shape (Y, X)

        # Threshold the MIP
        # mask = (mip >= self.vmin) & (mip <= self.vmax)
        # masked_mip = mip * mask

        # Detect 2D peaks
        coords = detect_local_max(mip)

        actors = []
        for c in coords:
            y, x = c

            # Find Z coordinate of maximum intensity along the column
            z = np.argmax(volume[:, y, x])

            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(x * self.spacing[0], y * self.spacing[1], z * self.spacing[2])
            sphere.SetRadius(radius)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.5,0.5, 0)
            actor.GetProperty().SetOpacity(1)
            self.renderer.AddActor(actor)
            actors.append(actor)

        return actors

    def add_peaks(self, volume, radius=5.0):
        """
        Detect local peaks and add spheres at each centroid.
        Returns a list of vtkActor spheres.
        """
        # Threshold volume to focus on high-value regions
        # mask = (volume >= self.vmin) & (volume <= self.vmax)
        # masked_volume = volume * mask

        # Detect local maxima (peaks)
        


        coords = detect_local_max(volume)

        actors = []
        for c in coords:
            z, y, x = c

            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(x*self.spacing[0], y*self.spacing[1], z*self.spacing[2])
            sphere.SetRadius(radius)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 0, 0)  # red
            actor.GetProperty().SetOpacity(0.95)

            self.renderer.AddActor(actor)
            actors.append(actor)

        return actors
    
    def update_frame(self, t):
        # Remove old actor
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
        if hasattr(self, "peak_actors"):
            for a in self.peak_actors:
                self.renderer.RemoveActor(a)
        if hasattr(self, "image_actor") and self.image_actor:
            self.renderer.RemoveActor(self.image_actor)
        if t in self.other_target_peaks:
            self.show_other_target_peak(self.other_target_peaks[t])
        self.peak_actors = []

        self.image_actor = None

        if hasattr(self, "cylinder_actors") and hasattr(self, "current_t"):
            if self.current_t in self.cylinder_actors:
                for a in self.cylinder_actors[self.current_t]:
                    self.renderer.RemoveActor(a)

        # Get frame data
        volume = self.volumes[t]   # shape (Z, Y, X)
        image_volume = self.images[t]

        if self.show_image and np.any(image_volume):
            # Image-only mode: do not overlay logits.
            self.current_actor = None
            self.image_actor = self.add_image_volume(image_volume)
        else:
            if np.any(volume):
                # Logits mode
                self.current_actor = self.add_logits_as_cubes(volume)
                self.peak_actors = self.add_peaks_mip(volume, radius=4.0)
            else:
                self.current_actor = None
            
        self.text_actor.SetInput(
            f"{self.filenames[t]}"
        )

        if not self.show_image:
            if t not in self.target_peaks:
                peak = self.compute_target_peak(volume)
                self.target_peaks[t] = peak

            # Show target marker only in logits mode.
            self.show_target_peak(self.target_peaks[t])
        elif self.target_peak_actor:
            self.renderer.RemoveActor(self.target_peak_actor)
            self.target_peak_actor = None
        if self.show_image and self.other_target_peak_actor:
            self.renderer.RemoveActor(self.other_target_peak_actor)
            self.other_target_peak_actor = None

        self.current_t = t
        if t in self.frame_boxes:
            bounds = self.frame_boxes[t]
            self.box_widget.PlaceWidget(bounds)
        else:
            # first time this frame → place with left split
            if self.vtk_img:
                bounds = self.vtk_img.GetBounds()
                self._place_box_widget_split(self.box_widget, bounds, side='left')
            else:
                self.box_widget.PlaceWidget()

        # Handle other_box_widget similarly
        if t in self.other_frame_boxes:
            other_bounds = self.other_frame_boxes[t]
            self.other_box_widget.PlaceWidget(other_bounds)
        else:
            # first time this frame → place with right split
            if self.vtk_img:
                bounds = self.vtk_img.GetBounds()
                self._place_box_widget_split(self.other_box_widget, bounds, side='right')
            else:
                self.other_box_widget.PlaceWidget()

        if hasattr(self, "cylinder_actors") and t in self.cylinder_actors:
            for a in self.cylinder_actors[t]:
                self.renderer.AddActor(a)

        self.render_window.Render()

    def make_slider(self, min_val, max_val, x1, y1, x2, y2, callback, is_int=False):
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

        def on_slider(obj, event):
            val = obj.GetRepresentation().GetValue()
            if is_int:
                val = int(round(val))
                obj.GetRepresentation().SetValue(val) 
            callback(val)

        slider_widget.AddObserver("InteractionEvent", on_slider)
        return slider_widget

    def start(self):
        self.render_window_interactor.Initialize()
        self.render_window.Render()
        self.render_window_interactor.Start()
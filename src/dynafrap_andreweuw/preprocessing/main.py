"""
Convert one Leica LIF package into TIFF Z-stacks.

Configuration is read from config.py in the same directory.

The output reproduces the supplied Fiji/ImageJ macro:

    OUTPUT_TIFF_DIRECTORY/
    └── <input filename.lif>/
        └── <series name>/
            ├── Channel_1_Time_1.tif
            ├── Channel_1_Time_2.tif
            └── ...

For each selected channel and timepoint, the complete Z-stack is written as
one ImageJ-compatible TIFF.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import tifffile
from liffile import LifFile, LifFlimImage, LifImage

import dynafrap_andreweuw.preprocessing.config as config

SPATIAL_DIM_IDS = {
    "X": 1,
    "Y": 2,
    "Z": 3,
}

MICROMETRE_FACTORS = {
    "m": 1_000_000.0,
    "meter": 1_000_000.0,
    "metre": 1_000_000.0,
    "mm": 1_000.0,
    "millimeter": 1_000.0,
    "millimetre": 1_000.0,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "nm": 0.001,
    "nanometer": 0.001,
    "nanometre": 0.001,
}


def validate_configuration() -> tuple[Path, Path, set[int] | None, bool]:
    """Validate and normalize values imported from config.py."""
    input_lif = config.INPUT_LIF
    output_directory = config.OUTPUT_TIFF_DIRECTORY
    include_bleach = bool(config.INCLUDE_BLEACH)

    if not input_lif.exists():
        raise FileNotFoundError(f"INPUT_LIF does not exist: {input_lif}")

    if not input_lif.is_file():
        raise ValueError(f"INPUT_LIF is not a file: {input_lif}")

    if input_lif.suffix != ".lif":
        raise ValueError(
            "INPUT_LIF must end with the case-sensitive suffix '.lif': "
            f"{input_lif}"
        )

    selected_channels_config = config.SELECTED_CHANNELS

    if selected_channels_config is None:
        selected_channels = None
    else:
        if not isinstance(selected_channels_config, (list, tuple, set)):
            raise TypeError(
                "SELECTED_CHANNELS must be None or a sequence of one-based "
                "integer channel numbers."
            )

        selected_channels = set()

        for channel_number in selected_channels_config:
            if isinstance(channel_number, bool) or not isinstance(
                channel_number, int
            ):
                raise TypeError(
                    "Every SELECTED_CHANNELS entry must be an integer."
                )

            if channel_number < 1:
                raise ValueError(
                    "SELECTED_CHANNELS uses one-based numbering; every value "
                    "must be at least 1."
                )

            selected_channels.add(channel_number)

        if not selected_channels:
            raise ValueError(
                "SELECTED_CHANNELS cannot be empty. Use None to export all "
                "channels."
            )

    output_directory.mkdir(parents=True, exist_ok=True)

    return (
        input_lif,
        output_directory,
        selected_channels,
        include_bleach,
    )


def series_directory_name(series_name: str) -> str:
    """
    Reproduce the macro's series-name transformations:

        replace(seriesName, "/", "_")
        replace(seriesName, ".lif", "")
    """
    return series_name.replace("/", "_").replace(".lif", "")


def spatial_spacing_um(image: LifImage, axis: str) -> float | None:
    """
    Return X, Y, or Z sampling in micrometres when present in Leica metadata.
    """
    dim_id = SPATIAL_DIM_IDS[axis]

    dimensions = image.xml_element.findall(
        "./Data/Image/ImageDescription/Dimensions/DimensionDescription"
    )

    for dimension in dimensions:
        if int(dimension.attrib["DimID"]) != dim_id:
            continue

        number_of_elements = int(dimension.attrib["NumberOfElements"])
        if number_of_elements <= 1:
            return None

        physical_length = abs(float(dimension.attrib["Length"]))
        unit = dimension.attrib.get("Unit", "").strip()
        factor = MICROMETRE_FACTORS.get(unit.casefold())

        if factor is None:
            return None

        return physical_length * factor / (number_of_elements - 1)

    return None


def extract_channel_timepoint(
    image: LifImage,
    channel_index: int,
    time_index: int,
) -> tuple[np.ndarray, str]:
    """
    Reproduce ImageJ Duplicate... for one C/T selection and all Z slices.

    channel_index and time_index are zero-based internally.
    """
    permitted_dimensions = {"X", "Y", "Z", "C", "T"}
    unsupported_dimensions = set(image.sizes) - permitted_dimensions

    if unsupported_dimensions:
        raise NotImplementedError(
            f"Series {image.name!r} contains dimensions outside the macro's "
            f"XYCZT model: {sorted(unsupported_dimensions)}"
        )

    selection: dict[str, int] = {}

    if "C" in image.sizes:
        selection["C"] = channel_index
    elif channel_index != 0:
        raise IndexError(
            f"Series {image.name!r} has no explicit C dimension."
        )

    if "T" in image.sizes:
        selection["T"] = time_index
    elif time_index != 0:
        raise IndexError(
            f"Series {image.name!r} has no explicit T dimension."
        )

    selected = image.frames(**selection)
    # Memory-map each read; LIF packages are often several hundred MB.
    data = selected.asarray(out="memmap")
    dimensions = list(selected.dims)

    # Fixed C and T selections are normally squeezed. Remove any singleton
    # remnants defensively.
    for dimension in ("C", "T"):
        if dimension not in dimensions:
            continue

        axis = dimensions.index(dimension)

        if data.shape[axis] != 1:
            raise ValueError(
                f"Expected singleton {dimension} after selection, but got "
                f"shape {data.shape} with axes {dimensions}."
            )

        data = np.take(data, 0, axis=axis)
        dimensions.pop(axis)

    if set(dimensions) == {"Y", "X"}:
        target_dimensions = ["Y", "X"]
        output_axes = "YX"
    elif set(dimensions) == {"Z", "Y", "X"}:
        target_dimensions = ["Z", "Y", "X"]
        output_axes = "ZYX"
    else:
        raise ValueError(
            f"Cannot reduce series {image.name!r} to YX or ZYX. "
            f"Resulting axes: {dimensions}; shape: {data.shape}."
        )

    permutation = [
        dimensions.index(dimension)
        for dimension in target_dimensions
    ]
    data = np.transpose(data, axes=permutation)

    return np.ascontiguousarray(data), output_axes


def save_imagej_tiff(
    output_path: Path,
    data: np.ndarray,
    axes: str,
    image: LifImage,
) -> None:
    """
    Save raw values as an ImageJ-compatible TIFF and preserve calibration.
    """
    metadata: dict[str, object] = {"axes": axes}
    write_options: dict[str, object] = {
        "imagej": True,
        "photometric": "minisblack",
        "metadata": metadata,
    }

    x_spacing_um = spatial_spacing_um(image, "X")
    y_spacing_um = spatial_spacing_um(image, "Y")
    z_spacing_um = spatial_spacing_um(image, "Z")

    if x_spacing_um is not None and y_spacing_um is not None:
        write_options["resolution"] = (
            1.0 / x_spacing_um,
            1.0 / y_spacing_um,
        )
        metadata["unit"] = "um"

    if axes == "ZYX" and z_spacing_um is not None:
        metadata["spacing"] = z_spacing_um
        metadata["unit"] = "um"

    tifffile.imwrite(
        output_path,
        data,
        **write_options,
    )


def channels_for_series(
    channel_count: int,
    selected_channels: set[int] | None,
    series_name: str,
) -> list[int]:
    """
    Return zero-based channels for a series.

    Configuration uses one-based channel numbers to match ImageJ.
    """
    if selected_channels is None:
        return list(range(channel_count))

    unavailable = sorted(
        channel_number
        for channel_number in selected_channels
        if channel_number > channel_count
    )

    if unavailable:
        raise ValueError(
            f"Series {series_name!r} has {channel_count} channel(s), but "
            f"SELECTED_CHANNELS requests unavailable channel(s): "
            f"{unavailable}"
        )

    return sorted(channel_number - 1 for channel_number in selected_channels)


def process_lif_file(
    input_lif: Path,
    output_directory: Path,
    selected_channels: set[int] | None,
    include_bleach: bool,
) -> int:
    """Process the configured LIF package."""
    lif_directory = output_directory / input_lif.stem
    lif_directory.mkdir(parents=True, exist_ok=True)

    written_count = 0

    with LifFile(input_lif, squeeze=True) as lif:
        series_count = len(lif.images)

        if series_count == 0:
            raise ValueError(f"No image series found in {input_lif}")

        print(f"Input LIF: {input_lif}")
        print(f"Output root: {lif_directory}")
        print(f"Series count: {series_count}")

        for series_index, image in enumerate(lif.images, start=1):
            if isinstance(image, LifFlimImage):
                raise NotImplementedError(
                    f"Series {series_index} in {input_lif.name} is a raw "
                    "FLIM series."
                )

            # The ImageJ macro uses the Bio-Formats window title. Here we use
            # the actual Leica series name stored in the LIF.
            series_name = image.name

            # Match indexOf(seriesName, "Bleach") case-sensitively.
            if not include_bleach and "Bleach" in series_name:
                print(
                    f"[{series_index}/{series_count}] "
                    f"Skipping Bleach series: {series_name}"
                )
                continue

            channel_count = int(image.sizes.get("C", 1))
            slice_count = int(image.sizes.get("Z", 1))
            frame_count = int(image.sizes.get("T", 1))

            channel_indices = channels_for_series(
                channel_count=channel_count,
                selected_channels=selected_channels,
                series_name=series_name,
            )

            safe_series_name = series_directory_name(series_name)
            series_directory = lif_directory / safe_series_name
            series_directory.mkdir(parents=True, exist_ok=True)

            print(
                f"[{series_index}/{series_count}] {series_name}: "
                f"C={channel_count}, Z={slice_count}, T={frame_count}"
            )

            for channel_index in channel_indices:
                for time_index in range(frame_count):
                    data, axes = extract_channel_timepoint(
                        image=image,
                        channel_index=channel_index,
                        time_index=time_index,
                    )

                    output_path = series_directory / (
                        f"Channel_{channel_index + 1}"
                        f"_Time_{time_index + 1}.tif"
                    )

                    save_imagej_tiff(
                        output_path=output_path,
                        data=data,
                        axes=axes,
                        image=image,
                    )

                    written_count += 1
                    print(f"  Saved {output_path}")

    return written_count


def run() -> int:
    try:
        (
            input_lif,
            output_directory,
            selected_channels,
            include_bleach,
        ) = validate_configuration()

        written_count = process_lif_file(
            input_lif=input_lif,
            output_directory=output_directory,
            selected_channels=selected_channels,
            include_bleach=include_bleach,
        )

        print(f"Macro ends: wrote {written_count} TIFF file(s)")
        return 0

    except Exception as error:
        traceback.print_exc()
        print(
            f"\nConversion failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

def main() -> None:
    raise SystemExit(run())

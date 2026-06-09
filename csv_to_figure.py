import argparse
import ast
import json
from pathlib import Path

import pandas as pd

TARGET_CENTER_COL = "target_center[z=depth,y,x]"
CONTROL_CENTER_COL = "control_center[z=depth,y,x]"
IMAGE_SHAPE_COL = "image_shape[z=depth,y,x]"

REQUIRED_COLUMNS = [
    "time",
    "target_mean",
    "control_mean",
    TARGET_CENTER_COL,
    "target_radius",
    "target_height",
    CONTROL_CENTER_COL,
    "control_radius",
    "control_height",
    "target_mean_likelihood",
    "control_mean_likelihood",
    IMAGE_SHAPE_COL,
]


def parse_triplet(value, name):
    if pd.isna(value):
        raise ValueError(f"Encountered empty {name} value.")

    parsed = ast.literal_eval(value.strip()) if isinstance(value, str) else value

    if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
        raise ValueError(f"Invalid {name} format: {value}. Expected [z,y,x].")

    z, y, x = parsed
    return float(z), float(y), float(x)


def parse_center(value):
    return parse_triplet(value, "center")


def parse_image_shape(value):
    z, y, x = parse_triplet(value, "image_shape")
    return int(z), int(y), int(x)


def unique_or_range(values):
    values = [v for v in values if pd.notna(v)]
    unique_values = sorted(set(values))

    if not unique_values:
        return "unknown"

    if len(unique_values) == 1:
        return str(unique_values[0])

    return f"{min(unique_values)}–{max(unique_values)}"


def load_and_prepare(csv_path: Path):
    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
            + "\n\nColumns containing commas must be quoted, for example:\n"
            + '"[7,147,40]" or "[13,512,512]"'
        )

    target_centers = df[TARGET_CENTER_COL].apply(parse_center)
    control_centers = df[CONTROL_CENTER_COL].apply(parse_center)
    image_shapes = df[IMAGE_SHAPE_COL].apply(parse_image_shape)

    df["target_z"] = target_centers.apply(lambda p: p[0])
    df["target_y"] = target_centers.apply(lambda p: p[1])
    df["target_x"] = target_centers.apply(lambda p: p[2])

    df["control_z"] = control_centers.apply(lambda p: p[0])
    df["control_y"] = control_centers.apply(lambda p: p[1])
    df["control_x"] = control_centers.apply(lambda p: p[2])

    df["image_shape_z"] = image_shapes.apply(lambda p: p[0])
    df["image_shape_y"] = image_shapes.apply(lambda p: p[1])
    df["image_shape_x"] = image_shapes.apply(lambda p: p[2])

    numeric_columns = [
        "time",
        "target_mean",
        "control_mean",
        "target_radius",
        "target_height",
        "control_radius",
        "control_height",
        "target_mean_likelihood",
        "control_mean_likelihood",
        "target_z",
        "target_y",
        "target_x",
        "control_z",
        "control_y",
        "control_x",
        "image_shape_z",
        "image_shape_y",
        "image_shape_x",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["target_mean_likelihood"] = df["target_mean_likelihood"].clip(0, 100)
    df["control_mean_likelihood"] = df["control_mean_likelihood"].clip(0, 100)

    unique_shapes = (
        df[["image_shape_z", "image_shape_y", "image_shape_x"]]
        .drop_duplicates()
        .to_records(index=False)
        .tolist()
    )

    if len(unique_shapes) != 1:
        raise ValueError(
            "Multiple image shapes were found in the CSV. "
            "This script expects one common image volume shape for all time points."
        )

    image_shape_z, image_shape_y, image_shape_x = unique_shapes[0]

    metadata = {
        "target_radius": unique_or_range(df["target_radius"]),
        "target_height": unique_or_range(df["target_height"]),
        "control_radius": unique_or_range(df["control_radius"]),
        "control_height": unique_or_range(df["control_height"]),
        "image_shape_z": int(image_shape_z),
        "image_shape_y": int(image_shape_y),
        "image_shape_x": int(image_shape_x),
        "image_shape_text": f"[z={int(image_shape_z)}, y={int(image_shape_y)}, x={int(image_shape_x)}]",
    }

    keep_columns = [
        "time",
        "target_mean",
        "control_mean",
        "target_mean_likelihood",
        "control_mean_likelihood",
        "target_z",
        "target_y",
        "target_x",
        "control_z",
        "control_y",
        "control_x",
        "target_radius",
        "target_height",
        "control_radius",
        "control_height",
    ]

    return df[keep_columns].to_dict(orient="records"), metadata


def make_html(data, metadata):
    data_json = json.dumps(data, indent=2)
    metadata_json = json.dumps(metadata, indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>FRAP D3 + 3D Volume Trajectory Visualization</title>

  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

  <style>
    body {{
      margin: 24px;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #fafafa;
      color: #222;
    }}

    h1 {{
      margin-bottom: 6px;
    }}

    .subtitle {{
      color: #666;
      margin-bottom: 20px;
    }}

    .panel {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 24px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }}

    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 18px;
      margin-top: 10px;
    }}

    label {{
      cursor: pointer;
      user-select: none;
      font-size: 14px;
    }}

    .legend-color {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      margin-right: 5px;
      vertical-align: -1px;
    }}

    svg {{
      width: 100%;
      height: auto;
      overflow: visible;
    }}

    .axis path,
    .axis line {{
      stroke: #999;
    }}

    .axis text {{
      fill: #555;
      font-size: 12px;
    }}

    .grid line {{
      stroke: #e5e5e5;
    }}

    .grid path {{
      display: none;
    }}

    .line {{
      fill: none;
      stroke-width: 2.5;
    }}

    .point {{
      stroke: white;
      stroke-width: 1.2;
    }}

    .chart-title {{
      font-weight: 700;
      font-size: 16px;
    }}

    .axis-label {{
      fill: #444;
      font-size: 13px;
    }}

    .plot3d {{
      width: 100%;
      height: 720px;
    }}

    .tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(20, 20, 20, 0.92);
      color: white;
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 13px;
      opacity: 0;
      z-index: 1000;
      max-width: 300px;
      line-height: 1.35;
    }}
  </style>
</head>

<body>
  <h1>FRAP Time Series and 3D Center Tracking</h1>
  <div class="subtitle">
    Generated from CSV. Target and control trajectories are
    shown together inside the full image volume.
  </div>

  <div class="panel">
    <strong>Visibility controls</strong>

    <div class="controls">
      <label>
        <input type="checkbox" class="toggle" data-series="target_mean" checked />
        <span class="legend-color" style="background:#1f77b4"></span>
        target_mean
      </label>

      <label>
        <input type="checkbox" class="toggle" data-series="control_mean" checked />
        <span class="legend-color" style="background:#ff7f0e"></span>
        control_mean
      </label>

      <label>
        <input type="checkbox" class="toggle" data-series="target_mean_likelihood" checked />
        <span class="legend-color" style="background:#2ca02c"></span>
        target_mean_likelihood
      </label>

      <label>
        <input type="checkbox" class="toggle" data-series="control_mean_likelihood" checked />
        <span class="legend-color" style="background:#d62728"></span>
        control_mean_likelihood
      </label>

      <label>
        <input type="checkbox" class="toggle" data-series="target_trajectory" checked />
        <span class="legend-color" style="background:#1f77b4"></span>
        target trajectory
      </label>

      <label>
        <input type="checkbox" class="toggle" data-series="control_trajectory" checked />
        <span class="legend-color" style="background:#ff7f0e"></span>
        control trajectory
      </label>
    </div>
  </div>

  <div class="panel">
    <svg id="meansChart"></svg>
  </div>

  <div class="panel">
    <svg id="likelihoodChart"></svg>
  </div>

  <div class="panel">
    <div id="trajectory3dChart" class="plot3d"></div>
  </div>

  <div id="tooltip" class="tooltip"></div>

  <script>
    const data = {data_json};
    const metadata = {metadata_json};

    const colors = {{
      target_mean: "#1f77b4",
      control_mean: "#ff7f0e",
      target_mean_likelihood: "#2ca02c",
      control_mean_likelihood: "#d62728",
      target_trajectory: "#1f77b4",
      control_trajectory: "#ff7f0e"
    }};

    const tooltip = d3.select("#tooltip");

    function isVisible(seriesName) {{
      const checkbox = document.querySelector(`.toggle[data-series="${{seriesName}}"]`);
      return checkbox && checkbox.checked;
    }}

    function showTooltip(event, html) {{
      tooltip
        .style("opacity", 1)
        .html(html)
        .style("left", `${{event.clientX + 12}}px`)
        .style("top", `${{event.clientY + 12}}px`);
    }}

    function hideTooltip() {{
      tooltip.style("opacity", 0);
    }}

    function drawTimeSeriesChart(config) {{
      const {{
        svgId,
        title,
        yLabel,
        yDomain,
        series
      }} = config;

      const svg = d3.select(svgId);
      svg.selectAll("*").remove();

      const width = 1000;
      const height = 360;
      const margin = {{ top: 48, right: 32, bottom: 56, left: 76 }};

      svg
        .attr("viewBox", `0 0 ${{width}} ${{height}}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;

      const g = svg
        .append("g")
        .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

      const xExtent = d3.extent(data, d => d.time);
      const x = d3.scaleLinear()
        .domain(xExtent[0] === xExtent[1] ? [xExtent[0] - 1, xExtent[1] + 1] : xExtent)
        .nice()
        .range([0, innerWidth]);

      const visibleValues = [];

      for (const s of series) {{
        if (!isVisible(s.key)) continue;

        for (const d of data) {{
          const v = d[s.key];
          if (Number.isFinite(v)) visibleValues.push(v);
        }}
      }}

      let finalYDomain = yDomain;

      if (!finalYDomain) {{
        if (visibleValues.length === 0) {{
          finalYDomain = [0, 1];
        }} else {{
          const extent = d3.extent(visibleValues);
          finalYDomain = extent[0] === extent[1]
            ? [extent[0] - 1, extent[1] + 1]
            : extent;
        }}
      }}

      const y = d3.scaleLinear()
        .domain(finalYDomain)
        .nice()
        .range([innerHeight, 0]);

      g.append("g")
        .attr("class", "grid")
        .call(
          d3.axisLeft(y)
            .tickSize(-innerWidth)
            .tickFormat("")
        );

      g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${{innerHeight}})`)
        .call(d3.axisBottom(x));

      g.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y));

      svg.append("text")
        .attr("class", "chart-title")
        .attr("x", margin.left)
        .attr("y", 25)
        .text(title);

      svg.append("text")
        .attr("class", "axis-label")
        .attr("x", margin.left + innerWidth / 2)
        .attr("y", height - 14)
        .attr("text-anchor", "middle")
        .text("time");

      svg.append("text")
        .attr("class", "axis-label")
        .attr("transform", "rotate(-90)")
        .attr("x", -(margin.top + innerHeight / 2))
        .attr("y", 22)
        .attr("text-anchor", "middle")
        .text(yLabel);

      const lineGenerator = d3.line()
        .x(d => x(d.time))
        .y(d => y(d.value))
        .defined(d => Number.isFinite(d.value));

      for (const s of series) {{
        if (!isVisible(s.key)) continue;

        const values = data.map(d => ({{
          time: d.time,
          value: d[s.key],
          key: s.key,
          label: s.label
        }}));

        g.append("path")
          .datum(values)
          .attr("class", "line")
          .attr("stroke", s.color)
          .attr("d", lineGenerator);

        g.selectAll(`circle.${{s.key}}`)
          .data(values.filter(d => Number.isFinite(d.value)))
          .join("circle")
          .attr("class", `point ${{s.key}}`)
          .attr("cx", d => x(d.time))
          .attr("cy", d => y(d.value))
          .attr("r", 4)
          .attr("fill", s.color)
          .on("mousemove", (event, d) => {{
            showTooltip(
              event,
              `<strong>${{d.label}}</strong><br>` +
              `time: ${{d.time}}<br>` +
              `value: ${{d.value.toFixed(4)}}`
            );
          }})
          .on("mouseleave", hideTooltip);
      }}
    }}

    function makeVolumeSurfaceTrace() {{
      const xMax = metadata.image_shape_x;
      const yMax = metadata.image_shape_y;
      const zMax = metadata.image_shape_z;

      return {{
        type: "mesh3d",
        name: "image volume",
        x: [0, xMax, xMax, 0, 0, xMax, xMax, 0],
        y: [0, 0, yMax, yMax, 0, 0, yMax, yMax],
        z: [0, 0, 0, 0, zMax, zMax, zMax, zMax],
        i: [0, 0, 0, 4, 4, 4, 0, 1, 2, 3, 0, 1],
        j: [1, 2, 3, 5, 6, 7, 1, 2, 3, 0, 4, 5],
        k: [2, 3, 0, 6, 7, 4, 5, 6, 7, 4, 5, 6],
        opacity: 0.045,
        color: "rgb(120,120,120)",
        hoverinfo: "skip",
        showscale: false,
        showlegend: false
      }};
    }}

    function makeVolumeEdgesTrace() {{
      const xMax = metadata.image_shape_x;
      const yMax = metadata.image_shape_y;
      const zMax = metadata.image_shape_z;

      const edges = [
        [[0, 0, 0], [xMax, 0, 0]],
        [[xMax, 0, 0], [xMax, yMax, 0]],
        [[xMax, yMax, 0], [0, yMax, 0]],
        [[0, yMax, 0], [0, 0, 0]],

        [[0, 0, zMax], [xMax, 0, zMax]],
        [[xMax, 0, zMax], [xMax, yMax, zMax]],
        [[xMax, yMax, zMax], [0, yMax, zMax]],
        [[0, yMax, zMax], [0, 0, zMax]],

        [[0, 0, 0], [0, 0, zMax]],
        [[xMax, 0, 0], [xMax, 0, zMax]],
        [[xMax, yMax, 0], [xMax, yMax, zMax]],
        [[0, yMax, 0], [0, yMax, zMax]]
      ];

      const xs = [];
      const ys = [];
      const zs = [];

      for (const edge of edges) {{
        xs.push(edge[0][0], edge[1][0], null);
        ys.push(edge[0][1], edge[1][1], null);
        zs.push(edge[0][2], edge[1][2], null);
      }}

      return {{
        type: "scatter3d",
        mode: "lines",
        name: "volume bounds",
        x: xs,
        y: ys,
        z: zs,
        line: {{
          color: "rgba(80,80,80,0.55)",
          width: 3
        }},
        hoverinfo: "skip",
        showlegend: false
      }};
    }}

    function makeTrajectoryTrace(prefix, name, color) {{
      return {{
        type: "scatter3d",
        mode: "lines+markers",
        name: name,
        x: data.map(d => d[`${{prefix}}_x`]),
        y: data.map(d => d[`${{prefix}}_y`]),
        z: data.map(d => d[`${{prefix}}_z`]),
        text: data.map(d =>
          `series: ${{name}}<br>` +
          `time: ${{d.time}}<br>` +
          `x: ${{d[`${{prefix}}_x`]}}<br>` +
          `y: ${{d[`${{prefix}}_y`]}}<br>` +
          `z/depth: ${{d[`${{prefix}}_z`]}}`
        ),
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{
          size: 5,
          color: color
        }},
        line: {{
          width: 6,
          color: color
        }},
        visible: isVisible(`${{prefix}}_trajectory`) ? true : "legendonly"
      }};
    }}

    function makeStartEndMarkers(prefix, name, color) {{
      if (data.length === 0) return null;

      const first = data[0];
      const last = data[data.length - 1];

      return {{
        type: "scatter3d",
        mode: "markers+text",
        name: `${{name}} start/end`,
        x: [first[`${{prefix}}_x`], last[`${{prefix}}_x`]],
        y: [first[`${{prefix}}_y`], last[`${{prefix}}_y`]],
        z: [first[`${{prefix}}_z`], last[`${{prefix}}_z`]],
        text: [`${{name}} start`, `${{name}} end`],
        textposition: "top center",
        marker: {{
          size: [8, 8],
          color: [color, color],
          symbol: ["circle", "diamond"]
        }},
        hovertemplate: "%{{text}}<extra></extra>",
        showlegend: false,
        visible: isVisible(`${{prefix}}_trajectory`) ? true : "legendonly"
      }};
    }}

    function drawCombined3DTrajectory() {{
      const traces = [
        makeVolumeSurfaceTrace(),
        makeVolumeEdgesTrace(),
        makeTrajectoryTrace("target", "target trajectory", colors.target_trajectory),
        makeTrajectoryTrace("control", "control trajectory", colors.control_trajectory),
        makeStartEndMarkers("target", "target", colors.target_trajectory),
        makeStartEndMarkers("control", "control", colors.control_trajectory)
      ].filter(Boolean);

      const layout = {{
        title: {{
          text:
            `Target radius: ${{metadata.target_radius}}, height: ${{metadata.target_height}} | ` +
            `Control radius: ${{metadata.control_radius}}, height: ${{metadata.control_height}} | `+
            `Volume: ${{metadata.image_shape_text}}`,
          x: 0.02
        }},
        margin: {{ l: 0, r: 0, t: 72, b: 0 }},
        paper_bgcolor: "white",
        scene: {{
          xaxis: {{
            title: "x",
            range: [0, metadata.image_shape_x],
            zeroline: false
          }},
          yaxis: {{
            title: "y",
            range: [0, metadata.image_shape_y],
            zeroline: false
          }},
          zaxis: {{
            title: "z / depth",
            range: [0, metadata.image_shape_z],
            zeroline: false
          }},
          aspectmode: "data",
          camera: {{
            eye: {{ x: 1.65, y: 1.65, z: 1.05 }}
          }}
        }},
        legend: {{
          x: 0.02,
          y: 0.98,
          bgcolor: "rgba(255,255,255,0.75)"
        }}
      }};

      const plotConfig = {{
        responsive: true,
        displaylogo: false
      }};

      Plotly.react("trajectory3dChart", traces, layout, plotConfig);
    }}

    function redrawAll() {{
      drawTimeSeriesChart({{
        svgId: "#meansChart",
        title: "Mean intensity over time",
        yLabel: "mean intensity",
        series: [
          {{
            key: "target_mean",
            label: "target_mean",
            color: colors.target_mean
          }},
          {{
            key: "control_mean",
            label: "control_mean",
            color: colors.control_mean
          }}
        ]
      }});

      drawTimeSeriesChart({{
        svgId: "#likelihoodChart",
        title: "Mean likelihood over time",
        yLabel: "likelihood [%]",
        yDomain: [0, 100],
        series: [
          {{
            key: "target_mean_likelihood",
            label: "target_mean_likelihood",
            color: colors.target_mean_likelihood
          }},
          {{
            key: "control_mean_likelihood",
            label: "control_mean_likelihood",
            color: colors.control_mean_likelihood
          }}
        ]
      }});

      drawCombined3DTrajectory();
    }}

    document.querySelectorAll(".toggle").forEach(cb => {{
      cb.addEventListener("change", redrawAll);
    }});

    redrawAll();
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate a D3.js HTML visualization with "
        "target/control 3D FRAP trajectories in one shared volume."
    )

    parser.add_argument("csv", type=Path, help="Input CSV file.")

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("frap_visualization.html"),
        help="Output HTML file. Default: frap_visualization.html",
    )

    args = parser.parse_args()

    data, metadata = load_and_prepare(args.csv)
    html = make_html(data, metadata)

    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote visualization to: {args.output}")


if __name__ == "__main__":
    main()

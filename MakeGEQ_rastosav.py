"""
Created 30. Juli 2026 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

from math import sin, pi
from pathlib import Path
import os

MAP = {
    "NTHREADS": lambda h: os.cpu_count(),

    "VERZERR": "AUTO",

    "GEOMETRY": lambda h:
        "TRANSMISSION" if "Transmission" in h.get("*HW_SAMPLE_NAME", "")
        else "CAPILLARY" if (
            "Capillary" in h.get("*HW_SAMPLE_NAME", "")
            or "PB-" in h.get("*MEAS_COND_OPT_ATTR", "")
        )
        else "REFLEXION",

    "R": lambda h:
        h.get("*MEAS_COND_COUNTER_DISTANCE")
        or h.get("*HW_GONIOMETER_RADIUS-2"),

    "FocusH": lambda h:
        float(
            h["*HW_XG_FOCUS"]
            .split(" x ")[1]
            .replace("mm", "")
        ),

    "FocusW": lambda h:
        float(
            h["*HW_XG_FOCUS"]
            .split(" x ")[0]
            .replace("mm", "")
        ) / 10,

    "HSlitR": lambda h:
        (
            float(h["*HW_GONIOMETER_RADIUS-3"])
            - float(h["*HW_GONIOMETER_RADIUS-2"])
            if float(h["*HW_GONIOMETER_RADIUS-3"]) > 0
            else float(h["*HW_GONIOMETER_RADIUS-2"])
            - float(h["*HW_GONIOMETER_RADIUS-1"])
        ),

    "irr": lambda h:
        ""
        if h.get("*MEAS_COND_OPT_ATTR") == "CB"
        else (h.get("*MEAS_COND_IRRADIATION_WIDTH") or "20"),

    "div": lambda h:
        (
            axis_position(h, "IncidentSlitBox")
            .replace("deg", "")
            if "MiniFlex" in h.get("*HW_GONIOMETER_NAME", "")
            else ""
            if h.get("*MEAS_COND_OPT_ATTR") == "CB"
            else "atan(11/(R-HSlitR)/2)*360/pi"
            if (h.get("*MEAS_COND_IRRADIATION_WIDTH") or "")
            else "0.625"
        ),

    "HSlitW0": lambda h:
        "%(2*(R-HSlitR)*irr*sin(pi*zweiTheta/360))/(2*R+irr*cos(pi*zweiTheta/360))"
        if h.get("*MEAS_COND_OPT_ATTR") == "CB"
        else
        "(2*(R-HSlitR)*irr*sin(pi*zweiTheta/360))/(2*R+irr*cos(pi*zweiTheta/360))",

    "HSlitW1": lambda h:
        "%2*tan(div*pi/360)*(R-HSlitR)"
        if h.get("*MEAS_COND_OPT_ATTR") == "CB"
        else
        "2*tan(div*pi/360)*(R-HSlitR)",

    "HSlitW": lambda h:
        (
            axis_position(h, "IncidentSlitBox").replace("mm", "")
            if h.get("*MEAS_COND_OPT_ATTR") == "CB"
            else "ifthenelse(lt(HSlitW0,HSlitW1),HSlitW0,HSlitW1)"
        ),

    "RoundSlitR": lambda h:
        float(h["*HW_GONIOMETER_RADIUS-3"])
        - float(h["*HW_GONIOMETER_RADIUS-2"])
        - 85.0
        if "Collimator" in axis_position(h, "IncidentAxdSlit")
        else "",

    "RoundSlitD": lambda h:
        float(
            axis_position(h, "IncidentAxdSlit")
            .replace("Collimator", "")
            .replace("mm", "")
        )
        if "Collimator" in axis_position(h, "IncidentAxdSlit")
        else "",

    "TSlitR": lambda h:
        (
            float(h["*HW_GONIOMETER_RADIUS-2"])
            - float(h["*HW_GONIOMETER_RADIUS-0"])
            if "MiniFlex" in h.get("*HW_GONIOMETER_NAME", "")
            else float(h["*HW_GONIOMETER_RADIUS-4"])
            if float(h.get("*HW_GONIOMETER_RADIUS-4", 0))
            else ""
        ),

    "TSlitH": lambda h:
        float(
            axis_position(h, "IncidentAxdSlit")
            .replace("mm", "")
        )
        if "mm" in axis_position(h, "IncidentAxdSlit")
        else "",


    "PColl": lambda h:
        (
            float(
                axis_position(h, "IncidentSollerSlit")
                .split("_")[-1]
                .replace("deg", "")
            ) * pi / 180
        )
        if "deg" in axis_position(h, "IncidentSollerSlit")
        else "",

    "VSlitR": lambda h:
        (
            float(h["*HW_GONIOMETER_RADIUS-3"])
            - float(h["*HW_GONIOMETER_RADIUS-2"])
            if float(h["*HW_GONIOMETER_RADIUS-3"]) > 0
            else float(h["*HW_GONIOMETER_RADIUS-2"])
            - float(h["*HW_GONIOMETER_RADIUS-1"])
        ),

    "VSlitH": lambda h:
        float(
            axis_position(h, "IncidentAxdSlit")
            .replace("mm", "")
        ),

    "SSlitR": lambda h:
        (
            float(h["*HW_GONIOMETER_RADIUS-4"])
            if float(h.get("*HW_GONIOMETER_RADIUS-4", 0))
            else 103.4
        ),

    "SSlitW": lambda h:
        next(
            float(v.replace("mm", ""))
            for v in axis_position(h, "ReceivingSlitBox1", return_all=True)
            if v not in ("Open", "")
        ),

    "SColl": lambda h:
        (
            float(
                axis_position(h, "ReceivingSollerSlit")
                .split("_")[-1]
                .replace("deg", "")
            ) * pi / 180
        )
        if (
            "deg" in axis_position(h, "ReceivingSollerSlit")
            and "PSA" not in axis_position(h, "ReceivingSollerSlit")
        )
        else "",

    "SCollA": lambda h:
        (
            float(
                axis_position(h, "ReceivingSollerSlit")
                .split("_")[-1]
                .replace("deg", "")
            ) * pi / 180
        )
        if "PSA" in axis_position(h, "ReceivingSollerSlit")
        else "",

    "DetW": lambda h:
        0.1
        if "MiniFlex" in h.get("*HW_GONIOMETER_NAME", "")
        else h["*MEAS_COND_COUNTER_PITCH_X"],

    "DetH": lambda h:
        (
            13.0
            if "MiniFlex" in h.get("*HW_GONIOMETER_NAME", "")
            else float(h["*MEAS_COND_COUNTER_PITCH_Y"])
            * float(h["*MEAS_COND_COUNTER_VALIDWIDTH_Y"])
        ),

    "DetArrayW": lambda h:
        (
            20.0
            if "MiniFlex" in h.get("*HW_GONIOMETER_NAME", "")
            else float(h["*MEAS_COND_COUNTER_PITCH_X"])
            * float(h["*MEAS_COND_COUNTER_VALIDWIDTH_X"])
        ),

    "GSUM": "Y",

    "WMIN": "*MEAS_SCAN_START",
    "WMAX": "*MEAS_SCAN_STOP",
    "WSTEP": "2*sin(pi*zweiTheta/180)",
    "SAVE": "N",
    "pi": "2*acos(0)",

    # ------------------------------------------------------------------
    # TODO: future raytracing / instrument keywords
    #
    # Optional BGMN variables not yet mapped
    # ---------------------------------------
    # TubeTails
    # PCollA
    # SamplD
    # DeltaOmega
    # AirScat
    # SamplW
    # SamplH
    # MonR
    # MonH
    # EPSG
    # FocusS
    # FocusA
    # GEQ
    # D
    # T
    # STANDARDPAR
    # VAL[x]
    #
    # Project-specific placeholders / open questions
    # ----------------------------------------------
    # RSoll
    # VFDIV
    # RSollA
    # FDIV
    #
    # PSD geometry-specific terms
    # variable divergence slit support
    # variable anti-scatter slit support
    # detector point-spread / strip-integration corrections
    #
    # Also convert any remaining
    # *MEAS_COND_AXIS_POSITION-N
    # references to axis_position(h, "<name>")
    # once all physical mappings are finalized.
    # ------------------------------------------------------------------
}


def axis_position(h, axis_name, return_all=False):
    values = []

    for k, v in h.items():
        if (
            k.startswith("*MEAS_COND_AXIS_NAME_INTERNAL-")
            and v == axis_name
        ):
            idx = k.rsplit("-", 1)[1]
            values.append(
                h.get(f"*MEAS_COND_AXIS_POSITION-{idx}")
            )

    if return_all:
        return values

    if not values:
        raise KeyError(
            f"Axis '{axis_name}' not found"
        )

    return values[0]

def parse_ras_header(filename):
    h = {}
    with open(filename, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "*RAS_HEADER_END":
                break
            if not line.startswith("*"):
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            key, value = parts
            h[key] = value.strip().strip('"')
    return h

def make_sav_string(h):
    lines = []

    values = {}

    for key, value in MAP.items():
        if key == "VERZERR":
            value = f"{Path.cwd().name}.ger"
        elif callable(value):
            value = value(h)
        elif isinstance(value, str) and value.startswith("*"):
            value = h.get(value, "")

        values[key] = value

        if value != "":
            lines.append(f"{key}={value}")

    wmin = float(values["WMIN"])
    wmax = float(values["WMAX"])

    zwei_theta = [round(wmin)]

    while zwei_theta[-1] < wmax:
        step = 1 + 14 * sin(zwei_theta[-1] * pi / 180)
        zt = round(zwei_theta[-1] + step)

        if zt <= zwei_theta[-1]:
            zt = zwei_theta[-1] + 1

        zwei_theta.append(zt)

    while zwei_theta[-1] > wmax:
        zwei_theta.pop()

    if zwei_theta[-1] != round(wmax):
        zwei_theta.append(round(wmax))

    for i, zt in enumerate(zwei_theta, start=1):
        lines.append(f"zweiTheta[{i}]={zt}")

    return "\n".join(lines)

ras_files = sorted(Path(".").glob("*.ras"))

generated = []

for ras in ras_files:
    h = parse_ras_header(ras)
    h["_rasfile"] = str(ras)

    generated.append(
        (ras, make_sav_string(h))
    )

if generated:
    first = generated[0][1]

    if any(
        sav_text != first
        for _, sav_text in generated[1:]
    ):
        answer = input(
            "Generated SAV files differ.\n"
            "Create individual SAV files instead? [y/N] "
        )

        if answer.lower().startswith("y"):
            for ras, sav_text in generated:
                ras.with_name(f"MakeGEQ_{ras.stem}").with_suffix(".sav").write_text(
                    sav_text,
                    encoding="utf-8",
                )
                print(
                    f"created {ras.with_suffix('.sav').name}"
                )

            raise SystemExit

        raise RuntimeError(
            "generated SAV files are not identical"
        )

    folder = Path(".").resolve().name

    Path('MakeGEQ_'+f"{folder}.sav").write_text(
        first,
        encoding="utf-8",
    )

    print(f"created {folder}.sav")

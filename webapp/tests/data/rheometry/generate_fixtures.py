"""Write the synthetic rheometer files the rheometry tests read.

    python generate_fixtures.py        # (re)writes every fixture next to this script

These are NOT real instrument exports. They follow the layouts TRIOS and RheoCompass are
commonly documented to write (a TRIOS text export: key/value header, then `[step]` sections with
a units row; a RheoCompass CSV: Project/Test/Result lines, then "Interval and data points" /
"Interval data" tables with bracketed units, semicolons and decimal commas). Replace or add to
them with real exports when those are available; the tests only assume what is asserted.

The data are physical but simple: a single Maxwell mode shifted with a WLF law for the
frequency sweeps, a softening gel for the amplitude sweep, a Carreau-like flow curve and an
exponential relaxation, so derived values can be checked by hand.
"""

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
G0 = 1e5  # Pa
TAU = 1.0  # s at the reference temperature
T_REF = 150.0
C1, C2 = 8.86, 101.6


def a_t(temperature: float) -> float:
    return 10 ** (-C1 * (temperature - T_REF) / (C2 + temperature - T_REF))


def maxwell(omega: float, temperature: float) -> tuple[float, float]:
    x = omega * TAU * a_t(temperature)
    return G0 * x * x / (1 + x * x), G0 * x / (1 + x * x)


OMEGA = [0.1 * 10 ** (i / 4) for i in range(13)]  # 0.1 .. 100 rad/s


def trios() -> str:
    lines = [
        "Filename\tPS_sweeps.tri",
        "Instrument serial number\t5343-0001",
        "Instrument name\tDHR-2",
        "Procedure name\tFrequency sweeps 150-170 C",
        "",
    ]
    for n, temperature in enumerate((150.0, 160.0, 170.0), start=1):
        lines += [
            "[step]",
            f"Frequency sweep ({temperature:.0f} °C) - {n}",
            "Angular frequency\tStorage modulus\tLoss modulus\tTan(delta)\tTemperature\tStep time",
            "rad/s\tPa\tPa\t\t°C\ts",
        ]
        for i, w in enumerate(OMEGA):
            g1, g2 = maxwell(w, temperature)
            jitter = 0.02 * (i % 3)
            lines.append(
                f"{w:.6g}\t{g1:.6g}\t{g2:.6g}\t{g2 / g1:.6g}\t{temperature + jitter:.3f}\t{10 * i}"
            )
        lines.append("")
    return "\n".join(lines)


STRAIN = [0.01 * 10 ** (i / 3) for i in range(10)]  # 0.01 .. 10 %


def anton_paar() -> str:
    lines = [
        "Project:;Gel study",
        "Test:;Amplitude sweep 25 °C",
        "Result:;Gel A",
        "",
        f"Interval and data points:;1;{len(STRAIN)}",
        "Interval data:;Point No.;Time;Shear Strain;Storage Modulus;Loss Modulus;Loss Factor;"
        "Temperature",
        ";;[s];[%];[kPa];[kPa];[1];[°C]",
    ]
    for i, s in enumerate(STRAIN):
        g1 = 12.5 / (1 + s / 50)
        g2 = 1.1 + 0.01 * s
        row = f";{i + 1};{10 * (i + 1)};{s:.4g};{g1:.4f};{g2:.4f};{g2 / g1:.4f};25.0"
        lines.append(row.replace(".", ","))
    return "\n".join(lines)


SHEAR_RATE = [0.01 * 10 ** (i / 2) for i in range(9)]  # 0.01 .. 100 1/s


def flow_curve() -> str:
    lines = ["Sample,PEO 2 %", "Shear rate (1/s),Viscosity (mPa.s),Shear stress (Pa)"]
    for rate in SHEAR_RATE:
        eta = 1000 / (1 + rate) ** 0.6  # mPa·s
        lines.append(f"{rate:.5g},{eta:.5g},{rate * eta / 1000:.5g}")
    return "\n".join(lines)


def relaxation() -> str:
    lines = ["# Stress relaxation, entered by hand", "# Temperature: 25 C", "time   G(t)"]
    for i in range(8):
        t = 0.01 * 10 ** (i / 3)
        lines.append(f"{t:.4e}   {1e4 * math.exp(-t):.4e}")
    return "\n".join(lines)


def workbook(path: Path) -> None:
    import openpyxl

    book = openpyxl.Workbook()
    first = book.active
    first.title = "Sample A"
    first.append(["Frequency sweep of sample A"])
    first.append([])
    first.append(["ω (rad/s)", "G' (Pa)", "G'' (Pa)"])
    for i in range(8):
        w = 0.1 * 10 ** (i / 2)
        first.append([w, 1e3 * w, 5e2 * w**0.5])
    second = book.create_sheet("Sample B")
    second.append(["Angular frequency", "Storage modulus", "Loss modulus"])
    second.append(["rad/s", "MPa", "MPa"])
    for i in range(8):
        w = 0.1 * 10 ** (i / 2)
        second.append([w, 0.001 * w, 0.0005 * w])
    book.save(path)


def main() -> None:
    (HERE / "trios_frequency_sweeps.txt").write_text(trios(), encoding="utf-8")
    (HERE / "anton_paar_amplitude_sweep.csv").write_text(anton_paar(), encoding="cp1252")
    (HERE / "flow_curve.csv").write_text(flow_curve(), encoding="utf-8")
    (HERE / "relaxation.dat").write_text(relaxation(), encoding="utf-8")
    (HERE / "not_a_table.txt").write_text(
        "These are notes about the sample.\nNo numbers here.\n", encoding="utf-8"
    )
    workbook(HERE / "two_samples.xlsx")


if __name__ == "__main__":
    main()

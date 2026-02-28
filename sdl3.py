# Save as sdl3.py
# Run GUI:
#   python sdl3.py gui
# CLI still works:
#   python sdl3.py point c2p --x 3 --y 4 --units deg
#   python sdl3.py curve --r-expr "1+cos(theta)" --units rad --save-prefix cardioid

import math
import csv
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

# matplotlib optional
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# tkinter GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# embed matplotlib in tkinter (optional, only works if matplotlib is installed)
HAS_MPL_EMBED = False
try:
    if plt is not None:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        HAS_MPL_EMBED = True
except Exception:
    HAS_MPL_EMBED = False


@dataclass
class PointXY:
    x: float
    y: float


@dataclass
class PointPolar:
    r: float
    theta: float  # radians


# ---------- helpers ----------
def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def normalize_angle(theta: float, mode: str = "0_to_2pi") -> float:
    two_pi = 2.0 * math.pi
    if mode == "0_to_2pi":
        return theta % two_pi
    if mode == "negpi_to_pi":
        return (theta + math.pi) % two_pi - math.pi
    return theta  # none


def normalize_negative_r(p: PointPolar) -> PointPolar:
    if p.r >= 0:
        return p
    return PointPolar(-p.r, p.theta + math.pi)


# ---------- conversions ----------
def polar_to_cart(p: PointPolar) -> PointXY:
    return PointXY(p.r * math.cos(p.theta), p.r * math.sin(p.theta))


def cart_to_polar(p: PointXY, angle_mode: str = "0_to_2pi") -> PointPolar:
    r = math.hypot(p.x, p.y)
    theta = 0.0 if r == 0 else math.atan2(p.y, p.x)
    theta = normalize_angle(theta, angle_mode)
    return PointPolar(r, theta)


def round_trip_error_xy(p: PointXY, angle_mode: str = "0_to_2pi") -> float:
    pol = cart_to_polar(p, angle_mode=angle_mode)
    xy2 = polar_to_cart(pol)
    return math.hypot(p.x - xy2.x, p.y - xy2.y)


def round_trip_error_polar(
    p: PointPolar, angle_mode: str = "0_to_2pi", fix_neg_r: bool = True
) -> Tuple[float, float]:
    if fix_neg_r:
        p = normalize_negative_r(p)
    xy = polar_to_cart(p)
    pol2 = cart_to_polar(xy, angle_mode=angle_mode)
    dr = abs(p.r - pol2.r)
    dtheta = abs(normalize_angle(p.theta - pol2.theta, "negpi_to_pi"))
    return dr, dtheta


# ---------- plotting ----------
def ensure_matplotlib():
    if plt is None:
        raise RuntimeError("matplotlib not installed. Install it with: pip install matplotlib")


def plot_points_cart(points: List[PointXY], title: str, save: Optional[str] = None):
    ensure_matplotlib()
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    plt.figure()
    plt.scatter(xs, ys)
    plt.axhline(0)
    plt.axvline(0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_points_polar(points: List[PointPolar], title: str, save: Optional[str] = None):
    ensure_matplotlib()
    thetas = [p.theta for p in points]
    rs = [p.r for p in points]
    plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.scatter(thetas, rs)
    ax.set_title(title)
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# Safe-ish expression eval for curve mode
ALLOWED = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "pi": math.pi,
    "abs": abs,
    "log": math.log,
    "exp": math.exp,
}


def eval_r_expr(expr: str, theta: float) -> float:
    env = dict(ALLOWED)
    env["theta"] = theta
    return float(eval(expr, {"__builtins__": {}}, env))


def plot_curve_polar_and_cart(
    thetas: List[float], rs: List[float], title: str, save_prefix: Optional[str] = None
):
    ensure_matplotlib()

    # polar
    plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.plot(thetas, rs)
    ax.set_title(title + " (polar)")
    if save_prefix:
        plt.savefig(f"{save_prefix}_polar.png", dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    # cartesian
    xy = [polar_to_cart(PointPolar(r, th)) for r, th in zip(rs, thetas)]
    plt.figure()
    plt.plot([p.x for p in xy], [p.y for p in xy])
    plt.axhline(0)
    plt.axvline(0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title + " (cartesian)")
    plt.xlabel("x")
    plt.ylabel("y")
    if save_prefix:
        plt.savefig(f"{save_prefix}_cart.png", dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ---------- CSV ----------
def read_csv_dicts(path: str) -> Tuple[List[str], List[dict]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no headers.")
        headers = [h.strip() for h in reader.fieldnames]
        rows = list(reader)
    return headers, rows


def write_csv_dicts(path: str, fieldnames: List[str], rows: List[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def detect_theta_units(headers_lower: List[str]) -> str:
    if "theta_rad" in headers_lower:
        return "rad"
    if "theta_deg" in headers_lower:
        return "deg"
    if "theta" in headers_lower:
        return "deg"
    return "deg"


# ---------- core operations ----------
def compute_point_p2c(
    r: float, theta_in: float, units: str, fix_neg_r: bool
) -> Tuple[PointPolar, PointXY]:
    th = deg2rad(theta_in) if units == "deg" else theta_in
    pol = PointPolar(r, th)
    if fix_neg_r:
        pol = normalize_negative_r(pol)
    xy = polar_to_cart(pol)
    return pol, xy


def compute_point_c2p(
    x: float, y: float, units: str, angle_mode: str
) -> Tuple[PointXY, PointPolar, float]:
    xy = PointXY(x, y)
    pol = cart_to_polar(xy, angle_mode=angle_mode)
    theta_out = rad2deg(pol.theta) if units == "deg" else pol.theta
    return xy, pol, theta_out


def do_batch(
    input_path: str,
    output_path: str,
    angle_mode: str,
    fix_neg_r: bool,
    plot: bool = False,
    save_prefix: Optional[str] = None,
) -> Tuple[int, str]:
    headers, rows = read_csv_dicts(input_path)
    headers_lower = [h.lower() for h in headers]

    out_rows = []
    xy_points: List[PointXY] = []
    pol_points: List[PointPolar] = []

    if "x" in headers_lower and "y" in headers_lower:
        # Cartesian -> polar
        for row in rows:
            x = float(row[headers[headers_lower.index("x")]])
            y = float(row[headers[headers_lower.index("y")]])
            xy = PointXY(x, y)
            pol = cart_to_polar(xy, angle_mode=angle_mode)
            out = dict(row)
            out["r"] = pol.r
            out["theta_deg"] = rad2deg(pol.theta)
            out["theta_rad"] = pol.theta
            out_rows.append(out)
            xy_points.append(xy)
            pol_points.append(pol)

        fieldnames = list(rows[0].keys()) + ["r", "theta_deg", "theta_rad"]
        write_csv_dicts(output_path, fieldnames, out_rows)

    elif "r" in headers_lower and (
        "theta" in headers_lower or "theta_deg" in headers_lower or "theta_rad" in headers_lower
    ):
        theta_units = detect_theta_units(headers_lower)

        def get_val(key: str) -> str:
            return headers[headers_lower.index(key)]

        for row in rows:
            r = float(row[get_val("r")])
            if theta_units == "rad":
                th = float(row[get_val("theta_rad")])
            elif "theta_deg" in headers_lower:
                th = deg2rad(float(row[get_val("theta_deg")]))
            else:
                th = deg2rad(float(row[get_val("theta")]))

            pol = PointPolar(r, th)
            if fix_neg_r:
                pol = normalize_negative_r(pol)
            xy = polar_to_cart(pol)

            out = dict(row)
            out["x"] = xy.x
            out["y"] = xy.y
            out_rows.append(out)
            xy_points.append(xy)
            pol_points.append(pol)

        fieldnames = list(rows[0].keys()) + ["x", "y"]
        write_csv_dicts(output_path, fieldnames, out_rows)
    else:
        raise ValueError("CSV must have x,y OR r with theta/theta_deg/theta_rad headers.")

    if plot:
        cart_save = f"{save_prefix}_cart.png" if save_prefix else None
        polar_save = f"{save_prefix}_polar.png" if save_prefix else None
        if xy_points:
            plot_points_cart(xy_points, "Batch points (cartesian)", save=cart_save)
        if pol_points:
            plot_points_polar(pol_points, "Batch points (polar)", save=polar_save)

    return len(rows), output_path


def do_curve(
    r_expr: str,
    units: str,
    theta_start: float,
    theta_end: float,
    samples: int,
    angle_mode: str,
    fix_neg_r: bool,
    save_prefix: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    th0 = deg2rad(theta_start) if units == "deg" else theta_start
    th1 = deg2rad(theta_end) if units == "deg" else theta_end

    thetas = [th0 + (th1 - th0) * i / (samples - 1) for i in range(samples)]
    rs = []
    for th in thetas:
        r = eval_r_expr(r_expr, th)
        pol = PointPolar(r, th)
        if fix_neg_r:
            pol = normalize_negative_r(pol)
        rs.append(pol.r)

    if angle_mode != "none":
        thetas = [normalize_angle(th, angle_mode) for th in thetas]

    title = f"r = {r_expr}"
    plot_curve_polar_and_cart(thetas, rs, title, save_prefix=save_prefix)
    return thetas, rs


def do_test(angle_mode: str, fix_neg_r: bool) -> str:
    lines = []
    demo_xy = [PointXY(3, 4), PointXY(-2, 5), PointXY(-3, -3), PointXY(0, 0)]
    lines.append("Round-trip error (x,y -> polar -> x,y):")
    for p in demo_xy:
        err = round_trip_error_xy(p, angle_mode=angle_mode)
        lines.append(f"  ({p.x:.3f}, {p.y:.3f}) error = {err:.6e}")

    demo_pol = [PointPolar(2, deg2rad(30)), PointPolar(-5, deg2rad(45)), PointPolar(0, 0)]
    lines.append("\nRound-trip error (r,theta -> x,y -> r,theta):")
    for p in demo_pol:
        dr, dth = round_trip_error_polar(p, angle_mode=angle_mode, fix_neg_r=fix_neg_r)
        lines.append(
            f"  (r={p.r:.3f}, theta={rad2deg(p.theta):.3f}deg) dr={dr:.6e}, dtheta(rad)={dth:.6e}"
        )
    return "\n".join(lines)


# ---------- GUI ----------
class SDL3Gui(tk.Tk):
    def __init__(self, angle_mode: str, fix_neg_r: bool):
        super().__init__()
        self.title("SDL3 Coordinate Tool (GUI)")
        self.geometry("980x650")
        self.angle_mode = angle_mode
        self.fix_neg_r = fix_neg_r

        # top options
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Angle mode:").pack(side="left")
        self.angle_mode_var = tk.StringVar(value=self.angle_mode)
        ttk.Combobox(top, textvariable=self.angle_mode_var, state="readonly",
                     values=["0_to_2pi", "negpi_to_pi", "none"], width=12).pack(side="left", padx=6)

        self.fix_var = tk.BooleanVar(value=self.fix_neg_r)
        ttk.Checkbutton(top, text="Fix negative r", variable=self.fix_var).pack(side="left", padx=10)

        ttk.Label(top, text=("Plots in-app" if HAS_MPL_EMBED else "Install matplotlib for plots")).pack(side="right")

        # tabs
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_point = ttk.Frame(nb)
        self.tab_batch = ttk.Frame(nb)
        self.tab_curve = ttk.Frame(nb)
        self.tab_test = ttk.Frame(nb)

        nb.add(self.tab_point, text="Point Convert")
        nb.add(self.tab_batch, text="Batch CSV")
        nb.add(self.tab_curve, text="Curve")
        nb.add(self.tab_test, text="Tests")

        self._build_point_tab()
        self._build_batch_tab()
        self._build_curve_tab()
        self._build_test_tab()

    # ----- Point Tab -----
    def _build_point_tab(self):
        f = self.tab_point

        left = ttk.Frame(f)
        left.pack(side="left", fill="y", padx=8, pady=8)

        self.dir_var = tk.StringVar(value="c2p")
        self.units_var = tk.StringVar(value="deg")

        ttk.Label(left, text="Direction").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.dir_var, state="readonly",
                     values=["c2p", "p2c"], width=10).pack(anchor="w", pady=4)

        ttk.Label(left, text="Units").pack(anchor="w", pady=(10, 0))
        ttk.Combobox(left, textvariable=self.units_var, state="readonly",
                     values=["deg", "rad"], width=10).pack(anchor="w", pady=4)

        # inputs
        self.x_var = tk.StringVar(value="3")
        self.y_var = tk.StringVar(value="4")
        self.r_var = tk.StringVar(value="5")
        self.theta_var = tk.StringVar(value="53.130102")

        grid = ttk.Frame(left)
        grid.pack(anchor="w", pady=10)

        ttk.Label(grid, text="x").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.x_var, width=14).grid(row=0, column=1, padx=6, pady=2)

        ttk.Label(grid, text="y").grid(row=1, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.y_var, width=14).grid(row=1, column=1, padx=6, pady=2)

        ttk.Label(grid, text="r").grid(row=2, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.r_var, width=14).grid(row=2, column=1, padx=6, pady=2)

        ttk.Label(grid, text="theta").grid(row=3, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.theta_var, width=14).grid(row=3, column=1, padx=6, pady=2)

        ttk.Button(left, text="Compute", command=self._point_compute).pack(anchor="w", pady=8)

        self.point_out = tk.Text(left, height=8, width=42)
        self.point_out.pack(anchor="w", pady=6)

        # plot area
        right = ttk.Frame(f)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.point_plot_container = right
        self.point_canvas_polar = None
        self.point_canvas_cart = None

        note = ttk.Label(right, text="Plots will appear here (if matplotlib is installed).")
        note.pack(anchor="w")

        self.point_plot_frame = ttk.Frame(right)
        self.point_plot_frame.pack(fill="both", expand=True, pady=8)

    def _clear_frame(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    def _point_compute(self):
        try:
            direction = self.dir_var.get()
            units = self.units_var.get()
            angle_mode = self.angle_mode_var.get()
            fix_neg = self.fix_var.get()

            self.point_out.delete("1.0", "end")

            if direction == "p2c":
                r = float(self.r_var.get())
                theta = float(self.theta_var.get())
                pol, xy = compute_point_p2c(r, theta, units, fix_neg)
                self.point_out.insert("end", f"Input (r, theta) = ({r}, {theta} {units})\n")
                self.point_out.insert("end", f"Normalized polar: r={pol.r:.6f}, theta(rad)={pol.theta:.6f}\n")
                self.point_out.insert("end", f"Output (x, y) = ({xy.x:.6f}, {xy.y:.6f})\n")

                if HAS_MPL_EMBED:
                    self._plot_point_embedded(pol=pol, xy=xy, title="Point p2c")

            else:
                x = float(self.x_var.get())
                y = float(self.y_var.get())
                xy, pol, theta_out = compute_point_c2p(x, y, units, angle_mode)
                self.point_out.insert("end", f"Input (x, y) = ({x}, {y})\n")
                self.point_out.insert("end", f"Output r = {pol.r:.6f}\n")
                self.point_out.insert("end", f"Output theta = {theta_out:.6f} {units}\n")

                if HAS_MPL_EMBED:
                    self._plot_point_embedded(pol=pol, xy=xy, title="Point c2p")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _plot_point_embedded(self, pol: PointPolar, xy: PointXY, title: str):
        self._clear_frame(self.point_plot_frame)

        # Polar figure
        fig1 = Figure(figsize=(4.2, 3.6), dpi=100)
        ax1 = fig1.add_subplot(111, projection="polar")
        ax1.scatter([pol.theta], [pol.r])
        ax1.set_title(title + " (polar)")

        canvas1 = FigureCanvasTkAgg(fig1, master=self.point_plot_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side="left", fill="both", expand=True)

        # Cartesian figure
        fig2 = Figure(figsize=(4.2, 3.6), dpi=100)
        ax2 = fig2.add_subplot(111)
        ax2.scatter([xy.x], [xy.y])
        ax2.axhline(0)
        ax2.axvline(0)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_title(title + " (cartesian)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        canvas2 = FigureCanvasTkAgg(fig2, master=self.point_plot_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side="left", fill="both", expand=True)

    # ----- Batch Tab -----
    def _build_batch_tab(self):
        f = self.tab_batch

        row = ttk.Frame(f)
        row.pack(fill="x", padx=10, pady=10)

        self.batch_in = tk.StringVar(value="points_xy.csv")
        self.batch_out = tk.StringVar(value="converted.csv")
        self.batch_saveprefix = tk.StringVar(value="")

        ttk.Label(row, text="Input CSV:").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.batch_in, width=60).grid(row=0, column=1, padx=6)
        ttk.Button(row, text="Browse", command=self._browse_in).grid(row=0, column=2, padx=6)

        ttk.Label(row, text="Output CSV:").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(row, textvariable=self.batch_out, width=60).grid(row=1, column=1, padx=6, pady=6)
        ttk.Button(row, text="Browse", command=self._browse_out).grid(row=1, column=2, padx=6)

        self.batch_plot_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Plot points", variable=self.batch_plot_var).grid(row=2, column=1, sticky="w", pady=6)

        ttk.Label(row, text="Save plot prefix (optional):").grid(row=3, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.batch_saveprefix, width=30).grid(row=3, column=1, sticky="w", padx=6)

        ttk.Button(row, text="Run Batch Convert", command=self._run_batch).grid(row=4, column=1, sticky="w", pady=10)

        self.batch_log = tk.Text(f, height=12)
        self.batch_log.pack(fill="both", expand=True, padx=10, pady=10)

    def _browse_in(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.batch_in.set(path)

    def _browse_out(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.batch_out.set(path)

    def _run_batch(self):
        try:
            angle_mode = self.angle_mode_var.get()
            fix_neg = self.fix_var.get()
            plot = self.batch_plot_var.get()
            prefix = self.batch_saveprefix.get().strip() or None

            n, outp = do_batch(self.batch_in.get(), self.batch_out.get(), angle_mode, fix_neg, plot=plot, save_prefix=prefix)
            self.batch_log.insert("end", f"Converted {n} rows -> {outp}\n")
            if plot:
                self.batch_log.insert("end", "Plots shown or saved.\n")
        except Exception as e:
            messagebox.showerror("Batch Error", str(e))

    # ----- Curve Tab -----
    def _build_curve_tab(self):
        f = self.tab_curve

        top = ttk.Frame(f)
        top.pack(fill="x", padx=10, pady=10)

        self.expr_var = tk.StringVar(value="1+cos(theta)")
        self.curve_units = tk.StringVar(value="rad")
        self.th0_var = tk.StringVar(value="0")
        self.th1_var = tk.StringVar(value=str(2 * math.pi))
        self.samples_var = tk.StringVar(value="1200")
        self.curve_prefix = tk.StringVar(value="")

        ttk.Label(top, text="r(theta) =").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.expr_var, width=50).grid(row=0, column=1, padx=6, sticky="w")

        ttk.Label(top, text="Units").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(top, textvariable=self.curve_units, state="readonly", values=["rad", "deg"], width=8)\
            .grid(row=1, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(top, text="theta start").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.th0_var, width=14).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(top, text="theta end").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Entry(top, textvariable=self.th1_var, width=14).grid(row=3, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(top, text="samples").grid(row=4, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.samples_var, width=14).grid(row=4, column=1, sticky="w", padx=6)

        ttk.Label(top, text="save prefix (optional)").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Entry(top, textvariable=self.curve_prefix, width=20).grid(row=5, column=1, sticky="w", padx=6, pady=6)

        ttk.Button(top, text="Plot Curve", command=self._plot_curve).grid(row=6, column=1, sticky="w", pady=10)

        self.curve_log = tk.Text(f, height=10)
        self.curve_log.pack(fill="both", expand=True, padx=10, pady=10)

    def _plot_curve(self):
        try:
            angle_mode = self.angle_mode_var.get()
            fix_neg = self.fix_var.get()

            expr = self.expr_var.get().strip()
            units = self.curve_units.get()
            th0 = float(self.th0_var.get())
            th1 = float(self.th1_var.get())
            samples = int(self.samples_var.get())
            prefix = self.curve_prefix.get().strip() or None

            do_curve(expr, units, th0, th1, samples, angle_mode, fix_neg, save_prefix=prefix)
            self.curve_log.insert("end", f"Plotted curve: r = {expr}\n")
            if prefix:
                self.curve_log.insert("end", f"Saved: {prefix}_polar.png and {prefix}_cart.png\n")
            else:
                self.curve_log.insert("end", "Displayed plot windows.\n")
        except Exception as e:
            messagebox.showerror("Curve Error", str(e))

    # ----- Tests Tab -----
    def _build_test_tab(self):
        f = self.tab_test
        ttk.Button(f, text="Run Tests", command=self._run_tests).pack(anchor="w", padx=10, pady=10)
        self.test_out = tk.Text(f)
        self.test_out.pack(fill="both", expand=True, padx=10, pady=10)

    def _run_tests(self):
        try:
            angle_mode = self.angle_mode_var.get()
            fix_neg = self.fix_var.get()
            txt = do_test(angle_mode, fix_neg)
            self.test_out.delete("1.0", "end")
            self.test_out.insert("end", txt)
        except Exception as e:
            messagebox.showerror("Test Error", str(e))


# ---------- CLI parser ----------
def build_parser():
    p = argparse.ArgumentParser(description="SDL3: Coordinate Conversion Tool (Polar ↔ Cartesian)")
    p.add_argument("--angle-mode", choices=["0_to_2pi", "negpi_to_pi", "none"], default="0_to_2pi")
    p.add_argument("--fix-negative-r", action="store_true", help="If r<0, flip r positive and add pi to theta.")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("point")
    sp.add_argument("direction", choices=["p2c", "c2p"])
    sp.add_argument("--units", choices=["deg", "rad"], default="deg")
    sp.add_argument("--r", type=float, default=1.0)
    sp.add_argument("--theta", type=float, default=0.0)
    sp.add_argument("--x", type=float, default=0.0)
    sp.add_argument("--y", type=float, default=0.0)
    sp.add_argument("--plot", action="store_true")
    sp.add_argument("--save-cart", default=None)
    sp.add_argument("--save-polar", default=None)

    sb = sub.add_parser("batch")
    sb.add_argument("--input", required=True)
    sb.add_argument("--output", default="converted.csv")
    sb.add_argument("--plot", action="store_true")
    sb.add_argument("--save-prefix", default=None)

    sc = sub.add_parser("curve")
    sc.add_argument("--r-expr", required=True)
    sc.add_argument("--units", choices=["deg", "rad"], default="rad")
    sc.add_argument("--theta-start", type=float, default=0.0)
    sc.add_argument("--theta-end", type=float, default=2 * math.pi)
    sc.add_argument("--samples", type=int, default=1200)
    sc.add_argument("--save-prefix", default=None)

    sub.add_parser("test")
    sub.add_parser("gui")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    angle_mode = args.angle_mode
    fix_neg_r = args.fix_negative_r

    if args.cmd == "gui":
        app = SDL3Gui(angle_mode, fix_neg_r)
        app.mainloop()
        return

    # CLI mode
    if args.cmd == "point":
        if args.direction == "p2c":
            pol, xy = compute_point_p2c(args.r, args.theta, args.units, fix_neg_r)
            print(f"(x, y) = ({xy.x:.6f}, {xy.y:.6f})")
            if args.plot:
                plot_points_polar([pol], "Point (polar)", save=args.save_polar)
                plot_points_cart([xy], "Point (cartesian)", save=args.save_cart)
        else:
            xy, pol, theta_out = compute_point_c2p(args.x, args.y, args.units, angle_mode)
            print(f"(r, theta) = ({pol.r:.6f}, {theta_out:.6f} {args.units})")
            if args.plot:
                plot_points_cart([xy], "Point (cartesian)", save=args.save_cart)
                plot_points_polar([pol], "Point (polar)", save=args.save_polar)

    elif args.cmd == "batch":
        n, outp = do_batch(args.input, args.output, angle_mode, fix_neg_r, plot=args.plot, save_prefix=args.save_prefix)
        print(f"Converted {n} rows -> {outp}")

    elif args.cmd == "curve":
        do_curve(args.r_expr, args.units, args.theta_start, args.theta_end, args.samples, angle_mode, fix_neg_r, save_prefix=args.save_prefix)
        print("Done. (Plots shown or saved.)")

    elif args.cmd == "test":
        print(do_test(angle_mode, fix_neg_r))
    else:
        # default: open GUI (better UX than old menu)
        app = SDL3Gui(angle_mode, fix_neg_r)
        app.mainloop()


if __name__ == "__main__":
    main()
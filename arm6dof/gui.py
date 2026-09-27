"""Interactive control panel for the 6-DOF arm — styled to the user's
design-scheme "Night" register (see arm6dof/theme.py).

Six joint sliders drive, live:
  * a dark 3D plot of the arm (matplotlib embedded in Tk),
  * the tool-tip pose readout (XYZ in mm),
  * a HUD-style per-joint servo-sizing table (lime PASS / red FAIL) at the
    CURRENT pose.

Runs with numpy + matplotlib + Tkinter. Launch:  python -m arm6dof.gui
"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from math import degrees

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import theme as T
from .model import default_arm
from .forward import fk_chain, joint_frames
from .torque import sizing_report as static_report, SAFETY_FACTOR
from .dynamics import sizing_report as dynamic_report, DEFAULT_PEAK_ACCEL


def _bracket(canvas: tk.Canvas, x, y, size, corner, colour):
    """Draw one L-shaped bracket corner (design-scheme .bracket), square caps."""
    s = size
    if corner == "tl":
        canvas.create_line(x, y, x + s, y, fill=colour, width=2)
        canvas.create_line(x, y, x, y + s, fill=colour, width=2)
    elif corner == "br":
        canvas.create_line(x, y, x - s, y, fill=colour, width=2)
        canvas.create_line(x, y, x, y - s, fill=colour, width=2)


class BracketBox(tk.Frame):
    """A hairline box with tl/br bracket corners and an optional tag label —
    the design-scheme .box component."""

    def __init__(self, master, tag: str = "", **kw):
        super().__init__(master, bg=T.INK,
                         highlightbackground=T.HAIR, highlightthickness=1, **kw)
        self._brackets = tk.Canvas(self, bg=T.INK, highlightthickness=0)
        self._brackets.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.body = tk.Frame(self, bg=T.INK)
        self.body.pack(fill="both", expand=True, padx=T.U * 2, pady=T.U * 2)
        if tag:
            lbl = tk.Label(self, text=f" {tag} ", bg=T.INK, fg=T.FG_DIM,
                           font=(T.SERIF[0], 10, "italic"))
            lbl.place(x=T.U * 3, y=-9)
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, _e):
        c = self._brackets
        c.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        _bracket(c, 2, 2, 20, "tl", T.GREY)
        _bracket(c, w - 2, h - 2, 20, "br", T.GREY)


class ArmGUI:
    def __init__(self, root: tk.Tk, model=None):
        self.model = model or default_arm()
        self.root = root
        self.serif_family = T.register_matplotlib_font()
        root.title("LA PAGINA BARBARA — arm6dof")
        root.configure(bg=T.INK)

        self._init_mpl_style()

        self.q = np.zeros(self.model.dof)
        self.dynamic_mode = tk.BooleanVar(value=False)
        self.peak_accel = tk.DoubleVar(value=DEFAULT_PEAK_ACCEL)

        # named tk fonts (italic serif, per Night register)
        self.f_title = tkfont.Font(family=self.serif_family, size=30, slant="italic")
        self.f_body = tkfont.Font(family=self.serif_family, size=13, slant="italic")
        self.f_small = tkfont.Font(family=self.serif_family, size=11, slant="italic")
        self.f_mono = tkfont.Font(family=T.MONO[0], size=10)
        self.f_mono_b = tkfont.Font(family=T.MONO[0], size=10, weight="bold")

        # ---- header bar --------------------------------------------------
        bar = tk.Frame(root, bg=T.INK)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew",
                 padx=T.U * 3, pady=(T.U * 2, T.U))
        tk.Label(bar, text="arm6dof", bg=T.INK, fg=T.GREY,
                 font=self.f_title).pack(side="left")
        tk.Label(bar, text="6-AXIS · 500 g PAYLOAD · SERVO SIZING", bg=T.INK,
                 fg=T.FG_DIM, font=self.f_mono).pack(side="right", pady=(14, 0))
        tk.Frame(root, bg=T.HAIR, height=1).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=T.U * 3)

        # ---- layout ------------------------------------------------------
        left = tk.Frame(root, bg=T.INK)
        left.grid(row=2, column=0, sticky="nsew", padx=(T.U * 3, T.U), pady=T.U * 2)
        right = tk.Frame(root, bg=T.INK)
        right.grid(row=2, column=1, sticky="nsew", padx=(T.U, T.U * 3), pady=T.U * 2)
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(2, weight=1)

        # ---- 3D figure in a bracket box ----------------------------------
        plot_box = BracketBox(left, tag="SCAN")
        plot_box.pack(fill="both", expand=True)
        self.fig = Figure(figsize=(6, 6), dpi=100, facecolor=T.INK)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_box.body)
        self.canvas.get_tk_widget().configure(bg=T.INK, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ---- joint sliders box -------------------------------------------
        ctrl = BracketBox(right, tag="JOINTS")
        ctrl.pack(fill="x", pady=(6, T.U * 2))
        self.slider_vars: list[tk.DoubleVar] = []
        self.slider_labels: list[tk.Label] = []
        for i, j in enumerate(self.model.joints):
            row = tk.Frame(ctrl.body, bg=T.INK)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=j.name, bg=T.INK, fg=T.GREY, width=14,
                     anchor="w", font=self.f_small).pack(side="left")
            var = tk.DoubleVar(value=0.0)
            s = tk.Scale(row, from_=degrees(j.q_min), to=degrees(j.q_max),
                         variable=var, orient="horizontal", length=150,
                         showvalue=False, bg=T.INK, fg=T.GREY, troughcolor=T.CHAR,
                         highlightthickness=0, bd=0, sliderrelief="flat",
                         activebackground=T.FG_HI,
                         command=lambda _v, idx=i: self._on_slide(idx))
            s.pack(side="left", fill="x", expand=True, padx=6)
            lbl = tk.Label(row, text="0°", bg=T.INK, fg=T.FG_DIM, width=6,
                           anchor="e", font=self.f_mono)
            lbl.pack(side="right")
            self.slider_vars.append(var)
            self.slider_labels.append(lbl)

        home = tk.Label(ctrl.body, text="◦ HOME (all 0°)", bg=T.INK, fg=T.FG_DIM,
                        font=self.f_mono, cursor="hand2")
        home.pack(anchor="w", pady=(T.U, 0))
        home.bind("<Button-1>", lambda _e: self._home())
        home.bind("<Enter>", lambda _e: home.config(fg=T.FG_HI))
        home.bind("<Leave>", lambda _e: home.config(fg=T.FG_DIM))

        # ---- pose readout ------------------------------------------------
        pose_box = BracketBox(right, tag="TCP")
        pose_box.pack(fill="x", pady=T.U * 2)
        self.pose_lbl = tk.Label(pose_box.body, text="", bg=T.INK, fg=T.GREY,
                                 font=self.f_mono, justify="left", anchor="w")
        self.pose_lbl.pack(fill="x")

        # ---- torque / sizing table ---------------------------------------
        tbl_box = BracketBox(right, tag="SERVO SIZING")
        tbl_box.pack(fill="both", expand=True, pady=(T.U * 2, 6))

        # mode toggle: STATIC (hold) vs DYNAMIC (hold + accelerate)
        mode_row = tk.Frame(tbl_box.body, bg=T.INK)
        mode_row.pack(fill="x")
        self.mode_lbl = tk.Label(mode_row, text="", bg=T.INK, fg=T.GREY,
                                 font=self.f_mono, cursor="hand2")
        self.mode_lbl.pack(side="left")
        self.mode_lbl.bind("<Button-1>", lambda _e: self._toggle_mode())
        self.mode_lbl.bind("<Enter>", lambda _e: self.mode_lbl.config(fg=T.FG_HI))
        self.mode_lbl.bind("<Leave>", lambda _e: self.mode_lbl.config(fg=T.GREY))

        # peak-acceleration slider (only meaningful in dynamic mode)
        self.accel_row = tk.Frame(tbl_box.body, bg=T.INK)
        self.accel_row.pack(fill="x", pady=(2, 2))
        self.accel_lbl = tk.Label(self.accel_row, text="", bg=T.INK, fg=T.FG_DIM,
                                  width=18, anchor="w", font=self.f_mono)
        self.accel_lbl.pack(side="left")
        self.accel_scale = tk.Scale(
            self.accel_row, from_=0.0, to=40.0, resolution=0.5,
            variable=self.peak_accel, orient="horizontal", length=140,
            showvalue=False, bg=T.INK, fg=T.GREY, troughcolor=T.CHAR,
            highlightthickness=0, bd=0, sliderrelief="flat",
            activebackground=T.FG_HI, command=lambda _v: self._redraw())
        self.accel_scale.pack(side="left", fill="x", expand=True, padx=6)

        self.table = tk.Frame(tbl_box.body, bg=T.INK)
        self.table.pack(fill="both", expand=True, pady=(4, 0))

        self._redraw()

    # -- matplotlib dark style ------------------------------------------
    def _init_mpl_style(self):
        import matplotlib as mpl
        mpl.rcParams.update({
            "font.family": self.serif_family,
            "text.color": T.GREY,
            "axes.edgecolor": T.HAIR,
            "axes.labelcolor": T.FG_DIM,
            "xtick.color": T.FG_DIM,
            "ytick.color": T.FG_DIM,
        })

    def _toggle_mode(self):
        self.dynamic_mode.set(not self.dynamic_mode.get())
        self._redraw()

    # -- callbacks -------------------------------------------------------
    def _on_slide(self, idx: int):
        deg = self.slider_vars[idx].get()
        self.q[idx] = np.radians(deg)
        self.slider_labels[idx].config(text=f"{deg:.0f}°")
        self._redraw()

    def _home(self):
        for i, var in enumerate(self.slider_vars):
            var.set(0.0)
            self.q[i] = 0.0
            self.slider_labels[i].config(text="0°")
        self._redraw()

    # -- rendering -------------------------------------------------------
    def _redraw(self):
        self._draw_arm()
        self._update_pose()
        self._update_table()

    def _draw_arm(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(T.INK)
        try:
            ax.xaxis.set_pane_color((0, 0, 0, 0))
            ax.yaxis.set_pane_color((0, 0, 0, 0))
            ax.zaxis.set_pane_color((0, 0, 0, 0))
        except Exception:
            pass

        frames = joint_frames(self.model, self.q)
        pts = np.array([f[:3, 3] for f in frames])
        tip = fk_chain(self.model, self.q)[:3, 3]
        pts = np.vstack([pts, tip])

        # links: grey; joints: small grey squares (point-marker component)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", lw=2, color=T.GREY)
        ax.scatter(pts[:-1, 0], pts[:-1, 1], pts[:-1, 2],
                   c=T.GREY, s=28, marker="s")
        # TCP + payload: scan-red X (signal)
        ax.scatter(*tip, c=T.RED, s=80, marker="X")

        reach = sum(abs(j.a) + abs(j.d) for j in self.model.joints) + 0.1
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_zlim(0, reach)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.grid(True, color=T.HAIR, linewidth=0.5)
        self.canvas.draw_idle()

    def _update_pose(self):
        T4 = fk_chain(self.model, self.q)
        x, y, z = T4[:3, 3] * 1000.0
        self.pose_lbl.config(
            text=f"X {x:8.1f}\nY {y:8.1f}\nZ {z:8.1f}   [mm]")

    def _update_table(self):
        dyn = self.dynamic_mode.get()
        pa = self.peak_accel.get()

        # mode + accel control labels
        self.mode_lbl.config(
            text=("▸ MODE: DYNAMIC (hold + accelerate)" if dyn
                  else "▸ MODE: STATIC (hold only)"))
        if dyn:
            if not self.accel_row.winfo_ismapped():
                self.accel_row.pack(fill="x", pady=(2, 2))
            self.accel_lbl.config(text=f"peak accel {pa:4.1f} rad/s²")
        else:
            self.accel_row.pack_forget()

        # gather rows for the active mode
        if dyn:
            rows = dynamic_report(self.model, self.q, pa)
            headers = ["joint", "stat", "accel", "total", "need", "rated", "st"]
            keys = ["static_Nm", "accel_Nm", "total_Nm",
                    "required_with_margin_Nm", "servo_rated_Nm"]
        else:
            rows = static_report(self.model, self.q)
            headers = ["joint", "static", "need", "rated", "st"]
            keys = ["static_Nm", "required_with_margin_Nm", "servo_rated_Nm"]

        # rebuild the table grid (column count differs between modes)
        for w in self.table.winfo_children():
            w.destroy()
        for c, name in enumerate(headers):
            tk.Label(self.table, text=name, bg=T.INK, fg=T.FG_DIM,
                     font=self.f_mono, anchor="w").grid(
                row=0, column=c, sticky="w", padx=4)
        tk.Frame(self.table, bg=T.HAIR, height=1).grid(
            row=1, column=0, columnspan=len(headers), sticky="ew", pady=2)

        for r, info in enumerate(rows):
            short = info["joint"].split("_", 1)[-1][:10]
            cells = [short] + [f"{info[k]:.2f}" for k in keys]
            cells.append("PASS" if info["ok"] else "FAIL")
            colour = T.LIME if info["ok"] else T.RED
            last = len(cells) - 1
            for c, t in enumerate(cells):
                fg = colour if c == last else T.GREY
                fnt = self.f_mono_b if c == last else self.f_mono
                tk.Label(self.table, text=t, bg=T.INK, fg=fg, font=fnt,
                         anchor="w").grid(row=2 + r, column=c, sticky="w", padx=4)


def main():
    root = tk.Tk()
    ArmGUI(root)
    root.geometry("1020x680")
    root.mainloop()


if __name__ == "__main__":
    main()

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
from dataclasses import replace
from math import degrees
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import theme as T
from .model import default_arm, preset as model_preset, PRESETS as M_PRESETS
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
    the design-scheme .box component.

    The hairline border is drawn on an internal canvas (NOT a frame border),
    with a gap left on the top edge for the tag, so the tag's italic ascenders
    are never clipped by a border line.
    """

    def __init__(self, master, tag: str = "", **kw):
        super().__init__(master, bg=T.INK, highlightthickness=0, **kw)
        self.tag = tag
        self._border = tk.Canvas(self, bg=T.INK, highlightthickness=0)
        self._border.place(relx=0, rely=0, relwidth=1, relheight=1)
        # top padding leaves room for the tag that straddles the top edge
        self.body = tk.Frame(self, bg=T.INK)
        self.body.pack(fill="both", expand=True,
                       padx=T.U * 2, pady=(T.U * 3, T.U * 2))
        if tag:
            self._tag = tk.Label(self, text=f" {tag} ", bg=T.INK, fg=T.FG_DIM,
                                 font=(T.SERIF[0], 11, "italic"), padx=2)
            self._tag.place(x=T.U * 3, y=T.U + 2, anchor="w")
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, _e):
        c = self._border
        c.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 2 or h <= 2:
            return
        ty = T.U + 2                       # vertical centre of the tag row
        # tag gap on the top edge (skip the border under the tag text)
        if self.tag:
            gap0 = T.U * 3 - 4
            gap1 = gap0 + self._tag.winfo_reqwidth() + 4
        else:
            gap0 = gap1 = -1
        # top edge, split around the tag gap
        c.create_line(2, ty, max(2, gap0), ty, fill=T.HAIR, width=1)
        c.create_line(gap1, ty, w - 2, ty, fill=T.HAIR, width=1)
        # other three edges
        c.create_line(2, ty, 2, h - 2, fill=T.HAIR, width=1)          # left
        c.create_line(w - 2, ty, w - 2, h - 2, fill=T.HAIR, width=1)  # right
        c.create_line(2, h - 2, w - 2, h - 2, fill=T.HAIR, width=1)   # bottom
        # bracket corners (brighter, thicker) — the .bracket component
        _bracket(c, 2, ty, 20, "tl", T.GREY)
        _bracket(c, w - 2, h - 2, 20, "br", T.GREY)


class ArmGUI:
    def __init__(self, root: tk.Tk, model=None):
        self.base_model = model or default_arm()
        self.model = self.base_model
        self.root = root
        self.serif_family = T.register_matplotlib_font()
        root.title("LA PAGINA BARBARA — arm6dof")
        root.configure(bg=T.INK)

        self._init_mpl_style()

        self.q = np.array(self.model.home_q(), dtype=float)
        self._ready = False   # gate: traces must not redraw until UI is built
        self.dynamic_mode = tk.BooleanVar(value=False)
        self.peak_accel = tk.DoubleVar(value=DEFAULT_PEAK_ACCEL)
        # live TCP payload (grams) and per-joint external gear ratio, both
        # seeded from the base model and applied via _apply_overrides()
        self.payload_g = tk.DoubleVar(value=self.base_model.payload * 1000.0)
        self.gear_vars = [tk.DoubleVar(value=j.gear_ratio)
                          for j in self.base_model.joints]
        # editable DH distances per joint, in mm: a = link length (along X),
        # d = link offset / off-centre (along Z). Seeded from the base model.
        self.a_vars = [tk.DoubleVar(value=round(j.a * 1000.0, 1))
                       for j in self.base_model.joints]
        self.d_vars = [tk.DoubleVar(value=round(j.d * 1000.0, 1))
                       for j in self.base_model.joints]
        # DH angles per joint, in degrees: alpha = link twist (about X),
        # theta_offset = constant added to the commanded joint angle.
        self.alpha_vars = [tk.StringVar(value=f"{round(degrees(j.alpha)):d}")
                           for j in self.base_model.joints]
        self.toff_vars = [tk.DoubleVar(value=round(degrees(j.theta_offset), 1))
                          for j in self.base_model.joints]
        self.preset_var = tk.StringVar(value="IRB 1300-7/1.4")
        self.preset_var.trace_add(
            "write", lambda *_a: self._on_preset(self.preset_var.get()))
        # register live-recompute traces once (rebuilding rows must not re-add)
        for _vs in (self.a_vars, self.d_vars, self.alpha_vars, self.toff_vars):
            for _v in _vs:
                _v.trace_add("write", lambda *_a: self._safe_redraw())

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
        home_deg = [degrees(v) for v in self.model.home_q()]
        for i, j in enumerate(self.model.joints):
            row = tk.Frame(ctrl.body, bg=T.INK)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=j.name, bg=T.INK, fg=T.GREY, width=14,
                     anchor="w", font=self.f_small).pack(side="left")
            var = tk.DoubleVar(value=home_deg[i])
            s = tk.Scale(row, from_=degrees(j.q_min), to=degrees(j.q_max),
                         variable=var, orient="horizontal", length=150,
                         showvalue=False, bg=T.INK, fg=T.GREY, troughcolor=T.CHAR,
                         highlightthickness=0, bd=0, sliderrelief="flat",
                         activebackground=T.FG_HI,
                         command=lambda _v, idx=i: self._on_slide(idx))
            s.pack(side="left", fill="x", expand=True, padx=6)
            lbl = tk.Label(row, text=f"{home_deg[i]:.0f}°", bg=T.INK, fg=T.FG_DIM,
                           width=6, anchor="e", font=self.f_mono)
            lbl.pack(side="right")
            self.slider_vars.append(var)
            self.slider_labels.append(lbl)

        home = tk.Label(ctrl.body, text="◦ HOME (all 0°)", bg=T.INK,
                        fg=T.FG_DIM, font=self.f_mono, cursor="hand2")
        home.pack(anchor="w", pady=(T.U, 0))
        home.bind("<Button-1>", lambda _e: self._home())
        home.bind("<Enter>", lambda _e: home.config(fg=T.FG_HI))
        home.bind("<Leave>", lambda _e: home.config(fg=T.FG_DIM))

        # ---- DH geometry editor ------------------------------------------
        # a = link length (along X_i), d = link offset / off-centre (along Z_i),
        # both in mm. alpha = link twist (deg), toff = theta-offset (deg).
        dh_box = BracketBox(right, tag="DH GEOMETRY  ·  a·d [mm]  α·θ₀ [deg]")
        dh_box.pack(fill="x", pady=T.U * 2)

        # preset picker
        preset_row = tk.Frame(dh_box.body, bg=T.INK)
        preset_row.pack(fill="x", pady=(0, T.U))
        tk.Label(preset_row, text="preset", bg=T.INK, fg=T.FG_DIM, width=12,
                 anchor="w", font=self.f_mono).pack(side="left")
        om = tk.OptionMenu(preset_row, self.preset_var, *M_PRESETS.keys())
        om.config(bg=T.CHAR, fg=T.GREY, activebackground=T.CHAR,
                  activeforeground=T.FG_HI, highlightthickness=0, bd=0,
                  font=self.f_mono, anchor="w")
        om["menu"].config(bg=T.CHAR, fg=T.GREY, activebackground=T.FG_DIM,
                          font=self.f_mono)
        om.pack(side="left", fill="x", expand=True)

        hdr = tk.Frame(dh_box.body, bg=T.INK)
        hdr.pack(fill="x")
        for txt, w in (("joint", 10), ("a", 6), ("d", 6), ("α", 6), ("θ₀", 6)):
            tk.Label(hdr, text=txt, bg=T.INK, fg=T.FG_DIM, width=w,
                     anchor="w", font=self.f_mono).pack(side="left", padx=(0, 4))
        self.dh_rows = tk.Frame(dh_box.body, bg=T.INK)
        self.dh_rows.pack(fill="x")
        self._build_dh_rows()

        # config actions: save-as-default / save-as / load
        cfg_row = tk.Frame(dh_box.body, bg=T.INK)
        cfg_row.pack(fill="x", pady=(T.U, 0))

        def _action(parent, text, cmd):
            lb = tk.Label(parent, text=text, bg=T.INK, fg=T.FG_DIM,
                          font=self.f_mono, cursor="hand2")
            lb.pack(side="left", padx=(0, T.U * 2))
            lb.bind("<Button-1>", lambda _e: cmd())
            lb.bind("<Enter>", lambda _e: lb.config(fg=T.FG_HI))
            lb.bind("<Leave>", lambda _e: lb.config(fg=T.FG_DIM))
            return lb

        _action(cfg_row, "▸ SAVE AS DEFAULT", self._save_default)
        _action(cfg_row, "▸ SAVE AS…", self._save_as)
        _action(cfg_row, "▸ LOAD…", self._load_config)
        self.cfg_status = tk.Label(dh_box.body, text="", bg=T.INK, fg=T.LIME,
                                   font=self.f_mono, anchor="w")
        self.cfg_status.pack(fill="x")

        # ---- pose readout ------------------------------------------------
        pose_box = BracketBox(right, tag="TCP")
        pose_box.pack(fill="x", pady=T.U * 2)
        self.pose_lbl = tk.Label(pose_box.body, text="", bg=T.INK, fg=T.GREY,
                                 font=self.f_mono, justify="left", anchor="w")
        self.pose_lbl.pack(fill="x")

        # ---- TCP payload control -----------------------------------------
        pay_box = BracketBox(right, tag="PAYLOAD")
        pay_box.pack(fill="x", pady=T.U * 2)
        pay_row = tk.Frame(pay_box.body, bg=T.INK)
        pay_row.pack(fill="x")
        self.payload_lbl = tk.Label(pay_row, text="", bg=T.INK, fg=T.GREY,
                                    width=14, anchor="w", font=self.f_mono)
        self.payload_lbl.pack(side="left")
        tk.Scale(
            pay_row, from_=0.0, to=2000.0, resolution=25.0,
            variable=self.payload_g, orient="horizontal", length=150,
            showvalue=False, bg=T.INK, fg=T.GREY, troughcolor=T.CHAR,
            highlightthickness=0, bd=0, sliderrelief="flat",
            activebackground=T.FG_HI, command=lambda _v: self._redraw()
        ).pack(side="left", fill="x", expand=True, padx=6)

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

        # per-joint external gear ratio (>1 = reduction, multiplies servo torque)
        gear_hdr = tk.Label(tbl_box.body, text="▸ GEAR RATIO  (× reduction per joint)",
                            bg=T.INK, fg=T.FG_DIM, font=self.f_mono, anchor="w")
        gear_hdr.pack(fill="x", pady=(6, 2))
        gear_grid = tk.Frame(tbl_box.body, bg=T.INK)
        gear_grid.pack(fill="x")
        for i, j in enumerate(self.base_model.joints):
            cell = tk.Frame(gear_grid, bg=T.INK)
            cell.grid(row=i // 3, column=i % 3, sticky="w", padx=4, pady=1)
            tk.Label(cell, text=j.name.split("_", 1)[-1][:8], bg=T.INK,
                     fg=T.GREY, width=8, anchor="w", font=self.f_mono).pack(side="left")
            tk.Spinbox(
                cell, from_=1.0, to=50.0, increment=0.5, width=5,
                textvariable=self.gear_vars[i], font=self.f_mono,
                bg=T.CHAR, fg=T.GREY, buttonbackground=T.CHAR,
                insertbackground=T.GREY, relief="flat", justify="right",
            ).pack(side="left")
            # fires on arrow clicks AND typed edits; guard bad input
            self.gear_vars[i].trace_add("write", lambda *_a: self._safe_redraw())

        self.table = tk.Frame(tbl_box.body, bg=T.INK)
        self.table.pack(fill="both", expand=True, pady=(4, 0))

        self._ready = True
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
        home_deg = [degrees(v) for v in self.model.home_q()]
        for i, var in enumerate(self.slider_vars):
            var.set(home_deg[i])
            self.q[i] = np.radians(home_deg[i])
            self.slider_labels[i].config(text=f"{home_deg[i]:.0f}°")
        self._redraw()

    # -- rendering -------------------------------------------------------
    def _build_dh_rows(self):
        """(Re)build the per-joint DH editor rows into self.dh_rows."""
        for w in self.dh_rows.winfo_children():
            w.destroy()
        for i, j in enumerate(self.base_model.joints):
            row = tk.Frame(self.dh_rows, bg=T.INK)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=j.name.split("_", 1)[-1][:9], bg=T.INK,
                     fg=T.GREY, width=10, anchor="w",
                     font=self.f_mono).pack(side="left", padx=(0, 4))
            for var in (self.a_vars[i], self.d_vars[i]):
                tk.Spinbox(
                    row, from_=0.0, to=2000.0, increment=5.0, width=6,
                    textvariable=var, font=self.f_mono,
                    bg=T.CHAR, fg=T.GREY, buttonbackground=T.CHAR,
                    insertbackground=T.GREY, relief="flat", justify="right",
                ).pack(side="left", padx=(0, 4))
            # alpha: only physical revolute values
            am = tk.OptionMenu(row, self.alpha_vars[i], "-90", "0", "90", "180")
            am.config(bg=T.CHAR, fg=T.GREY, activebackground=T.CHAR,
                      activeforeground=T.FG_HI, highlightthickness=0, bd=0,
                      font=self.f_mono, width=4, anchor="w")
            am["menu"].config(bg=T.CHAR, fg=T.GREY, activebackground=T.FG_DIM,
                              font=self.f_mono)
            am.pack(side="left", padx=(0, 4))
            tk.Spinbox(
                row, from_=-180.0, to=180.0, increment=5.0, width=6,
                textvariable=self.toff_vars[i], font=self.f_mono,
                bg=T.CHAR, fg=T.GREY, buttonbackground=T.CHAR,
                insertbackground=T.GREY, relief="flat", justify="right",
            ).pack(side="left")

    def _on_preset(self, name: str):
        """Swap the base model to the chosen preset, reseed every control from
        it, rebuild the DH rows and joint sliders, and redraw."""
        if not getattr(self, "_ready", False):
            return
        self.base_model = model_preset(name)
        # reseed control variables from the new base model
        for i, j in enumerate(self.base_model.joints):
            self.a_vars[i].set(round(j.a * 1000.0, 1))
            self.d_vars[i].set(round(j.d * 1000.0, 1))
            self.alpha_vars[i].set(f"{round(degrees(j.alpha)):d}")
            self.toff_vars[i].set(round(degrees(j.theta_offset), 1))
            self.gear_vars[i].set(j.gear_ratio)
        self.payload_g.set(round(self.base_model.payload * 1000.0, 0))
        # reset pose to the preset's home and resync sliders
        self.q = np.array(self.base_model.home_q(), dtype=float)
        home_deg = [degrees(v) for v in self.base_model.home_q()]
        for i, var in enumerate(self.slider_vars):
            var.set(home_deg[i])
            self.slider_labels[i].config(text=f"{home_deg[i]:.0f}°")
        self._build_dh_rows()
        self._flash(f"preset ← {name} ✓")
        self._safe_redraw()

    def _apply_overrides(self):
        """Rebuild self.model from the base model plus the live payload and
        per-joint gear-ratio controls. Immutable: uses dataclasses.replace,
        never mutates the base model."""
        new_joints = tuple(
            replace(
                j,
                gear_ratio=max(0.1, self.gear_vars[i].get()),
                a=self.a_vars[i].get() / 1000.0,
                d=self.d_vars[i].get() / 1000.0,
                alpha=np.radians(float(self.alpha_vars[i].get())),
                theta_offset=np.radians(self.toff_vars[i].get()),
            )
            for i, j in enumerate(self.base_model.joints)
        )
        self.model = replace(
            self.base_model,
            joints=new_joints,
            payload=max(0.0, self.payload_g.get()) / 1000.0,
        )

    def _redraw(self):
        self._apply_overrides()
        self._draw_arm()
        self._update_pose()
        self._update_table()

    def _safe_redraw(self):
        """Redraw, ignoring transient errors while a spinbox holds a half-typed
        value, and skipping entirely until the UI is fully built (traces can
        fire during __init__ before all widgets exist)."""
        if not getattr(self, "_ready", False):
            return
        try:
            self._redraw()
        except (tk.TclError, AttributeError):
            pass

    # -- config save / load ----------------------------------------------
    def _flash(self, msg: str):
        """Briefly show a status message on the config row."""
        if hasattr(self, "cfg_status"):
            self.cfg_status.config(text=msg)
            self.root.after(2500, lambda: self.cfg_status.config(text=""))

    def _save_default(self):
        from . import config as C
        self._apply_overrides()
        try:
            C.save_config(self.model, C.DEFAULT_CONFIG)
            self._flash("saved as default ✓")
        except Exception as e:  # pragma: no cover - filesystem edge
            self._flash(f"save failed: {e}")

    def _save_as(self):
        from tkinter import filedialog
        from . import config as C
        self._apply_overrides()
        path = filedialog.asksaveasfilename(
            title="Save arm config as…", defaultextension=".json",
            initialfile="arm_config.json",
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            C.save_config(self.model, path)
            self._flash(f"saved → {Path(path).name} ✓")
        except Exception as e:  # pragma: no cover
            self._flash(f"save failed: {e}")

    def _load_config(self):
        from tkinter import filedialog
        from . import config as C
        path = filedialog.askopenfilename(
            title="Load arm config…",
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            cfg = C.load_config(path)
            self._apply_config_to_controls(cfg)
            self._flash(f"loaded ← {Path(path).name} ✓")
        except Exception as e:  # pragma: no cover
            self._flash(f"load failed: {e}")

    def _apply_config_to_controls(self, cfg: dict):
        """Push a loaded config dict into the live control variables so the UI
        reflects it, then redraw."""
        cfg_joints = cfg.get("joints", [])
        for i in range(self.base_model.dof):
            cj = cfg_joints[i] if i < len(cfg_joints) else {}
            if "a_mm" in cj:
                self.a_vars[i].set(round(float(cj["a_mm"]), 1))
            if "d_mm" in cj:
                self.d_vars[i].set(round(float(cj["d_mm"]), 1))
            if "alpha_deg" in cj:
                self.alpha_vars[i].set(f"{round(float(cj['alpha_deg'])):d}")
            if "theta_offset_deg" in cj:
                self.toff_vars[i].set(round(float(cj["theta_offset_deg"]), 1))
            if "gear_ratio" in cj:
                self.gear_vars[i].set(round(float(cj["gear_ratio"]), 2))
        if "payload_g" in cfg:
            self.payload_g.set(round(float(cfg["payload_g"]), 0))
        self._safe_redraw()

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

        # per-joint coordinate frames: X=red, Y=green, Z=blue (RGB convention).
        # axis length scales with reach so triads stay readable.
        alen = 0.06 * (1 + sum(abs(j.a) + abs(j.d) for j in self.model.joints))
        alen = min(alen, 0.09)
        axis_cols = ("#FF5A4D", "#7CE07C", "#5AA0FF")  # X, Y, Z (RGB, HUD-muted)
        for f in frames:
            o = f[:3, 3]
            for k in range(3):
                d = f[:3, k] * alen
                ax.plot([o[0], o[0] + d[0]], [o[1], o[1] + d[1]],
                        [o[2], o[2] + d[2]], "-", lw=1.4, color=axis_cols[k])

        reach = sum(abs(j.a) + abs(j.d) for j in self.model.joints) + 0.1
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_zlim(-reach * 0.25, reach)
        try:
            ax.set_box_aspect((1, 1, 0.625))   # keep arms undistorted
        except Exception:
            pass
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.grid(True, color=T.HAIR, linewidth=0.5)
        # RGB axis-convention legend, top-left of the 3D panel
        ax.text2D(0.02, 0.98, "X", transform=ax.transAxes,
                  color="#FF5A4D", fontsize=9, va="top")
        ax.text2D(0.05, 0.98, "Y", transform=ax.transAxes,
                  color="#7CE07C", fontsize=9, va="top")
        ax.text2D(0.08, 0.98, "Z", transform=ax.transAxes,
                  color="#5AA0FF", fontsize=9, va="top")
        self.canvas.draw_idle()

    def _update_pose(self):
        T4 = fk_chain(self.model, self.q)
        x, y, z = T4[:3, 3] * 1000.0
        self.pose_lbl.config(
            text=f"X {x:8.1f}\nY {y:8.1f}\nZ {z:8.1f}   [mm]")
        self.payload_lbl.config(text=f"{self.payload_g.get():.0f} g")

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
    root.geometry("1020x820")
    root.mainloop()


if __name__ == "__main__":
    main()

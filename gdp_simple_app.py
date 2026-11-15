"""
Predicting GDP growth with linear algebra - a small demo (4 steps)

  1. Set up  Ax = b  and use Gaussian elimination to test for an exact solution
  2. Least squares: project b onto the column space (normal equations  A^T A x = A^T b)
  3. SVD: remove the unstable directions that make least squares wobble
  4. Prediction: compare both models on years they have not seen

Run:  pip install numpy matplotlib   then   python gdp_simple_app.py
CSV format (optional): first column Year, second column GDP growth, other columns indicators.
"""
import csv
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

NAVY, BLUE, ORANGE, GREY = "#1F3864", "#2E75B6", "#E36C0A", "#6B7280"
TEST_YEARS = 8                                   # last 8 years are hidden from the models


# ------------------------------------------------------------------ data
def demo_data():
    """SYNTHETIC data (not real figures): exports, imports and investment move together."""
    rng = np.random.default_rng(19)
    years = np.arange(1991, 2024)
    n = len(years)
    g, d = rng.normal(size=(2, n))
    shock = np.zeros(n)
    shock[years == 2008], shock[years == 2009] = -0.8, -1.8
    shock[years == 2020], shock[years == 2021] = -4.2, 2.4
    infl = 7 + 2.2 * rng.normal(size=n)
    X = np.column_stack([
        8 + 5.0 * g + 1.0 * d + 0.3 * rng.normal(size=n) + 3.0 * shock,
        9 + 4.5 * g + 2.5 * d + 0.3 * rng.normal(size=n) + 3.5 * shock,
        8 + 3.0 * g + 4.5 * d + 0.3 * rng.normal(size=n) + 3.0 * shock,
        infl])
    gdp = 6.2 + 0.55 * g + 0.45 * d - 0.14 * (infl - 7) + shock + 0.55 * rng.normal(size=n)
    return years, gdp, X, ["Exports", "Imports", "Investment", "Inflation"], "Synthetic demo data"


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], [r for r in rows[1:] if len(r) == len(rows[0]) and all(c.strip() for c in r)]
    data = np.array([[float(c) for c in r] for r in body])
    order = np.argsort(data[:, 0])
    data = data[order]
    return data[:, 0].astype(int), data[:, 1], data[:, 2:], head[2:], "CSV: " + path


# ------------------------------------------------------------ the maths
def rref(M, tol=1e-9):
    """Gauss-Jordan elimination: returns the reduced matrix and its pivot columns."""
    R, piv, r = np.array(M, dtype=float), [], 0
    for c in range(R.shape[1]):
        if r == R.shape[0]:
            break
        p = r + np.argmax(abs(R[r:, c]))
        if abs(R[p, c]) < tol:
            continue
        R[[r, p]] = R[[p, r]]
        R[r] /= R[r, c]
        for i in range(R.shape[0]):
            if i != r:
                R[i] -= R[i, c] * R[r]
        piv.append(c)
        r += 1
    return R, piv


def least_squares(Z, b):
    """Normal equations A^T A x = A^T b with A = [1 | Z]. Returns (intercept, coefficients)."""
    A = np.column_stack([np.ones(len(b)), Z])
    x = np.linalg.solve(A.T @ A, A.T @ b)
    return x[0], x[1:]


def svd_fit(Z, b, k):
    """Keep only the k biggest singular directions: x = sum_i (u_i . b / s_i) v_i."""
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    coef = (U.T @ (b - b.mean())) / S
    return b.mean(), Vt[:k].T @ coef[:k]


def rmse(y, yhat):
    return np.linalg.norm(y - yhat) / np.sqrt(len(y))        # 2-norm of the error vector


def leave_one_out(Z, b, fit):
    """Refit with each year removed: how much do the coefficients move?"""
    return np.array([fit(Z[np.arange(len(b)) != i], b[np.arange(len(b)) != i])[1] for i in range(len(b))])


class Model:
    def __init__(self, years, y, X, names, k):
        self.years, self.y, self.names, self.k = years, y, names, k
        self.ntr = len(y) - TEST_YEARS
        mu, sd = X[:self.ntr].mean(0), X[:self.ntr].std(0, ddof=1)
        self.Z = (X - mu) / sd                                   # standardised indicators
        Ztr, b = self.Z[:self.ntr], y[:self.ntr]
        self.A = np.column_stack([np.ones(self.ntr), Ztr])       # the matrix A
        self.b = b
        self.R_aug = rref(np.column_stack([self.A, b]))[0]
        self.rank_A = len(rref(self.A)[1])
        self.rank_Ab = len(rref(np.column_stack([self.A, b]))[1])
        self.corr = np.corrcoef(Ztr.T)
        self.b0, self.beta_ols = least_squares(Ztr, b)
        self.e = b - (self.b0 + Ztr @ self.beta_ols)
        self.S = np.linalg.svd(Ztr - Ztr.mean(0), compute_uv=False)
        self.c0, self.beta_svd = svd_fit(Ztr, b, k)
        self.spread_ols = leave_one_out(Ztr, b, least_squares).std(0)
        self.spread_svd = leave_one_out(Ztr, b, lambda Z_, b_: svd_fit(Z_, b_, k)).std(0)
        self.pred_ols = self.b0 + self.Z @ self.beta_ols
        self.pred_svd = self.c0 + self.Z @ self.beta_svd


# ------------------------------------------------------------------- GUI
def bars(ax, names, a, b, ea=None, eb=None):
    x = np.arange(len(names))
    ax.bar(x - 0.2, a, 0.4, yerr=ea, capsize=3, color=BLUE, label="Least squares")
    ax.bar(x + 0.2, b, 0.4, yerr=eb, capsize=3, color=ORANGE, label="SVD (k components)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, names, fontsize=8)
    ax.legend(frameon=False, fontsize=8)


def step1(m, t):
    t.h("THE MATH")
    t.p("Each year gives one equation:  b = x0 + x1*exports + x2*imports + ...")
    t.p("In matrix form  A x = b,  with A = [1 | indicators]  (one row per year).")
    t.p("Gaussian elimination reduces [A | b] and shows whether an exact")
    t.p("solution exists:  it does only if  rank(A) = rank([A | b]).")
    t.h("RESULT")
    t.p(f"A has {m.A.shape[0]} rows (years) and {m.A.shape[1]} columns (unknowns)")
    t.p(f"rank(A)      = {m.rank_A}")
    t.p(f"rank([A|b])  = {m.rank_Ab}")
    t.h("TAKEAWAY")
    t.p("The rank goes up when b is added, so b is not in the column space:" if m.rank_Ab > m.rank_A else "Consistent system.")
    t.p("there is NO exact solution. We need the best approximate one.")
    t.p("")
    t.p("Also: strongly correlated indicators (see plot) are almost copies")
    t.p("of each other. Nearly dependent columns cause trouble in step 2.")


def plot1(fig, m):
    ax = fig.add_subplot(111)
    im = ax.imshow(m.corr, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(m.names)):
        for j in range(len(m.names)):
            ax.text(j, i, f"{m.corr[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(m.corr[i, j]) > 0.6 else "black")
    ax.set_xticks(range(len(m.names)), m.names)
    ax.set_yticks(range(len(m.names)), m.names)
    ax.set_title("Correlation between indicators (columns of A)")
    fig.colorbar(im, shrink=0.8)


def step2(m, t):
    t.h("THE MATH")
    t.p("No exact solution, so find x that makes the error smallest.")
    t.p("Project b onto the column space of A:  p = A x,  error e = b - p.")
    t.p("The best error is perpendicular to every column:  A^T e = 0")
    t.p("which gives the NORMAL EQUATIONS:   (A^T A) x = A^T b")
    t.h("RESULT")
    t.p(f"max |A^T e| = {np.abs(m.A.T @ m.e).max():.1e}   (error is perpendicular)")
    t.p("")
    t.p("Coefficients (standardised):")
    for n, c, s in zip(m.names, m.beta_ols, m.spread_ols):
        t.p(f"  {n:<11}{c:>7.2f}   (moves by +/-{s:.2f} if one year is dropped)")
    t.h("TAKEAWAY")
    t.p("Look at the error bars: with nearly dependent columns, A^T A is")
    t.p("almost singular, so small changes in the data can swing the")
    t.p("coefficients a lot (even flip their sign).")


def plot2(fig, m):
    ax = fig.add_subplot(111)
    x = np.arange(len(m.names))
    ax.bar(x, m.beta_ols, 0.5, yerr=m.spread_ols, capsize=4, color=BLUE)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, m.names)
    ax.set_title("Least squares coefficients\n(error bars: spread when one year is dropped)", fontsize=10)


def step3(m, t):
    t.h("THE MATH")
    t.p("Singular value decomposition:   A = U S V^T")
    t.p("Least squares then becomes   x = sum_i (u_i . b / s_i) v_i")
    t.p("A tiny singular value s_i divides by almost zero, so noise is")
    t.p("blown up along that direction v_i. FIX: keep only the first k")
    t.p("terms (truncated SVD). This is PCA regression.")
    t.h("RESULT")
    t.p("Singular values: " + "  ".join(f"{s:.2f}" for s in m.S))
    t.p(f"Condition number s_max / s_min = {m.S[0] / m.S[-1]:.1f}")
    t.p(f"Keeping k = {m.k} of {len(m.S)} directions:")
    for n, c, s in zip(m.names, m.beta_svd, m.spread_svd):
        t.p(f"  {n:<11}{c:>7.2f}   (moves by +/-{s:.2f})")
    t.h("TAKEAWAY")
    t.p("Dropping the small directions shrinks the swings in the")
    t.p("coefficients (compare the error bars). Change k below to see")
    t.p("the trade-off between stability and fit.")


def plot3(fig, m):
    a1, a2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.6]})
    a1.bar(range(1, len(m.S) + 1), m.S, color=[BLUE if i < m.k else GREY for i in range(len(m.S))])
    a1.set(title="Singular values\n(blue = kept)", xlabel="direction")
    bars(a2, m.names, m.beta_ols, m.beta_svd, m.spread_ols, m.spread_svd)
    a2.set_title("Coefficients with and without\nthe unstable directions", fontsize=10)


def step4(m, t):
    n = m.ntr
    t.h("THE MATH")
    t.p("Prediction for a year:   GDP = x0 + x . (standardised indicators)")
    t.p("Error vector  e = actual - predicted,   RMSE = |e| / sqrt(m)")
    t.p(f"Models are fitted on {m.years[0]}-{m.years[n - 1]} only; the last")
    t.p(f"{TEST_YEARS} years ({m.years[n]}-{m.years[-1]}) are unseen.")
    t.h("RESULT (RMSE, percentage points)")
    rows = [("Least squares", m.pred_ols), (f"SVD, k={m.k}", m.pred_svd)]
    t.p(f"{'':<16}{'train':>8}{'test':>8}")
    for name, p in rows:
        t.p(f"{name:<16}{rmse(m.y[:n], p[:n]):>8.2f}{rmse(m.y[n:], p[n:]):>8.2f}")
    t.h("TAKEAWAY")
    better = rmse(m.y[n:], m.pred_svd[n:]) < rmse(m.y[n:], m.pred_ols[n:])
    t.p("On unseen years the SVD model is " + ("more accurate and also more stable." if better else
        "not more accurate here; its advantage is stability."))
    t.p("Neither can foresee a big shock like 2020: a linear model only")
    t.p("captures average relationships.")


def plot4(fig, m):
    ax = fig.add_subplot(111)
    ax.axvspan(m.years[m.ntr] - 0.5, m.years[-1] + 0.5, color="#FFF4DB", zorder=0)
    ax.plot(m.years, m.y, "ko-", ms=3.5, lw=1.4, label="actual GDP growth")
    ax.plot(m.years, m.pred_ols, "--", color=BLUE, label="least squares")
    ax.plot(m.years, m.pred_svd, color=ORANGE, lw=1.8, label=f"SVD (k={m.k})")
    ax.text(m.years[m.ntr], 0.97, " unseen years", va="top", fontsize=8, color=GREY, transform=ax.get_xaxis_transform())
    ax.set(ylabel="GDP growth (%)", title="Actual vs predicted GDP growth")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.25)


STEPS = [("1  Set up Ax = b", step1, plot1), ("2  Least squares", step2, plot2),
         ("3  SVD", step3, plot3), ("4  Prediction", step4, plot4)]


class Page(ttk.Frame):
    def __init__(self, parent, step, plot):
        super().__init__(parent)
        self.step, self.plot = step, plot
        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=8)
        self.box = tk.Text(panes, width=10, height=10, font=("Consolas", 10), wrap="none", relief="flat",
                           bg="#FBFCFE", padx=12, pady=10)
        self.box.tag_configure("h", font=("Consolas", 11, "bold"), foreground=NAVY, spacing1=8)
        self.fig = Figure(figsize=(5, 4), layout="constrained")
        right = ttk.Frame(panes)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        w = self.canvas.get_tk_widget()
        w.configure(width=60, height=60)
        w.pack(fill="both", expand=True)
        for ev in ("<Configure>", "<Map>"):                              # keep figure = widget size (HiDPI)
            w.bind(ev, lambda e: w.after(40, self._fit), add="+")
        panes.add(self.box, weight=1)
        panes.add(right, weight=1)

    def _fit(self):
        w, h = self.canvas.get_tk_widget().winfo_width(), self.canvas.get_tk_widget().winfo_height()
        d = self.fig.dpi
        if w > 50 and (abs(self.fig.get_figwidth() * d - w) > 1 or abs(self.fig.get_figheight() * d - h) > 1):
            self.fig.set_size_inches(w / d, h / d, forward=False)
            self.canvas.draw_idle()

    def h(self, s):
        self.box.insert("end", s + "\n", "h")

    def p(self, s):
        self.box.insert("end", s + "\n")

    def show(self, m):
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")
        self.step(m, self)
        self.box.configure(state="disabled")
        self.fig.clear()
        self.plot(self.fig, m)
        self.canvas.draw_idle()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Predicting GDP growth with linear algebra")
        self.geometry("1150x640")
        self.k = tk.IntVar(value=2)
        bar = tk.Frame(self, bg=NAVY)
        bar.pack(fill="x")
        tk.Label(bar, text="Predicting GDP growth with linear algebra", bg=NAVY, fg="white",
                 font=("Segoe UI", 15, "bold")).pack(side="left", padx=14, pady=10)
        ttk.Button(bar, text="Load CSV...", command=self.pick).pack(side="right", padx=10)
        self.note = tk.Label(self, anchor="w", padx=14, font=("Segoe UI", 9, "bold"))
        self.note.pack(fill="x")
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.pages = [Page(nb, s, p) for _, s, p in STEPS]
        for (title, _, _), pg in zip(STEPS, self.pages):
            nb.add(pg, text=title)
        ctl = tk.Frame(self)
        ctl.pack(fill="x", pady=4)
        tk.Label(ctl, text="SVD directions kept (k):", font=("Segoe UI", 10)).pack(side="left", padx=(14, 6))
        self.spin = ttk.Spinbox(ctl, from_=1, to=4, width=4, textvariable=self.k, command=self.refresh)
        self.spin.pack(side="left")
        self.load(*demo_data())

    def load(self, years, y, X, names, source):
        self.data = (years, y, X, names)
        synthetic = source.startswith("Synthetic")
        self.note.configure(text=("  DEMO: synthetic data, not real figures. Use 'Load CSV' for real data."
                                  if synthetic else "  " + source),
                            bg="#FFF4DB" if synthetic else "#E3F4E6", fg="#7A4B00" if synthetic else "#1B5E20")
        self.spin.configure(to=X.shape[1])
        self.k.set(min(2, X.shape[1]))
        self.refresh()

    def refresh(self):
        try:
            k = max(1, min(int(self.k.get()), self.data[2].shape[1]))
            m = Model(*self.data, k)
        except Exception as ex:
            messagebox.showerror("Cannot analyse this data", str(ex))
            return
        for pg in self.pages:
            pg.show(m)

    def pick(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            try:
                self.load(*load_csv(path))
            except Exception as ex:
                messagebox.showerror("Could not read the CSV", f"{ex}\n\nExpected: Year, GDP growth, then indicator columns.")


if __name__ == "__main__":
    try:                                       # sharp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()

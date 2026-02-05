import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit, prange
from multiprocessing import Pool, cpu_count

# Work in geometric units: set G = c = 1 in formulas, and use M = M_geom (meters)
G = 1.0
c = 1.0
M = 1.0      # M in black hole mass units
r_s = 2*M


#DEFINES COVARIANT METRIC RETURNING IT AT SOME POSITION
def g_covariant(x, rs=r_s):
    t, r, theta, phi = x
    g = np.zeros((4,4), dtype=float)
    g[0,0] = -(1-rs/r)
    g[1,1] = ((1-rs/r)**(-1))
    g[2,2] = (r**2)
    g[3,3] = ((r**2)*(np.sin(theta))**2)
    return g

#DEFINES CONTRAVARIANT METRIC RETURNING IT AT SOME POSITION
def g_contravariant(x, rs=r_s):
    t, r, theta, phi = x
    g = np.zeros((4,4), dtype=float)
    g[0,0] = -1/(1-rs/r)
    g[1,1] = (1-rs/r)
    g[2,2] = 1/(r**2)
    g[3,3] = 1/((r**2)*(np.sin(theta))**2)
    return g












#COMPUTES THE CHRISTOFFEL COEFICIENTS FOR A GIVEN POSITION
def christoffel_schwarzschild(x, rs=r_s):

    t, r, theta, phi = x
    f = 1-rs/r
    Gam = np.zeros((4,4,4), dtype=float) #CALCULATED BY HAND AND AI
    Gam[1,0,0] = 0.5*(f)*(rs/r**2)
    Gam[0,0,1] = 0.5*(rs/r**2)/(f)
    Gam[0,1,0] = Gam[0,0,1]
    Gam[1,1,1] = -0.5*(rs/r**2)/(f)
    Gam[1,2,2] = -r*(f)
    Gam[1,3,3] = -r*(f)*(np.sin(theta)**2)
    Gam[2,1,2] = 1.0/r
    Gam[2,2,1] = Gam[2,1,2]
    Gam[2,3,3] = -np.sin(theta)*np.cos(theta)
    Gam[3,1,3] = 1.0/r
    Gam[3,3,1] = Gam[3,1,3]
    Gam[3,2,3] = np.cos(theta)/np.sin(theta)
    Gam[3,3,2] = Gam[3,2,3]
    return Gam

def null_constraint(x, p):
    g = g_covariant(x)
    return float(np.dot(p, g.dot(p)))





# ============================================================
# 2. Numba-accelerated Christoffels and RHS
# ============================================================

@njit
def christoffel_schwarzschild_nb(x, rs):
    """
    Numba version of Christoffel symbols Γ^μ_{αβ} for Schwarzschild.
    x = (t, r, θ, φ)"""


    t, r, theta, phi = x
    f = 1-rs/r
    Gam = np.zeros((4,4,4), dtype=float) #CALCULATED BY HAND AND AI
    Gam[1,0,0] = 0.5*(f)*(rs/r**2)
    Gam[0,0,1] = 0.5*(rs/r**2)/(f)
    Gam[0,1,0] = Gam[0,0,1]
    Gam[1,1,1] = -0.5*(rs/r**2)/(f)
    Gam[1,2,2] = -r*(f)
    Gam[1,3,3] = -r*(f)*(np.sin(theta)**2)
    Gam[2,1,2] = 1.0/r
    Gam[2,2,1] = Gam[2,1,2]
    Gam[2,3,3] = -np.sin(theta)*np.cos(theta)
    Gam[3,1,3] = 1.0/r
    Gam[3,3,1] = Gam[3,1,3]
    Gam[3,2,3] = np.cos(theta)/np.sin(theta)
    Gam[3,3,2] = Gam[3,2,3]
    return Gam


@njit
def geodesic_rhs_core(x, p, rs):
    """
    Numba core for geodesic RHS.

    Inputs:
        x : array(4,)  = [t, r, θ, φ]
        p : array(4,)  = [p^t, p^r, p^θ, p^φ]

    Output:
        dydλ : array(8,) = [dx^μ/dλ, dp^μ/dλ]
    """
    Gam = christoffel_schwarzschild_nb(x, rs)

    dx = np.empty(4)
    dp = np.empty(4)

    # dx^μ = p^μ
    for mu in range(4):
        dx[mu] = p[mu]

    # dp^μ = -Γ^μ_{αβ} p^α p^β
    for mu in range(4):
        s = 0.0
        for a in range(4):
            pa = p[a]
            for b in range(4):
                pb = p[b]
                s += Gam[mu, a, b] * pa * pb
        dp[mu] = -s

    out = np.empty(8)
    for i in range(4):
        out[i] = dx[i]
        out[4 + i] = dp[i]

    return out


@njit
def rk4_step_nb(x, p, h, rs):
    """
    One RK4 step in affine parameter λ for (x, p).

    x, p : arrays of shape (4,)
    h    : step size in λ
    rs   : Schwarzschild radius
    """
    # k1
    k1 = geodesic_rhs_core(x, p, rs)
    # k1x = k1[:4], k1p = k1[4:]

    x2 = np.empty(4)
    p2 = np.empty(4)
    for i in range(4):
        x2[i] = x[i] + 0.5 * h * k1[i]
        p2[i] = p[i] + 0.5 * h * k1[4 + i]

    # k2
    k2 = geodesic_rhs_core(x2, p2, rs)

    x3 = np.empty(4)
    p3 = np.empty(4)
    for i in range(4):
        x3[i] = x[i] + 0.5 * h * k2[i]
        p3[i] = p[i] + 0.5 * h * k2[4 + i]

    # k3
    k3 = geodesic_rhs_core(x3, p3, rs)

    x4 = np.empty(4)
    p4 = np.empty(4)
    for i in range(4):
        x4[i] = x[i] + h * k3[i]
        p4[i] = p[i] + h * k3[4 + i]

    # k4
    k4 = geodesic_rhs_core(x4, p4, rs)

    x_new = np.empty(4)
    p_new = np.empty(4)
    for i in range(4):
        x_new[i] = x[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
        p_new[i] = p[i] + (h / 6.0) * (k1[4 + i] + 2.0 * k2[4 + i] + 2.0 * k3[4 + i] + k4[4 + i])

    return x_new, p_new


@njit
def trace_ray_fast_nb(x0, p0, r_far, max_steps, h_base, rs):
    """
    Semi-adaptive RK4 integrator for ray classification.

    Returns:
        0 = escaped
        1 = horizon
        2 = maxsteps
    """
    x = x0.copy()
    p = p0.copy()
    L = 0
    for _ in range(max_steps):
        r = x[1]

        # Horizon detection
        if r <= rs:
            return 1, L

        # Escape detection
        if r >= r_far:
            return 0, L

        # -------------------------
        # Semi-adaptive step size
        # -------------------------

        h = h_base

        # Strong curvature region near photon sphere and horizon
        if r < 6.0:
            h *= 0.25
        if r < 4.0:
            h *= 0.1

        # Turning point detection (p^r ≈ 0)
        if abs(p[1]) < 0.02:
            h *= 0.2

        # Curvature spike detector (θ, φ change too fast not currently measured but we can add)
        
        # Single RK4 step
        x_prev = x.copy()
        x, p = rk4_step_nb(x, p, h, rs)
        t_prev, r_prev, theta_prev, phi_prev = x_prev
        t_new,  r_new,  theta_new,  phi_new  = x
        # Numerical failure check
        if not np.isfinite(x[1]):
            return 1, L  # assume plunged


        # disk vertical condition
        is_in_prev = (r_in <= r_prev <= r_out) and (abs(theta_prev - np.pi/2) <= disk_half_thickness)
        is_in_new  = (r_in <= r_new  <= r_out) and (abs(theta_new  - np.pi/2) <= disk_half_thickness)

        if is_in_prev or is_in_new:
            return 2, float(emissivity_nb(r_new))
        
    return 3, L

def classify_ray_fast(x0, p0, r_far, h=0.05, max_steps=200000):
    return trace_ray_fast_nb(x0, p0, r_far, max_steps, h, r_s)

@njit
def trace_ray_fast_nb_store(x0, p0, r_far, max_steps, h_base, rs):
    x = x0.copy()
    p = p0.copy()

    traj = np.zeros((max_steps, 4))
    
    for i in range(max_steps):
        traj[i, :] = x[:]  # store position only

        r = x[1]
        if r <= rs:
            return traj[:i+1], 1  # horizon
        if r >= r_far:
            return traj[:i+1], 0  # escaped

        h = h_base
        if r < 6.0: h *= 0.25
        if r < 4.0: h *= 0.1
        if abs(p[1]) < 0.02: h *= 0.2

        x, p = rk4_step_nb(x, p, h, rs)

        if not np.isfinite(x[1]):
            return traj[:i+1], 1

    return traj, 2  # max steps


def trace_ray_fast_with_path(x0, p0, r_far, h=0.02, max_steps=200000):
    traj, code = trace_ray_fast_nb_store(x0, p0, r_far, max_steps, h, r_s)
    status = ["escaped", "horizon", "maxsteps"][code]
    return {"x": traj, "status": status}


def geodesic_rhs_lambda(lam, y):

    """
    Wrapper for SciPy: y = [t, r, θ, φ, p^t, p^r, p^θ, p^φ].
    Heavy maths is in geodesic_rhs_core (Numba).
    """
    x = y[:4]
    p = y[4:]
    return geodesic_rhs_core(x, p, r_s)


EPS_HORIZON = 1e-6 * r_s
def event_horizon(lam, y):
    # zero when r = r_s, we want to detect inward crossing
    r = y[1]
    return r - (r_s+EPS_HORIZON)
event_horizon.terminal  = True
event_horizon.direction = -1   # only trigger when r decreases through r_s

disk_half_thickness = np.pi/80
r_in = 6.0 * M
r_out = 12.0 * M


@njit
def emissivity_nb(r):
    return r**(-2.0)



def event_far(lam, y, r_far):
    # zero when r = r_far, detect outward crossing
    r = y[1]
    return r - r_far
# we'll attach terminal/direction when we wrap this in a closure below

def trace_ray_scipy(x0, p0, r_far, lam_max=500.0,rtol=1e-7, atol=1e-7, max_step=0.5):
    """
    Adaptive RK45 tracer using scipy.solve_ivp.

    Parameters
    ----------
    x0 : array-like, shape (4,)
        Initial coordinates [t, r, theta, phi].
    p0 : array-like, shape (4,)
        Initial 4-momentum components [pt, pr, ptheta, pphi].
    r_far : float
        Radius at which we declare 'escaped'.
    lam_max : float
        Maximum affine parameter to integrate to.
    rtol, atol : float
        Relative and absolute tolerances for solve_ivp.
    max_step : float
        Max step in lambda.

    Returns
    -------
    result : dict with keys
        'status' : 'horizon', 'escaped', or 'lam_max'
        'x'      : array shape (N,4) of coordinates along the ray
        'p'      : array shape (N,4) of momenta along the ray
    """
    y0 = np.concatenate((x0, p0))

    # make a closure for the far event with this r_far
    def _event_far(lam, y):
        return event_far(lam, y, r_far)
    _event_far.terminal  = True
    _event_far.direction = +1

    sol = solve_ivp(geodesic_rhs_lambda,(0.0, lam_max),y0,method="RK45",rtol=rtol,atol=atol,max_step=max_step,events=[event_horizon, _event_far],dense_output=False,)

    # trajectory arrays
    y = sol.y.T

    # work out why it stopped
    status = "lam_max"

    for k in range(len(y)):
        r     = y[k, 1]
        theta = y[k, 2]

        if (r_in <= r <= r_out) and (abs(theta - np.pi/2) <= disk_half_thickness):
            # ray entered the disk volume
            L = float(emissivity_nb(r))
            return 2, L

    if sol.status == 1:
        if len(sol.t_events[0]) > 0:
            # horizon event
            return 1, 0
        if len(sol.t_events[1]) > 0:
            # far-away event
            return 0, 0
    # Disk check
    


# Otherwise lam_max
    return 3, 0



def trace_ray_hybrid(x0, p0, r_far,h_fast=0.02, max_steps_fast=200000,lam_max_scipy=500.0,rtol=1e-7, atol=1e-7, max_step=0.5, b_crit = 3.0 * np.sqrt(3.0) * M,
    delta_b = 0.1 * M ):
    """
    Hybrid ray tracer:
      - Uses fast RK4 for large-|b| rays (far from photon sphere)
      - Uses SciPy adaptive integrator near the critical impact parameter.

    Returns a dict:
      {
        'status': 'escaped' / 'horizon' / 'lam_max' / 'maxsteps',
        'x': trajectory or None,
        'b': impact_parameter,
        'backend': 'fast' or 'scipy'
      }
    """
    # Compute impact parameter
    E, L = energy_and_angular_momentum(x0, p0)
    b = L / E

    # Decide which backend to use
    use_fast = (abs(b) > b_crit + delta_b)

    if use_fast:
        code = classify_ray_fast(x0, p0, r_far, h=h_fast, max_steps=max_steps_fast)
    else:
        code = trace_ray_scipy(x0, p0, r_far=r_far,lam_max=lam_max_scipy, rtol=rtol,atol=atol, max_step=max_step)
    return code





@njit
def static_observer_tetrad_nb(r0, theta0, phi0):
    rs = r_s  # global or pass as argument

    # 4-velocity
    u_t = 1.0 / np.sqrt(1 - rs/r0)
    u = np.array([u_t, 0.0, 0.0, 0.0])

    # spatial basis vectors (coordinate components)
    e_r     = np.array([0.0, np.sqrt(1 - rs/r0), 0.0, 0.0])
    e_theta = np.array([0.0, 0.0, 1.0/r0, 0.0])
    e_phi   = np.array([0.0, 0.0, 0.0, 1.0/(r0*np.sin(theta0))])

    return u, e_r, e_theta, e_phi

def photon_momentum_from_angles(u_cam, e_r, e_theta, e_phi, alpha_deg, beta_deg=0.0):
    """
    Build photon 4-momentum from camera tetrad and screen angles.

    alpha: angle on camera screen (in degrees) in φ-direction
    beta:  tilt out of equatorial plane (not really used yet)
    """
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)

    # spatial unit direction in tetrad frame (n^hat{i})
    nhat = np.array([
        -np.cos(beta) * np.cos(alpha),   # along -e_r
        np.sin(beta),                    # e_theta
        np.cos(beta) * np.sin(alpha)     # e_phi
    ])

    # photon with local energy E=1: p^{hat{μ}} = (1, n^hat{i})
    p = (
        u_cam * 1.0
        + e_r     * nhat[0]
        + e_theta * nhat[1]
        + e_phi   * nhat[2]
    )
    return p

def energy_and_angular_momentum(x0, p0):
    """Compute conserved E and L_z for Schwarzschild (up to sign)."""
    g = g_covariant(x0)
    # p_t = g_tt * p^t, so E = -p_t
    p_t = g[0, 0] * p0[0]
    E = -p_t
    # p_phi = g_phi,phi * p^phi
    p_phi = g[3, 3] * p0[3]
    L = p_phi
    return E, L

@njit
def camera_ray_direction_nb(i, j, Nx, Ny, fov_x, fov_y, forward, right, up):
    # Normalized Device Coordinates in [-0.5, 0.5]
    x_ndc = (i+0.5)/Nx - 0.5
    y_ndc = (j+0.5)/Ny - 0.5

    angle_x = x_ndc * fov_x
    angle_y = y_ndc * fov_y

    # coefficients in the orthonormal basis
    a = 1.0
    b = np.tan(angle_x)
    c = np.tan(angle_y)

    # normalise in coefficient space: g(n,n) = a^2 + b^2 + c^2
    norm = np.sqrt(a*a + b*b + c*c)
    a /= norm
    b /= norm
    c /= norm

    dir_vec = a*forward + b*right + c*up
    return dir_vec




def shoot_rays(N, cam_pos, alphas, betas, r_far=300, h_fast=0.02, max_steps_fast=200000, b_crit=1.5*np.sqrt(3)*r_s):
    t0 = cam_pos[0]
    camera_r, camera_theta, camera_phi = cam_pos[1], cam_pos[2], cam_pos[3]

    x_cam = np.array([t0, camera_r, camera_theta, camera_phi], float)
    u_cam, e_r, e_theta, e_phi = static_observer_tetrad_nb(camera_r, camera_theta, camera_phi)

    trajs = []
    stats = []
    b_vals = []
    backends = []

    for i in range(N):
        alpha = alphas[i]
        beta  = betas[i]

        # camera position and tetrad assumed already defined
        x0 = x_cam.copy()
        p0 = photon_momentum_from_angles(u_cam, e_r, e_theta, e_phi,
                                     alpha_deg=alpha, beta_deg=beta)

        res = trace_ray_hybrid(x0, p0, r_far,
                           h_fast=0.02, max_steps_fast=200000,
                           lam_max_scipy=500.0, b_crit=0, delta_b=0)

        trajs.append(res["x"])          # may be None for fast rays
        stats.append(res["status"])
        b_vals.append(res["b"])
        backends.append(res["backend"])

        print(i,"status =", res["status"],"backend =", res["backend"],"b =", res["b"])
    return x_cam, trajs, stats, np.array(b_vals)

def plot_bundle(x_cam, trajectories, statuses, N, camera_r):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    for traj, status in zip(trajectories, statuses):

        # ----------------------------------------------------
        # Skip rays that used the fast integrator (traj=None)
        # ----------------------------------------------------
        if traj is None:
            continue

        r_vals = traj[:, 1]
        theta_vals = traj[:, 2]
        phi_vals = traj[:, 3]

        x_vals = r_vals * np.sin(theta_vals) * np.cos(phi_vals)
        y_vals = r_vals * np.sin(theta_vals) * np.sin(phi_vals)
        z_vals = r_vals * np.cos(theta_vals)

        if status == "horizon":
            color = "red"
        elif status == "escaped":
            color = "C0"
        else:
            color = "gray"

        ax.plot(x_vals, y_vals, z_vals, alpha=0.8, color=color)

    # ----------------------------------------------------
    # Plot the event horizon as a sphere
    # ----------------------------------------------------
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    xh = r_s * np.outer(np.cos(u), np.sin(v))
    yh = r_s * np.outer(np.sin(u), np.sin(v))
    zh = r_s * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xh, yh, zh, alpha=0.4, linewidth=0, color="steelblue")

    # ----------------------------------------------------
    # Camera location
    # ----------------------------------------------------
    _, r_cam, th_cam, ph_cam = x_cam
    cam_x = r_cam * np.sin(th_cam) * np.cos(ph_cam)
    cam_y = r_cam * np.sin(th_cam) * np.sin(ph_cam)
    cam_z = r_cam * np.cos(th_cam)
    ax.scatter([cam_x], [cam_y], [cam_z], s=50, color="k")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{N} photons from r={camera_r}")

    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_zlim(-30, 30)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=40)

    plt.tight_layout()
    plt.show()




def render_row(args):
    """
    Worker function for multiprocessing.
    Computes one image row j.
    """
    (j, Nx, Ny, fov, x0, u_cam, e_r, e_theta, e_phi,
     r_far, h_fast, max_steps_fast, b_crit, delta_b,
     col_hole, col_bg) = args

    forward = -e_r
    right   = e_phi
    up      = e_theta

    row = np.zeros((Nx, 3), dtype=float)

    for i in range(Nx):
        # local direction for this pixel
        n_hat = camera_ray_direction_nb(i, j, Nx, Ny, fov, fov, forward, right, up)

        # photon 4-momentum: p = u + n_hat
        p0 = u_cam + n_hat

        # hybrid tracer (fast or SciPy)
        code, L = trace_ray_hybrid(
            x0, p0, r_far,
            h_fast=h_fast,
            max_steps_fast=max_steps_fast,
            lam_max_scipy=500.0,
            rtol=1e-6,
            atol=1e-6,
            max_step=0.5,
            b_crit=0, 
            delta_b=0
        )

        # classify + colour
        if code == 1:      # horizon
            color = col_hole
        elif code == 0:    # escaped
            color = col_bg
        elif code == 2:    # hit disc
            bright = min(1.0, 10.0 * L)
            color = np.array([bright, 0.3 * bright, 0.1 * bright])
        else:
            color = np.array([0.0, 0.0, 1.0])  # debug (should be rare)

        row[i, :] = color

    return j, row



if __name__ == "__main__":
    N = 100
    Nx = 250
    Ny = 250
    fov_deg = 90.0
    fov = np.deg2rad(fov_deg)

    camera_4pos = [0, 25, (np.pi/2)-(np.pi/8), np.pi/6]
    r_far = 100.0

    b_crit = 3.0 * np.sqrt(3.0) * M
    delta_b = 0.1 * M   # narrower SciPy band

    col_hole = np.array([0.0, 0.0, 0.0])
    col_bg   = np.array([0.05, 0.05, 0.1])

    img = np.zeros((Ny, Nx, 3), dtype=float)

    # camera tetrad and initial 4-position
    u_cam, e_r, e_theta, e_phi = static_observer_tetrad_nb(
        camera_4pos[1], camera_4pos[2], camera_4pos[3]
    )
    x0 = np.array(camera_4pos, dtype=float)

    h_fast = 0.02
    max_steps_fast = 200_000

    # Build argument list for each row
    jobs = []
    for j in range(Ny):
        jobs.append((
            j, Nx, Ny, fov, x0, u_cam, e_r, e_theta, e_phi,
            r_far, h_fast, max_steps_fast, b_crit, delta_b,
            col_hole, col_bg
        ))

    # Use all available CPU cores (or set processes=<some number>)
    with Pool(processes=cpu_count()) as pool:
        for j, row in pool.imap_unordered(render_row, jobs):
            # place row into image, flipping vertically
            img[Ny - 1 - j, :, :] = row

    # Show image
    plt.figure(figsize=(6, 6))
    plt.imshow(img, origin="lower")
    plt.axis("off")
    plt.title(
        f"Schwarzschild BH with thin emitting ring: r={camera_4pos[1]}, "
        f"theta={camera_4pos[2]}, phi={camera_4pos[3]}"
    )
    plt.show()

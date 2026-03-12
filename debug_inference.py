#!/usr/bin/env python3
"""Step-by-step inference debug — reproduces the Rust Jacobi fixed-point."""

import numpy as np
from scipy.special import digamma, polygamma, gammaln

# ══════════════════════════════════════════════════════════════════════
# Distribution helpers
# ══════════════════════════════════════════════════════════════════════

def gaussian_expected_T(eta1, eta2):
    """E[T] = (E[x], E[x²]) for Gaussian."""
    sigma2 = -1.0 / (2.0 * eta2)
    mu = eta1 * sigma2
    return np.array([mu, mu**2 + sigma2])

def gaussian_fisher(eta1, eta2):
    """Fisher info for Gaussian."""
    sigma2 = -1.0 / (2.0 * eta2)
    mu = eta1 * sigma2
    return np.array([
        [sigma2,       2*mu*sigma2],
        [2*mu*sigma2,  2*sigma2**2 + 4*mu**2*sigma2]
    ])

def beta_expected_T(eta):
    """E[T] = (E[ln x], E[ln(1-x)]) for Beta/Dirichlet K=2.
    eta = (α-1, β-1), so α = eta[0]+1, β = eta[1]+1."""
    a = eta[0] + 1.0
    b = eta[1] + 1.0
    psi_ab = digamma(a + b)
    return np.array([digamma(a) - psi_ab, digamma(b) - psi_ab])

def beta_fisher(eta):
    """Fisher info for Beta."""
    a = eta[0] + 1.0
    b = eta[1] + 1.0
    tri_a = float(polygamma(1, a))
    tri_b = float(polygamma(1, b))
    tri_ab = float(polygamma(1, a + b))
    return np.array([
        [tri_a - tri_ab, -tri_ab],
        [-tri_ab,         tri_b - tri_ab]
    ])

def beta_canonical(eta):
    """Return (α, β, mean, std) from eta."""
    a = eta[0] + 1.0
    b = eta[1] + 1.0
    m = a / (a + b) if (a + b) > 0 else 0.5
    v = (a * b) / ((a + b)**2 * (a + b + 1)) if (a + b + 1) > 0 else 0
    return a, b, m, np.sqrt(max(v, 0))

def gaussian_canonical(eta1, eta2):
    """Return (mu, sigma2) from eta."""
    sigma2 = -1.0 / (2.0 * eta2)
    mu = eta1 * sigma2
    return mu, sigma2

# ══════════════════════════════════════════════════════════════════════
# Node definitions — from nodes.yaml
# ══════════════════════════════════════════════════════════════════════

EPS = 1e-8  # NATURAL_PARAM_EPS from constants.rs

nodes = {
    "temperature": {"family": "gaussian", "eta": np.array([37.0/0.25, -1.0/(2*0.25)]), "tau": 7.0},
    "headache":    {"family": "beta", "eta": np.array([1.5-1, 13.5-1]), "tau": 3.0},
    "sore_throat": {"family": "beta", "eta": np.array([1.0-1, 19.0-1]), "tau": 5.0},
    "body_aches":  {"family": "beta", "eta": np.array([1.0-1, 19.0-1]), "tau": 3.0},
    "flu":         {"family": "beta", "eta": np.array([6.0-1, 34.0-1]), "tau": 14.0},
    "tonsillitis": {"family": "beta", "eta": np.array([2.0-1, 18.0-1]), "tau": 10.0},
}

node_names = ["temperature", "headache", "sore_throat", "body_aches", "flu", "tonsillitis"]

# Initialize: epidemio = prev = relax = post = eta (since delta_t=0, relax=prev)
for n in nodes.values():
    n["relax"] = n["eta"].copy()
    n["post"] = n["eta"].copy()

# ══════════════════════════════════════════════════════════════════════
# Edges — from _calibration_result.yaml + edges.yaml
# Calibration result edges are stored as (node_a, node_b) with B_ab
# But inline edges from nodes.yaml declare (declaring_node, neighbor)
# The validate.rs fix transposes when merging reversed order.
#
# Let's just store edges as (node_a, node_b, B) where B is d_a × d_b.
# The adjacency list will handle transpose.
# ══════════════════════════════════════════════════════════════════════

# From _calibration_result.yaml — these are the "raw" edges as emitted
# by the calibration solver. node_a is the TARGET (symptom), node_b is
# the SOURCE (disease).
calibration_edges_raw = {
    ("temperature", "flu"):         np.array([[3.14e-05, -2.1715], [-0, -0]]),
    ("temperature", "tonsillitis"): np.array([[1.89e-05, -1.3029], [-0, -0]]),
    ("headache", "flu"):            np.array([[4.72e-05, -3.2572], [-4.72e-05, 3.2572]]),
    ("headache", "tonsillitis"):    np.array([[9.44e-06, -0.6514], [-9.44e-06, 0.6514]]),
    ("sore_throat", "tonsillitis"): np.array([[9.44e-05, -6.5145], [-9.44e-05, 6.5145]]),
    ("sore_throat", "flu"):         np.array([[6.29e-06, -0.4343], [-6.29e-06, 0.4343]]),
    ("body_aches", "flu"):          np.array([[8.18e-05, -5.6459], [-8.18e-05, 5.6459]]),
    ("flu", "tonsillitis"):         np.array([[-3.79e-03, 1.7374], [3.79e-03, -1.7374]]),
}

# From edges.yaml — symptom↔symptom edge
extra_edges_raw = {
    ("temperature", "headache"): np.array([[0.005, 0.0], [0.0, 0.0]]),
}

# Now, the inline edges in nodes.yaml are:
# flu declares coupled_with: [temperature, headache, body_aches, tonsillitis]
#   → inline edges: (flu, temperature), (flu, headache), (flu, body_aches), (flu, tonsillitis)
# tonsillitis declares coupled_with: [temperature, sore_throat, headache]
#   → inline edges: (tonsillitis, temperature), (tonsillitis, sore_throat), (tonsillitis, headache)
#
# When merging calibration edge (temperature, flu) into inline (flu, temperature),
# the FIX transposes: B_inline = B_calibration^T
#
# Let's build the final edge list as the Rust code does:

edges = []  # Each: (node_a, node_b, B_ab) where B is d_a × d_b

# 1. Inline edges from flu's coupled_with
inline_from_flu = ["temperature", "headache", "body_aches", "tonsillitis"]
for neighbor in inline_from_flu:
    # Look for calibration edge in either order
    key_fwd = ("flu", neighbor)
    key_rev = (neighbor, "flu")
    if key_fwd in calibration_edges_raw:
        B = calibration_edges_raw[key_fwd]
        edges.append(("flu", neighbor, B))
    elif key_rev in calibration_edges_raw:
        # Reversed! Must transpose
        B = calibration_edges_raw[key_rev].T
        edges.append(("flu", neighbor, B))
    else:
        print(f"WARNING: no coupling for flu-{neighbor}")

# 2. Inline edges from tonsillitis's coupled_with
inline_from_tonsillitis = ["temperature", "sore_throat", "headache"]
for neighbor in inline_from_tonsillitis:
    key_fwd = ("tonsillitis", neighbor)
    key_rev = (neighbor, "tonsillitis")
    if key_fwd in calibration_edges_raw:
        B = calibration_edges_raw[key_fwd]
        edges.append(("tonsillitis", neighbor, B))
    elif key_rev in calibration_edges_raw:
        B = calibration_edges_raw[key_rev].T
        edges.append(("tonsillitis", neighbor, B))
    else:
        print(f"WARNING: no coupling for tonsillitis-{neighbor}")

# 3. Extra edge from edges.yaml: (temperature, headache)
edges.append(("temperature", "headache", extra_edges_raw[("temperature", "headache")]))

print(f"Total edges: {len(edges)}")
for a, b, B in edges:
    print(f"  ({a}, {b}): B =")
    for row in B:
        print(f"    [{row[0]:+12.6f}, {row[1]:+12.6f}]")

# ══════════════════════════════════════════════════════════════════════
# Build adjacency list
# ══════════════════════════════════════════════════════════════════════

# For each node, list of (neighbor_name, B_matrix, transposed)
# If edge is (A,B) with B_AB: A sees (B, B_AB, False), B sees (A, B_AB, True)
adjacency = {n: [] for n in node_names}
for edge_idx, (a, b, B_ab) in enumerate(edges):
    adjacency[a].append((b, B_ab, False))
    adjacency[b].append((a, B_ab, True))

# ══════════════════════════════════════════════════════════════════════
# Observations — from patient_flu.yaml + instruments.yaml
# ══════════════════════════════════════════════════════════════════════

# thermometre → temperature: gaussian_noise, noise_var=0.04, value=39.5
#   η_obs = (39.5/0.04, -1/(2*0.04)) = (987.5, -12.5)
# questionnaire_cephalee → headache: beta_obs, kappa=50, value=0.999
#   η_obs = (50*0.999 - 1, 50*(1-0.999) - 1) = (48.95, -0.95)
# questionnaire_gorge → sore_throat: beta_obs, kappa=50, value=0.001
#   η_obs = (50*0.001 - 1, 50*0.999 - 1) = (-0.95, 48.95)
# questionnaire_courbatures → body_aches: beta_obs, kappa=50, value=0.999
#   η_obs = (50*0.999 - 1, 50*(1-0.999) - 1) = (48.95, -0.95)

observations = {
    "temperature": [np.array([39.5/0.04, -1.0/(2*0.04)])],
    "headache":    [np.array([50*0.999 - 1, 50*0.001 - 1])],
    "sore_throat": [np.array([50*0.001 - 1, 50*0.999 - 1])],
    "body_aches":  [np.array([50*0.999 - 1, 50*0.001 - 1])],
    "flu":         [],
    "tonsillitis": [],
}

print("\nObservations:")
for name, obs_list in observations.items():
    for obs in obs_list:
        print(f"  {name}: η_obs = {obs}")

# ══════════════════════════════════════════════════════════════════════
# Helper: compute E[T] and Fisher for a node given its eta
# ══════════════════════════════════════════════════════════════════════

def expected_T(name, eta):
    fam = nodes[name]["family"]
    if fam == "gaussian":
        return gaussian_expected_T(eta[0], eta[1])
    else:
        return beta_expected_T(eta)

def fisher(name, eta):
    fam = nodes[name]["family"]
    if fam == "gaussian":
        return gaussian_fisher(eta[0], eta[1])
    else:
        return beta_fisher(eta)

def spectral_norm(M):
    return np.linalg.svd(M, compute_uv=False).max()

def project(name, eta):
    """Clamp to valid domain."""
    fam = nodes[name]["family"]
    if fam == "gaussian":
        eta[1] = min(eta[1], -EPS)  # η₂ < 0
    elif fam == "beta":
        eta[0] = max(eta[0], -1.0 + EPS)  # α-1 > -1
        eta[1] = max(eta[1], -1.0 + EPS)
    return eta

def describe(name, eta):
    fam = nodes[name]["family"]
    if fam == "gaussian":
        mu, s2 = gaussian_canonical(eta[0], eta[1])
        return f"mu={mu:.4f}, σ²={s2:.4f}"
    else:
        a, b, m, s = beta_canonical(eta)
        return f"α={a:.4f}, β={b:.4f}, mean={m:.4f}±{s:.4f}"

# ══════════════════════════════════════════════════════════════════════
# Inference loop — Jacobi fixed-point with damping
# ══════════════════════════════════════════════════════════════════════

entropy_scale = 1.0
MIN_ALPHA = 0.001

# Initialize buffers
ET_buffer = {}
Fisher_buffer = {}
post = {}
for name in node_names:
    post[name] = nodes[name]["post"].copy()
    ET_buffer[name] = expected_T(name, post[name])
    Fisher_buffer[name] = fisher(name, post[name])

# Denominators
denoms = {}
for name in node_names:
    n_obs = len(observations.get(name, []))
    denoms[name] = 1.0 + n_obs + entropy_scale

print("\n" + "="*80)
print("INITIAL STATE")
print("="*80)
for name in node_names:
    print(f"  {name:15s}: η={post[name]}, E[T]={ET_buffer[name]}, denom={denoms[name]}")
    print(f"  {'':15s}  {describe(name, post[name])}")

NUM_ITERS = 5

for iter_num in range(NUM_ITERS):
    print(f"\n{'='*80}")
    print(f"ITERATION {iter_num}")
    print(f"{'='*80}")
    
    ET_read = {k: v.copy() for k, v in ET_buffer.items()}
    Fisher_read = {k: v.copy() for k, v in Fisher_buffer.items()}
    
    new_post = {}
    new_ET = {}
    new_Fisher = {}
    
    max_change = 0.0
    
    for name in node_names:
        denom = denoms[name]
        relax = nodes[name]["relax"]
        
        # ── Compute damping α ──
        sum_j_norms = 0.0
        for (neighbor, B_ab, transposed) in adjacency[name]:
            B = B_ab.T if transposed else B_ab
            F_j = Fisher_read[neighbor]
            J_block = B @ F_j
            sn = spectral_norm(J_block)
            sum_j_norms += sn / denom
        
        alpha = max(1.0 / max(sum_j_norms, 1.0), MIN_ALPHA)
        
        # ── Compute numerator: η_relax + Σ B·E[T_j] + Σ η_obs ──
        numerator = relax.copy()
        
        coupling_details = []
        for (neighbor, B_ab, transposed) in adjacency[name]:
            B = B_ab.T if transposed else B_ab
            et_j = ET_read[neighbor]
            contribution = B @ et_j
            coupling_details.append((neighbor, transposed, B, et_j, contribution))
            numerator += contribution
        
        obs_list = observations.get(name, [])
        for eta_obs in obs_list:
            numerator += eta_obs
        
        # ── Fixed point ──
        eta_star = numerator / denom
        
        # ── Damped update ──
        old_eta = post[name]
        new_eta = (1.0 - alpha) * old_eta + alpha * eta_star
        new_eta = project(name, new_eta)
        
        change = np.linalg.norm(new_eta - old_eta)
        max_change = max(max_change, change)
        
        # ── Print details ──
        print(f"\n  --- {name} (denom={denom}, α_damp={alpha:.6f}) ---")
        print(f"    η_relax     = {relax}")
        for (neighbor, transposed, B, et_j, contrib) in coupling_details:
            t_label = "^T" if transposed else ""
            print(f"    coupling from {neighbor:15s}: B{t_label}·E[T] = {contrib}  (E[T_{neighbor}]={et_j})")
        for eta_obs in obs_list:
            print(f"    observation: η_obs = {eta_obs}")
        print(f"    numerator   = {numerator}")
        print(f"    η*          = {eta_star}")
        print(f"    old η       = {old_eta}")
        print(f"    new η       = {new_eta}  (change={change:.6f})")
        print(f"    → {describe(name, new_eta)}")
        
        new_post[name] = new_eta
        new_ET[name] = expected_T(name, new_eta)
        new_Fisher[name] = fisher(name, new_eta)
    
    # Swap buffers
    post = new_post
    ET_buffer = new_ET
    Fisher_buffer = new_Fisher
    
    print(f"\n  max_change = {max_change:.6f}")
    print(f"\n  Summary after iter {iter_num}:")
    for name in node_names:
        print(f"    {name:15s}: {describe(name, post[name])}  η={post[name]}")

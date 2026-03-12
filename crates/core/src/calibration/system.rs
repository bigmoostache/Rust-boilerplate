//! Linear system assembly and solving for coupling calibration.
//!
//! Pipeline:
//! 1. Compute `T_B(x_B)` — sufficient statistics at a sample-space point.
//! 2. Identify hard-zero entries (from `±∞` in `T_B`).
//! 3. Convert each [`StatementDef`] into linear equations in `vec(B)`.
//! 4. Assemble `A · b_vec = rhs`, check DOF, solve.

use nalgebra::{DMatrix, DVector};

use crate::schema::raw::FamilyDef;

use super::parse::{ResolvedEdge, StatementDef, TargetMean};
use super::suff_stats::{at_point, eta_from_canonical, scalar_target, vec_target};

// ───────────────────────────────────────────────────────────────────
// Result type
// ───────────────────────────────────────────────────────────────────

/// The result of calibrating one edge.
#[derive(Debug)]
pub struct CalibratedEdge {
    /// Node A name.
    pub name_a: String,
    /// Node B name.
    pub name_b: String,
    /// Coupling matrix `B` as row-major `d_A × d_B`.
    pub coupling: DMatrix<f64>,
    /// Number of unknowns.
    pub n_unknowns: usize,
    /// Number of constraints.
    pub n_constraints: usize,
    /// Residual norm (0.0 for exact solve, >0 for least squares).
    pub residual: f64,
}

// ───────────────────────────────────────────────────────────────────
// Emitter context — bundles the repeated arguments
// ───────────────────────────────────────────────────────────────────

/// Shared context for constraint emitters, avoiding long param lists.
struct EmitCtx<'ctx> {
    /// Dimension of `T_A` (rows of `B`).
    d_a: usize,
    /// Dimension of `T_B` (columns of `B`).
    d_b: usize,
    /// Number of free (non-hard-zero) variables.
    n_free: usize,
    /// Mapping: flat index → position in free-variable vector.
    flat_to_free: &'ctx [Option<usize>],
    /// Accumulated constraint rows.
    rows: &'ctx mut Vec<DVector<f64>>,
    /// Accumulated right-hand-side values.
    rhs: &'ctx mut Vec<f64>,
}

impl EmitCtx<'_> {
    /// Compute flat index `B[row, col]` in row-major order.
    const fn flat(&self, row: usize, col: usize) -> usize {
        row.saturating_mul(self.d_b).saturating_add(col)
    }
}

// ───────────────────────────────────────────────────────────────────
// Main calibration pipeline
// ───────────────────────────────────────────────────────────────────

/// Calibrate one edge: assemble and solve the linear system.
///
/// # Errors
///
/// Returns an error if:
/// - A statement references an invalid `x_b` dimension.
/// - The system is under-determined (more unknowns than constraints).
/// - A `conditional_mean` is used without matching
///   `variance_unchanged` / `concentration_unchanged`.
pub fn calibrate_edge(edge: &ResolvedEdge) -> Result<CalibratedEdge, String> {
    let d_a = edge.dim_a;
    let d_b = edge.dim_b;
    let n_total = d_a.saturating_mul(d_b);

    // Step 1: Hard-zero entries from ±∞ in T_B.
    let mut is_hard_zero = vec![false; n_total];
    collect_hard_zeros(edge, d_a, d_b, &mut is_hard_zero)?;

    // Step 2: Free variable mapping.
    let free_vars: Vec<usize> = (0..n_total)
        .filter(|&idx| !is_hard_zero.get(idx).copied().unwrap_or(true))
        .collect();
    let n_free = free_vars.len();

    let mut flat_to_free = vec![None::<usize>; n_total];
    for (pos, &flat) in free_vars.iter().enumerate() {
        if let Some(slot) = flat_to_free.get_mut(flat) {
            *slot = Some(pos);
        }
    }

    // Step 3: Assemble constraint rows.
    let eta_a = eta_from_canonical(&edge.family_a);
    let mut rows: Vec<DVector<f64>> = Vec::new();
    let mut rhs_vals: Vec<f64> = Vec::new();

    for stmt in &edge.statements {
        let mut ctx = EmitCtx {
            d_a,
            d_b,
            n_free,
            flat_to_free: &flat_to_free,
            rows: &mut rows,
            rhs: &mut rhs_vals,
        };
        emit_statement(stmt, &edge.family_a, &edge.family_b, &eta_a, &mut ctx)?;
    }

    let n_constraints = rows.len();

    // Step 4: DOF check.
    if n_free == 0 {
        return Ok(CalibratedEdge {
            name_a: edge.name_a.clone(),
            name_b: edge.name_b.clone(),
            coupling: DMatrix::zeros(d_a, d_b),
            n_unknowns: 0,
            n_constraints,
            residual: 0.0,
        });
    }

    if n_constraints < n_free {
        return Err(format!(
            "under-determined: {n_free} unknowns but only \
             {n_constraints} constraints \
             ({} hard zeros eliminated {n_total}→{n_free})",
            n_total.saturating_sub(n_free),
        ));
    }

    // Step 5: Solve.
    let (solution, residual) = solve_system(&rows, &rhs_vals)?;

    // Step 6: Reconstruct B matrix.
    let mut b_matrix = DMatrix::zeros(d_a, d_b);
    for (pos, &flat) in free_vars.iter().enumerate() {
        // Recover (row, col) from flat index without integer division.
        let mut row: usize = 0;
        let mut remainder = flat;
        while remainder >= d_b {
            remainder = remainder.saturating_sub(d_b);
            row = row.saturating_add(1);
        }
        if let Some(&val) = solution.get(pos)
            && let Some(entry) = b_matrix.get_mut((row, remainder))
        {
            *entry = val;
        }
    }

    Ok(CalibratedEdge {
        name_a: edge.name_a.clone(),
        name_b: edge.name_b.clone(),
        coupling: b_matrix,
        n_unknowns: n_free,
        n_constraints,
        residual,
    })
}

// ───────────────────────────────────────────────────────────────────
// Hard-zero collection
// ───────────────────────────────────────────────────────────────────

/// Scan all statements for `T_B` entries that are `±∞`, and mark the
/// corresponding B entries as hard-zero.
fn collect_hard_zeros(
    edge: &ResolvedEdge,
    d_a: usize,
    d_b: usize,
    is_hard_zero: &mut [bool],
) -> Result<(), String> {
    for stmt in &edge.statements {
        let x_b = match stmt {
            StatementDef::ConditionalMean { x_b, .. }
            | StatementDef::VarianceUnchanged { x_b }
            | StatementDef::ConcentrationUnchanged { x_b }
            | StatementDef::NoEffect { x_b } => Some(x_b),
            StatementDef::EntryZero { .. } => None,
        };

        if let Some(obs) = x_b {
            let t_b = at_point(&edge.family_b, obs)?;
            for (j, &t_j) in t_b.iter().enumerate() {
                if t_j.is_infinite() {
                    for i in 0..d_a {
                        let flat = i.saturating_mul(d_b).saturating_add(j);
                        if let Some(flag) = is_hard_zero.get_mut(flat) {
                            *flag = true;
                        }
                    }
                }
            }
        }

        if let StatementDef::EntryZero { row, col } = stmt {
            let flat = row.saturating_mul(d_b).saturating_add(*col);
            if let Some(flag) = is_hard_zero.get_mut(flat) {
                *flag = true;
            }
        }
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// Statement dispatch
// ───────────────────────────────────────────────────────────────────

/// Emit constraint rows for a single statement.
fn emit_statement(
    stmt: &StatementDef,
    family_a: &FamilyDef,
    family_b: &FamilyDef,
    eta_a: &[f64],
    ctx: &mut EmitCtx<'_>,
) -> Result<(), String> {
    match stmt {
        StatementDef::EntryZero { .. } => Ok(()),
        StatementDef::NoEffect { x_b } => {
            let t_b = at_point(family_b, x_b)?;
            emit_no_effect(&t_b, ctx);
            Ok(())
        }
        StatementDef::VarianceUnchanged { x_b } => {
            let t_b = at_point(family_b, x_b)?;
            emit_variance_unchanged(family_a, &t_b, ctx)
        }
        StatementDef::ConcentrationUnchanged { x_b } => {
            let t_b = at_point(family_b, x_b)?;
            emit_concentration_unchanged(family_a, &t_b, ctx)
        }
        StatementDef::ConditionalMean { x_b, target_mean } => {
            let t_b = at_point(family_b, x_b)?;
            emit_conditional_mean(family_a, eta_a, &t_b, target_mean, ctx)
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Constraint emitters
// ───────────────────────────────────────────────────────────────────

/// Emit `d_A` equations for `B · T_B = 0` (one per row of `B`).
fn emit_no_effect(t_b: &[f64], ctx: &mut EmitCtx<'_>) {
    for i in 0..ctx.d_a {
        let mut row = DVector::zeros(ctx.n_free);
        let mut any_nonzero = false;
        for j in 0..ctx.d_b {
            let t_j = t_b.get(j).copied().unwrap_or(0.0);
            if t_j.is_infinite() || t_j == 0.0 {
                continue;
            }
            if let Some(Some(pos)) = ctx.flat_to_free.get(ctx.flat(i, j))
                && let Some(entry) = row.get_mut(*pos)
            {
                *entry = t_j;
                any_nonzero = true;
            }
        }
        if any_nonzero {
            ctx.rows.push(row);
            ctx.rhs.push(0.0);
        }
    }
}

/// Emit variance-unchanged constraint.
///
/// Gaussian / Gamma: `B[1,:] · T_B = 0`.
///
/// # Errors
///
/// Returns `Err` for Dirichlet families.
fn emit_variance_unchanged(
    family_a: &FamilyDef,
    t_b: &[f64],
    ctx: &mut EmitCtx<'_>,
) -> Result<(), String> {
    let eta2_row = match family_a {
        FamilyDef::Gaussian { .. } | FamilyDef::Gamma { .. } => 1,
        FamilyDef::Beta { .. } | FamilyDef::Dirichlet { .. } => {
            return Err("variance_unchanged not valid for Dirichlet; \
                 use concentration_unchanged"
                .into());
        }
    };
    emit_row_zero(eta2_row, t_b, ctx);
    Ok(())
}

/// Emit concentration-unchanged constraint (Dirichlet only).
///
/// `Σ_k Σ_j B[k,j] · t_j = 0`  (1 scalar equation).
///
/// # Errors
///
/// Returns `Err` for non-Dirichlet families.
fn emit_concentration_unchanged(
    family_a: &FamilyDef,
    t_b: &[f64],
    ctx: &mut EmitCtx<'_>,
) -> Result<(), String> {
    match family_a {
        FamilyDef::Beta { .. } | FamilyDef::Dirichlet { .. } => {}
        FamilyDef::Gaussian { .. } | FamilyDef::Gamma { .. } => {
            return Err("concentration_unchanged only valid for \
                 Dirichlet/Beta families"
                .into());
        }
    }

    let mut row = DVector::zeros(ctx.n_free);
    for k in 0..ctx.d_a {
        for j in 0..ctx.d_b {
            let t_j = t_b.get(j).copied().unwrap_or(0.0);
            if t_j.is_infinite() {
                continue;
            }
            if let Some(Some(pos)) = ctx.flat_to_free.get(ctx.flat(k, j))
                && let Some(entry) = row.get_mut(*pos)
            {
                *entry += t_j;
            }
        }
    }
    ctx.rows.push(row);
    ctx.rhs.push(0.0);
    Ok(())
}

/// Emit conditional-mean equations.
///
/// Assumes the matching shape constraint has been stated.
///
/// # Errors
///
/// Returns `Err` if target dimension mismatches.
fn emit_conditional_mean(
    family_a: &FamilyDef,
    eta_a: &[f64],
    t_b: &[f64],
    target: &TargetMean,
    ctx: &mut EmitCtx<'_>,
) -> Result<(), String> {
    match family_a {
        FamilyDef::Gaussian { sigma2, .. } => {
            let target_val = scalar_target(target, "Gaussian")?;
            let eta1 = eta_a.first().copied().unwrap_or(0.0);
            let rhs_val = target_val / sigma2 - eta1;
            emit_row_equation(0, t_b, rhs_val, ctx);
        }
        FamilyDef::Gamma { beta, .. } => {
            let target_val = scalar_target(target, "Gamma")?;
            let eta1 = eta_a.first().copied().unwrap_or(0.0);
            let rhs_val = target_val.mul_add(*beta, -eta1 - 1.0);
            emit_row_equation(0, t_b, rhs_val, ctx);
        }
        FamilyDef::Beta { alpha, beta } => {
            let target_val = scalar_target(target, "Beta")?;
            let sum_alpha = alpha + beta;
            let eta1 = eta_a.first().copied().unwrap_or(0.0);
            let rhs_val = target_val.mul_add(sum_alpha, -eta1 - 1.0);
            emit_row_equation(0, t_b, rhs_val, ctx);
        }
        FamilyDef::Dirichlet { alpha } => {
            let target_vec = vec_target(target, ctx.d_a, "Dirichlet")?;
            let sum_alpha: f64 = alpha.iter().sum();
            for k in 0..(ctx.d_a.saturating_sub(1)) {
                let eta_k = eta_a.get(k).copied().unwrap_or(0.0);
                let t_k = target_vec.get(k).copied().unwrap_or(0.0);
                let rhs_val = t_k.mul_add(sum_alpha, -eta_k - 1.0);
                emit_row_equation(k, t_b, rhs_val, ctx);
            }
        }
    }
    Ok(())
}

/// Emit `B[b_row,:] · t_b = rhs_val`.
fn emit_row_equation(b_row: usize, t_b: &[f64], rhs_val: f64, ctx: &mut EmitCtx<'_>) {
    let mut row = DVector::zeros(ctx.n_free);
    for j in 0..ctx.d_b {
        let t_j = t_b.get(j).copied().unwrap_or(0.0);
        if t_j.is_infinite() {
            continue;
        }
        if let Some(Some(pos)) = ctx.flat_to_free.get(ctx.flat(b_row, j))
            && let Some(entry) = row.get_mut(*pos)
        {
            *entry = t_j;
        }
    }
    ctx.rows.push(row);
    ctx.rhs.push(rhs_val);
}

/// Emit `B[b_row,:] · t_b = 0`.
fn emit_row_zero(b_row: usize, t_b: &[f64], ctx: &mut EmitCtx<'_>) {
    emit_row_equation(b_row, t_b, 0.0, ctx);
}

// ───────────────────────────────────────────────────────────────────
// Linear solver
// ───────────────────────────────────────────────────────────────────

/// Assemble and solve the linear system `A · x = b`.
fn solve_system(rows: &[DVector<f64>], rhs_vals: &[f64]) -> Result<(DVector<f64>, f64), String> {
    let n_constraints = rows.len();

    let transposed: Vec<_> = rows.iter().map(nalgebra::Matrix::transpose).collect();
    let a_mat = DMatrix::from_rows(&transposed);
    let b_vec = DVector::from_vec(rhs_vals.to_vec());
    let n_free = a_mat.ncols();

    if n_constraints == n_free {
        a_mat.lu().solve(&b_vec).map_or_else(
            || Err("singular system — cannot solve".into()),
            |sol| Ok((sol, 0.0)),
        )
    } else {
        let svd = a_mat.clone().svd(true, true);
        svd.solve(&b_vec, 1e-12).map_or_else(
            |_| Err("SVD solve failed".into()),
            |sol| {
                let residual_vec = &a_mat * &sol - &b_vec;
                Ok((sol, residual_vec.norm()))
            },
        )
    }
}

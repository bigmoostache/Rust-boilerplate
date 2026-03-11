//! Validation and construction of the domain model from raw YAML types.
//!
//! Converts canonical parameters to natural parameters, checks all
//! constraints, and builds the [`Graph`] structure.

use std::collections::{HashMap, HashSet};

use nalgebra::DMatrix;

use crate::distributions::NaturalParams;
use crate::graph::{Edge, Graph, Node, NodeId};
use crate::observation::Observation;

use super::diag::{SchemaError, SchemaErrors};
use super::observation_compat::{check_obs_family_compat, validate_observation};
use super::raw::{EdgeDef, FamilyDef, GraphConfig};

// ---------------------------------------------------------------------------
// Validated output
// ---------------------------------------------------------------------------

/// Fully validated and converted graph configuration.
#[derive(Debug)]
pub struct ValidatedConfig {
    /// The constructed graph (nodes + edges + observations).
    pub graph: Graph,
    /// Maximum coordinate-ascent iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Time gap for relaxation.
    pub delta_t: f64,
}

// ---------------------------------------------------------------------------
// Parsing entry point
// ---------------------------------------------------------------------------

/// Parse a YAML string into a validated graph configuration.
///
/// Returns all validation errors at once rather than failing on the
/// first error.
///
/// # Errors
///
/// Returns [`SchemaErrors`] if:
/// - The YAML is malformed or has unknown fields.
/// - Any parameter is out of range.
/// - Node IDs are not unique.
/// - Edges reference non-existent nodes.
/// - Coupling matrix dimensions don't match node families.
/// - Observations reference non-existent nodes.
/// - Observation types are incompatible with node families.
pub fn parse_yaml(yaml: &str) -> Result<ValidatedConfig, SchemaErrors> {
    let raw: GraphConfig = serde_yaml::from_str(yaml).map_err(|err| SchemaErrors {
        errors: vec![SchemaError {
            path: String::new(),
            message: format!("YAML parse error: {err}"),
        }],
    })?;

    validate_and_build(&raw)
}

/// Parse multiple YAML strings and merge them into a single validated
/// configuration.
///
/// Files are merged in order: `nodes`, `edges`, and `observations`
/// lists are appended; `inference` is overridden by the last file
/// that provides it.
///
/// # Errors
///
/// Returns [`SchemaErrors`] if any file is malformed or the merged
/// result fails validation.
pub fn parse_yamls(yamls: &[&str]) -> Result<ValidatedConfig, SchemaErrors> {
    let mut merged: Option<GraphConfig> = None;

    for (i, yaml) in yamls.iter().enumerate() {
        let raw: GraphConfig = serde_yaml::from_str(yaml).map_err(|err| SchemaErrors {
            errors: vec![SchemaError {
                path: format!("file[{i}]"),
                message: format!("YAML parse error: {err}"),
            }],
        })?;
        if let Some(ref mut m) = merged {
            m.merge(raw);
        } else {
            merged = Some(raw);
        }
    }

    let Some(config) = merged else {
        return Err(SchemaErrors {
            errors: vec![SchemaError {
                path: String::new(),
                message: "no input files provided".to_owned(),
            }],
        });
    };

    validate_and_build(&config)
}

// ---------------------------------------------------------------------------
// Validation + construction
// ---------------------------------------------------------------------------

/// Validate the raw config and build the domain model.
fn validate_and_build(raw: &GraphConfig) -> Result<ValidatedConfig, SchemaErrors> {
    let mut errors: Vec<SchemaError> = Vec::new();

    // ── Validate nodes ──────────────────────────────────────────
    let mut seen_ids = HashSet::new();
    let mut nodes: Vec<Node> = Vec::with_capacity(raw.nodes.len());
    let mut node_families: HashMap<&str, &FamilyDef> = HashMap::new();

    for (idx, raw_node) in raw.nodes.iter().enumerate() {
        let prefix = format!("nodes[{idx}]");

        if !seen_ids.insert(&raw_node.name) {
            errors.push(SchemaError {
                path: format!("{prefix}.name"),
                message: format!("duplicate node name \"{}\"", raw_node.name),
            });
        }

        if raw_node.tau <= 0.0 {
            errors.push(SchemaError {
                path: format!("{prefix}.tau"),
                message: format!("tau must be > 0, got {}", raw_node.tau),
            });
        }

        match validate_family(&raw_node.family, &format!("{prefix}.family")) {
            Ok(params) => {
                let node = Node {
                    name: raw_node.name.clone(),
                    epidemio: params.clone(),
                    prev: params.clone(),
                    relax: params.clone(),
                    post: params,
                    tau: raw_node.tau,
                };
                nodes.push(node);
                let _prev = node_families.insert(&raw_node.name, &raw_node.family);
            }
            Err(mut errs) => errors.append(&mut errs),
        }
    }

    let node_dim: HashMap<&str, usize> = nodes
        .iter()
        .map(|n| (n.name.as_str(), n.epidemio.suff_stat_dim()))
        .collect();

    // ── Collect inline edges from nodes ────────────────────────
    let mut all_edges: Vec<EdgeDef> = Vec::new();

    for raw_node in &raw.nodes {
        for inline in &raw_node.edges_to {
            all_edges.push(EdgeDef {
                from: raw_node.name.clone(),
                to: inline.node.clone(),
                coupling: inline.coupling.clone(),
            });
        }
        for inline in &raw_node.edges_from {
            all_edges.push(EdgeDef {
                from: inline.node.clone(),
                to: raw_node.name.clone(),
                coupling: inline.coupling.clone(),
            });
        }
    }

    // Append top-level edges after inline edges.
    for raw_edge in &raw.edges {
        all_edges.push(EdgeDef {
            from: raw_edge.from.clone(),
            to: raw_edge.to.clone(),
            coupling: raw_edge.coupling.clone(),
        });
    }

    // ── Validate edges ──────────────────────────────────────────
    let mut edges: Vec<Edge> = Vec::with_capacity(all_edges.len());

    for (idx, raw_edge) in all_edges.iter().enumerate() {
        let prefix = format!("edges[{idx}]");

        let from_exists = seen_ids.contains(&raw_edge.from);
        let to_exists = seen_ids.contains(&raw_edge.to);

        if !from_exists {
            errors.push(SchemaError {
                path: format!("{prefix}.from"),
                message: format!("unknown node \"{}\"", raw_edge.from),
            });
        }
        if !to_exists {
            errors.push(SchemaError {
                path: format!("{prefix}.to"),
                message: format!("unknown node \"{}\"", raw_edge.to),
            });
        }

        match parse_coupling(&raw_edge.coupling, &format!("{prefix}.coupling")) {
            Ok(coupling) => {
                if from_exists
                    && to_exists
                    && let (Some(&di), Some(&dj)) = (
                        node_dim.get(raw_edge.from.as_str()),
                        node_dim.get(raw_edge.to.as_str()),
                    )
                    && (coupling.nrows() != di || coupling.ncols() != dj)
                {
                    errors.push(SchemaError {
                        path: format!("{prefix}.coupling"),
                        message: format!(
                            "expected {di}×{dj} matrix, got {}×{}",
                            coupling.nrows(),
                            coupling.ncols()
                        ),
                    });
                }
                edges.push(Edge {
                    i: raw_edge.from.clone(),
                    j: raw_edge.to.clone(),
                    coupling,
                });
            }
            Err(mut errs) => errors.append(&mut errs),
        }
    }

    // ── Validate observations ───────────────────────────────────
    let mut obs_map: HashMap<NodeId, Vec<Observation>> = HashMap::new();

    for (idx, raw_obs) in raw.observations.iter().enumerate() {
        let prefix = format!("observations[{idx}]");
        let (node_id, obs_result) = validate_observation(raw_obs, &prefix);

        if !seen_ids.contains(&node_id) {
            errors.push(SchemaError {
                path: format!("{prefix}.node"),
                message: format!("unknown node \"{node_id}\""),
            });
        }

        match obs_result {
            Ok(obs) => {
                if let Some(family) = node_families.get(node_id.as_str())
                    && let Some(err) = check_obs_family_compat(&obs, family, &prefix)
                {
                    errors.push(err);
                }
                obs_map.entry(node_id).or_default().push(obs);
            }
            Err(mut errs) => errors.append(&mut errs),
        }
    }

    // ── Validate inference config ───────────────────────────────
    let Some(inference) = &raw.inference else {
        errors.push(SchemaError {
            path: "inference".to_owned(),
            message: "inference settings are required (provide in at least one file)".to_owned(),
        });
        return Err(SchemaErrors { errors });
    };

    if inference.tolerance <= 0.0 {
        errors.push(SchemaError {
            path: "inference.tolerance".to_owned(),
            message: format!("tolerance must be > 0, got {}", inference.tolerance),
        });
    }
    if inference.delta_t < 0.0 {
        errors.push(SchemaError {
            path: "inference.delta_t".to_owned(),
            message: format!("delta_t must be >= 0, got {}", inference.delta_t),
        });
    }

    // ── Return ──────────────────────────────────────────────────
    if !errors.is_empty() {
        return Err(SchemaErrors { errors });
    }

    let mut graph = Graph::new(nodes, edges);
    graph.observations = obs_map;

    Ok(ValidatedConfig {
        graph,
        max_iter: inference.max_iter,
        tolerance: inference.tolerance,
        delta_t: inference.delta_t,
    })
}

// ---------------------------------------------------------------------------
// Family validation + canonical → natural conversion
// ---------------------------------------------------------------------------

/// Validate canonical params and convert to natural params.
pub(super) fn validate_family(
    family: &FamilyDef,
    path: &str,
) -> Result<NaturalParams, Vec<SchemaError>> {
    let mut errors = Vec::new();

    let params = match family {
        FamilyDef::Gaussian { mu, sigma2 } => {
            if *sigma2 <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.sigma2"),
                    message: format!("sigma2 must be > 0, got {sigma2}"),
                });
            }
            errors.is_empty().then(|| NaturalParams::Gaussian {
                eta1: mu / sigma2,
                eta2: -1.0 / (2.0 * sigma2),
            })
        }
        FamilyDef::Gamma { alpha, beta } => {
            if *alpha <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.alpha"),
                    message: format!("alpha must be > 0, got {alpha}"),
                });
            }
            if *beta <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.beta"),
                    message: format!("beta must be > 0, got {beta}"),
                });
            }
            errors.is_empty().then(|| NaturalParams::Gamma {
                eta1: alpha - 1.0,
                eta2: -beta,
            })
        }
        FamilyDef::Beta { alpha, beta } => {
            if *alpha <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.alpha"),
                    message: format!("alpha must be > 0, got {alpha}"),
                });
            }
            if *beta <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.beta"),
                    message: format!("beta must be > 0, got {beta}"),
                });
            }
            errors.is_empty().then(|| NaturalParams::Beta {
                eta1: alpha - 1.0,
                eta2: beta - 1.0,
            })
        }
        FamilyDef::Poisson { lambda } => {
            if *lambda <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.lambda"),
                    message: format!("lambda must be > 0, got {lambda}"),
                });
            }
            errors
                .is_empty()
                .then(|| NaturalParams::Poisson { eta1: lambda.ln() })
        }
        FamilyDef::Bernoulli { p } => {
            if *p <= 0.0 || *p >= 1.0 {
                errors.push(SchemaError {
                    path: format!("{path}.p"),
                    message: format!("p must be in (0, 1), got {p}"),
                });
            }
            errors.is_empty().then(|| NaturalParams::Bernoulli {
                eta1: (p / (1.0 - p)).ln(),
            })
        }
        FamilyDef::Categorical { probs } => {
            if probs.len() < 2 {
                errors.push(SchemaError {
                    path: format!("{path}.probs"),
                    message: "categorical needs at least 2 categories".to_owned(),
                });
            }
            for (k, pk) in probs.iter().enumerate() {
                if *pk <= 0.0 {
                    errors.push(SchemaError {
                        path: format!("{path}.probs[{k}]"),
                        message: format!("probability must be > 0, got {pk}"),
                    });
                }
            }
            let sum: f64 = probs.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                errors.push(SchemaError {
                    path: format!("{path}.probs"),
                    message: format!("probabilities must sum to 1, got {sum}"),
                });
            }
            if errors.is_empty() {
                probs.split_last().map(|(&last, rest)| {
                    let eta: Vec<f64> = rest.iter().map(|pk: &f64| (pk / last).ln()).collect();
                    NaturalParams::Categorical { eta }
                })
            } else {
                None
            }
        }
        FamilyDef::Dirichlet { alpha } => {
            if alpha.len() < 2 {
                errors.push(SchemaError {
                    path: format!("{path}.alpha"),
                    message: "dirichlet needs at least 2 components".to_owned(),
                });
            }
            for (k, ak) in alpha.iter().enumerate() {
                if *ak <= 0.0 {
                    errors.push(SchemaError {
                        path: format!("{path}.alpha[{k}]"),
                        message: format!("alpha must be > 0, got {ak}"),
                    });
                }
            }
            errors.is_empty().then(|| NaturalParams::Dirichlet {
                alpha: alpha.clone(),
            })
        }
    };

    if errors.is_empty() {
        Ok(params.unwrap_or_else(|| {
            debug_assert!(false, "params should be Some when no errors");
            NaturalParams::Gaussian {
                eta1: 0.0,
                eta2: -0.5,
            }
        }))
    } else {
        Err(errors)
    }
}

// ---------------------------------------------------------------------------
// Coupling matrix parsing
// ---------------------------------------------------------------------------

/// Parse a row-major nested vector into a `DMatrix`.
fn parse_coupling(rows: &[Vec<f64>], path: &str) -> Result<DMatrix<f64>, Vec<SchemaError>> {
    if rows.is_empty() {
        return Err(vec![SchemaError {
            path: path.to_owned(),
            message: "coupling matrix must have at least one row".to_owned(),
        }]);
    }

    let nrows = rows.len();
    let ncols = if let Some(first) = rows.first() {
        first.len()
    } else {
        return Err(vec![SchemaError {
            path: path.to_owned(),
            message: "coupling matrix must have at least one column".to_owned(),
        }]);
    };

    if ncols == 0 {
        return Err(vec![SchemaError {
            path: path.to_owned(),
            message: "coupling matrix must have at least one column".to_owned(),
        }]);
    }

    let mut errors = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if row.len() != ncols {
            errors.push(SchemaError {
                path: format!("{path}[{i}]"),
                message: format!(
                    "row has {} columns, expected {ncols} (from first row)",
                    row.len()
                ),
            });
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    let data: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Ok(DMatrix::from_row_slice(nrows, ncols, &data))
}

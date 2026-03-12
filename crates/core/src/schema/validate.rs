//! Validation and construction of the domain model from raw YAML types.
//!
//! Converts canonical parameters to natural parameters, checks all
//! constraints, and builds the [`Graph`] structure.

use std::collections::{HashMap, HashSet};

use crate::graph::{Edge, Graph, Node, NodeId};

use super::convert::{SchemaError, SchemaErrors, parse_coupling, validate_family};
use super::observation_compat::{
    ValidatedInstrument, check_model_family_compat, resolve_observation, validate_instrument,
};
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
    /// Entropy scaling factor `λ` (1.0 = standard VI).
    pub entropy_scale: f64,
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
        for inline in &raw_node.coupled_with {
            all_edges.push(EdgeDef {
                node_a: raw_node.name.clone(),
                node_b: inline.node.clone(),
                coupling: inline.coupling.clone(),
            });
        }
    }

    // Append top-level edges after inline edges.
    for raw_edge in &raw.edges {
        all_edges.push(EdgeDef {
            node_a: raw_edge.node_a.clone(),
            node_b: raw_edge.node_b.clone(),
            coupling: raw_edge.coupling.clone(),
        });
    }

    // ── Validate edges ──────────────────────────────────────────
    let mut edges: Vec<Edge> = Vec::with_capacity(all_edges.len());

    for (idx, raw_edge) in all_edges.iter().enumerate() {
        let prefix = format!("edges[{idx}]");

        let from_exists = seen_ids.contains(&raw_edge.node_a);
        let to_exists = seen_ids.contains(&raw_edge.node_b);

        if !from_exists {
            errors.push(SchemaError {
                path: format!("{prefix}.node_a"),
                message: format!("unknown node \"{}\"", raw_edge.node_a),
            });
        }
        if !to_exists {
            errors.push(SchemaError {
                path: format!("{prefix}.node_b"),
                message: format!("unknown node \"{}\"", raw_edge.node_b),
            });
        }

        match parse_coupling(&raw_edge.coupling, &format!("{prefix}.coupling")) {
            Ok(coupling) => {
                if from_exists
                    && to_exists
                    && let (Some(&di), Some(&dj)) = (
                        node_dim.get(raw_edge.node_a.as_str()),
                        node_dim.get(raw_edge.node_b.as_str()),
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
                    node_a: raw_edge.node_a.clone(),
                    node_b: raw_edge.node_b.clone(),
                    coupling,
                });
            }
            Err(mut errs) => errors.append(&mut errs),
        }
    }

    // ── Validate instruments ─────────────────────────────────────
    let mut validated_instruments: HashMap<String, ValidatedInstrument> = HashMap::new();
    let mut seen_instrument_names: HashSet<String> = HashSet::new();

    for (idx, inst) in raw.instruments.iter().enumerate() {
        let prefix = format!("instruments[{idx}]");

        if !seen_instrument_names.insert(inst.name.clone()) {
            errors.push(SchemaError {
                path: format!("{prefix}.name"),
                message: format!("duplicate instrument name \"{}\"", inst.name),
            });
            continue;
        }

        // Validate model parameter ranges
        if let Err(mut errs) = validate_instrument(inst, &prefix) {
            errors.append(&mut errs);
            continue;
        }

        // Check node existence
        if !seen_ids.contains(&inst.node) {
            errors.push(SchemaError {
                path: format!("{prefix}.node"),
                message: format!("unknown node \"{}\"", inst.node),
            });
            continue;
        }

        // Check model–family compatibility
        if let Some(family) = node_families.get(inst.node.as_str())
            && let Some(err) = check_model_family_compat(&inst.model, family, &prefix)
        {
            errors.push(err);
            continue;
        }

        // Extract num_categories for categorical nodes
        let num_categories = node_families.get(inst.node.as_str()).and_then(|f| {
            if let FamilyDef::Categorical { probs } = f {
                Some(probs.len())
            } else {
                None
            }
        });

        let _prev = validated_instruments.insert(
            inst.name.clone(),
            ValidatedInstrument {
                node: inst.node.clone(),
                model: inst.model,
                num_categories,
            },
        );
    }

    // ── Resolve observations via instruments ────────────────────
    let mut obs_map: HashMap<NodeId, Vec<crate::distributions::NaturalParams>> = HashMap::new();

    for (idx, raw_obs) in raw.observations.iter().enumerate() {
        let prefix = format!("observations[{idx}]");
        let (node_id, obs_result) = resolve_observation(raw_obs, &validated_instruments, &prefix);

        match obs_result {
            Ok(obs) => {
                if let Some(nid) = node_id {
                    obs_map.entry(nid).or_default().push(obs);
                }
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
    if let Some(es) = inference.entropy_scale
        && es < 0.0
    {
        errors.push(SchemaError {
            path: "inference.entropy_scale".to_owned(),
            message: format!("entropy_scale must be >= 0, got {es}"),
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
        entropy_scale: inference.entropy_scale.unwrap_or(1.0),
    })
}

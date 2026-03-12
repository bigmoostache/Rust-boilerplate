# Grippe vs Angine — example

Two patients with overlapping symptoms but different diagnoses.

## Patient 1: Grippe (flu)
```
cargo run -- infer -i examples/grippe_angine/nodes.yaml -i examples/grippe_angine/edges.yaml -i examples/grippe_angine/instruments.yaml -i examples/grippe_angine/patient_flu.yaml
```

## Patient 2: Angine (tonsillitis)
```
cargo run -- infer -i examples/grippe_angine/nodes.yaml -i examples/grippe_angine/edges.yaml -i examples/grippe_angine/instruments.yaml -i examples/grippe_angine/patient_angine.yaml
```
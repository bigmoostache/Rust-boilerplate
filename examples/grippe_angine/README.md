# Grippe vs Angine — example

Two patients with overlapping symptoms but different diagnoses.

## Calibration example

```bash
cargo run -p app -- calibrate -i examples/grippe_angine/nodes.yaml -i examples/grippe_angine/calibration.yaml --yaml > examples/grippe_angine/_calibration_result.yaml
```

## Patient 1: Grippe (flu)

```bash
cargo run -- infer -i examples/grippe_angine/nodes.yaml -i examples/grippe_angine/edges.yaml -i examples/grippe_angine/instruments.yaml -i examples/grippe_angine/patient_flu.yaml -i examples/grippe_angine/_calibration_result.yaml
```

## Patient 2: Angine (tonsillitis)

```bash
cargo run -- infer -i examples/grippe_angine/nodes.yaml -i examples/grippe_angine/edges.yaml -i examples/grippe_angine/instruments.yaml -i examples/grippe_angine/patient_angine.yaml -i examples/grippe_angine/_calibration_result.yaml
```

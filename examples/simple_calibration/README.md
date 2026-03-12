# Simple calibration example — Grippe vs. Angine

Two diseases (flu, tonsillitis) and three symptoms (temperature, headache,
sore\_throat).  The calibration system computes the coupling matrices `B`
analytically from clinical statements.

## Run

```bash
cargo run -p app -- calibrate -i examples/simple_calibration/nodes.yaml -i examples/simple_calibration/calibration.yaml
```

## Run the tests

```bash
cargo test -p app-core calibration_tests
```

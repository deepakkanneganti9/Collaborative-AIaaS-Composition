# MLaaS Collaborative Composition Experiment

This is a minimal reviewer-facing package. It contains only the files needed to
understand the dataset, the final collaborative service requests, the benchmark
techniques, and the figure generated for the paper.

## Folder structure

- `datasets/`
  - `MLaaS_Dataset_F1.csv`: final MLaaS dataset used for the experiment.

- `service_request/`
  - `Service_Request_V3_collaborative_only.json`: final collaborative-only service request file used in the benchmark.

- `benchmark/`
  - `composability_model.py`: shared collaborative composition objective used by every optimizer.
  - `canonical_full_benchmark.py`: canonical full benchmark runner that uses the package-local dataset, service request, and update-signature files.
  - `run_lightweight_split_benchmark.py`: readable split-code benchmark runner. This is useful for inspecting individual optimizer structure, but the canonical runner is the one aligned with the archived paper results.
  - `plot_final_average_quality.py`: script that recreates the final average quality plot from the archived benchmark output.
  - `optimizers/`: one Python file per benchmark technique.

- `outputs/`
  - `baseline_benchmark_results.csv`: benchmark output used to create the paper figure. It contains only the baseline methods plotted in the figure.
  - `old_values_average_service_quality_without_mab_ymin_0_5_inside_legend.csv`: averaged values by composition length.
  - `old_values_average_service_quality_without_mab_ymin_0_5_inside_legend.pdf`: final paper-ready plot.
  - `old_values_average_service_quality_without_mab_ymin_0_5_inside_legend.png`: PNG version of the same plot.

## Benchmark techniques included

- Random Search
- Greedy
- Epsilon-Greedy
- Genetic Algorithm (GA)
- DAAGA
- MWOA
- CSSA
- SDFGA
- BPSC-GA
- PK-IDPSO

The MAB-based method is treated as the proposed solution generator and is not
included as a plotted baseline in this reviewer-facing output.

## Recreate the final plot

From the parent directory:

```bash
python3 -m MLaaS_exp_.benchmark.plot_final_average_quality
```

The regenerated PDF and PNG are written back to `outputs/`.

## Re-run the full optimization

From the parent directory:

```bash
python3 -m MLaaS_exp_.benchmark.canonical_full_benchmark
```

This recomputes the full signature-based collaborative benchmark and writes:

`outputs/canonical_recomputed_full_benchmark.csv`

This run is substantially slower than plotting because it evaluates all 2000
service requests across all benchmark optimizers.

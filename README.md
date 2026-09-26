# Collaborative AIaaS Composition Benchmark

This directory contains the data, service requests, benchmark implementation, and
result files for the paper:

**A Collaborative Artificial Intelligence as a Service Composition Dataset**

The materials support the collaborative AIaaS composition experiment reported in
the paper. The benchmark evaluates multiple service-selection and optimization
techniques over collaborative AIaaS service requests and reports average service
quality by service composition length.

## Contents

```text
datasets/
  AIaaS_Service_Dataset.csv
  update_signatures/

service_request/
  AIaaS_Collaborative_Service_Request_Dataset.json

benchmark/
  canonical_full_benchmark.py
  plot_final_average_quality.py

outputs/
  AIaaS_Composition_Benchmark_Results.csv
  AIaaS_Composition_Quality_by_Length.csv
  AIaaS_Composition_Quality_Comparison.pdf
  AIaaS_Composition_Quality_Comparison.png
```

## Data files

- `datasets/AIaaS_Service_Dataset.csv` contains the AIaaS service records used by
  the benchmark.
- `datasets/update_signatures/` contains update-signature vectors used to compute
  interface/model compatibility between AIaaS services.
- `service_request/AIaaS_Collaborative_Service_Request_Dataset.json` contains the
  collaborative service requests used in the benchmark.

## Benchmark code

- `benchmark/canonical_full_benchmark.py` contains the full benchmark
  implementation, including the objective calculation and the optimization
  techniques used in the experiment.
- `benchmark/plot_final_average_quality.py` regenerates the average service
  quality plot from the archived benchmark result file.

The optimization techniques included in the benchmark are:

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

The MAB-based method is used as the proposed solution-generation strategy and is
not plotted as a baseline in the final average-quality comparison figure.

## Composability objective used in the benchmark

The benchmark computes a weighted collaborative composability score using the
preference weights in each service request. The implemented terms are:

- `quality score`: normalized task performance, corresponding to the accuracy or
  task-quality component.
- `response time`: bounded latency score using the request latency threshold and
  the summed latency of the selected services.
- `tail latency`: bounded tail-latency score using the request tail-latency
  threshold and the summed tail latency of the selected services.
- `resource cost score`: resource/cost component used in the request weights.
- `interface compatibility`: pairwise compatibility computed from update-signature
  vectors using cosine similarity.
- `update signature availability`: availability of update-signature information
  for the selected services.

The final score is the weighted sum of the active terms listed in the service
request's `preference_weights`. The service dataset also includes additional
attributes such as mean compute time and reliability score. Those attributes remain
available in the dataset, but they are not active terms in this benchmark unless
they are included in the service-request weights.

## Recreate the final plot

Run the following command from the directory that contains
`AIaaS_Collaborative_Composition_Benchmark/`:

```bash
python3 -m AIaaS_Collaborative_Composition_Benchmark.benchmark.plot_final_average_quality
```

This regenerates:

```text
outputs/AIaaS_Composition_Quality_by_Length.csv
outputs/AIaaS_Composition_Quality_Comparison.pdf
outputs/AIaaS_Composition_Quality_Comparison.png
```

## Re-run the full optimization

Run the following command from the directory that contains
`AIaaS_Collaborative_Composition_Benchmark/`:

```bash
python3 -m AIaaS_Collaborative_Composition_Benchmark.benchmark.canonical_full_benchmark
```

This recomputes the full signature-based collaborative benchmark and writes:

```text
outputs/canonical_recomputed_full_benchmark.csv
outputs/canonical_recomputed_full_benchmark_summary.json
```

The full optimization run is slow because it evaluates all collaborative service
requests across all benchmark techniques and recomputes signature-based interface
compatibility. During execution, progress messages are printed, for example:

```text
processed 100/2000 collaborative requests
```

## Notes

The `datasets/update_signatures/` directory is required for a self-contained full
optimization rerun. Removing it reduces the directory size, but the full benchmark
will no longer be able to compute signature-based interface compatibility.

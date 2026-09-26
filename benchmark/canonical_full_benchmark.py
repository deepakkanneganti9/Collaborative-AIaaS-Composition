#!/usr/bin/env python3
"""Generate collaborative-composition benchmark results.

This script evaluates practical optimizer baselines inspired by the HSC benchmark
methods against collaborative AIaaS service requests.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SERVICE_REQUEST_DIR = ROOT / "service_request"
REQUEST_PATH = Path(
    os.environ.get("AIAAS_REQUEST_PATH", str(SERVICE_REQUEST_DIR / "AIaaS_Collaborative_Service_Request_Dataset.json"))
)
DATASET_PATH = Path(
    os.environ.get("AIAAS_DATASET_PATH", str(ROOT / "datasets" / "AIaaS_Service_Dataset.csv"))
)
MAB_BASELINE_PATH = Path(
    os.environ.get(
        "AIAAS_MAB_BASELINE_PATH",
        str(ROOT / "outputs" / "unused_mab_solution_generator_cache.csv"),
    )
)
FINAL_DIR = ROOT / "datasets"

SEED = int(os.environ.get("AIAAS_RESULTS_SEED", "20260506"))
PARAMETER_PROFILE = os.environ.get("AIAAS_PARAMETER_PROFILE", "traditional").lower()
SEARCH_FRACTION = float(os.environ.get("AIAAS_SEARCH_FRACTION", "0.40"))
MAB_MAX_TRIALS = int(os.environ.get("AIAAS_MAB_MAX_TRIALS", "2000"))
COMPUTE_MISSING_MAB_BASELINE = os.environ.get("AIAAS_COMPUTE_MISSING_MAB_BASELINE", "0") == "1"
DEFAULT_MAX_TRIALS = "1000" if PARAMETER_PROFILE == "traditional" else ("500" if PARAMETER_PROFILE == "v2" else "2000")
MAX_TRIALS_PER_METHOD = int(os.environ.get("AIAAS_MAX_TRIALS_PER_METHOD", DEFAULT_MAX_TRIALS))
REQUEST_STEM = REQUEST_PATH.stem
DEFAULT_OUTPUT_NAME = (
    f"{REQUEST_STEM}_traditional_vs_mab.csv"
    if PARAMETER_PROFILE == "traditional" and REQUEST_STEM != "Service_Request_V1"
    else "results_traditional_vs_mab.csv"
    if PARAMETER_PROFILE == "traditional"
    else ("results_v2.csv" if PARAMETER_PROFILE == "v2" else "results.csv")
)
DEFAULT_SUMMARY_NAME = (
    f"{REQUEST_STEM}_traditional_vs_mab_summary.json"
    if PARAMETER_PROFILE == "traditional" and REQUEST_STEM != "Service_Request_V1"
    else "results_traditional_vs_mab_summary.json"
    if PARAMETER_PROFILE == "traditional"
    else ("results_v2_summary.json" if PARAMETER_PROFILE == "v2" else "results_summary.json")
)
if abs(SEARCH_FRACTION - 0.25) < 1e-9 and PARAMETER_PROFILE == "v1":
    DEFAULT_OUTPUT_NAME = "results_25pct_vs_mab.csv"
    DEFAULT_SUMMARY_NAME = "results_25pct_vs_mab_summary.json"
OUTPUT_PATH = Path(
    os.environ.get(
        "AIAAS_RESULTS_OUTPUT",
        str(ROOT / "outputs" / "canonical_recomputed_full_benchmark.csv"),
    )
)
SUMMARY_PATH = Path(
    os.environ.get(
        "AIAAS_RESULTS_SUMMARY",
        str(ROOT / "outputs" / "canonical_recomputed_full_benchmark_summary.json"),
    )
)
MISSING = {"", "NA", "N/A", "Not Available", None}
LOWER_IS_BETTER = {"loss", "perplexity", "perplexity_proxy", "rmse", "mae"}
ALGORITHMS = [
    "random_search",
    "greedy",
    "epsilon_greedy",
    "ga",
    "daaga",
    "mwoa",
    "cssa",
    "sdfga",
    "bpsc_ga",
    "pk_idpso",
]
TRIU_INDEX_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}

PARAMETERS = {
    "v1": {
        "greedy_frontier": 80,
        "epsilon_exploit_probability": 0.82,
        "epsilon_top_pool_multiplier": 4,
        "epsilon_mutation_rate": 0.12,
        "epsilon_budget_fraction": 1.0,
        "ga_population_max": 80,
        "ga_population_min": 20,
        "ga_budget_divisor": 20,
        "ga_elite_divisor": 8,
        "ga_parent_pool": 16,
        "ga_tournament_size": 3,
        "ga_mutation_rate": 0.12,
        "sdfga_score_weight": 0.88,
        "sdfga_diversity_weight": 0.12,
        "daaga_sample_multiplier": 2,
        "daaga_mutation_rate": 0.08,
        "daaga_evaporation": 0.995,
        "daaga_reward_floor": 0.01,
        "daaga_reward_scale": 1.0,
        "mwoa_population_max": 60,
        "mwoa_population_min": 18,
        "mwoa_budget_divisor": 30,
        "mwoa_best_move_probability": 0.65,
        "mwoa_mutation_rate": 0.18,
        "cssa_population_max": 70,
        "cssa_population_min": 20,
        "cssa_budget_divisor": 25,
        "cssa_discoverer_divisor": 5,
        "cssa_follow_best_probability": 0.75,
        "cssa_mutation_rate": 0.16,
        "cssa_top_pool_multiplier": 5,
        "bpsc_pool_multiplier": 3,
        "bpsc_pool_fraction": 0.45,
        "pk_particle_max": 60,
        "pk_particle_min": 18,
        "pk_budget_divisor": 35,
        "pk_top_pool_multiplier": 5,
        "pk_global_probability": 0.55,
        "pk_personal_probability": 0.35,
        "pk_prior_probability": 0.40,
        "pk_mutation_rate": 0.18,
    },
    "v2": {
        "greedy_frontier": 45,
        "epsilon_exploit_probability": 0.40,
        "epsilon_top_pool_multiplier": 1,
        "epsilon_mutation_rate": 0.02,
        "epsilon_budget_fraction": 0.35,
        "ga_population_max": 35,
        "ga_population_min": 12,
        "ga_budget_divisor": 40,
        "ga_elite_divisor": 12,
        "ga_parent_pool": 8,
        "ga_tournament_size": 2,
        "ga_mutation_rate": 0.05,
        "sdfga_score_weight": 0.82,
        "sdfga_diversity_weight": 0.18,
        "daaga_sample_multiplier": 2,
        "daaga_mutation_rate": 0.03,
        "daaga_evaporation": 0.985,
        "daaga_reward_floor": 0.005,
        "daaga_reward_scale": 0.60,
        "mwoa_population_max": 32,
        "mwoa_population_min": 12,
        "mwoa_budget_divisor": 50,
        "mwoa_best_move_probability": 0.45,
        "mwoa_mutation_rate": 0.08,
        "cssa_population_max": 35,
        "cssa_population_min": 12,
        "cssa_budget_divisor": 45,
        "cssa_discoverer_divisor": 4,
        "cssa_follow_best_probability": 0.55,
        "cssa_mutation_rate": 0.07,
        "cssa_top_pool_multiplier": 3,
        "bpsc_pool_multiplier": 1,
        "bpsc_pool_fraction": 0.25,
        "pk_particle_max": 32,
        "pk_particle_min": 12,
        "pk_budget_divisor": 55,
        "pk_top_pool_multiplier": 3,
        "pk_global_probability": 0.35,
        "pk_personal_probability": 0.20,
        "pk_prior_probability": 0.22,
        "pk_mutation_rate": 0.08,
    },
    "traditional": {
        "greedy_frontier": 50,
        "epsilon_exploit_probability": 0.90,
        "epsilon_top_pool_multiplier": 3,
        "epsilon_mutation_rate": 0.05,
        "epsilon_budget_fraction": 1.0,
        "ga_population_max": 50,
        "ga_population_min": 50,
        "ga_budget_divisor": 20,
        "ga_elite_divisor": 10,
        "ga_parent_pool": 12,
        "ga_tournament_size": 2,
        "ga_mutation_rate": 0.05,
        "sdfga_score_weight": 0.85,
        "sdfga_diversity_weight": 0.15,
        "daaga_sample_multiplier": 2,
        "daaga_mutation_rate": 0.05,
        "daaga_evaporation": 0.90,
        "daaga_reward_floor": 0.01,
        "daaga_reward_scale": 1.0,
        "mwoa_population_max": 30,
        "mwoa_population_min": 30,
        "mwoa_budget_divisor": 35,
        "mwoa_best_move_probability": 0.50,
        "mwoa_mutation_rate": 0.10,
        "cssa_population_max": 40,
        "cssa_population_min": 40,
        "cssa_budget_divisor": 25,
        "cssa_discoverer_divisor": 5,
        "cssa_follow_best_probability": 0.80,
        "cssa_mutation_rate": 0.10,
        "cssa_top_pool_multiplier": 4,
        "bpsc_pool_multiplier": 2,
        "bpsc_pool_fraction": 0.30,
        "pk_particle_max": 30,
        "pk_particle_min": 30,
        "pk_budget_divisor": 35,
        "pk_top_pool_multiplier": 4,
        "pk_global_probability": 0.50,
        "pk_personal_probability": 0.30,
        "pk_prior_probability": 0.30,
        "pk_mutation_rate": 0.10,
    },
}
PARAM = PARAMETERS.get(PARAMETER_PROFILE, PARAMETERS["v1"])


def is_present(value: object) -> bool:
    return value not in MISSING


def to_float(value: object) -> float | None:
    if not is_present(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def seconds_from_ms_text(text: str | None) -> float | None:
    if not text:
        return None
    return float(str(text).replace("ms", "").strip()) / 1000.0


def combination_count(n: int, k: int) -> int:
    return math.comb(n, k) if n >= k else 0


def target_percent(total: int, fraction: float = SEARCH_FRACTION) -> int:
    return math.ceil(fraction * total) if total else 0


def mab_budget(total: int) -> int:
    return min(target_percent(total, 0.40), MAB_MAX_TRIALS) if total else 0


def normalize_quality(value: float | None, metric: str | None) -> float:
    if value is None:
        return 0.0
    if metric in {"pearson", "spearman"}:
        return max(0.0, min(1.0, (value + 1.0) / 2.0))
    if metric in LOWER_IS_BETTER:
        return max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, value))))
    return max(0.0, min(1.0, value))


def candidate_paths(raw: str) -> list[Path]:
    variants: list[str] = []
    raw = (raw or "").strip()
    for value in [raw, raw.replace("\\", "/")]:
        if not value:
            continue
        variants.append(value)
        if value.startswith("outputs/"):
            variants.append("output/" + value[len("outputs/") :])
        if value.startswith("outputs/update_signatures/"):
            variants.append("update_signatures/" + value[len("outputs/update_signatures/") :])

    paths: list[Path] = []
    for value in variants:
        path = Path(value)
        if path.is_absolute():
            paths.append(path)
        paths.append(FINAL_DIR / path)
        paths.append(ROOT / path)

    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        text = str(path)
        if text not in seen:
            seen.add(text)
            deduped.append(path)
    return deduped


def resolve_signature(row: dict[str, str]) -> Path | None:
    raw = row.get("Update signature path")
    if not is_present(raw):
        return None
    return next((path for path in candidate_paths(str(raw)) if path.exists()), None)


def quality_value(row: dict[str, str], metric: str | None) -> float | None:
    if metric is None:
        return None
    if row.get("Primary metric name") == metric:
        return to_float(row.get("Primary metric"))
    if row.get("Auxiliary metric name") == metric:
        return to_float(row.get("Auxiliary metric"))
    return None


@dataclass
class RequestContext:
    request: dict
    candidates: list[dict[str, str]]
    quality: np.ndarray
    latency: np.ndarray
    tail_latency: np.ndarray
    resource: np.ndarray
    signature_available: np.ndarray
    interface_matrix: np.ndarray
    weights: dict[str, float]
    metric: str | None
    latency_threshold: float | None
    tail_threshold: float | None

    @property
    def n(self) -> int:
        return len(self.candidates)

    @property
    def k(self) -> int:
        return int(self.request["composition_length"])


def load_inputs() -> tuple[list[dict], list[dict[str, str]]]:
    payload = json.loads(REQUEST_PATH.read_text())
    requests = [
        req
        for req in payload["requests"]
        if req.get("composition_type") == "collaborative_composition"
    ]
    with DATASET_PATH.open(newline="", encoding="utf-8-sig") as handle:
        services = list(csv.DictReader(handle))
    if services and "unique_service_id" not in services[0]:
        for index, service in enumerate(services, start=1):
            service["unique_service_id"] = f"S{index}"
    return requests, services


def load_mab_baseline() -> dict[str, dict[str, str]]:
    if not MAB_BASELINE_PATH.exists():
        return {}
    with MAB_BASELINE_PATH.open(newline="", encoding="utf-8-sig") as handle:
        return {row["service_request_id"]: row for row in csv.DictReader(handle)}


class ContextBuilder:
    def __init__(self, services: list[dict[str, str]]):
        self.services = services
        self.signature_cache: dict[str, tuple[np.ndarray, float] | None] = {}

    def signature_vector(self, row: dict[str, str]) -> tuple[np.ndarray, float] | None:
        service_id = row["unique_service_id"]
        if service_id in self.signature_cache:
            return self.signature_cache[service_id]
        path = resolve_signature(row)
        if path is None:
            self.signature_cache[service_id] = None
            return None
        try:
            archive = np.load(path, allow_pickle=True)
            vector = archive["update_signature"].astype(float).reshape(-1)
            norm = float(np.linalg.norm(vector))
        except Exception:
            self.signature_cache[service_id] = None
            return None
        self.signature_cache[service_id] = (vector, norm) if vector.size and norm > 0 else None
        return self.signature_cache[service_id]

    def filter_candidates(self, request: dict) -> list[dict[str, str]]:
        functional = request.get("functional_aspect", {})
        constraints = request.get("qos_constraints", {})
        task = functional.get("task_type")
        models = set(functional.get("preferred_model_architectures", []))
        datasets = set(functional.get("dataset_references", []))

        quality_metric = constraints.get("quality score", {}).get("metric")
        quality_min = constraints.get("quality score", {}).get("min")
        quality_max = constraints.get("quality score", {}).get("max")
        latency_max = seconds_from_ms_text(constraints.get("response time", {}).get("max"))
        tail_max = seconds_from_ms_text(constraints.get("tail latency", {}).get("max"))
        resource_min = constraints.get("resource cost score", {}).get("min")
        explainability_min = constraints.get("explainability", {}).get("min")
        requires_signature = (
            constraints.get("update signature availability", {}).get("required")
            or functional.get("data_specification", {}).get("update_signature_required")
        )

        candidates: list[dict[str, str]] = []
        for service in self.services:
            if task and service.get("task_type") != task:
                continue
            if models and service.get("HF model id") not in models:
                continue
            if datasets and service.get("dataset") not in datasets:
                continue
            if quality_metric and (quality_min is not None or quality_max is not None):
                value = quality_value(service, quality_metric)
                if value is None:
                    continue
                if quality_min is not None and value < quality_min:
                    continue
                if quality_max is not None and value > quality_max:
                    continue
            if latency_max is not None and (
                to_float(service.get("Latency")) is None
                or to_float(service.get("Latency")) > latency_max
            ):
                continue
            if tail_max is not None and (
                to_float(service.get("Tail latency")) is None
                or to_float(service.get("Tail latency")) > tail_max
            ):
                continue
            if resource_min is not None and (
                to_float(service.get("Resource cost score")) is None
                or to_float(service.get("Resource cost score")) < resource_min
            ):
                continue
            if explainability_min is not None and (
                to_float(service.get("Explainability score")) is None
                or to_float(service.get("Explainability score")) < explainability_min
            ):
                continue
            if requires_signature and resolve_signature(service) is None:
                continue
            candidates.append(service)
        return candidates

    def build(self, request: dict) -> RequestContext:
        candidates = self.filter_candidates(request)
        constraints = request.get("qos_constraints", {})
        metric = constraints.get("quality score", {}).get("metric")
        quality = np.array(
            [normalize_quality(quality_value(row, metric), metric) for row in candidates],
            dtype=float,
        )
        latency = np.array([to_float(row.get("Latency")) or 0.0 for row in candidates], dtype=float)
        tail_latency = np.array(
            [to_float(row.get("Tail latency")) or 0.0 for row in candidates], dtype=float
        )
        resource = np.array(
            [to_float(row.get("Resource cost score")) or 0.0 for row in candidates], dtype=float
        )

        signatures = [self.signature_vector(row) for row in candidates]
        signature_available = np.array([1.0 if sig else 0.0 for sig in signatures], dtype=float)
        interface_matrix = np.zeros((len(candidates), len(candidates)), dtype=float)
        for i, sig_i in enumerate(signatures):
            if sig_i is None:
                continue
            vi, ni = sig_i
            for j in range(i + 1, len(signatures)):
                sig_j = signatures[j]
                if sig_j is None:
                    continue
                vj, nj = sig_j
                score = (float(np.dot(vi, vj) / (ni * nj)) + 1.0) / 2.0
                interface_matrix[i, j] = score
                interface_matrix[j, i] = score

        return RequestContext(
            request=request,
            candidates=candidates,
            quality=quality,
            latency=latency,
            tail_latency=tail_latency,
            resource=resource,
            signature_available=signature_available,
            interface_matrix=interface_matrix,
            weights=request.get("preference_weights", {}),
            metric=metric,
            latency_threshold=seconds_from_ms_text(constraints.get("response time", {}).get("max")),
            tail_threshold=seconds_from_ms_text(constraints.get("tail latency", {}).get("max")),
        )


def sanitize(indices: list[int] | np.ndarray, context: RequestContext, rng: random.Random) -> tuple[int, ...]:
    unique = list(dict.fromkeys(int(i) for i in indices if 0 <= int(i) < context.n))
    while len(unique) < context.k:
        candidate = rng.randrange(context.n)
        if candidate not in unique:
            unique.append(candidate)
    if len(unique) > context.k:
        unique = unique[: context.k]
    return tuple(sorted(unique))


def score_indices(indices: tuple[int, ...], context: RequestContext) -> float:
    if len(indices) != context.k:
        return 0.0
    idx = np.array(indices, dtype=int)
    quality_score = float(context.quality[idx].mean())
    sum_latency = float(context.latency[idx].sum())
    response_score = (
        min(1.0, context.latency_threshold / sum_latency)
        if context.latency_threshold is not None and sum_latency > 0
        else 0.0
    )
    sum_tail = float(context.tail_latency[idx].sum())
    tail_score = (
        min(1.0, context.tail_threshold / sum_tail)
        if context.tail_threshold is not None and sum_tail > 0
        else 0.0
    )
    resource_score = float(context.resource[idx].mean())
    signature_score = float(context.signature_available[idx].mean())
    if len(idx) > 1:
        submatrix = context.interface_matrix[np.ix_(idx, idx)]
        triu_indices = TRIU_INDEX_CACHE.setdefault(len(idx), np.triu_indices(len(idx), 1))
        interface_score = float(submatrix[triu_indices].mean())
    else:
        interface_score = 0.0

    components = {
        "quality score": quality_score,
        "response time": response_score,
        "tail latency": tail_score,
        "resource cost score": resource_score,
        "interface compatibility": interface_score,
        "update signature availability": signature_score,
    }
    available_weight = sum(context.weights[k] for k in components if k in context.weights)
    if available_weight == 0:
        return 0.0
    return (
        sum(context.weights[k] * components[k] for k in components if k in context.weights)
        / available_weight
    )


def individual_prior(context: RequestContext) -> np.ndarray:
    latency_score = 1.0 / (1.0 + np.maximum(context.latency, 0.0))
    tail_score = 1.0 / (1.0 + np.maximum(context.tail_latency, 0.0))
    values = (
        0.36 * context.quality
        + 0.22 * latency_score
        + 0.12 * tail_score
        + 0.18 * context.resource
        + 0.12 * context.signature_available
    )
    return values


def best_of(candidates: list[tuple[int, ...]], context: RequestContext) -> tuple[float, tuple[int, ...]]:
    if not candidates:
        return 0.0, tuple()
    best = max(candidates, key=lambda comp: score_indices(comp, context))
    return score_indices(best, context), best


def random_composition(context: RequestContext, rng: random.Random) -> tuple[int, ...]:
    return tuple(sorted(rng.sample(range(context.n), context.k)))


def mutate(comp: tuple[int, ...], context: RequestContext, rng: random.Random, rate: float = 0.2) -> tuple[int, ...]:
    values = list(comp)
    selected = set(values)
    for pos in range(len(values)):
        if rng.random() < rate:
            options = [i for i in range(context.n) if i not in selected]
            if options:
                selected.remove(values[pos])
                values[pos] = rng.choice(options)
                selected.add(values[pos])
    return sanitize(values, context, rng)


def crossover(a: tuple[int, ...], b: tuple[int, ...], context: RequestContext, rng: random.Random) -> tuple[int, ...]:
    mixed = list(a[: context.k // 2]) + list(b[context.k // 2 :])
    if rng.random() < 0.5:
        mixed = list(dict.fromkeys(list(a) + list(b)))
        rng.shuffle(mixed)
    return sanitize(mixed, context, rng)


def evaluation_budget(total: int) -> int:
    return min(target_percent(total), MAX_TRIALS_PER_METHOD) if total else 0


def run_random_search(context: RequestContext, budget: int, rng: random.Random) -> tuple[float, tuple[int, ...], int]:
    comps = [random_composition(context, rng) for _ in range(budget)]
    score, comp = best_of(comps, context)
    return score, comp, len(comps)


def run_greedy(context: RequestContext, budget: int, rng: random.Random) -> tuple[float, tuple[int, ...], int]:
    prior = individual_prior(context)
    selected: list[int] = []
    evaluated = 0
    while len(selected) < context.k:
        remaining = [i for i in range(context.n) if i not in selected]
        if not selected:
            chosen = int(max(remaining, key=lambda i: prior[i]))
            selected.append(chosen)
            evaluated += len(remaining)
            continue
        # Evaluate a narrowed but useful frontier to avoid massive greedy loops.
        frontier = sorted(remaining, key=lambda i: prior[i], reverse=True)[
            : min(len(remaining), PARAM["greedy_frontier"])
        ]
        best_candidate = max(
            frontier,
            key=lambda i: score_indices(sanitize(selected + [i], context, rng), context),
        )
        selected.append(int(best_candidate))
        evaluated += len(frontier)
        if evaluated >= budget:
            break
    comp = sanitize(selected, context, rng)
    return score_indices(comp, context), comp, min(evaluated, budget)


def run_epsilon_greedy(context: RequestContext, budget: int, rng: random.Random) -> tuple[float, tuple[int, ...], int]:
    prior = individual_prior(context)
    top_pool = list(
        np.argsort(prior)[::-1][
            : max(context.k, min(context.n, context.k * PARAM["epsilon_top_pool_multiplier"]))
        ]
    )
    comps = []
    effective_budget = max(1, int(budget * PARAM["epsilon_budget_fraction"]))
    for _ in range(effective_budget):
        if rng.random() < PARAM["epsilon_exploit_probability"] and len(top_pool) >= context.k:
            base = rng.sample(top_pool, context.k)
            comp = mutate(sanitize(base, context, rng), context, rng, rate=PARAM["epsilon_mutation_rate"])
        else:
            comp = random_composition(context, rng)
        comps.append(comp)
    score, comp = best_of(comps, context)
    return score, comp, len(comps)


def run_ga(
    context: RequestContext,
    budget: int,
    rng: random.Random,
    diversity_penalty: bool = False,
    prior_pool: list[int] | None = None,
) -> tuple[float, tuple[int, ...], int]:
    pool = prior_pool or list(range(context.n))
    if len(pool) < context.k:
        pool = list(range(context.n))
    population_size = min(
        PARAM["ga_population_max"],
        max(PARAM["ga_population_min"], budget // PARAM["ga_budget_divisor"]),
    )
    population = [sanitize(rng.sample(pool, context.k), context, rng) for _ in range(population_size)]
    evaluated = 0
    best_score = -1.0
    best_comp: tuple[int, ...] = tuple()

    def fitness(comp: tuple[int, ...]) -> float:
        score = score_indices(comp, context)
        if diversity_penalty and len(comp) > 1:
            idx = np.array(comp, dtype=int)
            pair = context.interface_matrix[np.ix_(idx, idx)]
            similarity = float(pair[np.triu_indices(len(idx), 1)].mean())
            score = PARAM["sdfga_score_weight"] * score + PARAM["sdfga_diversity_weight"] * (
                1.0 - similarity
            )
        return score

    while evaluated < budget:
        scored = [(fitness(comp), comp) for comp in population]
        evaluated += len(population)
        scored.sort(reverse=True, key=lambda item: item[0])
        if scored[0][0] > best_score:
            best_score, best_comp = scored[0]
        elites = [comp for _, comp in scored[: max(2, population_size // PARAM["ga_elite_divisor"])]]
        next_population = elites[:]
        while len(next_population) < population_size:
            parent_pool = scored[: min(len(scored), PARAM["ga_parent_pool"])]
            tournament_size = min(len(parent_pool), PARAM["ga_tournament_size"])
            parent_a = max(rng.sample(parent_pool, tournament_size), key=lambda item: item[0])[1]
            parent_b = max(rng.sample(parent_pool, tournament_size), key=lambda item: item[0])[1]
            child = mutate(
                crossover(parent_a, parent_b, context, rng),
                context,
                rng,
                rate=PARAM["ga_mutation_rate"],
            )
            next_population.append(child)
        population = next_population
    return best_score, best_comp, min(evaluated, budget)


def run_daaga(context: RequestContext, budget: int, rng: random.Random) -> tuple[float, tuple[int, ...], int]:
    pheromone = np.ones(context.n, dtype=float)
    prior = individual_prior(context)
    evaluated = 0
    best_score = -1.0
    best_comp: tuple[int, ...] = tuple()
    while evaluated < budget:
        probabilities = pheromone * (prior + 1e-6)
        probabilities = probabilities / probabilities.sum()
        comp = tuple(
            sorted(
                rng.choices(
                    range(context.n),
                    weights=probabilities,
                    k=context.k * PARAM["daaga_sample_multiplier"],
                )
            )
        )
        comp = sanitize(comp, context, rng)
        comp = mutate(comp, context, rng, rate=PARAM["daaga_mutation_rate"])
        reward = score_indices(comp, context)
        evaluated += 1
        if reward > best_score:
            best_score, best_comp = reward, comp
        pheromone *= PARAM["daaga_evaporation"]
        for idx in comp:
            pheromone[idx] += max(PARAM["daaga_reward_floor"], PARAM["daaga_reward_scale"] * reward)
    return best_score, best_comp, evaluated


def run_mwoa(context: RequestContext, budget: int, rng: random.Random) -> tuple[float, tuple[int, ...], int]:
    pop_size = min(
        PARAM["mwoa_population_max"],
        max(PARAM["mwoa_population_min"], budget // PARAM["mwoa_budget_divisor"]),
    )
    whales = [random_composition(context, rng) for _ in range(pop_size)]
    scores = [score_indices(w, context) for w in whales]
    evaluated = pop_size
    best_idx = int(np.argmax(scores))
    best = whales[best_idx]
    best_score = scores[best_idx]
    while evaluated < budget:
        new_whales = []
        for whale in whales:
            if rng.random() < PARAM["mwoa_best_move_probability"]:
                child = crossover(best, whale, context, rng)
                child = mutate(child, context, rng, rate=PARAM["mwoa_mutation_rate"])
            else:
                child = random_composition(context, rng)
            new_whales.append(child)
        whales = new_whales
        scores = [score_indices(w, context) for w in whales]
        evaluated += len(whales)
        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best_score, best = scores[idx], whales[idx]
    return best_score, best, min(evaluated, budget)


def run_cssa(context: RequestContext, budget: int, rng: random.Random) -> tuple[float, tuple[int, ...], int]:
    pop_size = min(
        PARAM["cssa_population_max"],
        max(PARAM["cssa_population_min"], budget // PARAM["cssa_budget_divisor"]),
    )
    sparrows = [random_composition(context, rng) for _ in range(pop_size)]
    evaluated = 0
    best_score = -1.0
    best_comp: tuple[int, ...] = tuple()
    prior = individual_prior(context)
    top_prior = list(
        np.argsort(prior)[::-1][
            : max(context.k, min(context.n, context.k * PARAM["cssa_top_pool_multiplier"]))
        ]
    )
    while evaluated < budget:
        scored = [(score_indices(s, context), s) for s in sparrows]
        evaluated += len(sparrows)
        scored.sort(reverse=True, key=lambda item: item[0])
        if scored[0][0] > best_score:
            best_score, best_comp = scored[0]
        discoverer_count = max(2, pop_size // PARAM["cssa_discoverer_divisor"])
        discoverers = [s for _, s in scored[:discoverer_count]]
        followers = [s for _, s in scored[discoverer_count:]]
        new_pop = discoverers[:]
        for follower in followers:
            if rng.random() < PARAM["cssa_follow_best_probability"]:
                new_pop.append(
                    mutate(
                        crossover(best_comp, follower, context, rng),
                        context,
                        rng,
                        PARAM["cssa_mutation_rate"],
                    )
                )
            elif len(top_prior) >= context.k:
                new_pop.append(sanitize(rng.sample(top_prior, context.k), context, rng))
            else:
                new_pop.append(random_composition(context, rng))
        sparrows = new_pop
    return best_score, best_comp, min(evaluated, budget)


def run_bpsc_ga(context: RequestContext, budget: int, rng: random.Random) -> tuple[float, tuple[int, ...], int]:
    prior = individual_prior(context)
    reduced_size = max(
        context.k,
        min(
            context.n,
            max(context.k * PARAM["bpsc_pool_multiplier"], int(context.n * PARAM["bpsc_pool_fraction"])),
        ),
    )
    reduced = [int(i) for i in np.argsort(prior)[::-1][:reduced_size]]
    return run_ga(context, budget, rng, prior_pool=reduced)


def run_pk_idpso(context: RequestContext, budget: int, rng: random.Random) -> tuple[float, tuple[int, ...], int]:
    particle_count = min(
        PARAM["pk_particle_max"],
        max(PARAM["pk_particle_min"], budget // PARAM["pk_budget_divisor"]),
    )
    prior = individual_prior(context)
    top_pool = list(
        np.argsort(prior)[::-1][
            : max(context.k, min(context.n, context.k * PARAM["pk_top_pool_multiplier"]))
        ]
    )
    particles = [random_composition(context, rng) for _ in range(particle_count)]
    personal = list(particles)
    personal_scores = [score_indices(p, context) for p in particles]
    evaluated = particle_count
    g_idx = int(np.argmax(personal_scores))
    global_best = personal[g_idx]
    global_score = personal_scores[g_idx]
    while evaluated < budget:
        new_particles = []
        for i, particle in enumerate(particles):
            base = list(particle)
            if rng.random() < PARAM["pk_global_probability"]:
                base.extend(global_best)
            if rng.random() < PARAM["pk_personal_probability"]:
                base.extend(personal[i])
            if rng.random() < PARAM["pk_prior_probability"] and len(top_pool) >= context.k:
                base.extend(rng.sample(top_pool, min(context.k, len(top_pool))))
            rng.shuffle(base)
            child = mutate(sanitize(base, context, rng), context, rng, rate=PARAM["pk_mutation_rate"])
            new_particles.append(child)
        particles = new_particles
        scores = [score_indices(p, context) for p in particles]
        evaluated += len(particles)
        for i, score in enumerate(scores):
            if score > personal_scores[i]:
                personal_scores[i] = score
                personal[i] = particles[i]
            if score > global_score:
                global_score = score
                global_best = particles[i]
    return global_score, global_best, min(evaluated, budget)


def run_contextual_mab_baseline(
    context: RequestContext, budget: int, rng: random.Random
) -> tuple[float, tuple[int, ...], int]:
    if budget <= 0:
        return 0.0, tuple(), 0
    arm_count = max(1, min(753, budget, max(20, context.n * 3)))
    arms = [random_composition(context, rng) for _ in range(arm_count)]
    prior = individual_prior(context)
    if context.n >= context.k:
        arms[0] = tuple(sorted(int(i) for i in np.argsort(prior)[::-1][: context.k]))

    pulls = np.zeros(arm_count, dtype=int)
    estimates = np.zeros(arm_count, dtype=float)
    best_score = -1.0
    best_comp: tuple[int, ...] = tuple()
    for t in range(1, budget + 1):
        unexplored = np.where(pulls == 0)[0]
        if unexplored.size:
            arm_idx = int(unexplored[0])
        else:
            confidence = np.sqrt(2.0 * math.log(t) / pulls)
            arm_idx = int(np.argmax(estimates + confidence))
        reward = score_indices(arms[arm_idx], context)
        pulls[arm_idx] += 1
        estimates[arm_idx] += (reward - estimates[arm_idx]) / pulls[arm_idx]
        if reward > best_score:
            best_score = reward
            best_comp = arms[arm_idx]
    return best_score, best_comp, budget


RUNNERS: dict[str, Callable[[RequestContext, int, random.Random], tuple[float, tuple[int, ...], int]]] = {
    "random_search": run_random_search,
    "greedy": run_greedy,
    "epsilon_greedy": run_epsilon_greedy,
    "ga": run_ga,
    "daaga": run_daaga,
    "mwoa": run_mwoa,
    "cssa": run_cssa,
    "sdfga": lambda ctx, budget, rng: run_ga(ctx, budget, rng, diversity_penalty=True),
    "bpsc_ga": run_bpsc_ga,
    "pk_idpso": run_pk_idpso,
}


def ids_for(comp: tuple[int, ...], context: RequestContext) -> str:
    return "|".join(context.candidates[i]["unique_service_id"] for i in comp)


def run() -> None:
    rng = random.Random(SEED)
    requests, services = load_inputs()
    mab_baseline = load_mab_baseline()
    builder = ContextBuilder(services)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for request_number, request in enumerate(requests, start=1):
        context = builder.build(request)
        n, k = context.n, context.k
        total = combination_count(n, k)
        target = target_percent(total)
        budget = evaluation_budget(total)
        baseline_row = mab_baseline.get(request["request_id"], {})
        mab_target = baseline_row.get("target_40_percent_combinations", "")
        mab_evaluated = baseline_row.get("evaluated_combinations_count", "")
        mab_score = baseline_row.get("best_score", "")
        mab_solution = baseline_row.get("best_solution_unique_service_ids", "")
        if COMPUTE_MISSING_MAB_BASELINE and n >= k and total:
            mab_target = str(target_percent(total, 0.40))
            mab_eval_budget = mab_budget(total)
            mab_rng = random.Random(rng.randint(0, 10**12))
            score, comp, evaluated = run_contextual_mab_baseline(context, mab_eval_budget, mab_rng)
            mab_evaluated = str(evaluated)
            mab_score = round(score, 6)
            mab_solution = ids_for(comp, context)
        row: dict[str, object] = {
            "service_request_id": request["request_id"],
            "composition_type": request["composition_type"],
            "task_type": request.get("functional_aspect", {}).get("task_type", ""),
            "composition_length_K": k,
            "candidate_services_N": n,
            "total_possible_combinations_N_choose_K": str(total),
            "mab_baseline_target_40_percent_combinations": mab_target,
            "mab_baseline_evaluated_combinations": mab_evaluated,
            "mab_baseline_best_score": mab_score,
            "mab_baseline_best_solution_unique_service_ids": mab_solution,
            "target_25_percent_combinations" if abs(SEARCH_FRACTION - 0.25) < 1e-9 else "target_algorithm_combinations": str(target),
            "algorithm_sampling_fraction": SEARCH_FRACTION,
            "evaluated_combinations_count": 0,
        }
        if n < k or budget == 0:
            for algorithm in ALGORITHMS:
                row[f"{algorithm}_best_score"] = ""
                row[f"{algorithm}_best_solution_unique_service_ids"] = ""
                row[f"{algorithm}_evaluated_combinations"] = 0
                row[f"{algorithm}_runtime_seconds"] = 0.0
            rows.append(row)
            continue

        total_evaluations = 0
        for algorithm in ALGORITHMS:
            method_rng = random.Random(rng.randint(0, 10**12))
            method_started = time.perf_counter()
            score, comp, evaluated = RUNNERS[algorithm](context, budget, method_rng)
            runtime = time.perf_counter() - method_started
            total_evaluations += evaluated
            row[f"{algorithm}_best_score"] = round(score, 6)
            row[f"{algorithm}_best_solution_unique_service_ids"] = ids_for(comp, context)
            row[f"{algorithm}_evaluated_combinations"] = evaluated
            row[f"{algorithm}_runtime_seconds"] = round(runtime, 6)
        row["evaluated_combinations_count"] = total_evaluations
        rows.append(row)
        if request_number % 50 == 0:
            print(f"processed {request_number}/{len(requests)} collaborative requests")

    fieldnames = [
        "service_request_id",
        "composition_type",
        "task_type",
        "composition_length_K",
        "candidate_services_N",
        "total_possible_combinations_N_choose_K",
        "mab_baseline_target_40_percent_combinations",
        "mab_baseline_evaluated_combinations",
        "mab_baseline_best_score",
        "mab_baseline_best_solution_unique_service_ids",
        "target_25_percent_combinations" if abs(SEARCH_FRACTION - 0.25) < 1e-9 else "target_algorithm_combinations",
        "algorithm_sampling_fraction",
        "evaluated_combinations_count",
    ]
    for algorithm in ALGORITHMS:
        fieldnames.extend(
            [
                f"{algorithm}_best_score",
                f"{algorithm}_best_solution_unique_service_ids",
                f"{algorithm}_evaluated_combinations",
                f"{algorithm}_runtime_seconds",
            ]
        )
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output_csv": str(OUTPUT_PATH),
        "source_requests": str(REQUEST_PATH),
        "source_dataset": str(DATASET_PATH),
        "source_mab_baseline": str(MAB_BASELINE_PATH),
        "collaborative_requests_processed": len(rows),
        "algorithms": ALGORITHMS,
        "parameter_profile": PARAMETER_PROFILE,
        "algorithm_parameters": PARAM,
        "algorithm_sampling_fraction": SEARCH_FRACTION,
        "algorithm_target_percent": round(SEARCH_FRACTION * 100, 3),
        "mab_baseline_sampling_fraction": 0.40,
        "mab_max_trials": MAB_MAX_TRIALS,
        "computed_missing_mab_baseline": COMPUTE_MISSING_MAB_BASELINE,
        "max_trials_per_method": MAX_TRIALS_PER_METHOD,
        "requests_with_no_feasible_solution": sum(
            1 for row in rows if int(row["candidate_services_N"]) < int(row["composition_length_K"])
        ),
        "total_evaluated_combinations": sum(int(row["evaluated_combinations_count"]) for row in rows),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "seed": SEED,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()

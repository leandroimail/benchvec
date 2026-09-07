"""Configuration models and file loading utilities for BenchVec."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DistanceMetric = Literal["l2", "cosine"]
EngineName = Literal["faiss", "chromadb", "lancedb", "sqlite_vec", "turbovec"]


def _default_enabled_engines() -> list[EngineName]:
    return ["faiss", "chromadb", "lancedb"]


class TimeoutConfig(BaseModel):
    """Default timeout values in seconds for each benchmark phase."""

    model_config = ConfigDict(extra="forbid")

    ingest_batch_seconds: int = Field(default=1800, gt=0)
    build_index_seconds: int = Field(default=1800, gt=0)
    warmup_query_seconds: int = Field(default=300, gt=0)
    measured_query_seconds: int = Field(default=900, gt=0)
    reopen_seconds: int = Field(default=600, gt=0)


class RuntimePaths(BaseModel):
    """Filesystem paths used by benchmark runs."""

    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Field(default=Path(".data"))
    benchmark_dir: Path = Field(default=Path(".benchmarks"))
    cache_dir: Path = Field(default=Path(".cache"))


class DatasetConfig(BaseModel):
    """Dataset selection and preprocessing parameters."""

    model_config = ConfigDict(extra="forbid")

    primary: Literal[
        "cohere_wikipedia", "simple_wikipedia", "fast_wikipedia", "sift1m", "gist1m"
    ] = "fast_wikipedia"
    fallback_enabled: bool = True
    seed: int = Field(default=42, ge=0)
    sample_size: int = Field(default=5_000, gt=0)
    ground_truth_k: int = Field(default=100, gt=0)
    ground_truth_query_count: int = Field(default=100, gt=0)
    min_text_characters: int = Field(default=64, gt=0)
    chunk_size: int = Field(default=256, gt=0)
    chunk_overlap: int = Field(default=32, ge=0)
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cohere_dataset_name: str = "Cohere/wikipedia-22-12-simple-embeddings"
    cohere_dataset_config: str = "en"
    cohere_split: str = "train"
    simple_wikipedia_name: str = "wikipedia"
    simple_wikipedia_config: str = "20220301.simple"
    huggingface_token_env_var: str = "HF_TOKEN"
    fast_wikipedia_dataset_name: str = "not-lain/wikipedia-small-3000-embedded"
    fast_wikipedia_split: str = "train"
    texmex_base_url: str = "http://corpus-texmex.irisa.fr"
    texmex_download_timeout_seconds: int = Field(default=600, gt=0)


class LanceDbConfig(BaseModel):
    """LanceDB index and query-tuning parameters.

    The ``ivfpq`` default is the scientific-fairness fix: without it the
    adapter only ever runs full scans. The adapter constructor keeps its own
    conservative ``"none"`` default so legacy callers are unaffected.
    """

    model_config = ConfigDict(extra="forbid")

    index_type: Literal["none", "ivfpq"] = "ivfpq"
    num_partitions: int = Field(default=8, gt=0)
    num_sub_vectors: int = Field(default=16, gt=0)
    nprobes: int = Field(default=8, gt=0)
    # 0 or 1 means "no refinement" (lancedb's default); >=2 enables it.
    refine_factor: int = Field(default=1, ge=0)


class ChromaConfig(BaseModel):
    """ChromaDB HNSW search parameters.

    HNSW ``ef`` is fixed at collection creation, so sweeping it requires a
    rebuild per grid point; the grid stays disabled by default.
    """

    model_config = ConfigDict(extra="forbid")

    search_ef: int = Field(default=64, gt=0)
    ef_search_grid_enabled: bool = False
    ef_search_grid: list[int] = Field(default_factory=list)


class TurboVecConfig(BaseModel):
    """TurboVec quantization parameters.

    ``bit_width`` is a build-time knob (the index is constructed with it),
    so sweeping it requires a rebuild + re-ingest per grid point (mirrors
    the Chroma ``ef`` constraint). ``calibrate`` enables the optional TQ+
    per-coordinate calibration (~1024-row sample) at build time.
    """

    model_config = ConfigDict(extra="forbid")

    bit_width: Literal[2, 3, 4] = 4
    calibrate: bool = False


class ParamSweepConfig(BaseModel):
    """Search-parameter grids for the Recall-QPS Pareto sweep workload."""

    model_config = ConfigDict(extra="forbid")

    faiss_nprobe: list[int] = Field(default_factory=list)
    lancedb_nprobes: list[int] = Field(default_factory=list)
    lancedb_refine_factor: list[int] = Field(default_factory=list)
    turbovec_bit_width: list[int] = Field(default_factory=list)
    max_points: int = Field(default=20_000, gt=0)

    @model_validator(mode="after")
    def _validate_grid_values(self) -> ParamSweepConfig:
        """Reject invalid grid values (nprobe/nprobes >= 1, refine >= 0)."""

        for grid_name, minimum in (
            ("faiss_nprobe", 1),
            ("lancedb_nprobes", 1),
            ("lancedb_refine_factor", 0),
        ):
            grid = getattr(self, grid_name)
            if any(value < minimum for value in grid):
                raise ValueError(f"{grid_name} values must be >= {minimum}")
        # Bit widths are fixed by the library contract (probed: 2, 3, 4).
        if any(value not in (2, 3, 4) for value in self.turbovec_bit_width):
            raise ValueError("turbovec_bit_width values must be in {2, 3, 4}")
        return self


class EngineConfig(BaseModel):
    """Engine feature flags and ANN parameters."""

    model_config = ConfigDict(extra="forbid")

    enabled: list[EngineName] = Field(default_factory=_default_enabled_engines)
    distance_metric: DistanceMetric = "l2"
    faiss_index_type: Literal["flat", "ivfflat"] = "flat"
    faiss_nlist: int = Field(default=100, gt=0)
    # 0 means "FAISS library default" (only meaningful for ivfflat; flat
    # ignores it). The adapter translates the sentinel before touching the index.
    faiss_nprobe: int = Field(default=0, ge=0)
    lancedb: LanceDbConfig = Field(default_factory=LanceDbConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    turbovec: TurboVecConfig = Field(default_factory=TurboVecConfig)

    @model_validator(mode="after")
    def _validate_turbovec_metric(self) -> EngineConfig:
        """Fail fast when turbovec is enabled under an unsupported metric.

        TurboVec's kernel is inner-product-only (probed), so an ``l2``
        campaign would only fail mid-run inside the adapter (blast radius:
        the whole multi-engine run). Rejecting it at config load keeps the
        error at the earliest possible moment.
        """

        if "turbovec" in self.enabled and self.distance_metric == "l2":
            raise ValueError(
                "engines.turbovec requires engines.distance_metric='cosine' "
                "(turbovec's kernel is inner-product-only; l2 is not supported)"
            )
        return self


class WorkloadConfig(BaseModel):
    """Workload execution parameters derived from PRD03."""

    model_config = ConfigDict(extra="forbid")

    repetitions: int = Field(default=3, ge=1)
    warmup_queries: int = Field(default=10, ge=0)
    k_values: list[int] = Field(default_factory=lambda: [10])
    ingest_batch_sizes: list[int] = Field(default_factory=lambda: [1_000])
    ingest_rates_per_second: list[int] = Field(default_factory=lambda: [100])
    concurrent_workers: list[int] = Field(default_factory=lambda: [1])
    sample_sizes: list[int] = Field(default_factory=list)
    max_c2_queries: int = Field(default=20_000, gt=0)
    param_sweep: ParamSweepConfig = Field(default_factory=ParamSweepConfig)


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration document."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = "benchvec"
    seed: int = Field(default=42, ge=0)
    paths: RuntimePaths = Field(default_factory=RuntimePaths)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    engines: EngineConfig = Field(default_factory=EngineConfig)
    workloads: WorkloadConfig = Field(default_factory=WorkloadConfig)


def _read_config_file(config_path: Path) -> dict[str, Any]:
    """Read a YAML or TOML configuration file into a dictionary."""

    suffix = config_path.suffix.lower()
    content = config_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(content) or {}
    elif suffix == ".toml":
        loaded = tomllib.loads(content)
    else:
        msg = f"Unsupported configuration format: {suffix}"
        raise ValueError(msg)
    if not isinstance(loaded, dict):
        msg = "Configuration root must be a mapping"
        raise ValueError(msg)
    return loaded


def load_config(config_path: str | Path) -> BenchmarkConfig:
    """Load and validate a benchmark configuration file."""

    path = Path(config_path)
    raw_config = _read_config_file(path)
    return BenchmarkConfig.model_validate(raw_config)

from __future__ import annotations

import csv
import json
import mimetypes
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


INDEX_SCHEMA_VERSION = 1
RUN_FILES = (
    "args.json",
    "results.json",
    "record.jsonl",
    "record.json",
    "summary.csv",
    "chat_log.jsonl",
    "compressed_graph.pdf",
    "compressed_graph.svg",
)
RENDER_SUFFIXES = (".svg", ".pdf", ".png")


def _load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).resolve().parent / ".env")
_load_dotenv()


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _uri_join(base: str, relative_path: str) -> str:
    return f"{base.rstrip('/')}/{relative_path.lstrip('/')}"


def _artifact_key(relative_path: str) -> str:
    return relative_path.replace("/", "_").replace(".", "_").replace("-", "_")


def _round_sort_key(path_or_uri: str) -> tuple[int, str]:
    name = Path(path_or_uri).name
    match = re.search(r"round_(\d+)", name)
    return (int(match.group(1)) if match else 10**9, name)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    last_row: dict[str, str] | None = None
    max_round: int | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                last_row = row
                round_num = _safe_int(row.get("round"))
                if round_num is not None:
                    max_round = round_num if max_round is None else max(max_round, round_num)
    except OSError:
        return {}

    stats: dict[str, Any] = {}
    if max_round is not None:
        stats["rounds"] = max_round
    if last_row:
        obs_text = last_row.get("obs_text") or ""
        match = re.search(r"Total team score:\s*(-?\d+)", obs_text)
        if match:
            stats["score"] = int(match.group(1))
    return stats


def _metadata_from_label(label: str) -> dict[str, Any]:
    parts = label.split("/")
    if len(parts) >= 4 and parts[0].startswith("old_"):
        return {
            "archive": parts[0],
            "model": parts[1],
            "experiment": parts[2],
            "seed": parts[3],
        }
    if len(parts) >= 3:
        return {"model": parts[0], "experiment": parts[1], "seed": parts[2]}
    return {"model": parts[0] if parts else label, "experiment": "", "seed": ""}


def _latest_mtime(path: Path) -> str | None:
    latest = 0.0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                latest = max(latest, item.stat().st_mtime)
    except OSError:
        return None
    if latest <= 0:
        return None
    return datetime.fromtimestamp(latest, tz=timezone.utc).isoformat()


def build_index_records(
    data_root: str | Path,
    artifact_uri_prefix: str | None = None,
) -> list[dict[str, Any]]:
    root = Path(data_root)
    if not root.exists():
        return []

    records: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("**/summary.csv")):
        run_dir = summary_path.parent
        label = run_dir.relative_to(root).as_posix()
        args = _load_json(run_dir / "args.json")
        results = _load_json(run_dir / "results.json")
        label_meta = _metadata_from_label(label)

        base_uri = _uri_join(artifact_uri_prefix, label) if artifact_uri_prefix else run_dir.as_posix()
        artifacts: dict[str, str] = {}
        for filename in RUN_FILES:
            path = run_dir / filename
            if path.exists():
                artifacts[_artifact_key(filename)] = _uri_join(base_uri, filename)

        render_dir = run_dir / "renders"
        render_files = [
            path
            for path in render_dir.iterdir()
            if path.is_file() and path.suffix.lower() in RENDER_SUFFIXES
        ] if render_dir.exists() else []
        render_files = sorted(render_files, key=lambda p: _round_sort_key(p.name))
        if render_dir.exists():
            artifacts["renders_dir"] = _uri_join(base_uri, "renders")

        summary_stats = _summary_stats(summary_path)
        record = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "label": label,
            "run_id": label.replace("/", "__"),
            "model": args.get("model") or args.get("model_name") or label_meta.get("model"),
            "provider": args.get("provider"),
            "experiment": args.get("exp_name") or label_meta.get("experiment"),
            "seed": args.get("seed") if args.get("seed") is not None else label_meta.get("seed"),
            "preset": args.get("preset"),
            "map": args.get("preset"),
            "score": results.get("score", summary_stats.get("score")),
            "rounds": results.get("rounds", summary_stats.get("rounds")),
            "valid_action_rate": results.get("valid_action_rate") or results.get("action_success_rate"),
            "prompt_tokens": results.get("prompt_tokens"),
            "completion_tokens": results.get("completion_tokens"),
            "total_tokens": results.get("total_tokens"),
            "updated_at": _latest_mtime(run_dir),
            "artifact_base_uri": base_uri,
            "artifacts": artifacts,
            "render_count": len(render_files),
            "render_exts": sorted({path.suffix.lower().lstrip(".") for path in render_files}),
        }
        records.append(record)

    return records


def write_index_records(records: list[dict[str, Any]], output_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "runs.jsonl"
    csv_path = out_dir / "runs.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    columns = [
        "label",
        "run_id",
        "model",
        "provider",
        "experiment",
        "seed",
        "preset",
        "map",
        "score",
        "rounds",
        "valid_action_rate",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "updated_at",
        "artifact_base_uri",
        "render_count",
        "render_exts",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {column: record.get(column) for column in columns}
            row["render_exts"] = ",".join(record.get("render_exts") or [])
            writer.writerow(row)

    return jsonl_path, csv_path


def load_index_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


@dataclass(frozen=True)
class ExperimentRef:
    label: str
    source: str
    artifact_base_uri: str
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def artifact_uri(self, relative_path: str) -> str:
        key = _artifact_key(relative_path)
        if key in self.artifacts:
            return self.artifacts[key]
        return _uri_join(self.artifact_base_uri, relative_path)


def _ref_from_record(record: dict[str, Any], source: str) -> ExperimentRef:
    label = record.get("label") or record.get("run_path") or record.get("run_id")
    if not label:
        raise ValueError(f"Index record is missing a label: {record}")
    return ExperimentRef(
        label=label,
        source=source,
        artifact_base_uri=record["artifact_base_uri"],
        artifacts=dict(record.get("artifacts") or {}),
        metadata=dict(record),
    )


def _parse_gcs_uri(uri: str, default_bucket: str, default_prefix: str = "") -> tuple[str, str]:
    if uri.startswith("gs://"):
        parsed = urlparse(uri)
        return parsed.netloc, unquote(parsed.path.lstrip("/"))
    if uri.startswith("https://storage.googleapis.com/"):
        parsed = urlparse(uri)
        parts = parsed.path.lstrip("/").split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid Cloud Storage URL: {uri}")
        return parts[0], unquote(parts[1])

    key = uri.lstrip("/")
    if default_prefix:
        key = f"{default_prefix.strip('/')}/{key}"
    return default_bucket, key


def _gcs_client(project: str | None = None):
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError("Install google-cloud-storage to use Cloud Storage.") from exc

    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if credentials_json:
        from google.oauth2 import service_account

        info = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(info)
        return storage.Client(project=project or info.get("project_id"), credentials=credentials)
    return storage.Client(project=project)


class ExperimentStore:
    def __init__(
        self,
        experiments: dict[str, ExperimentRef],
        backend: str,
        cache_dir: str | Path = ".cache/tom-app",
        gcs_bucket: str | None = None,
        gcs_prefix: str = "",
        fallback_error: str | None = None,
    ) -> None:
        self.experiments = dict(sorted(experiments.items()))
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix.strip("/")
        self.fallback_error = fallback_error
        self._client = None

    @classmethod
    def from_env(cls) -> "ExperimentStore":
        source = (_env("TOM_APP_DATA_SOURCE", default="auto") or "auto").lower()
        cache_dir = _env("TOM_APP_CACHE_DIR", default=".cache/tom-app") or ".cache/tom-app"
        bucket = _env(
            "TOM_APP_GCS_BUCKET",
            "GCS_BUCKET_NAME",
            "LLM_MARL_GCS_BUCKET",
            "EXPERIMENT_GCS_BUCKET",
        )
        prefix = (_env("TOM_APP_GCS_PREFIX", "LLM_MARL_GCS_PREFIX", default="") or "").strip("/")
        index_blob = _env("TOM_APP_GCS_INDEX", default="metadata/runs.jsonl") or "metadata/runs.jsonl"

        if source in {"auto", "gcs", "cloud"} and bucket:
            try:
                store = cls({}, "gcs", cache_dir=cache_dir, gcs_bucket=bucket, gcs_prefix=prefix)
                records = store._download_index(index_blob)
                refs = {record.get("label") or record.get("run_path") or record.get("run_id"): _ref_from_record(record, "gcs") for record in records}
                refs = {label: ref for label, ref in refs.items() if label}
                if refs:
                    return cls(refs, "gcs", cache_dir=cache_dir, gcs_bucket=bucket, gcs_prefix=prefix)
                raise RuntimeError("Cloud index contained no experiments.")
            except Exception as exc:
                if source in {"gcs", "cloud"}:
                    return cls._local_from_env(cache_dir=cache_dir, fallback_error=str(exc))
                return cls._local_from_env(cache_dir=cache_dir, fallback_error=str(exc))

        return cls._local_from_env(cache_dir=cache_dir)

    @classmethod
    def _local_from_env(cls, cache_dir: str | Path, fallback_error: str | None = None) -> "ExperimentStore":
        explicit_root = _env("TOM_APP_DATA_ROOT")
        if explicit_root:
            roots = [Path(explicit_root)]
        else:
            roots = [Path("sample_data"), Path("data")]

        for root in roots:
            records = build_index_records(root)
            if records:
                refs = {record["label"]: _ref_from_record(record, "local") for record in records}
                return cls(refs, f"local:{root.as_posix()}", cache_dir=cache_dir, fallback_error=fallback_error)
        return cls({}, "local:empty", cache_dir=cache_dir, fallback_error=fallback_error)

    def refresh(self) -> "ExperimentStore":
        return ExperimentStore.from_env()

    def _ensure_client(self):
        if self._client is None:
            self._client = _gcs_client(project=_env("TOM_APP_GCP_PROJECT", "GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"))
        return self._client

    def _download_index(self, index_blob: str) -> list[dict[str, Any]]:
        if not self.gcs_bucket:
            raise RuntimeError("No Cloud Storage bucket configured.")
        index_uri = _uri_join(f"gs://{self.gcs_bucket}/{self.gcs_prefix}", index_blob) if self.gcs_prefix else f"gs://{self.gcs_bucket}/{index_blob}"
        text = self.read_bytes_from_gcs(index_uri).decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def read_bytes_from_gcs(self, uri: str) -> bytes:
        if not self.gcs_bucket:
            raise RuntimeError("No Cloud Storage bucket configured.")
        bucket_name, blob_name = _parse_gcs_uri(uri, self.gcs_bucket)
        client = self._ensure_client()
        return client.bucket(bucket_name).blob(blob_name).download_as_bytes()

    def get(self, label: str) -> ExperimentRef:
        return self.experiments[label]

    def artifact_path(self, label: str, relative_path: str) -> Path:
        ref = self.get(label)
        if ref.source == "local":
            return Path(ref.artifact_base_uri) / relative_path

        target = self.cache_dir / label / relative_path
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        uri = ref.artifact_uri(relative_path)
        target.write_bytes(self.read_bytes_from_gcs(uri))
        return target

    def artifact_exists(self, label: str, relative_path: str) -> bool:
        ref = self.get(label)
        if ref.source == "local":
            return (Path(ref.artifact_base_uri) / relative_path).exists()
        return _artifact_key(relative_path) in ref.artifacts

    def read_text(self, label: str, relative_path: str) -> str:
        return self.artifact_path(label, relative_path).read_text(encoding="utf-8")

    def list_render_files(self, label: str, suffixes: tuple[str, ...] = RENDER_SUFFIXES) -> list[str]:
        ref = self.get(label)
        suffixes = tuple(s.lower() for s in suffixes)
        if ref.source == "local":
            render_dir = Path(ref.artifact_base_uri) / "renders"
            if not render_dir.exists():
                return []
            return [
                path.name
                for path in sorted(render_dir.iterdir(), key=lambda p: _round_sort_key(p.name))
                if path.is_file() and path.suffix.lower() in suffixes
            ]

        render_dir = ref.artifacts.get("renders_dir")
        if not render_dir or not self.gcs_bucket:
            return []
        bucket_name, prefix = _parse_gcs_uri(render_dir, self.gcs_bucket)
        client = self._ensure_client()
        names = []
        for blob in client.list_blobs(bucket_name, prefix=prefix.rstrip("/") + "/"):
            name = Path(blob.name).name
            if Path(name).suffix.lower() in suffixes:
                names.append(name)
        return sorted(names, key=_round_sort_key)


def upload_file_to_gcs(
    local_path: str | Path,
    bucket_name: str,
    blob_name: str,
    client: Any | None = None,
) -> None:
    client = client or _gcs_client()
    content_type = mimetypes.guess_type(str(local_path))[0]
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(
        str(local_path),
        content_type=content_type,
    )

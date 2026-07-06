#!/usr/bin/env python3

"""Formatter for accumulating Structured3D assets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DEFAULT_WORKERS = int(os.environ.get("DEFAULT_WORKERS", "16"))
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
UINT8_VLEN = h5py.vlen_dtype(np.dtype("uint8"))
INT32_VLEN = h5py.vlen_dtype(np.dtype("int32"))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Pack Structured3D data into COMBINED_DATASET.")
	parser.add_argument(
		"--combined-dataset",
		default=os.environ.get("COMBINED_DATASET"),
		help="Root of the combined dataset.",
	)
	parser.add_argument(
		"--curr-dataset",
		default=os.environ.get("CURR_DATASET"),
		help="Root of the latest partial dataset copy.",
	)
	parser.add_argument(
		"--manifest-name",
		default="manifest.json",
		help="Name of the JSON manifest written into COMBINED_DATASET.",
	)
	parser.add_argument(
		"--delete-originals",
		action="store_true",
		help="Delete packed source files after writing the combined artifacts.",
	)
	parser.add_argument(
		"--input-tar-gz",
		default=os.environ.get("COMBINED_TAR_GZ"),
		help="Path to a .tar.gz archive to ingest in streaming mode.",
	)
	parser.add_argument(
		"--extract-root",
		default=os.environ.get("WORKDIR"),
		help="Temporary extraction root for streaming mode.",
	)
	parser.add_argument(
		"--keep-extracted",
		action="store_true",
		help="Keep extracted files in streaming mode (for debugging).",
	)
	parser.add_argument(
		"--overwrite-jsonl",
		action="store_true",
		help="Overwrite JSONL files instead of appending to them.",
	)
	parser.add_argument(
		"--skip-existing-artifacts",
		action="store_true",
		help="Skip recreating h5/json pairs that already exist.",
	)
	parser.add_argument(
		"--tar-list-file",
		default=os.environ.get("TAR_LIST_FILE"),
		help="Optional precomputed tar listing file used to estimate progress total.",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=int(os.environ.get("SPAR7M_WORKERS", str(DEFAULT_WORKERS))),
		help="Number of parallel workers to use in tar streaming mode.",
	)
	return parser.parse_args()


def strip_tar_suffix(name: str) -> str:
	if name.endswith(".tar.gz"):
		return name[:-7]
	if name.endswith(".tgz"):
		return name[:-4]
	return Path(name).stem


def sanitize_name(name: str) -> str:
	cleaned = [character if character.isalnum() or character in {"-", "_"} else "_" for character in name]
	return "".join(cleaned).strip("_") or "unknown"


def load_manifest(manifest_path: Path, input_tar_gz: str | None, curr_dataset: Path) -> dict[str, Any]:
	if manifest_path.exists():
		with manifest_path.open("r", encoding="utf-8") as manifest_file:
			manifest = json.load(manifest_file)
		if isinstance(manifest, dict):
			manifest.setdefault("schema_version", 1)
			manifest.setdefault("source_archives", [])
			manifest.setdefault("source_root", str(curr_dataset))
			manifest.setdefault("files", {})
			manifest.setdefault("tree", {})
			manifest.setdefault("artifacts", {})
			return manifest
	return {
		"schema_version": 1,
		"source_archives": [input_tar_gz] if input_tar_gz else [],
		"source_root": str(curr_dataset),
		"files": {},
		"tree": {},
		"artifacts": {},
	}


def write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	with manifest_path.open("w", encoding="utf-8") as manifest_file:
		json.dump(manifest, manifest_file, indent=2, sort_keys=True)
		manifest_file.write("\n")


def parse_layout_pairs(file_path: Path) -> np.ndarray:
	pairs: list[int] = []
	for line in file_path.read_text(encoding="utf-8").splitlines():
		stripped = line.strip()
		if not stripped:
			continue
		fields = stripped.split()
		if len(fields) != 2:
			raise ValueError(f"Expected 2 integers per line in {file_path}, got: {line!r}")
		pairs.extend(int(field) for field in fields)
	return np.asarray(pairs, dtype=np.int32)


def parse_camera_xyz(file_path: Path) -> np.ndarray:
	values = np.fromstring(file_path.read_text(encoding="utf-8").strip(), sep=" ", dtype=np.float32)
	if values.size != 3:
		raise ValueError(f"Expected exactly 3 floats in {file_path}, got {values.size}")
	return values


def parse_camera_pose(file_path: Path) -> np.ndarray:
	values = np.fromstring(file_path.read_text(encoding="utf-8").strip(), sep=" ", dtype=np.float32)
	if values.size != 12:
		raise ValueError(f"Expected exactly 12 floats in {file_path}, got {values.size}")
	return values


def parse_relative_path(relative_path: Path) -> dict[str, Any]:
	parts = list(relative_path.parts)
	info: dict[str, Any] = {
		"relative_path": relative_path.as_posix(),
		"path_parts": parts,
		"filename": parts[-1],
		"parent_path": relative_path.parent.as_posix(),
	}
	if parts and parts[0].startswith("scene_"):
		info["scene_id"] = parts[0]
	if len(parts) >= 2:
		info["leaf_parent"] = parts[-2]
	if len(parts) >= 3 and parts[1] == "2D_rendering":
		info["rendering_group"] = parts[1]
		info["room_id"] = parts[2]
	if "panorama" in parts:
		panorama_index = parts.index("panorama")
		info["view_type"] = "panorama"
		if panorama_index + 1 < len(parts) - 1:
			info["configuration"] = parts[panorama_index + 1]
	elif "perspective" in parts:
		perspective_index = parts.index("perspective")
		info["view_type"] = "perspective"
		if perspective_index + 1 < len(parts) - 1:
			info["configuration"] = parts[perspective_index + 1]
		if perspective_index + 2 < len(parts) - 1:
			info["position_id"] = parts[perspective_index + 2]
	return info


def classify_file(relative_path: Path) -> tuple[str, str]:
	name = relative_path.name
	stem = relative_path.stem
	if relative_path.suffix == ".png":
		return "png", f"images/{sanitize_name(stem)}.h5"
	if name == "layout.txt":
		return "layout_pairs", "text/layout_pairs.h5"
	if name == "camera_xyz.txt":
		return "camera_xyz", "text/camera_xyz.h5"
	if name == "camera_pose.txt":
		return "camera_pose", "text/camera_pose.h5"
	if relative_path.suffix == ".json":
		return "json", f"json/{sanitize_name(stem)}.h5"
	if relative_path.suffix == ".txt":
		return "text", f"text/{sanitize_name(stem)}.h5"
	raise ValueError(f"Unsupported file type: {relative_path}")


def ensure_dataset(h5_file: h5py.File, dataset_name: str, dtype: Any, value_shape: tuple[int, ...] = ()) -> h5py.Dataset:
	if dataset_name in h5_file:
		return h5_file[dataset_name]
	shape = (0,) + value_shape
	maxshape = (None,) + value_shape
	chunks = (1,) + value_shape if value_shape else (1,)
	return h5_file.create_dataset(dataset_name, shape=shape, maxshape=maxshape, dtype=dtype, chunks=chunks)


def append_records(artifact_path: Path, paths: list[str], values: list[Any], dtype: Any, value_shape: tuple[int, ...] = ()) -> list[int]:
	artifact_path.parent.mkdir(parents=True, exist_ok=True)
	with h5py.File(artifact_path, "a") as h5_file:
		paths_dataset = ensure_dataset(h5_file, "paths", STRING_DTYPE)
		payload_dataset = ensure_dataset(h5_file, "payload", dtype, value_shape)
		start_index = int(payload_dataset.shape[0])
		new_size = start_index + len(values)
		paths_dataset.resize((new_size,))
		payload_dataset.resize((new_size,) + value_shape)
		paths_dataset[start_index:new_size] = paths
		payload_dataset[start_index:new_size] = values
	return list(range(start_index, start_index + len(values)))


def record_manifest_entry(manifest: dict[str, Any], relative_path: Path, entry: dict[str, Any]) -> None:
	path_key = relative_path.as_posix()
	manifest["files"][path_key] = entry
	node = manifest["tree"]
	parts = list(relative_path.parts)
	for part in parts[:-1]:
		node = node.setdefault(part, {})
	node[parts[-1]] = entry


def update_artifact_summary(manifest: dict[str, Any], artifact_rel_path: str, count: int) -> None:
	artifacts = manifest.setdefault("artifacts", {})
	artifact_summary = artifacts.setdefault(artifact_rel_path, {"count": 0})
	artifact_summary["count"] = int(artifact_summary.get("count", 0)) + count


def pack_png(relative_path: Path, source_path: Path, artifact_root: Path) -> tuple[list[int], dict[str, Any], str]:
	raw_bytes = source_path.read_bytes()
	artifact_rel_path = f"images/{sanitize_name(relative_path.stem)}.h5"
	indices = append_records(
		artifact_root / artifact_rel_path,
		[relative_path.as_posix()],
		[np.frombuffer(raw_bytes, dtype=np.uint8).copy()],
		UINT8_VLEN,
	)
	entry = {
		"artifact": artifact_rel_path,
		"index": indices[0],
		"kind": "png",
		"byte_length": len(raw_bytes),
	}
	entry.update(parse_relative_path(relative_path))
	return indices, entry, artifact_rel_path


def pack_layout(relative_path: Path, source_path: Path, artifact_root: Path) -> tuple[list[int], dict[str, Any], str]:
	values = parse_layout_pairs(source_path)
	artifact_rel_path = "text/layout_pairs.h5"
	indices = append_records(artifact_root / artifact_rel_path, [relative_path.as_posix()], [values], INT32_VLEN)
	entry = {
		"artifact": artifact_rel_path,
		"index": indices[0],
		"kind": "layout_pairs",
		"pair_count": int(values.size // 2),
	}
	entry.update(parse_relative_path(relative_path))
	return indices, entry, artifact_rel_path


def pack_camera_xyz(relative_path: Path, source_path: Path, artifact_root: Path) -> tuple[list[int], dict[str, Any], str]:
	values = parse_camera_xyz(source_path)
	artifact_rel_path = "text/camera_xyz.h5"
	indices = append_records(artifact_root / artifact_rel_path, [relative_path.as_posix()], [values], np.float32, (3,))
	entry = {
		"artifact": artifact_rel_path,
		"index": indices[0],
		"kind": "camera_xyz",
		"value_count": int(values.size),
	}
	entry.update(parse_relative_path(relative_path))
	return indices, entry, artifact_rel_path


def pack_camera_pose(relative_path: Path, source_path: Path, artifact_root: Path) -> tuple[list[int], dict[str, Any], str]:
	values = parse_camera_pose(source_path)
	artifact_rel_path = "text/camera_pose.h5"
	indices = append_records(artifact_root / artifact_rel_path, [relative_path.as_posix()], [values], np.float32, (12,))
	entry = {
		"artifact": artifact_rel_path,
		"index": indices[0],
		"kind": "camera_pose",
		"value_count": int(values.size),
	}
	entry.update(parse_relative_path(relative_path))
	return indices, entry, artifact_rel_path


def pack_json(relative_path: Path, source_path: Path, artifact_root: Path) -> tuple[list[int], dict[str, Any], str]:
	raw_text = source_path.read_text(encoding="utf-8").strip()
	artifact_rel_path = f"json/{sanitize_name(relative_path.stem)}.h5"
	indices = append_records(artifact_root / artifact_rel_path, [relative_path.as_posix()], [raw_text], STRING_DTYPE)
	entry = {
		"artifact": artifact_rel_path,
		"index": indices[0],
		"kind": "json",
		"character_count": len(raw_text),
	}
	entry.update(parse_relative_path(relative_path))
	return indices, entry, artifact_rel_path


def pack_text(relative_path: Path, source_path: Path, artifact_root: Path) -> tuple[list[int], dict[str, Any], str]:
	raw_text = source_path.read_text(encoding="utf-8").strip()
	artifact_rel_path = f"text/{sanitize_name(relative_path.stem)}.h5"
	indices = append_records(artifact_root / artifact_rel_path, [relative_path.as_posix()], [raw_text], STRING_DTYPE)
	entry = {
		"artifact": artifact_rel_path,
		"index": indices[0],
		"kind": "text",
		"character_count": len(raw_text),
	}
	entry.update(parse_relative_path(relative_path))
	return indices, entry, artifact_rel_path


def iter_source_files(root: Path) -> list[Path]:
	return sorted(path for path in root.rglob("*") if path.is_file())


def should_skip(relative_path: Path, manifest: dict[str, Any]) -> bool:
	return relative_path.as_posix() in manifest["files"]


def pack_curr_dataset(curr_dataset: Path, combined_dataset: Path, manifest: dict[str, Any]) -> None:
	packed = 0
	for source_path in iter_source_files(curr_dataset):
		relative_path = source_path.relative_to(curr_dataset)
		if should_skip(relative_path, manifest):
			continue
		try:
			file_kind, _ = classify_file(relative_path)
		except ValueError:
			continue
		if file_kind == "png":
			indices, entry, artifact_rel_path = pack_png(relative_path, source_path, combined_dataset)
		elif file_kind == "layout_pairs":
			indices, entry, artifact_rel_path = pack_layout(relative_path, source_path, combined_dataset)
		elif file_kind == "camera_xyz":
			indices, entry, artifact_rel_path = pack_camera_xyz(relative_path, source_path, combined_dataset)
		elif file_kind == "camera_pose":
			indices, entry, artifact_rel_path = pack_camera_pose(relative_path, source_path, combined_dataset)
		elif file_kind == "json":
			indices, entry, artifact_rel_path = pack_json(relative_path, source_path, combined_dataset)
		else:
			indices, entry, artifact_rel_path = pack_text(relative_path, source_path, combined_dataset)
		entry["artifact_index"] = indices[0]
		record_manifest_entry(manifest, relative_path, entry)
		update_artifact_summary(manifest, artifact_rel_path, len(indices))
		packed += 1
		if packed % 250 == 0:
			print(f"Packed {packed} files...")


def delete_source_files(curr_dataset: Path) -> None:
	for source_path in sorted((path for path in curr_dataset.rglob("*") if path.is_file()), reverse=True):
		source_path.unlink(missing_ok=True)
	for directory in sorted((path for path in curr_dataset.rglob("*") if path.is_dir()), reverse=True):
		try:
			directory.rmdir()
		except OSError:
			pass


def main() -> None:
	args = parse_args()

	if args.combined_dataset is None:
		raise SystemExit("--combined-dataset or COMBINED_DATASET must be set.")
	if args.curr_dataset is None:
		raise SystemExit("--curr-dataset or CURR_DATASET must be set.")

	combined_dataset = Path(args.combined_dataset)
	curr_dataset = Path(args.curr_dataset)
	if not curr_dataset.exists():
		raise SystemExit(f"Current dataset does not exist: {curr_dataset}")
	if not curr_dataset.is_dir():
		raise SystemExit(f"Current dataset is not a directory: {curr_dataset}")

	combined_dataset.mkdir(parents=True, exist_ok=True)
	manifest_path = combined_dataset / args.manifest_name
	manifest = load_manifest(manifest_path, args.input_tar_gz, curr_dataset)
	if args.input_tar_gz and args.input_tar_gz not in manifest["source_archives"]:
		manifest["source_archives"].append(args.input_tar_gz)

	pack_curr_dataset(curr_dataset, combined_dataset, manifest)
	write_manifest(manifest_path, manifest)

	if args.delete_originals:
		delete_source_files(curr_dataset)


if __name__ == "__main__":
	main()
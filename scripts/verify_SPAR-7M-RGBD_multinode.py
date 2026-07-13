#!/usr/bin/env python3
"""Comprehensive Verification Suite & Dataloader for SPAR-7M-RGBD HDF5/JSON Dataset.

Evaluates:
1. Path Resolution & Dataloader (SPARH5Resolver & SPARH5DataLoader)
2. Existence, Readability, and Content Identicality across all asset categories
3. Known Truncated 4x4 -> 3x3 Extrinsic Matrix handling
4. School Report Card Scored Searchable Output (TEST: ... and SCORE: ...)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from PIL import Image

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(iterable, *args, **kwargs):
		return iterable


SCENE_ASSET_FOLDERS = ("image_color", "image_depth", "video_color", "video_depth", "intrinsic", "pose", "video_pose")
IMAGE_ASSET_FOLDERS = ("image_color", "image_depth", "video_color", "video_depth")
MATRIX_ASSET_FOLDERS = ("intrinsic", "pose", "video_pose")
PACKABLE_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
RAW_COPY_FILENAMES = ("video_idx.txt", "intrinsic_color.txt", "intrinsic_depth.txt")

IMAGE_PACKING = {
	"image_color": {"h5_name": "image_color.h5", "index_name": "image_color.json", "index_field": "order"},
	"image_depth": {"h5_name": "image_depth.h5", "index_name": "image_depth.json", "index_field": "order"},
	"video_color": {"h5_name": "video_color.h5", "index_name": "video_color.json", "index_field": "order"},
	"video_depth": {"h5_name": "video_depth.h5", "index_name": "video_depth.json", "index_field": "order"},
}

MATRIX_PACKING = {
	"intrinsic": {
		"h5_name": "intrinsic.h5",
		"index_name": "intrinsic.json",
		"index_field": "index",
		"expected_shape": (3, 3),
	},
	"pose": {
		"h5_name": "pose.h5",
		"index_name": "pose.json",
		"index_field": "index",
		"expected_shape": (4, 4),
	},
	"video_pose": {
		"h5_name": "video_pose.h5",
		"index_name": "video_pose.json",
		"index_field": "index",
		"expected_shape": (4, 4),
	},
}

DATASET_SUBSETS = ("rxr", "scannet", "scannetpp", "structured3d")


@dataclass
class ResolvedTarget:
	"""Represents the mapped target location of an original dataset file inside dataset_h5."""

	original_rel_path: Path
	subset: str
	sub_subset: str
	action: str  # "pack_image", "pack_matrix", "copy", "jsonl", "ignored"
	container_path: Path  # absolute path to .h5, combined .jsonl, or copied file
	index_path: Path | None = None  # absolute path to .json index if packed
	member_key: str | None = None  # dataset key inside HDF5
	expected_shape: tuple[int, int] | None = None


@dataclass
class VerificationTestRecord:
	"""Represents a single evaluated test output line with searchable score info."""

	subset: str
	sub_subset: str
	test_name: str
	description: str
	purpose: str
	passed: bool
	details: str
	error_trace: str | None = None

	def to_searchable_line(self) -> str:
		status_badge = "PASS ✓" if self.passed else "FAIL ✗"
		line = f"TEST: [{self.subset}/{self.sub_subset}] {self.test_name} - {self.purpose} -> {status_badge} ({self.details})"
		if not self.passed and self.error_trace:
			line += f" | Error trace: {self.error_trace}"
		return line


def classify_subset(relative_path: Path) -> str:
	parts = [p.lower() for p in relative_path.parts]
	for subset in DATASET_SUBSETS:
		if subset in parts:
			return subset
	if len(parts) > 1 and parts[0] == "spar" and len(parts) > 2:
		return parts[1]
	return "other"


def classify_sub_subset(action: str, asset_type: str | None) -> str:
	if action == "jsonl":
		return "qa_jsonl"
	if action == "pack_image":
		return "images"
	if action == "pack_matrix":
		return "matrices"
	if action == "copy":
		return "raw_copies"
	return "other"


def classify_scene_member_action(relative_path: Path) -> tuple[Path | None, str | None, str]:
	parts = relative_path.parts
	for index, part in enumerate(parts):
		if part in SCENE_ASSET_FOLDERS:
			if index == 0:
				return None, None, "ignored"
			scene_root = Path(*parts[:index])
			file_name = relative_path.name
			suffix = relative_path.suffix.lower()
			if part in IMAGE_ASSET_FOLDERS:
				if suffix in PACKABLE_IMAGE_SUFFIXES:
					return scene_root, part, "pack_image"
				return scene_root, part, "copy"
			if part in MATRIX_ASSET_FOLDERS:
				if suffix == ".txt" and file_name not in RAW_COPY_FILENAMES:
					return scene_root, part, "pack_matrix"
				return scene_root, part, "copy"
			return scene_root, part, "copy"

	if relative_path.name in RAW_COPY_FILENAMES:
		return relative_path.parent, None, "copy"

	return None, None, "ignored"


class SPARH5Resolver:
	"""Resolves any original dataset file path to its target container in dataset_h5."""

	def __init__(self, combined_dataset_root: Path):
		self.combined_dataset_root = combined_dataset_root.expanduser().resolve()

	def resolve(self, relative_path: Path | str) -> ResolvedTarget:
		rel_path = Path(relative_path)
		subset = classify_subset(rel_path)

		if rel_path.suffix == ".jsonl":
			sub_sub = classify_sub_subset("jsonl", None)
			container = self.combined_dataset_root / rel_path
			return ResolvedTarget(
				original_rel_path=rel_path,
				subset=subset,
				sub_subset=sub_sub,
				action="jsonl",
				container_path=container,
			)

		scene_root, asset_type, action = classify_scene_member_action(rel_path)
		if scene_root is None or action == "ignored":
			return ResolvedTarget(
				original_rel_path=rel_path,
				subset=subset,
				sub_subset="other",
				action="ignored",
				container_path=self.combined_dataset_root / rel_path,
			)

		sub_sub = classify_sub_subset(action, asset_type)
		destination_path = self.combined_dataset_root / rel_path

		if action == "pack_image":
			packing = IMAGE_PACKING[str(asset_type)]
			container = destination_path.parent / str(packing["h5_name"])
			index_path = destination_path.parent / str(packing["index_name"])
			return ResolvedTarget(
				original_rel_path=rel_path,
				subset=subset,
				sub_subset=sub_sub,
				action=action,
				container_path=container,
				index_path=index_path,
				member_key=rel_path.name,
			)

		if action == "pack_matrix":
			packing = MATRIX_PACKING[str(asset_type)]
			container = destination_path.parent / str(packing["h5_name"])
			index_path = destination_path.parent / str(packing["index_name"])
			return ResolvedTarget(
				original_rel_path=rel_path,
				subset=subset,
				sub_subset=sub_sub,
				action=action,
				container_path=container,
				index_path=index_path,
				member_key=rel_path.name,
				expected_shape=packing.get("expected_shape"), # type: ignore[arg-type]
			)

		# action == "copy"
		return ResolvedTarget(
			original_rel_path=rel_path,
			subset=subset,
			sub_subset=sub_sub,
			action=action,
			container_path=destination_path,
		)


class SPARH5DataLoader:
	"""DataLoader API to retrieve files/arrays from dataset_h5 given their original path or ResolvedTarget."""

	def __init__(self, combined_dataset_root: Path, orig_dataset_root: Path | None = None):
		self.resolver = SPARH5Resolver(combined_dataset_root)
		self.orig_dataset_root = orig_dataset_root.expanduser().resolve() if orig_dataset_root else None
		self._index_cache: dict[Path, dict[str, Any]] = {}

	def get_json_index(self, index_path: Path) -> dict[str, Any]:
		if index_path not in self._index_cache:
			if not index_path.exists():
				self._index_cache[index_path] = {}
			else:
				try:
					with index_path.open("r", encoding="utf-8") as handle:
						self._index_cache[index_path] = json.load(handle)
				except Exception:
					self._index_cache[index_path] = {}
		return self._index_cache[index_path]

	def load_from_h5(self, target: ResolvedTarget | str | Path) -> tuple[np.ndarray, dict[str, Any]]:
		resolved = target if isinstance(target, ResolvedTarget) else self.resolver.resolve(target)
		if resolved.action not in ("pack_image", "pack_matrix"):
			raise ValueError(f"Target is not packed in HDF5: {resolved.original_rel_path} (action={resolved.action})")

		if not resolved.container_path.exists():
			raise FileNotFoundError(f"HDF5 container missing: {resolved.container_path}")

		metadata: dict[str, Any] = {}
		if resolved.index_path and resolved.member_key:
			index_data = self.get_json_index(resolved.index_path)
			metadata = index_data.get(resolved.member_key, {})

		with h5py.File(resolved.container_path, "r") as h5_file:
			if not resolved.member_key or resolved.member_key not in h5_file:
				raise KeyError(f"Member {resolved.member_key} missing inside {resolved.container_path}")
			dataset = h5_file[resolved.member_key]
			arr = dataset[:]
			return arr, metadata


def verify_target(
	resolved: ResolvedTarget,
	dataloader: SPARH5DataLoader,
	check_content: bool = True,
) -> list[VerificationTestRecord]:
	"""Verifies a resolved target and returns structured VerificationTestRecord results."""
	records: list[VerificationTestRecord] = []
	orig_file = (
		dataloader.orig_dataset_root / resolved.original_rel_path
		if dataloader.orig_dataset_root
		else None
	)
	has_orig = orig_file is not None and orig_file.exists()

	# Case 1: Packed Image
	if resolved.action == "pack_image":
		# Test 1: Existence
		h5_exists = resolved.container_path.exists()
		json_exists = resolved.index_path.exists() if resolved.index_path else False
		records.append(
			VerificationTestRecord(
				subset=resolved.subset,
				sub_subset=resolved.sub_subset,
				test_name="Image Container Existence",
				description=f"Verify HDF5 archive and JSON index exist for {resolved.original_rel_path.name}",
				purpose="Confirm image asset container has been created",
				passed=h5_exists and json_exists,
				details=f"h5={h5_exists}, json={json_exists} at {resolved.container_path.name}",
				error_trace=None if (h5_exists and json_exists) else f"Missing: {resolved.container_path}",
			)
		)
		if not h5_exists or not json_exists:
			return records

		# Test 2: Readability & Identicality
		try:
			h5_arr, meta = dataloader.load_from_h5(resolved)
			if has_orig and check_content:
				with Image.open(orig_file) as orig_img:  # type: ignore[arg-type]
					orig_arr = np.array(orig_img)
				is_equal = np.array_equal(orig_arr, h5_arr)
				details = f"Shape {h5_arr.shape}, dtype {h5_arr.dtype} identical to original"
				if not is_equal:
					details = f"Mismatch: orig shape {orig_arr.shape} vs h5 shape {h5_arr.shape}"
				records.append(
					VerificationTestRecord(
						subset=resolved.subset,
						sub_subset=resolved.sub_subset,
						test_name="Image Readability & Identicality",
						description=f"Verify {resolved.original_rel_path.name} decodes and matches original array",
						purpose="Ensure pixel-perfect lossless or preserved array conversion",
						passed=is_equal,
						details=details,
						error_trace=None if is_equal else "Arrays not equal",
					)
				)
			else:
				records.append(
					VerificationTestRecord(
						subset=resolved.subset,
						sub_subset=resolved.sub_subset,
						test_name="Image Readability",
						description=f"Verify {resolved.original_rel_path.name} decodes from HDF5",
						purpose="Ensure dataset is readable and uncorrupted",
						passed=True,
						details=f"Read shape {h5_arr.shape}, dtype {h5_arr.dtype}",
					)
				)
		except Exception as exc:
			records.append(
				VerificationTestRecord(
					subset=resolved.subset,
					sub_subset=resolved.sub_subset,
					test_name="Image Readability & Identicality",
					description=f"Verify {resolved.original_rel_path.name} decodes from HDF5",
					purpose="Ensure dataset is readable and uncorrupted",
					passed=False,
					details=f"Exception: {type(exc).__name__}: {exc}",
					error_trace=traceback.format_exc(),
				)
			)
		return records

	# Case 2: Packed Matrix
	if resolved.action == "pack_matrix":
		h5_exists = resolved.container_path.exists()
		json_exists = resolved.index_path.exists() if resolved.index_path else False
		records.append(
			VerificationTestRecord(
				subset=resolved.subset,
				sub_subset=resolved.sub_subset,
				test_name="Matrix Container Existence",
				description=f"Verify HDF5 matrix archive and JSON index exist for {resolved.original_rel_path.name}",
				purpose="Confirm matrix asset container has been created",
				passed=h5_exists and json_exists,
				details=f"h5={h5_exists}, json={json_exists} at {resolved.container_path.name}",
				error_trace=None if (h5_exists and json_exists) else f"Missing: {resolved.container_path}",
			)
		)
		if not h5_exists or not json_exists:
			return records

		try:
			h5_arr, meta = dataloader.load_from_h5(resolved)
			if has_orig and check_content:
				orig_mat = np.loadtxt(orig_file, dtype=np.float32)  # type: ignore[arg-type]
				expected_shape = resolved.expected_shape or (3, 3)

				# Check for Known Issue: 16 elements (4x4) truncated to 3x3
				if orig_mat.size == 16 and expected_shape == (3, 3):
					truncated_orig = np.asarray(orig_mat, dtype=np.float32).reshape((4, 4))[:3, :3]
					is_equal = bool(np.allclose(truncated_orig, h5_arr, atol=1e-5))
					details = "Verified known 4x4->3x3 truncated extrinsic matrix exactly matches orig[:3,:3]"
				else:
					reshaped_orig = np.asarray(orig_mat, dtype=np.float32).reshape(expected_shape)
					is_equal = bool(np.allclose(reshaped_orig, h5_arr, atol=1e-5))
					details = f"Matrix shape {h5_arr.shape} numerical match verified"

				if not is_equal:
					details = f"Numerical mismatch between original matrix and HDF5 dataset {resolved.member_key}"

				records.append(
					VerificationTestRecord(
						subset=resolved.subset,
						sub_subset=resolved.sub_subset,
						test_name="Matrix Readability & Identicality",
						description=f"Verify {resolved.original_rel_path.name} decodes and matches expected matrix",
						purpose="Verify numerical precision and handle 4x4->3x3 truncation correctly",
						passed=is_equal,
						details=details,
						error_trace=None if is_equal else details,
					)
				)
			else:
				records.append(
					VerificationTestRecord(
						subset=resolved.subset,
						sub_subset=resolved.sub_subset,
						test_name="Matrix Readability",
						description=f"Verify {resolved.original_rel_path.name} decodes from HDF5",
						purpose="Ensure matrix dataset is readable and uncorrupted",
						passed=True,
						details=f"Read matrix shape {h5_arr.shape}, dtype {h5_arr.dtype}",
					)
				)
		except Exception as exc:
			records.append(
				VerificationTestRecord(
					subset=resolved.subset,
					sub_subset=resolved.sub_subset,
					test_name="Matrix Readability & Identicality",
					description=f"Verify {resolved.original_rel_path.name} decodes from HDF5",
					purpose="Ensure matrix dataset is readable and uncorrupted",
					passed=False,
					details=f"Exception: {type(exc).__name__}: {exc}",
					error_trace=traceback.format_exc(),
				)
			)
		return records

	# Case 3: Combined JSONL
	if resolved.action == "jsonl":
		exists = resolved.container_path.exists()
		records.append(
			VerificationTestRecord(
				subset=resolved.subset,
				sub_subset=resolved.sub_subset,
				test_name="Combined JSONL Existence",
				description=f"Verify combined JSONL file exists at {resolved.original_rel_path}",
				purpose="Confirm QA JSONL file is present in target dataset",
				passed=exists,
				details=f"Path: {resolved.container_path}",
				error_trace=None if exists else f"Missing: {resolved.container_path}",
			)
		)
		if not exists:
			return records

		if has_orig and check_content:
			try:
				orig_lines = [line.strip() for line in orig_file.read_text(encoding="utf-8").splitlines() if line.strip()]  # type: ignore[union-attr]
				target_lines = set(
					line.strip()
					for line in resolved.container_path.read_text(encoding="utf-8").splitlines()
					if line.strip()
				)
				missing_count = sum(1 for line in orig_lines if line not in target_lines)
				passed = missing_count == 0
				records.append(
					VerificationTestRecord(
						subset=resolved.subset,
						sub_subset=resolved.sub_subset,
						test_name="JSONL Row Containment",
						description=f"Verify all {len(orig_lines)} rows from original appear in combined JSONL",
						purpose="Confirm no QA rows were lost during combination",
						passed=passed,
						details=f"All {len(orig_lines)} rows found in combined JSONL" if passed else f"{missing_count} rows missing",
						error_trace=None if passed else f"Missing {missing_count} JSONL rows",
					)
				)
			except Exception as exc:
				records.append(
					VerificationTestRecord(
						subset=resolved.subset,
						sub_subset=resolved.sub_subset,
						test_name="JSONL Row Containment",
						description="Verify JSONL rows",
						purpose="Confirm no QA rows were lost",
						passed=False,
						details=f"Error reading JSONL: {exc}",
						error_trace=traceback.format_exc(),
					)
				)
		return records

	# Case 4: Copied file
	if resolved.action == "copy":
		exists = resolved.container_path.exists()
		records.append(
			VerificationTestRecord(
				subset=resolved.subset,
				sub_subset=resolved.sub_subset,
				test_name="Copied File Existence",
				description=f"Verify passthrough copied file {resolved.original_rel_path.name} exists",
				purpose="Confirm uncompressed/passthrough file exists",
				passed=exists,
				details=f"Path: {resolved.container_path}",
				error_trace=None if exists else f"Missing: {resolved.container_path}",
			)
		)
		if not exists:
			return records

		if has_orig and check_content:
			try:
				orig_bytes = orig_file.read_bytes()  # type: ignore[union-attr]
				dest_bytes = resolved.container_path.read_bytes()
				is_equal = orig_bytes == dest_bytes
				records.append(
					VerificationTestRecord(
						subset=resolved.subset,
						sub_subset=resolved.sub_subset,
						test_name="Copied File Identicality",
						description=f"Verify {resolved.original_rel_path.name} matches original bytes exactly",
						purpose="Confirm lossless copy",
						passed=is_equal,
						details=f"Size {len(dest_bytes)} bytes identical" if is_equal else f"Size mismatch orig {len(orig_bytes)} vs dest {len(dest_bytes)}",
						error_trace=None if is_equal else "Byte content difference",
					)
				)
			except Exception as exc:
				records.append(
					VerificationTestRecord(
						subset=resolved.subset,
						sub_subset=resolved.sub_subset,
						test_name="Copied File Identicality",
						description=f"Verify {resolved.original_rel_path.name} matches original",
						purpose="Confirm lossless copy",
						passed=False,
						details=f"Error reading bytes: {exc}",
						error_trace=traceback.format_exc(),
					)
				)
		return records

	return records


def format_grade(percentage: float) -> str:
	if percentage >= 97.0:
		return "A+"
	if percentage >= 93.0:
		return "A"
	if percentage >= 90.0:
		return "A-"
	if percentage >= 87.0:
		return "B+"
	if percentage >= 83.0:
		return "B"
	if percentage >= 80.0:
		return "B-"
	if percentage >= 70.0:
		return "C"
	if percentage >= 60.0:
		return "D"
	return "F"


def generate_school_report_card(records: list[VerificationTestRecord]) -> str:
	"""Generates a structured, searchable School Report Card format with TEST: and SCORE: lines."""
	lines: list[str] = []
	lines.append("=" * 80)
	lines.append("SPAR-7M-RGBD VERIFICATION SCHOOL REPORT CARD")
	lines.append("=" * 80)
	lines.append("")

	# Group records by subset and sub-subset
	grouped: dict[str, dict[str, list[VerificationTestRecord]]] = {}
	for rec in records:
		grouped.setdefault(rec.subset, {}).setdefault(rec.sub_subset, []).append(rec)

	total_passed = 0
	total_tests = len(records)

	subset_scores: list[tuple[str, str, int, int]] = []

	for subset in sorted(grouped.keys()):
		subset_total = 0
		subset_passed = 0
		for sub_subset in sorted(grouped[subset].keys()):
			sub_recs = grouped[subset][sub_subset]
			sub_passed = sum(1 for r in sub_recs if r.passed)
			sub_total = len(sub_recs)
			subset_passed += sub_passed
			subset_total += sub_total
			total_passed += sub_passed

			subset_scores.append((subset, sub_subset, sub_passed, sub_total))

			lines.append("-" * 80)
			lines.append(f"SUBSET: {subset} | SUB-SUBSET: {sub_subset}")
			lines.append("-" * 80)

			for rec in sub_recs:
				lines.append(rec.to_searchable_line())

			pct = (sub_passed / sub_total * 100.0) if sub_total > 0 else 100.0
			lines.append(f"SCORE: {sub_passed}/{sub_total} ({pct:.1f}%) - SUB-SUBSET {subset}/{sub_subset}")
			lines.append("")

		sub_pct = (subset_passed / subset_total * 100.0) if subset_total > 0 else 100.0
		lines.append(f"SCORE: {subset_passed}/{subset_total} ({sub_pct:.1f}%) - SUBSET {subset}")
		lines.append("")

	lines.append("=" * 80)
	lines.append("FINAL SCHOOL REPORT CARD SUMMARY")
	lines.append("=" * 80)
	lines.append(f"{'Subset':<15} | {'Sub-Subset':<15} | {'Passed':>8} | {'Total':>8} | {'Score (%)':>10} | Grade")
	lines.append("-" * 80)

	for subset, sub_subset, p, t in subset_scores:
		pct = (p / t * 100.0) if t > 0 else 100.0
		grade = format_grade(pct)
		lines.append(f"{subset:<15} | {sub_subset:<15} | {p:>8} | {t:>8} | {pct:>9.1f}% | {grade}")

	lines.append("-" * 80)
	overall_pct = (total_passed / total_tests * 100.0) if total_tests > 0 else 100.0
	overall_grade = format_grade(overall_pct)
	lines.append(
		f"SCORE: {total_passed}/{total_tests} ({overall_pct:.1f}%) [OVERALL GRADE: {overall_grade}]"
	)
	lines.append("=" * 80)

	return "\n".join(lines)


def worker_verify_paths(
	paths_chunk: list[str],
	combined_dataset_root: str,
	orig_dataset_root: str | None,
	check_content: bool,
) -> list[dict[str, Any]]:
	dataloader = SPARH5DataLoader(
		Path(combined_dataset_root),
		Path(orig_dataset_root) if orig_dataset_root else None,
	)
	results = []
	for rel_str in paths_chunk:
		resolved = dataloader.resolver.resolve(rel_str)
		recs = verify_target(resolved, dataloader, check_content=check_content)
		for r in recs:
			results.append(
				{
					"subset": r.subset,
					"sub_subset": r.sub_subset,
					"test_name": r.test_name,
					"description": r.description,
					"purpose": r.purpose,
					"passed": r.passed,
					"details": r.details,
					"error_trace": r.error_trace,
				}
			)
	return results


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Verify SPAR-7M-RGBD Multi-Node HDF5/JSON conversion.")
	parser.add_argument(
		"--dataset-h5",
		"-c",
		default="/scratch/indrisch/SPAR-7M-RGBD_data_combined_h5_multinode",
		help="Root of combined HDF5 dataset.",
	)
	parser.add_argument(
		"--dataset",
		"-d",
		default="/scratch/indrisch/spar-rgbd-full",
		help="Root of original dataset (optional for content identicality checks).",
	)
	parser.add_argument(
		"--tar-list-file",
		default="/scratch/indrisch/spar-rgbd-full-file-list.txt",
		help="File listing of original dataset paths.",
	)
	parser.add_argument(
		"--sample-count",
		type=int,
		default=500,
		help="Number of stratified files to sample and verify (0 for full scan).",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
		help="Number of multiprocessing workers.",
	)
	parser.add_argument(
		"--no-content-check",
		action="store_true",
		help="Skip pixel/content reading checks (existence + readability only).",
	)
	parser.add_argument(
		"--report-file",
		default="verification/verification_school_report_card.txt",
		help="Path to output School Report Card file.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	combined_dataset_root = Path(args.dataset_h5).expanduser().resolve()
	orig_dataset_root = Path(args.dataset).expanduser().resolve() if args.dataset else None

	if not combined_dataset_root.exists():
		print(f"ERROR: Target combined dataset missing: {combined_dataset_root}", file=sys.stderr)
		sys.exit(1)

	all_paths: list[str] = []
	tar_list_path = Path(args.tar_list_file).expanduser().resolve()
	if tar_list_path.exists():
		print(f"Reading file list from {tar_list_path}...")
		with tar_list_path.open("r", encoding="utf-8") as handle:
			for line in handle:
				p = line.strip()
				if p and not p.endswith("/"):
					all_paths.append(p)
	else:
		print("Tar list file not found; discovering relative files in dataset-h5...")
		for root, _, files in os.walk(combined_dataset_root):
			for f in files:
				abs_p = Path(root) / f
				all_paths.append(str(abs_p.relative_to(combined_dataset_root)))

	if args.sample_count > 0 and len(all_paths) > args.sample_count:
		print(f"Sampling {args.sample_count} files out of {len(all_paths)} total files...")
		random.seed(42)
		all_paths = random.sample(all_paths, args.sample_count)
	else:
		print(f"Evaluating all {len(all_paths)} files...")

	workers = max(1, args.workers)
	chunk_size = max(1, math.ceil(len(all_paths) / workers))
	chunks = [all_paths[i : i + chunk_size] for i in range(0, len(all_paths), chunk_size)]

	check_content = not args.no_content_check

	print(f"Launching {len(chunks)} workers across {workers} CPUs...")
	records: list[VerificationTestRecord] = []

	if workers == 1:
		dict_recs = worker_verify_paths(
			all_paths,
			str(combined_dataset_root),
			str(orig_dataset_root) if orig_dataset_root and orig_dataset_root.exists() else None,
			check_content,
		)
		for d in dict_recs:
			records.append(VerificationTestRecord(**d))
	else:
		with multiprocessing.Pool(workers) as pool:
			tasks = [
				(
					chunk,
					str(combined_dataset_root),
					str(orig_dataset_root) if orig_dataset_root and orig_dataset_root.exists() else None,
					check_content,
				)
				for chunk in chunks
			]
			chunk_results = pool.starmap(worker_verify_paths, tasks)
			for res_list in chunk_results:
				for d in res_list:
					records.append(VerificationTestRecord(**d))

	report_card_text = generate_school_report_card(records)
	print(report_card_text)

	report_path = Path(args.report_file)
	report_path.parent.mkdir(parents=True, exist_ok=True)
	report_path.write_text(report_card_text, encoding="utf-8")
	print(f"\nSaved School Report Card to {report_path.resolve()}")


if __name__ == "__main__":
	main()

#!/usr/bin/env python3

"""Formatter for accumulating SPAR-7M-RGBD text and scene assets.

The script supports two modes:
1) Directory mode: append .jsonl files and pack scene folders from CURR_DATASET.
2) Streaming mode: iterate files from a .tar.gz archive, skip already indexed
	scene assets, pack new files into HDF5/JSON artifacts, and delete extracted
	files immediately to keep disk usage low.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing
import os
import queue
import shutil
import signal
import subprocess
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

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
DEFAULT_WORKERS = 16
STREAMING_STATS_KEYS = ("jsonl_appended", "packed_new", "copied_passthrough", "skipped_existing", "ignored", "unsafe_member", "failed")

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


def iter_jsonl_files(root: Path) -> Iterable[Path]:
	for dirpath, _, filenames in os.walk(root):
		for filename in filenames:
			if filename.endswith(".jsonl"):
				yield Path(dirpath) / filename


def iter_scene_dirs(root: Path) -> Iterable[Path]:
	for scene_path in root.rglob("*"):
		if scene_path.is_dir() and any((scene_path / folder_name).is_dir() for folder_name in SCENE_ASSET_FOLDERS):
			yield scene_path


def map_to_combined_path(curr_dataset: Path, combined_dataset: Path, source_path: Path) -> Path:
	relative_path = source_path.relative_to(curr_dataset)
	return combined_dataset / relative_path


def append_file(source_path: Path, destination_path: Path) -> None:
	destination_path.parent.mkdir(parents=True, exist_ok=True)
	with source_path.open("r", encoding="utf-8") as source_file, destination_path.open(
		"a", encoding="utf-8"
	) as destination_file:
		shutil.copyfileobj(source_file, destination_file)


def write_file(source_path: Path, destination_path: Path) -> None:
	destination_path.parent.mkdir(parents=True, exist_ok=True)
	with source_path.open("r", encoding="utf-8") as source_file, destination_path.open(
		"w", encoding="utf-8"
	) as destination_file:
		shutil.copyfileobj(source_file, destination_file)


def write_json(destination_path: Path, data: dict) -> None:
	destination_path.parent.mkdir(parents=True, exist_ok=True)
	with destination_path.open("w", encoding="utf-8") as json_file:
		json.dump(data, json_file, indent=2, sort_keys=True)
		json_file.write("\n")


def read_json(source_path: Path) -> dict[str, dict[str, object]]:
	if not source_path.exists():
		return {}
	with source_path.open("r", encoding="utf-8") as json_file:
		data = json.load(json_file)
	if isinstance(data, dict):
		return data
	return {}


class IndexCache:
	def __init__(self) -> None:
		self._cache: dict[Path, dict[str, dict[str, object]]] = {}

	def get(self, index_path: Path) -> dict[str, dict[str, object]]:
		if index_path not in self._cache:
			self._cache[index_path] = read_json(index_path)
		return self._cache[index_path]

	def has(self, index_path: Path, key: str) -> bool:
		return key in self.get(index_path)

	def next_position(self, index_path: Path, position_field: str) -> int:
		max_position = -1
		for entry in self.get(index_path).values():
			position = entry.get(position_field)
			if isinstance(position, int):
				max_position = max(max_position, position)
		return max_position + 1

	def upsert(self, index_path: Path, key: str, entry: dict[str, object]) -> None:
		index_data = self.get(index_path)
		index_data[key] = entry
		write_json(index_path, index_data)


def cleanup_empty_parents(path: Path, stop_path: Path) -> None:
	current_path = path
	while current_path != stop_path and stop_path in current_path.parents:
		try:
			current_path.rmdir()
		except OSError:
			break
		current_path = current_path.parent


def ensure_directory(path: Path) -> None:
	for _ in range(3):
		try:
			path.mkdir(parents=True, exist_ok=True)
			return
		except FileExistsError:
			if path.is_dir():
				return
			if not path.exists():
				continue
			raise
		except FileNotFoundError:
			# A parallel cleanup can remove an intermediate parent while
			# pathlib is recursively creating this directory. Retry rather
			# than failing the whole streaming ingest on that transient race.
			continue
	if not path.is_dir():
		path.mkdir(parents=True, exist_ok=True)


def is_safe_relative_path(relative_path: Path) -> bool:
	return not relative_path.is_absolute() and ".." not in relative_path.parts


def worker_for_key(key: str, worker_count: int) -> int:
	digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
	return int.from_bytes(digest, "big") % worker_count


def sharding_key_for_member(relative_path: Path) -> str:
	if relative_path.suffix == ".jsonl":
		return f"jsonl:{relative_path.as_posix()}"

	scene_root, asset_type, action = classify_scene_member(relative_path)
	if scene_root is not None and action != "ignored":
		return f"scene:{scene_root.as_posix()}"

	return f"ignored:{relative_path.as_posix()}"


def member_assigned_to_node(relative_path: Path, node_count: int, node_index: int) -> bool:
	return worker_for_key(sharding_key_for_member(relative_path), node_count) == node_index


@contextmanager
def open_tar_stream(input_tar_gz: Path, gzip_workers: int = 1) -> Iterator[tarfile.TarFile]:
	gzip_process = None
	stream_handle = None
	consumer_exception: BaseException | None = None
	try:
		pigz_path = shutil.which("pigz")
		if gzip_workers > 1 and pigz_path:
			gzip_process = subprocess.Popen(
				[pigz_path, "-dc", "-p", str(gzip_workers), str(input_tar_gz)],
				stdout=subprocess.PIPE,
			)
			if gzip_process.stdout is None:
				raise RuntimeError("Failed to open pigz stdout.")
			stream_handle = gzip_process.stdout
		else:
			stream_handle = gzip.open(input_tar_gz, "rb")

		with tarfile.open(fileobj=stream_handle, mode="r|", ignore_zeros=True) as archive:
			try:
				yield archive
			except BaseException as exc:
				# Exceptions raised by the caller's `with` body are injected back
				# at this yield point by contextlib. Record and re-raise them so
				# the real failure is not swallowed or replaced by pigz SIGPIPE.
				consumer_exception = exc
				raise
    
	finally:
		if stream_handle is not None:
			stream_handle.close()
		if gzip_process is not None:
			return_code = gzip_process.wait()
			if return_code != 0 and return_code != -signal.SIGPIPE:
				if consumer_exception is not None:
					consumer_exception.add_note(f"pigz also exited with status {return_code}")
				else:
					raise RuntimeError(f"pigz exited with status {return_code}")
			elif return_code == -signal.SIGPIPE and consumer_exception is None:
				# This can happen when the Python side intentionally stops reading
				# before pigz finishes writing. It is not useful to fail on SIGPIPE.
				pass


def pack_jsonl_files(curr_dataset: Path, combined_dataset: Path, overwrite: bool) -> None:
	for source_path in tqdm(iter_jsonl_files(curr_dataset), desc="Packing JSONL files", unit="file"):
		destination_path = map_to_combined_path(curr_dataset, combined_dataset, source_path)
		if overwrite:
			write_file(source_path, destination_path)
		else:
			append_file(source_path, destination_path)


def copy_file(source_path: Path, destination_path: Path) -> None:
	destination_path.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(source_path, destination_path)


def pack_encoded_file(source_path: Path, destination_folder: Path, packing_config: dict[str, str], index_cache: IndexCache) -> bool:
	file_name = source_path.name
	index_path = destination_folder / packing_config["index_name"]
	if index_cache.has(index_path, file_name):
		return False

	destination_folder.mkdir(parents=True, exist_ok=True)
	h5_path = destination_folder / packing_config["h5_name"]
	with Image.open(source_path) as image_file:
		payload = np.array(image_file)

	with h5py.File(h5_path, "a") as h5_file:
		if file_name in h5_file:
			stored = h5_file[file_name]
			index_cache.upsert(
				index_path,
				file_name,
				{
					"dataset": file_name,
					"dtype": str(stored.dtype),
					packing_config["index_field"]: index_cache.next_position(index_path, packing_config["index_field"]),
					"shape": list(stored.shape),
				},
			)
			return False
		h5_file.create_dataset(file_name, data=payload, compression="gzip", compression_opts=9)

	index_cache.upsert(
		index_path,
		file_name,
		{
			"dataset": file_name,
			"dtype": str(payload.dtype),
			packing_config["index_field"]: index_cache.next_position(index_path, packing_config["index_field"]),
			"shape": list(payload.shape),
		},
	)
	return True


def pack_matrix_file(
	source_path: Path,
	destination_folder: Path,
	packing_config: dict[str, object],
	index_cache: IndexCache,
) -> bool:
	file_name = source_path.name
	index_path = destination_folder / str(packing_config["index_name"])
	if index_cache.has(index_path, file_name):
		return False

	matrix = np.loadtxt(source_path, dtype=np.float32)
	matrix = np.asarray(matrix, dtype=np.float32).reshape(tuple(packing_config["expected_shape"]))

	destination_folder.mkdir(parents=True, exist_ok=True)
	h5_path = destination_folder / str(packing_config["h5_name"])
	with h5py.File(h5_path, "a") as h5_file:
		if file_name in h5_file:
			stored = h5_file[file_name]
			index_cache.upsert(
				index_path,
				file_name,
				{
					"dataset": file_name,
					"dtype": str(stored.dtype),
					str(packing_config["index_field"]): index_cache.next_position(index_path, str(packing_config["index_field"])),
					"shape": list(stored.shape),
				},
			)
			return False
		h5_file.create_dataset(file_name, data=matrix, compression="gzip", compression_opts=9)

	index_cache.upsert(
		index_path,
		file_name,
		{
			"dataset": file_name,
			"dtype": str(matrix.dtype),
			str(packing_config["index_field"]): index_cache.next_position(index_path, str(packing_config["index_field"])),
			"shape": list(matrix.shape),
		},
	)
	return True


def pack_encoded_folder(source_folder: Path, destination_folder: Path, h5_name: str, index_name: str) -> None:
	items = sorted(path for path in source_folder.iterdir() if path.is_file())
	if not items:
		return

	index_cache = IndexCache()
	packing_config = {"h5_name": h5_name, "index_name": index_name, "index_field": "order"}
	for source_path in items:
		pack_encoded_file(source_path, destination_folder, packing_config, index_cache)


def pack_matrix_folder(source_folder: Path, destination_folder: Path, h5_name: str, index_name: str, expected_shape: tuple[int, int]) -> None:
	items = sorted(path for path in source_folder.iterdir() if path.is_file() and path.suffix == ".txt")
	if not items:
		return

	index_cache = IndexCache()
	packing_config = {
		"h5_name": h5_name,
		"index_name": index_name,
		"index_field": "index",
		"expected_shape": expected_shape,
	}
	for source_path in items:
		pack_matrix_file(source_path, destination_folder, packing_config, index_cache)


def classify_scene_member(relative_path: Path) -> tuple[Path | None, str | None, str]:
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
					return scene_root, part, "pack"
				return scene_root, part, "copy"
			if part in MATRIX_ASSET_FOLDERS:
				if suffix == ".txt" and file_name not in RAW_COPY_FILENAMES:
					return scene_root, part, "pack"
				return scene_root, part, "copy"
			return scene_root, part, "copy"

	if relative_path.name in RAW_COPY_FILENAMES:
		return relative_path.parent, None, "copy"

	return None, None, "ignored"


def pack_scene(scene_path: Path, combined_dataset: Path, curr_dataset: Path, delete_originals: bool, index_cache: IndexCache) -> None:
	for source_path in sorted(path for path in scene_path.rglob("*") if path.is_file()):
		relative_path = source_path.relative_to(curr_dataset)
		scene_root, asset_type, action = classify_scene_member(relative_path)
		if scene_root is None:
			continue

		destination_path = combined_dataset / relative_path
		if action == "pack":
			destination_folder = destination_path.parent
			if asset_type in IMAGE_ASSET_FOLDERS:
				pack_encoded_file(source_path, destination_folder, IMAGE_PACKING[asset_type], index_cache)
			elif asset_type in MATRIX_ASSET_FOLDERS:
				pack_matrix_file(source_path, destination_folder, MATRIX_PACKING[asset_type], index_cache)
			else:
				copy_file(source_path, destination_path)
		elif action == "copy":
			copy_file(source_path, destination_path)

		if delete_originals and action in {"pack", "copy"}:
			source_path.unlink()
			cleanup_empty_parents(source_path.parent, scene_path)


def pack_scenes(curr_dataset: Path, combined_dataset: Path, delete_originals: bool) -> None:
	scene_paths = list(iter_scene_dirs(curr_dataset))
	index_cache = IndexCache()
	for scene_path in tqdm(scene_paths, desc="Packing scenes", unit="scene"):
		pack_scene(scene_path, combined_dataset, curr_dataset, delete_originals, index_cache)


def process_streamed_file(
	extracted_path: Path,
	relative_path: Path,
	combined_dataset: Path,
	index_cache: IndexCache,
	overwrite_jsonl: bool,
	skip_existing_artifacts: bool,
) -> str:
	if relative_path.suffix == ".jsonl":
		jsonl_destination = combined_dataset / relative_path
		if overwrite_jsonl:
			write_file(extracted_path, jsonl_destination)
		else:
			append_file(extracted_path, jsonl_destination)
		return "jsonl_appended"

	scene_root, asset_type, action = classify_scene_member(relative_path)
	if scene_root is None:
		return "ignored"

	destination_path = combined_dataset / relative_path
	if action == "pack":
		destination_folder = destination_path.parent
		if asset_type in IMAGE_ASSET_FOLDERS:
			was_packed = pack_encoded_file(extracted_path, destination_folder, IMAGE_PACKING[asset_type], index_cache)
			return "packed_new" if was_packed else "skipped_existing"
		if asset_type in MATRIX_ASSET_FOLDERS:
			was_packed = pack_matrix_file(extracted_path, destination_folder, MATRIX_PACKING[asset_type], index_cache)
			return "packed_new" if was_packed else "skipped_existing"
		copy_file(extracted_path, destination_path)
		return "copied_passthrough"

	if action == "copy":
		copy_file(extracted_path, destination_path)
		return "copied_passthrough"

	return "ignored"


def stream_tar_and_pack(
	input_tar_gz: Path,
	combined_dataset: Path,
	extract_root: Path,
	keep_extracted: bool,
	total_regular_files: int | None = None,
	overwrite_jsonl: bool = False,
	skip_existing_artifacts: bool = False,
	node_count: int = 1,
	node_index: int = 0,
) -> None:
	extract_root.mkdir(parents=True, exist_ok=True)
	index_cache = IndexCache()
	stats = {
		"jsonl_appended": 0,
		"packed_new": 0,
		"copied_passthrough": 0,
		"skipped_existing": 0,
		"ignored": 0,
		"unsafe_member": 0,
	}

	with open_tar_stream(input_tar_gz) as archive, tqdm(
		desc="Streaming archive members",
		unit="file",
		total=total_regular_files,
	) as progress:
		for member in archive:
			if not member.isfile():
				continue
			progress.update(1)

			relative_path = Path(member.name)
			if not is_safe_relative_path(relative_path):
				stats["unsafe_member"] += 1
				continue
			if not member_assigned_to_node(relative_path, node_count, node_index):
				continue

			member_payload = archive.extractfile(member)
			if member_payload is None:
				continue

			extracted_path = extract_root / relative_path
			ensure_directory(extracted_path.parent)
			with extracted_path.open("wb") as extracted_file:
				shutil.copyfileobj(member_payload, extracted_file)

			status = process_streamed_file(
				extracted_path,
				relative_path,
				combined_dataset,
				index_cache,
				overwrite_jsonl,
				skip_existing_artifacts,
			)
			stats[status] = stats.get(status, 0) + 1

			if not keep_extracted:
				extracted_path.unlink(missing_ok=True)
				cleanup_empty_parents(extracted_path.parent, extract_root)

	print("Streaming ingest summary:")
	for key in STREAMING_STATS_KEYS:
		print(f"  {key}: {stats.get(key, 0)}")


def count_worker_assignments(tar_list_file: Path, worker_count: int, node_count: int, node_index: int) -> tuple[list[int], int]:
	worker_counts = [0 for _ in range(worker_count)]
	unsafe_members = 0

	with tar_list_file.open("r", encoding="utf-8") as list_handle:
		for line in list_handle:
			member_name = line.rstrip("\n")
			if not member_name or member_name.endswith("/"):
				continue

			relative_path = Path(member_name)
			if not is_safe_relative_path(relative_path):
				unsafe_members += 1
				continue
			if not member_assigned_to_node(relative_path, node_count, node_index):
				continue

			worker_index = worker_for_key(sharding_key_for_member(relative_path), worker_count)
			worker_counts[worker_index] += 1

	return worker_counts, unsafe_members


def pack_stream_queue_worker(
	worker_index: int,
	task_queue,
	result_queue,
	combined_dataset: Path,
	extract_root: Path,
	keep_extracted: bool,
	overwrite_jsonl: bool,
	skip_existing_artifacts: bool,
	node_count: int,
	node_index: int,
) -> None:
	worker_extract_root = extract_root / f"node_{node_index:02d}" / f"worker_{worker_index:02d}"
	index_cache = IndexCache()
	stats = {key: 0 for key in STREAMING_STATS_KEYS}
	error_message = ""

	try:
		ensure_directory(worker_extract_root)
		while True:
			task = task_queue.get()
			if task is None:
				break

			extracted_path_text, relative_path_text = task
			extracted_path = Path(extracted_path_text)
			relative_path = Path(relative_path_text)
			try:
				status = process_streamed_file(
					extracted_path,
					relative_path,
					combined_dataset,
					index_cache,
					overwrite_jsonl,
					skip_existing_artifacts,
				)
				stats[status] = stats.get(status, 0) + 1
			except Exception:
				stats["failed"] = stats.get("failed", 0) + 1
			finally:
				if not keep_extracted:
					extracted_path.unlink(missing_ok=True)
					# Do not remove empty parent directories here. The producer writes
					# later archive members into the same worker tree and can otherwise
					# race with this cleanup while calling mkdir(parents=True).

		if stats.get("failed", 0):
			error_message = f"{stats['failed']} task(s) failed"

		if not keep_extracted:
			shutil.rmtree(worker_extract_root, ignore_errors=True)
	except Exception as exc:
		error_message = f"{type(exc).__name__}: {exc}"

	result_queue.put({"worker_index": worker_index, "stats": stats, "error": error_message})


def collect_worker_results(processes: list[multiprocessing.Process], result_queue) -> list[dict[str, object]]:
	for process in processes:
		process.join()

	worker_results = []
	while True:
		try:
			worker_results.append(result_queue.get_nowait())
		except queue.Empty:
			break

	worker_errors = []
	for result in worker_results:
		error_message = result.get("error")
		if error_message:
			worker_errors.append(f"worker_{result.get('worker_index'):02d}: {error_message}")

	for process in processes:
		if process.exitcode != 0:
			worker_errors.append(f"worker process {process.pid} exited with status {process.exitcode}")

	if len(worker_results) != len(processes):
		worker_errors.append(f"received {len(worker_results)} worker results for {len(processes)} processes")

	if worker_errors:
		raise RuntimeError("; ".join(worker_errors))

	return worker_results


def merge_stats(stats_items: Iterable[dict[str, int]]) -> dict[str, int]:
	merged_stats = {key: 0 for key in STREAMING_STATS_KEYS}
	for stats in stats_items:
		for key, value in stats.items():
			merged_stats[key] = merged_stats.get(key, 0) + value
	return merged_stats


def stream_tar_and_pack_parallel(
	input_tar_gz: Path,
	combined_dataset: Path,
	extract_root: Path,
	keep_extracted: bool,
	tar_list_file: Path,
	worker_count: int,
	overwrite_jsonl: bool = False,
	skip_existing_artifacts: bool = False,
	node_count: int = 1,
	node_index: int = 0,
) -> None:
	if worker_count < 1:
		raise ValueError("worker_count must be at least 1")

	extract_root.mkdir(parents=True, exist_ok=True)
	worker_counts, unsafe_members = count_worker_assignments(tar_list_file, worker_count, node_count, node_index)
	total_regular_files = sum(worker_counts) + unsafe_members
	print(f"Parallel streaming ingest with {worker_count} local workers on node {node_index} of {node_count}")
	for worker_index, worker_count_for_manifest in enumerate(worker_counts):
		print(f"  worker_{worker_index:02d}: {worker_count_for_manifest} regular files")
	if unsafe_members:
		print(f"  unsafe_member: {unsafe_members} skipped from tar listing")

	task_queues = [multiprocessing.Queue(maxsize=max(2, worker_count)) for _ in range(worker_count)]
	result_queue = multiprocessing.Queue()
	processes = [
		multiprocessing.Process(
			target=pack_stream_queue_worker,
			args=(
				worker_index,
				task_queues[worker_index],
				result_queue,
				combined_dataset,
				extract_root,
				keep_extracted,
				overwrite_jsonl,
				skip_existing_artifacts,
				node_count,
				node_index,
			),
		)
		for worker_index in range(worker_count)
	]
	for process in processes:
		process.start()

	producer_stats = {key: 0 for key in STREAMING_STATS_KEYS}
	producer_error = None
	try:
		with open_tar_stream(input_tar_gz, worker_count) as archive, tqdm(
			desc="Streaming archive members",
			unit="file",
			total=total_regular_files,
		) as progress:
			for member in archive:
				if not member.isfile():
					continue
				progress.update(1)

				relative_path = Path(member.name)
				if not is_safe_relative_path(relative_path):
					producer_stats["unsafe_member"] += 1
					continue
				if not member_assigned_to_node(relative_path, node_count, node_index):
					continue

				member_payload = archive.extractfile(member)
				if member_payload is None:
					continue

				worker_index = worker_for_key(sharding_key_for_member(relative_path), worker_count)
				extracted_path = extract_root / f"node_{node_index:02d}" / f"worker_{worker_index:02d}" / relative_path
				ensure_directory(extracted_path.parent)
				with extracted_path.open("wb") as extracted_file:
					shutil.copyfileobj(member_payload, extracted_file)
				task_queues[worker_index].put((str(extracted_path), relative_path.as_posix()))
	except Exception as exc:
		producer_error = exc
	finally:
		for task_queue in task_queues:
			task_queue.put(None)
		worker_results = collect_worker_results(processes, result_queue)
		for task_queue in task_queues:
			task_queue.close()
		result_queue.close()

	if producer_error is not None:
		raise producer_error

	worker_stats = []
	for result in sorted(worker_results, key=lambda item: item["worker_index"]):
		worker_index = result["worker_index"]
		stats = result["stats"]
		worker_stats.append(stats)
		print(
			f"worker_{worker_index:02d} summary: "
			+ ", ".join(f"{key}={stats.get(key, 0)}" for key in STREAMING_STATS_KEYS)
		)

	stats = merge_stats(worker_stats + [producer_stats])
	print("Parallel streaming ingest summary:")
	for key in STREAMING_STATS_KEYS:
		print(f"  {key}: {stats.get(key, 0)}")


def count_regular_files_from_list(list_file: Path, node_count: int = 1, node_index: int = 0) -> int:
	with list_file.open("r", encoding="utf-8") as handle:
		count = 0
		for line in handle:
			member_name = line.rstrip("\n")
			if not member_name or member_name.endswith("/"):
				continue
			relative_path = Path(member_name)
			if not is_safe_relative_path(relative_path):
				continue
			if not member_assigned_to_node(relative_path, node_count, node_index):
				continue
			count += 1
		return count


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Pack SPAR-7M-RGBD data into COMBINED_DATASET.")
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
		default=int(os.environ.get("SPAR7M_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", str(DEFAULT_WORKERS)))),
		help="Number of parallel workers to use in tar streaming mode.",
	)
	parser.add_argument(
		"--node-count",
		type=int,
		default=int(os.environ.get("SPAR7M_NODE_COUNT", os.environ.get("SLURM_NNODES", "1"))),
		help="Total number of nodes assigned to this job.",
	)
	parser.add_argument(
		"--node-index",
		type=int,
		default=int(os.environ.get("SPAR7M_NODE_INDEX", os.environ.get("SLURM_PROCID", "0"))),
		help="Zero-based node index for this process.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not args.combined_dataset:
		raise SystemExit("Provide --combined-dataset/COMBINED_DATASET.")
	if args.node_count < 1:
		raise SystemExit("--node-count must be at least 1.")
	if args.node_index < 0 or args.node_index >= args.node_count:
		raise SystemExit("--node-index must satisfy 0 <= index < node-count.")

	combined_dataset = Path(args.combined_dataset).expanduser().resolve()

	if args.input_tar_gz:
		input_tar_gz = Path(args.input_tar_gz).expanduser().resolve()
		if not input_tar_gz.exists():
			raise SystemExit(f"Archive not found: {input_tar_gz}")
		extract_root = Path(args.extract_root).expanduser().resolve() if args.extract_root else (Path.cwd() / "stream_extract")
		worker_count = max(1, args.workers)
		if worker_count > 1:
			if not args.tar_list_file:
				raise SystemExit("Parallel tar streaming requires --tar-list-file/TAR_LIST_FILE.")
			tar_list_file = Path(args.tar_list_file).expanduser().resolve()
			if not tar_list_file.exists():
				raise SystemExit(f"Tar list file not found: {tar_list_file}")
			stream_tar_and_pack_parallel(
				input_tar_gz,
				combined_dataset,
				extract_root,
				args.keep_extracted,
				tar_list_file,
				worker_count,
				args.overwrite_jsonl,
				args.skip_existing_artifacts,
				args.node_count,
				args.node_index,
			)
		else:
			if args.tar_list_file:
				tar_list_file = Path(args.tar_list_file).expanduser().resolve()
				if not tar_list_file.exists():
					raise SystemExit(f"Tar list file not found: {tar_list_file}")
				regular_file_total = count_regular_files_from_list(tar_list_file, args.node_count, args.node_index)
			else:
				regular_file_total = None
			stream_tar_and_pack(
				input_tar_gz,
				combined_dataset,
				extract_root,
				args.keep_extracted,
				regular_file_total,
				args.overwrite_jsonl,
				args.skip_existing_artifacts,
				args.node_count,
				args.node_index,
			)
		return

	if not args.curr_dataset:
		raise SystemExit("Provide --curr-dataset/CURR_DATASET when --input-tar-gz is not used.")

	curr_dataset = Path(args.curr_dataset).expanduser().resolve()

	pack_jsonl_files(curr_dataset, combined_dataset, args.overwrite_jsonl)
	pack_scenes(curr_dataset, combined_dataset, args.delete_originals)


if __name__ == "__main__":
	main()

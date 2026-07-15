#!/usr/bin/env python3

"""Formatter for accumulating Structured3D text and scene assets into HDF5.

This script processes the extracted Structured3D native directories (e.g., 2D_rendering)
and packs the images and camera data into HDF5 files using a format compatible with SPAR-7M-RGBD.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from PIL import Image

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(iterable, *args, **kwargs):
		return iterable

IMAGE_PACKING = {
	"image_color": {"h5_name": "image_color.h5", "index_name": "image_color.json", "index_field": "order"},
	"image_depth": {"h5_name": "image_depth.h5", "index_name": "image_depth.json", "index_field": "order"},
}

MATRIX_PACKING = {
	"pose": {
		"h5_name": "pose.h5",
		"index_name": "pose.json",
		"index_field": "index",
		"expected_shape": (4, 4),
	},
}


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

def pack_encoded_file(source_path: Path, scene_root: Path, destination_folder: Path, packing_config: dict[str, str], index_cache: IndexCache) -> bool:
	# Use relative path to avoid overwriting files with the same name across rooms/perspectives
	file_name = str(source_path.relative_to(scene_root)).replace("/", "_")
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

def pack_matrix_file(source_path: Path, scene_root: Path, destination_folder: Path, packing_config: dict[str, object], index_cache: IndexCache) -> bool:
	file_name = str(source_path.relative_to(scene_root)).replace("/", "_")
	index_path = destination_folder / str(packing_config["index_name"])
	if index_cache.has(index_path, file_name):
		return False

	matrix = np.loadtxt(source_path, dtype=np.float32)
	try:
		matrix = np.asarray(matrix, dtype=np.float32).reshape(tuple(packing_config["expected_shape"]))
	except ValueError:
		print(f"WARNING: Reshaping matrix failed for {source_path}. Cutting to the expected shape {packing_config['expected_shape']}")
		x, y = tuple(packing_config["expected_shape"])
		try:
			matrix = np.asarray(matrix, dtype=np.float32).reshape((4, 4))[:x, :y]
		except ValueError:
			# For 3-element camera_xyz.txt, we pad it to 4x4
			if matrix.size == 3:
				padded = np.eye(4, dtype=np.float32)
				padded[:3, 3] = matrix
				matrix = padded
			elif matrix.size == 12:
				padded = np.eye(4, dtype=np.float32)
				padded[:3, :4] = matrix.reshape((3, 4))
				matrix = padded
			else:
				print(f"Skipping {source_path}: unrecognized matrix size {matrix.size}")
				return False

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


def copy_file(source_path: Path, scene_root: Path, destination_folder: Path) -> None:
	file_name = str(source_path.relative_to(scene_root)).replace("/", "_")
	destination_path = destination_folder / file_name
	destination_path.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(source_path, destination_path)


def classify_and_process(source_path: Path, scene_root: Path, combined_dataset: Path, index_cache: IndexCache) -> None:
	suffix = source_path.suffix.lower()
	name = source_path.name
	destination_folder = combined_dataset / scene_root.name

	if suffix in (".png", ".jpg", ".jpeg"):
		if name.startswith("rgb_"):
			pack_encoded_file(source_path, scene_root, destination_folder, IMAGE_PACKING["image_color"], index_cache)
		elif name == "depth.png":
			pack_encoded_file(source_path, scene_root, destination_folder, IMAGE_PACKING["image_depth"], index_cache)
		else:
			# Just copy other images (semantic, instance, normal, albedo) without packing
			copy_file(source_path, scene_root, destination_folder)
	elif suffix == ".txt":
		if name in ("camera_pose.txt", "camera_xyz.txt"):
			pack_matrix_file(source_path, scene_root, destination_folder, MATRIX_PACKING["pose"], index_cache)
		else:
			copy_file(source_path, scene_root, destination_folder)
	elif suffix == ".json":
		copy_file(source_path, scene_root, destination_folder)


def pack_scene(scene_path: Path, combined_dataset: Path, index_cache: IndexCache) -> None:
	for source_path in sorted(path for path in scene_path.rglob("*") if path.is_file()):
		classify_and_process(source_path, scene_path, combined_dataset, index_cache)


def iter_scene_dirs(root: Path) -> Iterable[Path]:
	for scene_path in root.rglob("scene_*"):
		if scene_path.is_dir():
			yield scene_path

def pack_scenes(extract_root: Path, combined_dataset: Path) -> None:
	scene_paths = list(iter_scene_dirs(extract_root))
	index_cache = IndexCache()
	for scene_path in tqdm(scene_paths, desc="Packing scenes", unit="scene"):
		pack_scene(scene_path, combined_dataset, index_cache)

def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--combined-dataset", type=Path, required=True)
	parser.add_argument("--extract-root", type=Path, required=True)
	args = parser.parse_args()

	pack_scenes(args.extract_root, args.combined_dataset)

if __name__ == "__main__":
	main()

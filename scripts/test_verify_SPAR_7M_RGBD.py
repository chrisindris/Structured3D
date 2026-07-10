#!/usr/bin/env python3
"""Pytest suite for verifying SPAR-7M-RGBD multi-node HDF5/JSON conversion logic and School Report Card output."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

_mod = importlib.import_module("verify_SPAR-7M-RGBD_multinode")
SPARH5DataLoader = _mod.SPARH5DataLoader
SPARH5Resolver = _mod.SPARH5Resolver
VerificationTestRecord = _mod.VerificationTestRecord
generate_school_report_card = _mod.generate_school_report_card
verify_target = _mod.verify_target


def test_resolver_image_packing(tmp_path: Path):
	resolver = SPARH5Resolver(tmp_path)
	resolved = resolver.resolve("spar/scannet/images/scene0000_00/image_color/00000.jpg")
	assert resolved.subset == "scannet"
	assert resolved.sub_subset == "images"
	assert resolved.action == "pack_image"
	assert resolved.member_key == "00000.jpg"
	assert resolved.container_path == tmp_path / "spar/scannet/images/scene0000_00/image_color/image_color.h5"
	assert resolved.index_path == tmp_path / "spar/scannet/images/scene0000_00/image_color/image_color.json"


def test_resolver_matrix_packing(tmp_path: Path):
	resolver = SPARH5Resolver(tmp_path)
	resolved = resolver.resolve("spar/scannet/images/scene0267_00/intrinsic/extrinsic_depth.txt")
	assert resolved.subset == "scannet"
	assert resolved.sub_subset == "matrices"
	assert resolved.action == "pack_matrix"
	assert resolved.member_key == "extrinsic_depth.txt"
	assert resolved.expected_shape == (3, 3)
	assert resolved.container_path == tmp_path / "spar/scannet/images/scene0267_00/intrinsic/intrinsic.h5"
	assert resolved.index_path == tmp_path / "spar/scannet/images/scene0267_00/intrinsic/intrinsic.json"


def test_resolver_jsonl_and_raw_copy(tmp_path: Path):
	resolver = SPARH5Resolver(tmp_path)
	res_jsonl = resolver.resolve("spar/qa_jsonl/train/depth_prediction_oo/fill/fill_76837.jsonl")
	assert res_jsonl.action == "jsonl"
	assert res_jsonl.sub_subset == "qa_jsonl"
	assert res_jsonl.container_path == tmp_path / "spar/qa_jsonl/train/depth_prediction_oo/fill/fill_76837.jsonl"

	res_copy = resolver.resolve("spar/scannet/images/scene0000_00/video_idx.txt")
	assert res_copy.action == "copy"
	assert res_copy.sub_subset == "raw_copies"
	assert res_copy.container_path == tmp_path / "spar/scannet/images/scene0000_00/video_idx.txt"


def test_report_card_formatting():
	records = [
		VerificationTestRecord(
			subset="scannet",
			sub_subset="images",
			test_name="Image Readability & Identicality",
			description="Verify 00000.jpg",
			purpose="Ensure lossless conversion",
			passed=True,
			details="Shape (1080, 1440, 3), dtype uint8 identical",
		),
		VerificationTestRecord(
			subset="scannet",
			sub_subset="matrices",
			test_name="Matrix Readability & Identicality",
			description="Verify extrinsic_depth.txt",
			purpose="Check 4x4->3x3 truncation",
			passed=True,
			details="Verified known 4x4->3x3 truncated extrinsic matrix",
		),
		VerificationTestRecord(
			subset="rxr",
			sub_subset="images",
			test_name="Image Container Existence",
			description="Verify image_color.h5",
			purpose="Container presence",
			passed=False,
			details="Missing container",
			error_trace="FileNotFoundError",
		),
	]
	report = generate_school_report_card(records)
	assert "SPAR-7M-RGBD VERIFICATION SCHOOL REPORT CARD" in report
	assert "TEST: [scannet/images] Image Readability & Identicality" in report
	assert "PASS ✓ (Shape (1080, 1440, 3), dtype uint8 identical)" in report
	assert "TEST: [rxr/images] Image Container Existence" in report
	assert "FAIL ✗" in report
	assert "SCORE: 1/1 (100.0%) - SUB-SUBSET scannet/images" in report
	assert "FINAL SCHOOL REPORT CARD SUMMARY" in report
	assert "OVERALL GRADE:" in report


def test_verify_known_truncated_matrix(tmp_path: Path):
	import h5py

	orig_dir = tmp_path / "orig/spar/scannet/images/scene0267_00/intrinsic"
	orig_dir.mkdir(parents=True)
	orig_file = orig_dir / "extrinsic_depth.txt"

	# Create a 4x4 matrix (16 elements)
	full_4x4 = np.arange(16, dtype=np.float32).reshape((4, 4))
	np.savetxt(orig_file, full_4x4)

	# In dataset_h5, create intrinsic/intrinsic.h5 storing the top-left 3x3
	h5_dir = tmp_path / "h5/spar/scannet/images/scene0267_00/intrinsic"
	h5_dir.mkdir(parents=True)
	h5_path = h5_dir / "intrinsic.h5"
	json_path = h5_dir / "intrinsic.json"

	truncated_3x3 = full_4x4[:3, :3]
	with h5py.File(h5_path, "w") as h5f:
		h5f.create_dataset("extrinsic_depth.txt", data=truncated_3x3)
	json_path.write_text(
		json.dumps(
			{
				"extrinsic_depth.txt": {
					"dataset": "extrinsic_depth.txt",
					"dtype": "float32",
					"index": 0,
					"shape": [3, 3],
				}
			}
		),
		encoding="utf-8",
	)

	dataloader = SPARH5DataLoader(tmp_path / "h5", tmp_path / "orig")
	resolved = dataloader.resolver.resolve("spar/scannet/images/scene0267_00/intrinsic/extrinsic_depth.txt")
	recs = verify_target(resolved, dataloader, check_content=True)

	assert all(r.passed for r in recs)
	matrix_rec = [r for r in recs if r.test_name == "Matrix Readability & Identicality"][0]
	assert "4x4->3x3 truncated" in matrix_rec.details

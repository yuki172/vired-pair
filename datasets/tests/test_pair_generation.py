"""
Smoke tests for the dataset-level pair generation and object type encoding.

These tests do NOT require any actual dataset files on disk.  They exercise:

  - YOLO class_id → object type mapping
  - Object type validation assertion
  - Candidate pair generation (new 3-class logic)
  - Canonical pair ordering (lower type-id first)
  - Pair labelling (1 / 0 against a ground-truth set)
  - File-parsing helpers (_parse_yolo_labels, _parse_pair_labels, _boxes_to_masks)

Run with:
    pytest datasets/tests/test_pair_generation.py -v
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import List

import pytest
import torch

from utils.pair_builder import (
    TEXT,
    SYMBOL,
    SYMBOL_TEXT,
    YOLO_CLASS_TO_OBJECT_TYPE,
    count_candidate_pairs,
    generate_candidate_pairs,
    is_feasible_pair,
    label_candidate_pairs,
    make_gt_pair_set,
    validate_object_types,
    yolo_class_id_to_object_type,
    yolo_class_ids_to_object_types,
)
from datasets.plan_relation_dataset import (
    _boxes_to_masks,
    _parse_pair_labels,
    _parse_yolo_labels,
)


# ────────────────────────────────────────────────────────────────────────── #
# YOLO class_id mapping                                                      #
# ────────────────────────────────────────────────────────────────────────── #

class TestYoloMapping:
    """Every YOLO class_id must map to the correct object type."""

    def test_text_classes(self):
        for cid in (0, 2, 5):
            assert yolo_class_id_to_object_type(cid) == TEXT, (
                f"class_id {cid} should map to TEXT({TEXT})"
            )

    def test_symbol_classes(self):
        for cid in (1, 3):
            assert yolo_class_id_to_object_type(cid) == SYMBOL, (
                f"class_id {cid} should map to SYMBOL({SYMBOL})"
            )

    def test_symbol_text_class(self):
        assert yolo_class_id_to_object_type(4) == SYMBOL_TEXT, (
            f"class_id 4 should map to SYMBOL_TEXT({SYMBOL_TEXT})"
        )

    def test_all_six_class_ids_covered(self):
        """All class_ids 0-5 must be in the mapping."""
        for cid in range(6):
            assert cid in YOLO_CLASS_TO_OBJECT_TYPE, (
                f"class_id {cid} missing from YOLO_CLASS_TO_OBJECT_TYPE"
            )

    def test_unknown_class_id_raises(self):
        with pytest.raises(KeyError, match="Unknown YOLO class_id"):
            yolo_class_id_to_object_type(99)

    def test_bulk_conversion(self):
        ids   = [0, 1, 2, 3, 4, 5]
        types = yolo_class_ids_to_object_types(ids)
        assert types == [TEXT, SYMBOL, TEXT, SYMBOL, SYMBOL_TEXT, TEXT]

    def test_type_constants_are_distinct(self):
        assert len({TEXT, SYMBOL, SYMBOL_TEXT}) == 3


# ────────────────────────────────────────────────────────────────────────── #
# Object type validation                                                     #
# ────────────────────────────────────────────────────────────────────────── #

class TestValidateObjectTypes:
    def test_valid_types_pass(self):
        validate_object_types([TEXT, SYMBOL, SYMBOL_TEXT, TEXT, SYMBOL])

    def test_invalid_type_raises(self):
        with pytest.raises(AssertionError, match="not a valid type"):
            validate_object_types([TEXT, SYMBOL, 99])

    def test_empty_list_passes(self):
        validate_object_types([])


# ────────────────────────────────────────────────────────────────────────── #
# Pair feasibility                                                           #
# ────────────────────────────────────────────────────────────────────────── #

class TestPairFeasibility:
    """is_feasible_pair must allow all cross-type and reject all same-type."""

    @pytest.mark.parametrize("ta, tb", [
        (SYMBOL,      TEXT),
        (TEXT,        SYMBOL),
        (SYMBOL,      SYMBOL_TEXT),
        (SYMBOL_TEXT, SYMBOL),
        (TEXT,        SYMBOL_TEXT),
        (SYMBOL_TEXT, TEXT),
    ])
    def test_cross_type_pairs_are_feasible(self, ta, tb):
        assert is_feasible_pair(ta, tb), f"({ta}, {tb}) should be feasible"

    @pytest.mark.parametrize("t", [TEXT, SYMBOL, SYMBOL_TEXT])
    def test_same_type_pairs_are_not_feasible(self, t):
        assert not is_feasible_pair(t, t), f"({t}, {t}) should NOT be feasible"


# ────────────────────────────────────────────────────────────────────────── #
# Candidate pair generation — core validation                                #
# ────────────────────────────────────────────────────────────────────────── #

class TestGenerateCandidatePairs:

    def test_three_distinct_types_gives_three_pairs(self):
        """
        Spec example:
            object_types = [TEXT, SYMBOL, SYMBOL_TEXT]
            expected pairs: (TEXT,SYMBOL), (TEXT,SYMBOL_TEXT), (SYMBOL,SYMBOL_TEXT)
            total = 3
        """
        types = [TEXT, SYMBOL, SYMBOL_TEXT]   # indices 0, 1, 2
        pairs = generate_candidate_pairs(types)
        assert len(pairs) == 3, f"Expected 3 pairs, got {len(pairs)}: {pairs}"

    def test_three_distinct_types_correct_pairs(self):
        types = [TEXT, SYMBOL, SYMBOL_TEXT]
        pairs = generate_candidate_pairs(types)
        pair_set = {frozenset(p) for p in pairs}
        assert frozenset({0, 1}) in pair_set, "TEXT(0) ↔ SYMBOL(1) missing"
        assert frozenset({0, 2}) in pair_set, "TEXT(0) ↔ SYMBOL_TEXT(2) missing"
        assert frozenset({1, 2}) in pair_set, "SYMBOL(1) ↔ SYMBOL_TEXT(2) missing"

    def test_canonical_ordering_lower_type_id_first(self):
        """The subject (first) index must always have type-id ≤ object (second)."""
        types = [TEXT, SYMBOL, SYMBOL_TEXT]
        pairs = generate_candidate_pairs(types)
        for (i, j) in pairs:
            ti, tj = types[i], types[j]
            assert ti <= tj, (
                f"Pair ({i},{j}) has type_i={ti} > type_j={tj}; "
                "expected lower type-id first"
            )

    def test_all_same_type_gives_zero_pairs(self):
        types = [TEXT, TEXT, TEXT, TEXT]
        pairs = generate_candidate_pairs(types)
        assert pairs == [], f"Expected 0 pairs, got {pairs}"

    def test_same_type_symbol_gives_zero_pairs(self):
        pairs = generate_candidate_pairs([SYMBOL, SYMBOL])
        assert pairs == []

    def test_same_type_symbol_text_gives_zero_pairs(self):
        pairs = generate_candidate_pairs([SYMBOL_TEXT, SYMBOL_TEXT])
        assert pairs == []

    def test_two_types_text_symbol(self):
        """N=2 with [SYMBOL, TEXT] → 1 pair, TEXT first."""
        types = [SYMBOL, TEXT]   # indices 0=SYMBOL, 1=TEXT
        pairs = generate_candidate_pairs(types)
        assert len(pairs) == 1
        i, j = pairs[0]
        assert types[i] == TEXT and types[j] == SYMBOL, (
            "Pair subject must be TEXT (lower type-id)"
        )

    def test_no_duplicate_pairs(self):
        types = [TEXT, SYMBOL, SYMBOL_TEXT, TEXT, SYMBOL]
        pairs = generate_candidate_pairs(types)
        pair_sets = [frozenset(p) for p in pairs]
        assert len(pair_sets) == len(set(pair_sets)), "Duplicate pairs detected"

    def test_no_same_type_pairs_in_result(self):
        types = [TEXT, SYMBOL, SYMBOL_TEXT, TEXT, SYMBOL, SYMBOL_TEXT]
        pairs = generate_candidate_pairs(types)
        for (i, j) in pairs:
            assert types[i] != types[j], (
                f"Same-type pair ({i},{j}) should not be generated"
            )

    def test_count_matches_generate(self):
        types = [TEXT, SYMBOL, SYMBOL_TEXT, TEXT, SYMBOL, SYMBOL_TEXT]
        assert count_candidate_pairs(types) == len(generate_candidate_pairs(types))

    def test_empty_object_list_gives_zero_pairs(self):
        assert generate_candidate_pairs([]) == []

    def test_single_object_gives_zero_pairs(self):
        assert generate_candidate_pairs([SYMBOL]) == []

    def test_larger_example(self):
        """
        6 objects: 3 TEXT, 2 SYMBOL, 1 SYMBOL_TEXT
        Feasible pairs:
          TEXT↔SYMBOL:       3*2 = 6
          TEXT↔SYMBOL_TEXT:  3*1 = 3
          SYMBOL↔SYMBOL_TEXT: 2*1 = 2
          Total = 11
        """
        types = [TEXT, TEXT, TEXT, SYMBOL, SYMBOL, SYMBOL_TEXT]
        pairs = generate_candidate_pairs(types)
        assert len(pairs) == 11, f"Expected 11 pairs, got {len(pairs)}"

    def test_architecture_pair_builder_consistency(self):
        """
        The architecture's PairBuilder._cross_type_indices should produce the
        same pairs as generate_candidate_pairs for any object type sequence.

        This test imports the architecture's helper directly to verify parity.
        """
        import torch
        from vired_model.models.pair_builder import PairBuilder

        types_list = [TEXT, SYMBOL, SYMBOL_TEXT, TEXT, SYMBOL]
        types_tensor = torch.tensor(types_list, dtype=torch.long)

        # Architecture's output: (row_idx, col_idx)
        row, col = PairBuilder._cross_type_indices(types_tensor)
        arch_pairs = set(
            frozenset({row[k].item(), col[k].item()}) for k in range(len(row))
        )

        # Dataset-level output
        ds_pairs = {frozenset(p) for p in generate_candidate_pairs(types_list)}

        assert arch_pairs == ds_pairs, (
            f"Architecture and dataset pair sets differ.\n"
            f"Arch: {arch_pairs}\nDataset: {ds_pairs}"
        )


# ────────────────────────────────────────────────────────────────────────── #
# Pair labelling                                                             #
# ────────────────────────────────────────────────────────────────────────── #

class TestLabelCandidatePairs:
    def test_known_pairs_get_label_one(self):
        gt = make_gt_pair_set([(0, 1), (2, 3)])
        candidates = [(0, 1), (0, 2), (1, 2), (2, 3)]
        labels = label_candidate_pairs(candidates, gt)
        assert labels == [1, 0, 0, 1]

    def test_reversed_gt_pair_still_matches(self):
        """Pair matching is unordered: (1,0) in GT matches candidate (0,1)."""
        gt = make_gt_pair_set([(1, 0)])
        labels = label_candidate_pairs([(0, 1)], gt)
        assert labels == [1]

    def test_empty_gt_gives_all_zeros(self):
        gt = make_gt_pair_set([])
        labels = label_candidate_pairs([(0, 1), (1, 2)], gt)
        assert labels == [0, 0]

    def test_empty_candidates_gives_empty(self):
        gt = make_gt_pair_set([(0, 1)])
        labels = label_candidate_pairs([], gt)
        assert labels == []

    def test_all_gt_gives_all_ones(self):
        candidates = [(0, 1), (0, 2), (1, 2)]
        gt = make_gt_pair_set(candidates)
        labels = label_candidate_pairs(candidates, gt)
        assert all(l == 1 for l in labels)


# ────────────────────────────────────────────────────────────────────────── #
# File-parsing helpers                                                       #
# ────────────────────────────────────────────────────────────────────────── #

class TestParsePairLabels:
    def test_valid_file(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("0 1\n2 3\n1 4\n")
        pairs = _parse_pair_labels(p)
        assert pairs == [(0, 1), (2, 3), (1, 4)]

    def test_missing_file_returns_empty(self, tmp_path):
        pairs = _parse_pair_labels(tmp_path / "missing.txt")
        assert pairs == []

    def test_blank_lines_are_skipped(self, tmp_path):
        p = tmp_path / "blank.txt"
        p.write_text("\n0 1\n\n2 3\n")
        assert _parse_pair_labels(p) == [(0, 1), (2, 3)]


class TestParseYoloLabels:
    def test_valid_file_parses_correctly(self, tmp_path):
        p = tmp_path / "test.txt"
        # class_id=1 (SYMBOL), centre (0.5, 0.5), size (0.2, 0.1)
        p.write_text("1 0.5 0.5 0.2 0.1\n")
        boxes, class_ids = _parse_yolo_labels(p, img_w=100.0, img_h=100.0)
        assert class_ids == [1]
        assert boxes.shape == (1, 4)
        # x1 = (0.5 - 0.1) * 100 = 40, y1 = (0.5 - 0.05) * 100 = 45
        # x2 = (0.5 + 0.1) * 100 = 60, y2 = (0.5 + 0.05) * 100 = 55
        assert torch.allclose(boxes[0], torch.tensor([40., 45., 60., 55.]))

    def test_missing_file_returns_empty(self, tmp_path):
        boxes, ids = _parse_yolo_labels(tmp_path / "missing.txt", 100, 100)
        assert boxes.shape == (0, 4)
        assert ids == []

    def test_all_six_class_ids_parse_without_error(self, tmp_path):
        lines = "\n".join(f"{cid} 0.5 0.5 0.1 0.1" for cid in range(6))
        p = tmp_path / "all.txt"
        p.write_text(lines)
        boxes, ids = _parse_yolo_labels(p, 200.0, 200.0)
        assert len(ids) == 6
        assert boxes.shape == (6, 4)

    def test_unknown_class_id_is_skipped(self, tmp_path):
        p = tmp_path / "unk.txt"
        p.write_text("0 0.5 0.5 0.1 0.1\n99 0.2 0.2 0.1 0.1\n1 0.8 0.8 0.1 0.1\n")
        boxes, ids = _parse_yolo_labels(p, 100.0, 100.0)
        # class_id 99 should be skipped
        assert ids == [0, 1]
        assert boxes.shape == (2, 4)


class TestBoxesToMasks:
    def test_single_box_fills_correct_region(self):
        boxes = torch.tensor([[10., 20., 50., 60.]])
        masks = _boxes_to_masks(boxes, H=100, W=100)
        assert masks.shape == (1, 100, 100)
        # Region inside the box must all be 1
        assert masks[0, 20:60, 10:50].all()
        # Corners outside the box must be 0
        assert masks[0, 0, 0].item() == 0.0
        assert masks[0, 99, 99].item() == 0.0

    def test_zero_boxes_returns_empty_masks(self):
        boxes = torch.zeros((0, 4))
        masks = _boxes_to_masks(boxes, H=64, W=64)
        assert masks.shape == (0, 64, 64)

    def test_mask_is_float32(self):
        boxes = torch.tensor([[0., 0., 10., 10.]])
        masks = _boxes_to_masks(boxes, H=32, W=32)
        assert masks.dtype == torch.float32

    def test_mask_values_are_binary(self):
        boxes = torch.tensor([[5., 5., 25., 25.]])
        masks = _boxes_to_masks(boxes, H=50, W=50)
        unique = masks.unique()
        assert set(unique.tolist()) <= {0.0, 1.0}, (
            f"Mask contains non-binary values: {unique.tolist()}"
        )


# ────────────────────────────────────────────────────────────────────────── #
# End-to-end smoke test (no file I/O)                                       #
# ────────────────────────────────────────────────────────────────────────── #

class TestEndToEndPairPipeline:
    """
    Simulate the full dataset pipeline on fabricated data to verify the
    complete flow from YOLO class_ids → object types → candidate pairs → labels.
    """

    def test_spec_example(self):
        """
        Reproduces the spec validation example:
            object_types = [TEXT, SYMBOL, SYMBOL_TEXT]
            Valid pairs = (TEXT,SYMBOL), (TEXT,SYMBOL_TEXT), (SYMBOL,SYMBOL_TEXT)
            Total = 3
        """
        class_ids = [0, 1, 4]   # TEXT, SYMBOL, SYMBOL_TEXT
        object_types = yolo_class_ids_to_object_types(class_ids)
        assert object_types == [TEXT, SYMBOL, SYMBOL_TEXT]

        validate_object_types(object_types)

        pairs = generate_candidate_pairs(object_types)
        assert len(pairs) == 3

        pair_type_combos = {
            (object_types[i], object_types[j]) for (i, j) in pairs
        }
        assert (TEXT, SYMBOL)      in pair_type_combos
        assert (TEXT, SYMBOL_TEXT) in pair_type_combos
        assert (SYMBOL, SYMBOL_TEXT) in pair_type_combos

    def test_object_types_contain_only_valid_values(self):
        class_ids = [0, 1, 2, 3, 4, 5]
        object_types = yolo_class_ids_to_object_types(class_ids)
        for ot in object_types:
            assert ot in {0, 1, 2}, f"object_type {ot} is not in {{0,1,2}}"

    def test_pair_labels_align_with_pairs(self):
        """pair_labels must have same length as candidate_pairs."""
        class_ids = [0, 1, 4, 0, 3]   # TEXT, SYMBOL, SYMBOL_TEXT, TEXT, SYMBOL
        object_types = yolo_class_ids_to_object_types(class_ids)
        pairs = generate_candidate_pairs(object_types)
        gt_set = make_gt_pair_set([(0, 1)])
        labels = label_candidate_pairs(pairs, gt_set)
        assert len(labels) == len(pairs)

    def test_known_gt_pair_gets_label_one(self):
        class_ids = [0, 1, 4]   # indices: 0=TEXT, 1=SYMBOL, 2=SYMBOL_TEXT
        object_types = yolo_class_ids_to_object_types(class_ids)
        pairs = generate_candidate_pairs(object_types)

        # Mark (TEXT, SYMBOL) = (0,1) as a natural pair in GT
        gt_set = make_gt_pair_set([(0, 1)])
        labels = label_candidate_pairs(pairs, gt_set)

        # Find which pair index corresponds to (TEXT, SYMBOL)
        for k, (i, j) in enumerate(pairs):
            if frozenset({i, j}) == frozenset({0, 1}):
                assert labels[k] == 1, f"GT pair (0,1) should have label 1"
            else:
                assert labels[k] == 0, f"Non-GT pair ({i},{j}) should have label 0"

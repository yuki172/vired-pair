"""
utils/pair_builder.py
=====================
Pure-Python utilities for object-type encoding and candidate pair generation.

This module is intentionally framework-agnostic (no PyTorch, no dataset I/O).
It contains the ground-truth of the pairing logic so that both the dataset
loader and the architecture can reference a single source.

Object type encoding (electrical-plan task)
-------------------------------------------
Three object types are recognised:

    TEXT         = 0   →  YOLO class_ids {0, 2, 5}
    SYMBOL       = 1   →  YOLO class_ids {1, 3}
    SYMBOL_TEXT  = 2   →  YOLO class_id  {4}

SYMBOL_TEXT objects can function as either a symbol or a text label; they are
treated as a distinct third type so the model can learn their dual role.

Feasible pair combinations
---------------------------
A candidate pair (i, j) is feasible iff the two objects have *different* types:

    SYMBOL  ↔ TEXT
    SYMBOL  ↔ SYMBOL_TEXT
    TEXT    ↔ SYMBOL_TEXT

Same-type pairs (TEXT↔TEXT, SYMBOL↔SYMBOL, SYMBOL_TEXT↔SYMBOL_TEXT) are not
generated.

Canonical pair ordering
-----------------------
Within each generated pair the object with the **lower type-id** is always
placed first.  This matches the ordering used by the architecture's
``PairBuilder._cross_type_indices`` so that dataset labels align one-to-one
with the logits returned by the model.

    TEXT(0)   < SYMBOL(1) < SYMBOL_TEXT(2)

Examples with objects [TEXT(0), SYMBOL(1), SYMBOL_TEXT(2)]:
    pair 0: (TEXT_idx,   SYMBOL_idx)       embedding: concat(TEXT,   SYMBOL)
    pair 1: (TEXT_idx,   SYMTEXT_idx)      embedding: concat(TEXT,   SYMBOL_TEXT)
    pair 2: (SYMBOL_idx, SYMTEXT_idx)      embedding: concat(SYMBOL, SYMBOL_TEXT)
    → P = 3 total pairs
"""

from __future__ import annotations

from typing import FrozenSet, List, Sequence, Set, Tuple


# ────────────────────────────────────────────────────────────────────────── #
# Object type constants                                                      #
# ────────────────────────────────────────────────────────────────────────── #

TEXT: int = 0
SYMBOL: int = 1
SYMBOL_TEXT: int = 2

OBJECT_TYPE_NAMES: dict[int, str] = {
    TEXT: "TEXT",
    SYMBOL: "SYMBOL",
    SYMBOL_TEXT: "SYMBOL_TEXT",
}

VALID_OBJECT_TYPES: frozenset[int] = frozenset({TEXT, SYMBOL, SYMBOL_TEXT})

# ────────────────────────────────────────────────────────────────────────── #
# YOLO class_id → object type mapping                                       #
# ────────────────────────────────────────────────────────────────────────── #

YOLO_CLASS_TO_OBJECT_TYPE: dict[int, int] = {
    0: TEXT,
    2: TEXT,
    5: TEXT,
    1: SYMBOL,
    3: SYMBOL,
    4: SYMBOL_TEXT,
}

# The full set of known YOLO class ids (for validation).
_KNOWN_YOLO_CLASS_IDS: frozenset[int] = frozenset(YOLO_CLASS_TO_OBJECT_TYPE.keys())


# ────────────────────────────────────────────────────────────────────────── #
# Type conversion                                                            #
# ────────────────────────────────────────────────────────────────────────── #

def yolo_class_id_to_object_type(class_id: int) -> int:
    """Map a single YOLO class_id to an object type integer.

    Args:
        class_id: YOLO integer class label in {0, 1, 2, 3, 4, 5}.

    Returns:
        Object type in {TEXT=0, SYMBOL=1, SYMBOL_TEXT=2}.

    Raises:
        KeyError: if class_id is not in the known mapping.
    """
    if class_id not in YOLO_CLASS_TO_OBJECT_TYPE:
        raise KeyError(
            f"Unknown YOLO class_id {class_id!r}.  "
            f"Expected one of {sorted(_KNOWN_YOLO_CLASS_IDS)}."
        )
    return YOLO_CLASS_TO_OBJECT_TYPE[class_id]


def yolo_class_ids_to_object_types(class_ids: Sequence[int]) -> List[int]:
    """Convert a sequence of YOLO class_ids to a list of object type integers.

    Args:
        class_ids: sequence of YOLO integer class labels.

    Returns:
        List of object types, same length as ``class_ids``.
    """
    return [yolo_class_id_to_object_type(cid) for cid in class_ids]


# ────────────────────────────────────────────────────────────────────────── #
# Validation                                                                 #
# ────────────────────────────────────────────────────────────────────────── #

def validate_object_types(object_types: Sequence[int]) -> None:
    """Assert that every element in ``object_types`` is in {0, 1, 2}.

    Raises:
        AssertionError: with a descriptive message if any invalid value is found.
    """
    for idx, ot in enumerate(object_types):
        assert ot in VALID_OBJECT_TYPES, (
            f"object_types[{idx}] = {ot!r} is not a valid type.  "
            f"Expected one of {sorted(VALID_OBJECT_TYPES)} "
            f"({', '.join(f'{k}={v}' for k, v in OBJECT_TYPE_NAMES.items())})."
        )


# ────────────────────────────────────────────────────────────────────────── #
# Pair feasibility                                                           #
# ────────────────────────────────────────────────────────────────────────── #

def is_feasible_pair(type_i: int, type_j: int) -> bool:
    """Return True iff the pair (i, j) is a feasible candidate pair.

    A pair is feasible when the two objects have *different* types:
        SYMBOL  ↔ TEXT
        SYMBOL  ↔ SYMBOL_TEXT
        TEXT    ↔ SYMBOL_TEXT
        SYMBOL_TEXT    ↔ SYMBOL_TEXT

    Same-type pairs are never feasible.
    """
    return type_i != type_j or (type_i == SYMBOL_TEXT and type_j == SYMBOL_TEXT)


# ────────────────────────────────────────────────────────────────────────── #
# Candidate pair generation                                                  #
# ────────────────────────────────────────────────────────────────────────── #

def generate_candidate_pairs(
    object_types: Sequence[int],
) -> List[Tuple[int, int]]:
    """Generate all feasible candidate pairs from a sequence of object types.

    Canonical ordering: the object with the **lower type-id** is always placed
    at index 0 of the returned tuple.  This matches the architecture's
    ``PairBuilder._cross_type_indices`` exactly, so that dataset pair_labels
    align one-to-one with the model's pair_logits.

    Args:
        object_types: sequence of N object type integers.

    Returns:
        Sorted list of (subject_idx, object_idx) tuples where subject always
        has a lower or equal type-id compared to object.

    Example::

        >>> generate_candidate_pairs([TEXT, SYMBOL, SYMBOL_TEXT])
        [(0, 1), (0, 2), (1, 2)]
        # pair 0: (TEXT_idx=0, SYMBOL_idx=1)
        # pair 1: (TEXT_idx=0, SYMTEXT_idx=2)
        # pair 2: (SYMBOL_idx=1, SYMTEXT_idx=2)
    """
    validate_object_types(object_types)

    n = len(object_types)
    pairs: List[Tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            ti = object_types[i]
            tj = object_types[j]

            if is_feasible_pair(ti, tj):
                # Place the lower-type-id object first for canonical ordering.
                if ti <= tj:
                    pairs.append((i, j))
                else:
                    pairs.append((j, i))

    return pairs


def count_candidate_pairs(object_types: Sequence[int]) -> int:
    """Return the number of feasible candidate pairs without allocating the list."""
    validate_object_types(object_types)
    n = len(object_types)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if object_types[i] != object_types[j]:
                count += 1
    return count


# ────────────────────────────────────────────────────────────────────────── #
# Pair labelling                                                             #
# ────────────────────────────────────────────────────────────────────────── #

def make_gt_pair_set(gt_pair_lines: Sequence[Tuple[int, int]]) -> Set[FrozenSet[int]]:
    """Convert a list of (i, j) ground truth pairs into an unordered set.

    Args:
        gt_pair_lines: list of (i, j) integer tuples as read from a
                       ``pair_labels/{name}.txt`` file.

    Returns:
        Set of frozenset({i, j}); order within each pair is irrelevant.
    """
    return {frozenset(pair) for pair in gt_pair_lines}


def label_candidate_pairs(
    candidate_pairs: Sequence[Tuple[int, int]],
    gt_pair_set: Set[FrozenSet[int]],
) -> List[int]:
    """Assign binary labels to candidate pairs.

    A candidate pair receives label **1** (natural pair) if the unordered pair
    {i, j} is present in ``gt_pair_set``, otherwise **0**.

    Pair matching is unordered: (i, j) and (j, i) are considered the same.

    Args:
        candidate_pairs: list of (i, j) tuples as returned by
                         ``generate_candidate_pairs``.
        gt_pair_set:     set of frozenset({i, j}) ground truth pairs as
                         returned by ``make_gt_pair_set``.

    Returns:
        List of 0/1 integers, same length as ``candidate_pairs``.
    """
    return [1 if frozenset(pair) in gt_pair_set else 0 for pair in candidate_pairs]

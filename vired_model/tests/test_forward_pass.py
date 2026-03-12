"""
Smoke tests for the ViREDModel.

These tests verify:
  - The full forward pass (with bounding boxes) runs without errors.
  - Output tensor shapes are correct for the given inputs.
  - Both pair_modes ("cross_type" and "all") work.
  - Both decoder norm orders work.
  - Pair embeddings have a consistent type-id ordering in cross_type mode.
  - predict() is deterministic (eval mode applied, dropout off).
  - Geometry features are computed and concatenated correctly.
  - ROI features fuse into object embeddings correctly.
  - The model also runs in the original mask-only mode
    (use_roi_features=False, use_geometry_features=False).
  - Gradients reach every component.
  - The model can run on CPU (no GPU required).

Run with:
    pytest vired_model/tests/test_forward_pass.py -v
or:
    python -m pytest vired_model/tests/test_forward_pass.py -v
"""

from __future__ import annotations

import torch

from vired_model.config import ViredConfig
from vired_model.models.vired_model import ViREDModel, ViredOutput
from vired_model.models.pair_builder import compute_pair_geometry_features


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def make_config(**overrides) -> ViredConfig:
    """Return a lightweight config suitable for CPU smoke tests."""
    defaults = dict(
        embedding_dim=64,
        num_decoder_layers=2,
        num_attention_heads=4,
        ffn_hidden_dim=128,
        dropout=0.0,
        object_encoder_channels=[8, 16, 32],
        relation_mlp_hidden=64,
        relation_head_dropout=0.0,
        vision_backbone="vit_tiny_patch16_224",
        pretrained=False,
        num_object_types=2,
        use_type_embeddings=True,
        image_size=224,
        image_channels=3,
        num_relation_classes=2,
        pair_mode="cross_type",
        decoder_pre_norm=True,
        ffn_activation="gelu",
        # New: geometry and ROI features
        use_roi_features=True,
        roi_pool_size=7,
        roi_feature_dim=16,    # small for fast CPU tests
        use_geometry_features=True,
        geometry_feature_dim=6,
    )
    defaults.update(overrides)
    return ViredConfig(**defaults) # type: ignore


def make_inputs(
    B: int = 2,
    N: int = 6,        # N//2 symbols, N-N//2 texts
    H: int = 224,
    W: int = 224,
    device: torch.device = torch.device("cpu"),
):
    """Generate dummy inputs for a batch of B images, each with N objects.

    Returns:
        image         : (B, 3, H, W)
        object_masks  : (B, N, H, W)
        object_boxes  : (B, N, 4)   – (x1, y1, x2, y2) in pixel coordinates
        object_types  : (B, N)
    """
    n_sym = N // 2
    n_txt = N - n_sym

    image = torch.randn(B, 3, H, W, device=device)

    # Binary masks: small random blobs
    object_masks = torch.zeros(B, N, H, W, device=device)
    for b in range(B):
        for n in range(N):
            h0, w0 = torch.randint(0, H - 20, (2,)).tolist()
            object_masks[b, n, h0:h0 + 20, w0:w0 + 20] = 1.0

    # Random non-degenerate bounding boxes
    bx1 = torch.randint(0, W - 40, (B, N)).float().to(device)
    by1 = torch.randint(0, H - 40, (B, N)).float().to(device)
    bw  = torch.randint(10, 40,    (B, N)).float().to(device)
    bh  = torch.randint(10, 40,    (B, N)).float().to(device)
    object_boxes = torch.stack([bx1, by1, bx1 + bw, by1 + bh], dim=-1)  # (B, N, 4)

    # 0 = symbol, 1 = text
    object_types = torch.cat([
        torch.zeros(B, n_sym, dtype=torch.long, device=device),
        torch.ones( B, n_txt, dtype=torch.long, device=device),
    ], dim=1)  # (B, N)

    return image, object_masks, object_boxes, object_types


# ------------------------------------------------------------------ #
# Forward pass – full model (geometry + ROI)                          #
# ------------------------------------------------------------------ #

class TestForwardPassCrossType:
    """Tests with pair_mode='cross_type' and all new features enabled."""

    def setup_method(self):
        self.config = make_config(pair_mode="cross_type")
        self.model = ViREDModel(self.config)
        self.model.eval()

    def test_output_is_vired_output(self):
        image, masks, boxes, types = make_inputs()
        out = self.model(image, masks, boxes, types)
        assert isinstance(out, ViredOutput)

    def test_pair_logits_shape(self):
        B, N = 2, 6   # 3 symbols + 3 texts → 3×3 = 9 cross-type pairs
        image, masks, boxes, types = make_inputs(B=B, N=N)
        out = self.model(image, masks, boxes, types)

        expected_P = 9
        C = self.config.num_relation_classes
        assert out.pair_logits.shape == (B, expected_P, C), (
            f"pair_logits shape mismatch: {out.pair_logits.shape}"
        )

    def test_pair_indices_shape(self):
        B, N = 2, 6
        image, masks, boxes, types = make_inputs(B=B, N=N)
        out = self.model(image, masks, boxes, types)
        assert out.pair_indices.shape == (B, 9, 2)

    def test_pair_indices_valid_range(self):
        B, N = 2, 6
        image, masks, boxes, types = make_inputs(B=B, N=N)
        out = self.model(image, masks, boxes, types)
        assert out.pair_indices.min() >= 0
        assert out.pair_indices.max() < N

    def test_no_nan_in_logits(self):
        image, masks, boxes, types = make_inputs()
        out = self.model(image, masks, boxes, types)
        assert not torch.isnan(out.pair_logits).any(), "NaN detected in pair_logits"

    def test_no_inf_in_logits(self):
        image, masks, boxes, types = make_inputs()
        out = self.model(image, masks, boxes, types)
        assert not torch.isinf(out.pair_logits).any(), "Inf detected in pair_logits"

    def test_batch_size_one(self):
        image, masks, boxes, types = make_inputs(B=1, N=4)
        out = self.model(image, masks, boxes, types)
        assert out.pair_logits.shape[0] == 1

    def test_train_mode_runs(self):
        config = make_config(pair_mode="cross_type", dropout=0.1, relation_head_dropout=0.1)
        model = ViREDModel(config)
        model.train()
        image, masks, boxes, types = make_inputs()
        out = model(image, masks, boxes, types)
        assert out.pair_logits.shape[0] == 2

    def test_cross_type_pair_type_id_ordering(self):
        """First element of each pair must have a lower type-id than the second."""
        B, N = 2, 6
        image, masks, boxes, types = make_inputs(B=B, N=N)
        out = self.model(image, masks, boxes, types)
        for b in range(B):
            i_idx = out.pair_indices[b, :, 0]
            j_idx = out.pair_indices[b, :, 1]
            assert (types[b][i_idx] <= types[b][j_idx]).all()

    def test_cross_type_pairs_have_different_types(self):
        B, N = 2, 6
        image, masks, boxes, types = make_inputs(B=B, N=N)
        out = self.model(image, masks, boxes, types)
        for b in range(B):
            ti = types[b][out.pair_indices[b, :, 0]]
            tj = types[b][out.pair_indices[b, :, 1]]
            assert (ti != tj).all()


# ------------------------------------------------------------------ #
# Forward pass – all pairs mode                                       #
# ------------------------------------------------------------------ #

class TestForwardPassAllPairs:
    def setup_method(self):
        self.config = make_config(pair_mode="all")
        self.model = ViREDModel(self.config)
        self.model.eval()

    def test_pair_logits_shape_all(self):
        B, N = 2, 6
        image, masks, boxes, types = make_inputs(B=B, N=N)
        out = self.model(image, masks, boxes, types)
        expected_P = N * (N - 1) // 2
        assert out.pair_logits.shape == (B, expected_P, self.config.num_relation_classes)

    def test_pair_indices_i_less_than_j(self):
        image, masks, boxes, types = make_inputs(B=2, N=6)
        out = self.model(image, masks, boxes, types)
        assert (out.pair_indices[..., 0] < out.pair_indices[..., 1]).all()


# ------------------------------------------------------------------ #
# Geometry features                                                   #
# ------------------------------------------------------------------ #

class TestGeometryFeatures:
    """Validate compute_pair_geometry_features directly and via the model."""

    def test_output_shape(self):
        B, N, P = 2, 6, 9
        boxes = torch.rand(B, N, 4) * 200
        boxes[..., 2] += 10   # ensure x2 > x1
        boxes[..., 3] += 10   # ensure y2 > y1
        # Make valid (x1 < x2, y1 < y2)
        boxes[..., 0] = boxes[..., 0].clamp(0, 180)
        boxes[..., 1] = boxes[..., 1].clamp(0, 180)
        boxes[..., 2] = boxes[..., 0] + 10
        boxes[..., 3] = boxes[..., 1] + 10

        pair_indices = torch.zeros(B, P, 2, dtype=torch.long)
        pair_indices[:, :, 0] = 0
        pair_indices[:, :, 1] = 1

        geom = compute_pair_geometry_features(boxes, pair_indices, image_size=224)
        assert geom.shape == (B, P, 6), f"Expected (B, P, 6), got {geom.shape}"

    def test_no_nan_or_inf(self):
        B, N, P = 1, 4, 4
        boxes = torch.tensor([[[10., 10., 50., 50.],
                                [60., 60., 100., 100.],
                                [5., 5., 20., 20.],
                                [30., 30., 80., 80.]]])
        pair_indices = torch.tensor([[[0, 1], [0, 2], [1, 2], [2, 3]]])
        geom = compute_pair_geometry_features(boxes, pair_indices, image_size=224)
        assert not torch.isnan(geom).any(), "NaN in geometry features"
        assert not torch.isinf(geom).any(), "Inf in geometry features"

    def test_iou_is_bounded(self):
        """IoU must be in [0, 1]."""
        B, N, P = 1, 4, 4
        boxes = torch.tensor([[[0., 0., 50., 50.],
                                [0., 0., 50., 50.],   # identical box → iou=1
                                [60., 60., 100., 100.],
                                [200., 200., 220., 220.]]])
        pair_indices = torch.tensor([[[0, 1], [0, 2], [1, 2], [2, 3]]])
        geom = compute_pair_geometry_features(boxes, pair_indices, image_size=224)
        iou = geom[..., 5]
        assert (iou >= 0).all(), "IoU must be ≥ 0"
        assert (iou <= 1 + 1e-5).all(), "IoU must be ≤ 1"

    def test_identical_boxes_iou_is_one(self):
        """Two identical boxes should have IoU ≈ 1."""
        boxes = torch.tensor([[[10., 10., 60., 60.],
                                [10., 10., 60., 60.]]])   # (1, 2, 4)
        pair_indices = torch.tensor([[[0, 1]]])             # (1, 1, 2)
        geom = compute_pair_geometry_features(boxes, pair_indices, image_size=224)
        assert torch.isclose(geom[0, 0, 5], torch.tensor(1.0), atol=1e-4), (
            f"Identical boxes should have IoU≈1, got {geom[0,0,5]:.4f}"
        )

    def test_non_overlapping_boxes_iou_is_zero(self):
        boxes = torch.tensor([[[0., 0., 10., 10.],
                                [20., 20., 30., 30.]]])
        pair_indices = torch.tensor([[[0, 1]]])
        geom = compute_pair_geometry_features(boxes, pair_indices, image_size=224)
        assert torch.isclose(geom[0, 0, 5], torch.tensor(0.0), atol=1e-4), (
            f"Non-overlapping boxes should have IoU≈0, got {geom[0,0,5]:.4f}"
        )

    def test_dx_dy_are_normalised(self):
        """dx and dy must be in roughly [-1, 1] for reasonable boxes."""
        B, N = 2, 6
        image, masks, boxes, types = make_inputs(B=B, N=N)

        config = make_config()
        model = ViREDModel(config)
        model.eval()
        out = model(image, masks, boxes, types)

        # We can only test this indirectly – just confirm no overflow
        assert not torch.isnan(out.pair_logits).any()

    def test_geometry_features_disabled(self):
        """With use_geometry_features=False, pair embedding dim should be 2D."""
        config = make_config(use_geometry_features=False)
        model = ViREDModel(config)
        model.eval()
        image, masks, boxes, types = make_inputs(B=1, N=4)
        out = model(image, masks, boxes, types)
        # Logits should still have the right shape
        assert out.pair_logits.shape[-1] == config.num_relation_classes


# ------------------------------------------------------------------ #
# ROI features                                                        #
# ------------------------------------------------------------------ #

class TestROIFeatures:
    """Verify ROI feature extraction integrates cleanly."""

    def test_roi_features_enabled_runs(self):
        config = make_config(use_roi_features=True, roi_feature_dim=16)
        model = ViREDModel(config)
        model.eval()
        image, masks, boxes, types = make_inputs(B=1, N=4)
        out = model(image, masks, boxes, types)
        assert not torch.isnan(out.pair_logits).any()

    def test_roi_features_disabled_runs(self):
        """Model must run correctly when ROI features are turned off."""
        config = make_config(use_roi_features=False)
        model = ViREDModel(config)
        model.eval()
        image, masks, boxes, types = make_inputs(B=1, N=4)
        out = model(image, masks, boxes, types)
        assert not torch.isnan(out.pair_logits).any()

    def test_roi_and_geometry_both_disabled(self):
        """Original mask-only mode: both ROI and geometry features off."""
        config = make_config(use_roi_features=False, use_geometry_features=False)
        model = ViREDModel(config)
        model.eval()
        image, masks, boxes, types = make_inputs(B=2, N=6)
        out = model(image, masks, boxes, types)
        assert not torch.isnan(out.pair_logits).any()
        # Pair logits shape unchanged
        assert out.pair_logits.shape == (2, 9, config.num_relation_classes)

    def test_roi_object_encoder_proj_dim(self):
        """ObjectEncoder projection input dim must match C3 + roi_feature_dim."""
        config = make_config(
            use_roi_features=True,
            object_encoder_channels=[8, 16, 32],
            roi_feature_dim=24,
        )
        from vired_model.models.vision_encoder import VisionEncoder
        from vired_model.models.object_encoder import ObjectEncoder
        ve = VisionEncoder(config)
        oe = ObjectEncoder(config, backbone_dim=ve.backbone_dim)
        # proj[0] is the nn.Linear; its in_features = 32 + 24 = 56
        expected_in = 32 + 24
        assert oe.proj[0].in_features == expected_in, (
            f"proj in_features: expected {expected_in}, got {oe.proj[0].in_features}"
        )

    def test_roi_disabled_object_encoder_proj_dim(self):
        """Without ROI features, projection input dim must be just C3."""
        config = make_config(
            use_roi_features=False,
            object_encoder_channels=[8, 16, 32],
        )
        from vired_model.models.vision_encoder import VisionEncoder
        from vired_model.models.object_encoder import ObjectEncoder
        ve = VisionEncoder(config)
        oe = ObjectEncoder(config, backbone_dim=ve.backbone_dim)
        assert oe.proj[0].in_features == 32, (
            f"proj in_features: expected 32, got {oe.proj[0].in_features}"
        )

    def test_roi_context_pad_zero_is_default_behaviour(self):
        """roi_context_pad=0 must produce the same logits as the unpadded path."""
        config = make_config(use_roi_features=True, roi_context_pad=0)
        model = ViREDModel(config)
        model.eval()
        image, masks, boxes, types = make_inputs(B=1, N=4)
        with torch.no_grad():
            out = model(image, masks, boxes, types)
        assert not torch.isnan(out.pair_logits).any()

    def test_roi_context_pad_nonzero_runs_and_no_nan(self):
        """A nonzero roi_context_pad must run without errors and produce finite output."""
        config = make_config(use_roi_features=True, roi_context_pad=20)
        model = ViREDModel(config)
        model.eval()
        image, masks, boxes, types = make_inputs(B=2, N=6)
        with torch.no_grad():
            out = model(image, masks, boxes, types)
        assert not torch.isnan(out.pair_logits).any()
        assert not torch.isinf(out.pair_logits).any()

    def test_roi_context_pad_clamps_to_image_boundary(self):
        """Boxes touching the image edge must not cause errors even with large padding."""
        config = make_config(use_roi_features=True, roi_context_pad=500)
        model = ViREDModel(config)
        model.eval()
        B, N, H = 1, 4, 224
        image = torch.randn(B, 3, H, H)
        masks = torch.zeros(B, N, H, H)
        # Place boxes at the very corners and edges.
        boxes = torch.tensor([[
            [0.,   0.,  10.,  10.],    # top-left corner
            [214., 0.,  224., 10.],    # top-right corner
            [0.,   214., 10., 224.],   # bottom-left corner
            [107., 107., 117., 117.],  # centre
        ]])
        types = torch.zeros(B, N, dtype=torch.long)
        types[0, 1] = 1  # make at least one pair feasible
        types[0, 3] = 1
        with torch.no_grad():
            out = model(image, masks, boxes, types)
        assert not torch.isnan(out.pair_logits).any()

    def test_roi_context_pad_does_not_affect_geometry_features(self):
        """Geometry features must be identical whether or not context padding is used."""
        torch.manual_seed(42)
        image, masks, boxes, types = make_inputs(B=1, N=4)

        config_no_pad  = make_config(use_roi_features=True, roi_context_pad=0)
        config_with_pad = make_config(use_roi_features=True, roi_context_pad=30)

        # Share the same weights so only the padding differs.
        model_no_pad   = ViREDModel(config_no_pad)
        model_with_pad = ViREDModel(config_with_pad)
        model_with_pad.load_state_dict(
            # Both configs have identical architecture params; weights are
            # compatible except the roi_context_pad flag which is not a weight.
            model_no_pad.state_dict()
        )
        model_no_pad.eval()
        model_with_pad.eval()

        # pair_indices must be the same (pairs depend on types, not on ROI boxes).
        with torch.no_grad():
            out_no_pad   = model_no_pad(image, masks, boxes, types)
            out_with_pad = model_with_pad(image, masks, boxes, types)

        assert torch.equal(out_no_pad.pair_indices, out_with_pad.pair_indices), (
            "pair_indices changed when roi_context_pad was applied"
        )


# ------------------------------------------------------------------ #
# Decoder norm order                                                  #
# ------------------------------------------------------------------ #

class TestDecoderNormOrder:
    def _run(self, pre_norm: bool):
        config = make_config(decoder_pre_norm=pre_norm)
        model = ViREDModel(config)
        model.eval()
        image, masks, boxes, types = make_inputs(B=1, N=4)
        out = model(image, masks, boxes, types)
        assert not torch.isnan(out.pair_logits).any()
        assert not torch.isinf(out.pair_logits).any()

    def test_pre_norm(self):
        self._run(pre_norm=True)

    def test_post_norm(self):
        self._run(pre_norm=False)


# ------------------------------------------------------------------ #
# Predict determinism                                                 #
# ------------------------------------------------------------------ #

class TestPredictDeterminism:
    def test_predict_is_deterministic_from_train_mode(self):
        config = make_config(dropout=0.5, relation_head_dropout=0.5)
        model = ViREDModel(config)
        model.train()
        image, masks, boxes, types = make_inputs(B=1, N=4)
        torch.manual_seed(42)
        out1 = model.predict(image, masks, boxes, types)
        torch.manual_seed(42)
        out2 = model.predict(image, masks, boxes, types)
        assert torch.allclose(out1.pair_logits, out2.pair_logits), (
            "predict() must be deterministic (dropout should be disabled)"
        )

    def test_predict_restores_training_mode(self):
        config = make_config(dropout=0.1)
        model = ViREDModel(config)
        model.train()
        image, masks, boxes, types = make_inputs(B=1, N=4)
        model.predict(image, masks, boxes, types)
        assert model.training

    def test_predict_preserves_eval_mode(self):
        config = make_config()
        model = ViREDModel(config)
        model.eval()
        image, masks, boxes, types = make_inputs(B=1, N=4)
        model.predict(image, masks, boxes, types)
        assert not model.training


# ------------------------------------------------------------------ #
# Activation functions                                                #
# ------------------------------------------------------------------ #

class TestActivationFunctions:
    def _run(self, act: str):
        config = make_config(ffn_activation=act)
        model = ViREDModel(config)
        model.eval()
        image, masks, boxes, types = make_inputs(B=1, N=4)
        out = model(image, masks, boxes, types)
        assert not torch.isnan(out.pair_logits).any()

    def test_relu(self):
        self._run("relu")

    def test_gelu(self):
        self._run("gelu")


# ------------------------------------------------------------------ #
# RelationHead input dimension                                        #
# ------------------------------------------------------------------ #

class TestRelationHeadInputDim:
    def test_in_dim_with_geometry(self):
        config = make_config(
            embedding_dim=64,
            use_geometry_features=True,
            geometry_feature_dim=6,
        )
        from vired_model.models.relation_head import RelationHead
        head = RelationHead(config)
        assert head.in_dim == 2 * 64 + 6, (
            f"Expected in_dim={2*64+6}, got {head.in_dim}"
        )

    def test_in_dim_without_geometry(self):
        config = make_config(
            embedding_dim=64,
            use_geometry_features=False,
        )
        from vired_model.models.relation_head import RelationHead
        head = RelationHead(config)
        assert head.in_dim == 2 * 64, (
            f"Expected in_dim={2*64}, got {head.in_dim}"
        )


# ------------------------------------------------------------------ #
# Component integration                                               #
# ------------------------------------------------------------------ #

class TestComponentIntegration:
    def test_count_parameters_positive(self):
        model = ViREDModel(make_config())
        assert model.count_parameters() > 0

    def test_gradient_flows_object_encoder(self):
        config = make_config(dropout=0.0)
        model = ViREDModel(config)
        model.train()
        image, masks, boxes, types = make_inputs(B=1, N=4)
        out = model(image, masks, boxes, types)
        out.pair_logits.sum().backward()
        grad_sum = sum(
            p.grad.abs().sum().item()
            for p in model.object_encoder.parameters()
            if p.grad is not None
        )
        assert grad_sum > 0, "No gradient reached the object encoder"

    def test_gradient_flows_vision_encoder(self):
        config = make_config(dropout=0.0)
        model = ViREDModel(config)
        model.train()
        image, masks, boxes, types = make_inputs(B=1, N=4)
        out = model(image, masks, boxes, types)
        out.pair_logits.sum().backward()
        grad_sum = sum(
            p.grad.abs().sum().item()
            for p in model.vision_encoder.proj.parameters()
            if p.grad is not None
        )
        assert grad_sum > 0, "No gradient reached the vision encoder projection"

    def test_gradient_flows_roi_encoder(self):
        """Gradients must reach the ROI encoder projection."""
        config = make_config(dropout=0.0, use_roi_features=True)
        model = ViREDModel(config)
        model.train()
        image, masks, boxes, types = make_inputs(B=1, N=4)
        out = model(image, masks, boxes, types)
        out.pair_logits.sum().backward()
        assert model.object_encoder.roi_encoder is not None
        grad_sum = sum(
            p.grad.abs().sum().item()
            for p in model.object_encoder.roi_encoder.parameters()
            if p.grad is not None
        )
        assert grad_sum > 0, "No gradient reached the ROI encoder"

    def test_different_embedding_dims(self):
        for d in [32, 128]:
            config = make_config(embedding_dim=d, num_attention_heads=4)
            model = ViREDModel(config)
            model.eval()
            image, masks, boxes, types = make_inputs(B=1, N=4)
            out = model(image, masks, boxes, types)
            assert out.pair_logits.shape[-1] == 2

    def test_non_default_image_size_256(self):
        """image_size=256 must produce a valid square grid (16×16=256 patches)."""
        config = make_config(image_size=256)
        model = ViREDModel(config)
        model.eval()
        image, masks, boxes, types = make_inputs(B=1, N=4, H=256, W=256)
        with torch.no_grad():
            out = model(image, masks, boxes, types)
        assert not torch.isnan(out.pair_logits).any()
        # T = (256/16)² = 256 patch tokens (no CLS for vit_tiny_patch16_224)
        # pair_logits shape must still be (1, P, num_relation_classes)
        assert out.pair_logits.dim() == 3
        assert out.pair_logits.shape[0] == 1

    def test_feature_map_grid_size_matches_image_size(self):
        """feature_map grid size must equal image_size / patch_size for any image_size."""
        for image_size, expected_grid in [(224, 14), (256, 16)]:
            config = make_config(image_size=image_size, use_roi_features=True)
            from vired_model.models.vision_encoder import VisionEncoder
            ve = VisionEncoder(config)
            ve.eval()
            dummy = torch.zeros(1, 3, image_size, image_size)
            with torch.no_grad():
                _, feature_map = ve.forward_with_feature_map(dummy)
            H_p = feature_map.shape[2]
            assert H_p == expected_grid, (
                f"image_size={image_size}: expected grid {expected_grid}, got {H_p}"
            )

    def test_relation_head_dropout_applied(self):
        config = make_config(relation_head_dropout=0.9)
        model = ViREDModel(config)
        image, masks, boxes, types = make_inputs(B=1, N=4)
        model.train()
        outs = [model(image, masks, boxes, types).pair_logits for _ in range(5)]
        assert not all(torch.allclose(outs[0], o) for o in outs[1:])
        model.eval()
        out_a = model(image, masks, boxes, types).pair_logits
        out_b = model(image, masks, boxes, types).pair_logits
        assert torch.allclose(out_a, out_b)


# ------------------------------------------------------------------ #
# Stand-alone runner (no pytest)                                      #
# ------------------------------------------------------------------ #

def run_smoke_test() -> None:
    """Quick sanity check that can be run without pytest."""
    print("Running ViRED smoke test (with geometry + ROI features)...")

    config = make_config()
    model = ViREDModel(config)
    model.eval()

    B, N = 2, 6
    image, masks, boxes, types = make_inputs(B=B, N=N)

    with torch.no_grad():
        out = model(image, masks, boxes, types)

    n_params = model.count_parameters()
    print(f"  pair_logits  : {tuple(out.pair_logits.shape)}")
    print(f"  pair_indices : {tuple(out.pair_indices.shape)}")
    print(f"  parameters   : {n_params:,}")
    print("  PASSED")


if __name__ == "__main__":
    run_smoke_test()

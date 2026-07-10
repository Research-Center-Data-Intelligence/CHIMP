"""Unit tests for yolo_pose dataset normalization (UT-05, UT-06)."""
import pytest
from app.plugins.yolo_pose.dataset import _normalize_bbox, _normalize_keypoints


class TestNormalizeBbox:
    """Tests for _normalize_bbox (UT-05)."""

    def test_center_and_size_are_normalized(self):
        # bbox [x, y, w, h] = [0, 0, 100, 200] on 200x400 image
        xc, yc, bw, bh = _normalize_bbox([0, 0, 100, 200], width=200, height=400)
        assert xc == pytest.approx(0.25)   # (0 + 50) / 200
        assert yc == pytest.approx(0.25)   # (0 + 100) / 400
        assert bw == pytest.approx(0.5)    # 100 / 200
        assert bh == pytest.approx(0.5)    # 200 / 400

    def test_bbox_at_image_origin(self):
        xc, yc, bw, bh = _normalize_bbox([0, 0, 640, 480], width=640, height=480)
        assert xc == pytest.approx(0.5)
        assert yc == pytest.approx(0.5)
        assert bw == pytest.approx(1.0)
        assert bh == pytest.approx(1.0)

    def test_bbox_offset(self):
        # bbox starting at (320, 240) with size 100x50 on 640x480
        xc, yc, bw, bh = _normalize_bbox([320, 240, 100, 50], width=640, height=480)
        assert xc == pytest.approx((320 + 50) / 640)
        assert yc == pytest.approx((240 + 25) / 480)
        assert bw == pytest.approx(100 / 640)
        assert bh == pytest.approx(50 / 480)

    def test_raises_on_wrong_length(self):
        with pytest.raises(RuntimeError, match="4 values"):
            _normalize_bbox([1, 2, 3], width=100, height=100)

    def test_raises_on_too_many_values(self):
        with pytest.raises(RuntimeError, match="4 values"):
            _normalize_bbox([1, 2, 3, 4, 5], width=100, height=100)

    def test_accepts_float_values(self):
        xc, yc, bw, bh = _normalize_bbox([10.5, 20.5, 50.0, 40.0], width=100, height=100)
        assert xc == pytest.approx((10.5 + 25.0) / 100)


class TestNormalizeKeypoints:
    """Tests for _normalize_keypoints (UT-05)."""

    def _make_keypoints(self, x_val=100.0, y_val=200.0, v_val=2.0):
        """Build 51 values: 17 triplets each (x, y, v)."""
        return [x_val, y_val, v_val] * 17

    def test_normalizes_x_and_y_preserves_visibility(self):
        kps = self._make_keypoints(x_val=100.0, y_val=200.0, v_val=1.0)
        result = _normalize_keypoints(kps, width=400, height=800)
        assert len(result) == 51
        # Every triplet: x normalized, y normalized, v unchanged
        for i in range(0, 51, 3):
            assert result[i] == pytest.approx(100.0 / 400)
            assert result[i + 1] == pytest.approx(200.0 / 800)
            assert result[i + 2] == pytest.approx(1.0)

    def test_zero_keypoints_yield_zero(self):
        kps = [0.0, 0.0, 0.0] * 17
        result = _normalize_keypoints(kps, width=640, height=480)
        assert all(v == 0.0 for v in result)

    def test_raises_on_fewer_than_51_values(self):
        with pytest.raises(RuntimeError, match="51 values"):
            _normalize_keypoints([0.0] * 48, width=640, height=480)

    def test_raises_on_more_than_51_values(self):
        with pytest.raises(RuntimeError, match="51 values"):
            _normalize_keypoints([0.0] * 54, width=640, height=480)

    def test_visibility_flag_is_not_normalized(self):
        kps = [0.0, 0.0, 2.0] * 17  # v=2 throughout
        result = _normalize_keypoints(kps, width=640, height=480)
        for i in range(2, 51, 3):
            assert result[i] == pytest.approx(2.0)

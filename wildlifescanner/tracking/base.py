from __future__ import annotations

from collections.abc import Sequence

from ..models import Detection


class RegionSelector:
    """
    Interface for future dynamic crops (cropping/tracking).
    In v0.1 no crops are applied yet — this class serves as a hook.
    """

    def suggest_crop(
        self,
        frame_size: tuple[int, int],  # (width, height)
        detections: Sequence[Detection],
    ) -> tuple[int, int, int, int] | None:
        """
        Optionally returns a crop rectangle (x, y, w, h); otherwise None.
        Default: None (no crop).
        """
        return None

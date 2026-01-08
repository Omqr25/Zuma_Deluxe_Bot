from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int


class FeatureWindowDetector:
    """
    Locate an anchor image inside a screenshot using ORB feature matching + homography.
    Returns a bounding box around the found anchor (or you can expand it to game ROI).
    """

    def __init__(
        self,
        anchor_path: str,
        nfeatures: int = 4000,         
        ratio: float = 0.75,    
        min_good_matches: int = 25,
        min_inliers: int = 15,
        ransac_reproj_thresh: float = 5.0,
    ):
        self.bbox = BBox(0,0,0,0)
        self.anchor_gray = cv2.imread(anchor_path, cv2.IMREAD_GRAYSCALE)
        if self.anchor_gray is None:
            raise FileNotFoundError(f"Cannot read anchor image: {anchor_path}")

        self.orb = cv2.ORB_create(nfeatures=nfeatures)

        self.kpA, self.desA = self.orb.detectAndCompute(self.anchor_gray, None)
        if self.desA is None or len(self.kpA) < 20:
            raise RuntimeError("Anchor has too few ORB features. Choose a more textured ROI.")

        # ORB descriptors are binary => use Hamming distance
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.ratio = ratio
        self.min_good_matches = min_good_matches
        self.min_inliers = min_inliers
        self.ransac_reproj_thresh = ransac_reproj_thresh

        h, w = self.anchor_gray.shape[:2]
        self.anchor_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)

    def locate_anchor(self, frame_bgr: np.ndarray) -> Optional[Tuple[BBox, float, np.ndarray]]:
        """
        Returns (bbox, inlier_ratio, quad_points) or None.
        quad_points is 4x1x2 transformed anchor corners in frame coordinates.
        """
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kpB, desB = self.orb.detectAndCompute(frame_gray, None)
        if desB is None or len(kpB) < 10:
            return None

        # KNN match (k=2) + ratio test
        knn = self.matcher.knnMatch(self.desA, desB, k=2)
        good = []
        for m_n in knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < self.ratio * n.distance:
                good.append(m)

        if len(good) < self.min_good_matches:
            return None

        src_pts = np.float32([self.kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj_thresh)
        if H is None or mask is None:
            return None

        inliers = int(mask.sum())
        inlier_ratio = inliers / max(1, len(good))
        if inliers < self.min_inliers:
            return None

        # Transform anchor corners into frame
        quad = cv2.perspectiveTransform(self.anchor_corners, H)  # 4 points
        xs = quad[:, 0, 0]
        ys = quad[:, 0, 1]

        x1, y1 = int(np.floor(xs.min())), int(np.floor(ys.min()))
        x2, y2 = int(np.ceil(xs.max())), int(np.ceil(ys.max()))

        # Clip to image bounds
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        self.bbox = BBox(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1))
        return self.bbox, float(inlier_ratio), quad
    
    def getBBox(self):
        return self.bbox

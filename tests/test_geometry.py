import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "dsgvo-pixeler.py"
SPEC = importlib.util.spec_from_file_location("dsgvo_pixeler", MODULE_PATH)
PIXELER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIXELER)


class GeometryTests(unittest.TestCase):
    def test_tiles_overlap_and_cover_frame(self):
        tiles = PIXELER.build_tiles(100, 80, 2, 0.2)
        self.assertEqual(
            tiles,
            [
                (0, 0, 56, 45),
                (44, 0, 100, 45),
                (0, 35, 56, 80),
                (44, 35, 100, 80),
            ],
        )
        self.assertGreater(tiles[0][2] - tiles[1][0], 0)
        self.assertGreater(tiles[0][3] - tiles[2][1], 0)

    def test_tiles_without_overlap_match_regular_grid(self):
        self.assertEqual(
            PIXELER.build_tiles(100, 80, 2, 0.0),
            [(0, 0, 50, 40), (50, 0, 100, 40), (0, 40, 50, 80), (50, 40, 100, 80)],
        )
        self.assertEqual(PIXELER.build_tiles(100, 80, 1, 0.5), [(0, 0, 100, 80)])

    def test_tiles_never_create_empty_regions(self):
        tiles = PIXELER.build_tiles(3, 2, 10, 0.2)
        self.assertTrue(tiles)
        for x1, y1, x2, y2 in tiles:
            self.assertTrue(0 <= x1 < x2 <= 3)
            self.assertTrue(0 <= y1 < y2 <= 2)

    def test_merge_unites_duplicates_but_not_separate_boxes(self):
        boxes = [(40, 40, 50, 50), (2, 1, 22, 21), (0, 0, 20, 20)]
        self.assertEqual(
            PIXELER.merge_overlapping_boxes(boxes),
            [(0, 0, 22, 21), (40, 40, 50, 50)],
        )
        self.assertEqual(
            PIXELER.merge_overlapping_boxes(list(reversed(boxes))),
            [(0, 0, 22, 21), (40, 40, 50, 50)],
        )

    def test_merge_closes_low_iou_overlaps_and_small_tile_seams(self):
        self.assertEqual(
            PIXELER.merge_overlapping_boxes([(0, 0, 10, 10), (9, 0, 19, 10)]),
            [(0, 0, 19, 10)],
        )
        self.assertEqual(
            PIXELER.merge_overlapping_boxes([(0, 0, 10, 10), (12, 1, 22, 9)]),
            [(0, 0, 22, 10)],
        )

    def test_detection_box_uses_independent_axis_scales(self):
        self.assertEqual(
            PIXELER.map_detection_box((0, 0, 4, 2), 0, 0, 4 / 7, 2 / 5, 7, 5),
            (0, 0, 7, 5),
        )

    def test_subtract_zone_returns_exact_remaining_pixels(self):
        box = (0, 0, 10, 10)
        zone = (3, 2, 7, 8)
        pieces = PIXELER.subtract_zones(box, [zone])
        self.assertEqual(sum(PIXELER.box_area(piece) for piece in pieces), 76)

        remaining = set()
        for x1, y1, x2, y2 in pieces:
            for y in range(y1, y2):
                for x in range(x1, x2):
                    self.assertNotIn((x, y), remaining)
                    remaining.add((x, y))
        expected = {
            (x, y)
            for y in range(box[1], box[3])
            for x in range(box[0], box[2])
            if not (zone[0] <= x < zone[2] and zone[1] <= y < zone[3])
        }
        self.assertEqual(remaining, expected)

    def test_zone_preserves_pixels_for_blur_and_pixelation(self):
        rng = np.random.default_rng(42)
        source = rng.integers(0, 256, size=(14, 14, 3), dtype=np.uint8)
        box = (1, 1, 13, 13)
        zone = (4, 3, 9, 10)
        for mode in ("blur", "pixelate"):
            with self.subTest(mode=mode):
                frame = source.copy()
                PIXELER.anonymize_box_excluding_zones(frame, box, [zone], mode, 7, 4)
                np.testing.assert_array_equal(
                    frame[zone[1]:zone[3], zone[0]:zone[2]],
                    source[zone[1]:zone[3], zone[0]:zone[2]],
                )
                changed = np.any(frame != source, axis=2)
                changed[zone[1]:zone[3], zone[0]:zone[2]] = False
                self.assertTrue(np.any(changed[box[1]:box[3], box[0]:box[2]]))
                self.assertFalse(np.any(changed[:box[1], :]))
                self.assertFalse(np.any(changed[box[3]:, :]))

    def test_full_zone_leaves_detected_box_unchanged(self):
        rng = np.random.default_rng(7)
        source = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
        for mode in ("blur", "pixelate"):
            with self.subTest(mode=mode):
                frame = source.copy()
                PIXELER.anonymize_box_excluding_zones(frame, (2, 2, 8, 8), [(0, 0, 10, 10)], mode, 7, 4)
                np.testing.assert_array_equal(frame, source)


class TemporalMaskTests(unittest.TestCase):
    @staticmethod
    def covers(masks, target):
        tx1, ty1, tx2, ty2 = target
        return any(
            x1 <= tx1 and y1 <= ty1 and x2 >= tx2 and y2 >= ty2
            for _, (x1, y1, x2, y2), _ in masks
        )

    def test_masks_bridge_exact_ttl_and_predict_motion(self):
        tracks = []
        first = PIXELER.update_temporal_masks([("faces", (10, 10, 20, 20))], tracks, 3, 100, 100)
        second = PIXELER.update_temporal_masks([("faces", (12, 10, 22, 20))], tracks, 3, 100, 100)
        missed_one = PIXELER.update_temporal_masks([], tracks, 3, 100, 100)
        missed_two = PIXELER.update_temporal_masks([], tracks, 3, 100, 100)
        missed_three = PIXELER.update_temporal_masks([], tracks, 3, 100, 100)
        expired = PIXELER.update_temporal_masks([], tracks, 3, 100, 100)

        self.assertEqual(first, [("faces", (10, 10, 20, 20), 0)])
        self.assertTrue(self.covers(second, (10, 10, 20, 20)))
        self.assertTrue(self.covers(second, (12, 10, 22, 20)))
        self.assertTrue(self.covers(missed_one, (12, 10, 22, 20)))
        self.assertGreaterEqual(max(box[2] for _, box, missed in missed_one if missed == 1), 24)
        self.assertTrue(any(missed == 2 for _, _, missed in missed_two))
        self.assertTrue(any(missed == 3 for _, _, missed in missed_three))
        self.assertEqual(expired, [])
        self.assertEqual(tracks, [])

    def test_last_observed_box_remains_covered_for_every_missed_frame(self):
        tracks = []
        observed = (80, 10, 90, 20)
        PIXELER.update_temporal_masks([("faces", observed)], tracks, 3, 100, 100)
        for _ in range(3):
            masks = PIXELER.update_temporal_masks([], tracks, 3, 100, 100)
            self.assertTrue(self.covers(masks, observed))
        self.assertEqual(PIXELER.update_temporal_masks([], tracks, 3, 100, 100), [])

    def test_motion_at_frame_edge_does_not_end_ttl_early(self):
        tracks = []
        PIXELER.update_temporal_masks([("plates", (80, 10, 92, 20))], tracks, 3, 100, 100)
        PIXELER.update_temporal_masks([("plates", (88, 10, 100, 20))], tracks, 3, 100, 100)
        for _ in range(3):
            masks = PIXELER.update_temporal_masks([], tracks, 3, 100, 100)
            self.assertTrue(self.covers(masks, (88, 10, 100, 20)))

    def test_size_change_keeps_valid_prediction(self):
        tracks = []
        PIXELER.update_temporal_masks([("faces", (0, 0, 20, 20))], tracks, 3, 100, 100)
        PIXELER.update_temporal_masks([("faces", (5, 5, 15, 15))], tracks, 3, 100, 100)
        masks = PIXELER.update_temporal_masks([], tracks, 3, 100, 100)
        self.assertTrue(self.covers(masks, (5, 5, 15, 15)))
        self.assertGreater(PIXELER.box_area(tracks[0]["box"]), 0)

    def test_adjacent_detection_cannot_consume_missing_objects_ttl(self):
        tracks = []
        old_box = (10, 10, 20, 20)
        PIXELER.update_temporal_masks([("faces", old_box)], tracks, 3, 100, 100)
        masks = PIXELER.update_temporal_masks([("faces", (20, 10, 30, 20))], tracks, 3, 100, 100)
        self.assertTrue(self.covers(masks, old_box))
        self.assertEqual(len(tracks), 2)

    def test_kinds_are_tracked_separately(self):
        tracks = []
        masks = PIXELER.update_temporal_masks(
            [("faces", (10, 10, 20, 20)), ("plates", (10, 10, 20, 20))],
            tracks,
            3,
            100,
            100,
        )
        self.assertEqual(len(masks), 2)
        self.assertEqual({track["kind"] for track in tracks}, {"faces", "plates"})

    def test_zero_ttl_disables_history(self):
        tracks = [{"kind": "faces", "box": (1, 1, 2, 2), "velocity": (0, 0, 0, 0), "missed": 0}]
        self.assertEqual(PIXELER.update_temporal_masks([], tracks, 0, 10, 10), [])
        self.assertEqual(tracks, [])


class TrackerResetTests(unittest.TestCase):
    def test_initialized_trackers_are_reset_once(self):
        class Tracker:
            def __init__(self):
                self.calls = 0

            def reset(self):
                self.calls += 1

        class Predictor:
            def __init__(self, trackers):
                self.trackers = trackers

        class Model:
            def __init__(self, predictor=None):
                self.predictor = predictor

        trackers = [Tracker(), Tracker()]
        models = [(Model(Predictor(trackers)), "faces"), (Model(), "plates")]
        ready, errors = PIXELER.reset_ultralytics_trackers(models)
        self.assertTrue(ready)
        self.assertEqual(errors, [])
        self.assertEqual([tracker.calls for tracker in trackers], [1, 1])

    def test_reset_failure_disables_ultralytics_tracking(self):
        class BrokenTracker:
            def reset(self):
                raise RuntimeError("reset failed")

        class Predictor:
            trackers = [BrokenTracker()]

        class Model:
            predictor = Predictor()

        ready, errors = PIXELER.reset_ultralytics_trackers([(Model(), "faces")])
        self.assertFalse(ready)
        self.assertEqual(errors, ["reset failed"])

    def test_raw_detection_capture_runs_before_tracker_filtering(self):
        class Tensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=float)

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        class Boxes:
            def __init__(self, values):
                self.xyxy = Tensor(values)

            def __len__(self):
                return len(self.xyxy.values)

        class Result:
            def __init__(self, values):
                self.boxes = Boxes(values)

        class Model:
            predictor = None

            def __init__(self, tracker_callback):
                self.callbacks = {"on_predict_postprocess_end": [tracker_callback]}

        def filter_unconfirmed_tracks(predictor):
            predictor.results[0] = Result([])

        model = Model(filter_unconfirmed_tracks)
        self.assertTrue(PIXELER.install_raw_detection_capture(model))
        self.assertTrue(PIXELER.install_raw_detection_capture(model))
        self.assertIs(model.callbacks["on_predict_postprocess_end"][0], PIXELER.capture_raw_detections)
        self.assertEqual(model.callbacks["on_predict_postprocess_end"].count(PIXELER.capture_raw_detections), 1)

        predictor = type("Predictor", (), {})()
        predictor.results = [Result([[10, 20, 30, 40]])]
        for callback in model.callbacks["on_predict_postprocess_end"]:
            callback(predictor)

        np.testing.assert_array_equal(predictor._dsgvo_raw_xyxy[0], [[10, 20, 30, 40]])
        self.assertEqual(len(predictor.results[0].boxes), 0)


if __name__ == "__main__":
    unittest.main()

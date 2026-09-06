from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ild_2d import ILDLayoutDataset2D, discover_records, mixed_condition_sampler


class MixedDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        patient = self.root / "patient_001"
        for folder in ("images", "lung_masks", "roi_masks"):
            (patient / folder).mkdir(parents=True)
        image = np.full((16, 16), 100, dtype=np.uint8)
        lung = np.zeros((16, 16), dtype=np.uint8)
        lung[2:14, 2:14] = 1
        for index in range(3):
            np.save(patient / "images" / f"slice_{index}.npy", image)
            np.save(patient / "lung_masks" / f"slice_{index}.npy", lung)
        empty = np.zeros((16, 16), dtype=np.uint8)
        labelled = empty.copy()
        labelled[5:8, 6:9] = 4
        np.save(patient / "roi_masks" / "slice_1.npy", empty)
        np.save(patient / "roi_masks" / "slice_2.npy", labelled)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_image_led_discovery_keeps_missing_and_empty_roi(self) -> None:
        records = discover_records(self.root, require_foreground=False)
        self.assertEqual(len(records), 3)
        self.assertEqual([record.labels for record in records], [(), (), (4,)])
        self.assertIsNone(records[0].roi_path)
        dataset = ILDLayoutDataset2D(records, image_size=16)
        self.assertFalse(bool(dataset[0]["has_disease_condition"]))
        self.assertEqual(int(dataset[0]["roi"].sum()), 0)
        self.assertTrue(bool(dataset[2]["has_disease_condition"]))

    def test_foreground_filter_and_mixed_sampler(self) -> None:
        foreground = discover_records(self.root, require_foreground=True)
        self.assertEqual(len(foreground), 1)
        records = discover_records(self.root, require_foreground=False)
        sampler = mixed_condition_sampler(
            records, seed=7, samples_per_epoch=4000, roi_sample_fraction=0.6
        )
        indexes = list(iter(sampler))
        observed = sum(records[index].has_disease_condition for index in indexes) / len(indexes)
        self.assertAlmostEqual(observed, 0.6, delta=0.04)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from vitalsense.io import load_sensor_csv, save_sensor_csv
from vitalsense.synthetic import generate_synthetic_session


class IoTests(unittest.TestCase):
    def test_sensor_csv_round_trip(self):
        batch = generate_synthetic_session(duration_s=5.0, sample_rate_hz=25.0, include_activity=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor.csv"
            save_sensor_csv(batch, path)
            loaded = load_sensor_csv(path)
        self.assertEqual(len(loaded.timestamp_s), len(batch.timestamp_s))
        self.assertAlmostEqual(loaded.sample_rate_hz, 25.0, places=4)


if __name__ == "__main__":
    unittest.main()


import csv
import tempfile
import unittest
from pathlib import Path
from qxvision.dataset import validate_dataset
from qxvision.labels import Label, TargetDefinition, label_next_candle
from qxvision.dataset import OHLCRow

class DatasetValidationTests(unittest.TestCase):
    def test_invalid_rows_and_naive_timestamps_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.csv"
            with path.open("w",newline="") as handle:
                writer=csv.writer(handle); writer.writerow(["timestamp","open","high","low","close","volume","asset","timeframe"])
                writer.writerow(["2025-01-01T00:01:00",1,2,0,1,1,"A","1m"])
                writer.writerow(["2025-01-01T00:00:00Z",1,0,2,1,1,"A","1m"])
            report=validate_dataset(path)
            self.assertFalse(report.valid); self.assertGreaterEqual(report.quarantined_rows,2); self.assertTrue(any("timezone" in error for error in report.errors))
            self.assertTrue(any("high" in error for error in report.errors))
    def test_target_doji_is_not_forced_direction(self):
        current=OHLCRow("2025-01-01T00:00:00Z",1,2,.5,1.5,None,"A","1m")
        following=OHLCRow("2025-01-01T00:01:00Z",1,2,.5,.8,None,"A","1m")
        self.assertEqual(label_next_candle(current,following),Label.DOWN)
        doji=OHLCRow("2025-01-01T00:01:00Z",1,2,.5,1,None,"A","1m")
        self.assertEqual(label_next_candle(current,doji),Label.DOJI)
        exact=OHLCRow("2025-01-01T00:01:00Z",1,2,.5,1,None,"A","1m")
        self.assertEqual(label_next_candle(exact,exact),Label.DOJI)

if __name__=='__main__': unittest.main()

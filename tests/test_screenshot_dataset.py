import csv,tempfile,unittest
from pathlib import Path
from qxvision.synthetic import OHLC,write
from qxvision.screenshot_dataset import import_manifest

class ScreenshotDatasetTests(unittest.TestCase):
    def test_import_preserves_hash_and_actual_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            image=Path(directory)/"chart.ppm"; rows=[OHLC(100+i*.1,101+i*.1,99+i*.1,100.5+i*.1) if i%2 else OHLC(100+i*.1,101+i*.1,99+i*.1,99.5+i*.1) for i in range(24)]; write(rows,image)
            manifest=Path(directory)/"manifest.csv"
            with manifest.open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=["screenshot_path","prediction_timestamp","actual_outcome","asset"]);writer.writeheader();writer.writerow({"screenshot_path":str(image),"prediction_timestamp":"2025-01-01T00:00:00Z","actual_outcome":"UP","asset":"TEST"})
            output=Path(directory)/"candidate.jsonl"; self.assertEqual(import_manifest(manifest,output),1); text=output.read_text(); self.assertIn('"actual_outcome": "UP"',text); self.assertIn('"screenshot_hash"',text)

if __name__=='__main__':unittest.main()

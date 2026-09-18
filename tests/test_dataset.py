import csv, tempfile, unittest
from pathlib import Path
from qxvision.dataset import load_ohlc_csv, chronological_split, leakage_check

class DatasetTests(unittest.TestCase):
    def test_chronological_split_and_leakage_detector(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['timestamp','open','high','low','close','volume','asset','timeframe'])
                for i in range(30): w.writerow([f'2025-01-01T00:{i:02d}:00Z',100,101,99,100+(i%2),1,'X','1m'])
            rows=load_ohlc_csv(p); a,b,c=chronological_split(rows,purge=1)
            self.assertLess(a[-1].timestamp,b[0].timestamp); self.assertLess(b[-1].timestamp,c[0].timestamp)
        self.assertTrue(leakage_check(['return_1','target_next_close']))

if __name__ == '__main__': unittest.main()

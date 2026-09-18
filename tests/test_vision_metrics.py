import unittest
from qxvision.synthetic import OHLC, render
from qxvision.vision import analyze
from qxvision.vision_metrics import reconstruction_metrics

class VisionMetricTests(unittest.TestCase):
    def test_synthetic_vision_metrics_are_not_market_metrics(self):
        rows=[OHLC(100+i*.2,101+i*.2,99+i*.2,100.6+i*.2) if i%2 else OHLC(100+i*.2,101+i*.2,99+i*.2,99.6+i*.2) for i in range(20)]
        report=reconstruction_metrics(rows,analyze(render(rows)))
        self.assertGreater(report["candle_detection_precision"],.5); self.assertIn("body_geometry_mae",report)

if __name__=='__main__': unittest.main()

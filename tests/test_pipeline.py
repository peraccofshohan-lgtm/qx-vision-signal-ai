import unittest
from qxvision.pipeline import VisionSignalPipeline
from qxvision.synthetic import OHLC, render

class PipelineTests(unittest.TestCase):
    def test_end_to_end_is_measured_and_fails_closed_without_weights(self):
        rows=[OHLC(100+i*.1,101+i*.1,99+i*.1,100.7+i*.1) if i%2 else OHLC(100+i*.1,101+i*.1,99+i*.1,99.4+i*.1) for i in range(24)]
        r=VisionSignalPipeline().analyze_frame(render(rows))
        self.assertEqual(r.decision.direction.value,"NO_TRADE")
        self.assertIn("MODEL_ARTIFACT_UNAVAILABLE",r.decision.abstention_reasons)
        self.assertGreater(r.stage_ms["total"],0); self.assertEqual(len(r.features.values),len(r.features.names))

if __name__=='__main__': unittest.main()

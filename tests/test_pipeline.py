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

    def test_running_red_or_green_never_forces_direction(self):
        complete=[OHLC(100+i*.1,100.8+i*.1,99.8+i*.1,100.5+i*.1) for i in range(23)]
        red=VisionSignalPipeline().analyze_frame(render(complete+[OHLC(102,102.3,101.5,101.7)]))
        green=VisionSignalPipeline().analyze_frame(render(complete+[OHLC(102,102.5,101.8,102.4)]))
        self.assertTrue(red.sequence.candles[-1].partial_observation); self.assertTrue(green.sequence.candles[-1].partial_observation)
        self.assertEqual(red.decision.direction,green.decision.direction)
        self.assertEqual(red.decision.direction.value,"NO_TRADE")

if __name__=='__main__': unittest.main()

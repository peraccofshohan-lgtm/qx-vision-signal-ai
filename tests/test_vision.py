import unittest
from qxvision.synthetic import OHLC, render
from qxvision.vision import inspect, reconstruct, analyze


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [OHLC(100+i*.3, 101+i*.3, 99.5+i*.3, 100.7+i*.3) if i%2==0 else OHLC(100+i*.3,100.8+i*.3,99.1+i*.3,99.5+i*.3) for i in range(24)]
        self.frame = render(self.rows)

    def test_adaptive_colors_and_count(self):
        chart=inspect(self.frame)
        self.assertIsNotNone(chart.bullish_color)
        self.assertIsNotNone(chart.bearish_color)
        seq=reconstruct(self.frame,chart)
        self.assertGreaterEqual(len(seq.candles), 20)
        self.assertFalse(seq.candles[-1].is_complete)
        self.assertTrue(seq.candles[-1].partial_observation)
        self.assertEqual(seq.running_candle_index,len(seq.candles)-1)

    def test_theme_changes_do_not_break_geometry(self):
        f=render(self.rows,bullish=(50,220,70),bearish=(240,150,30),background=(238,238,238),grid=False)
        seq=analyze(f)
        self.assertGreaterEqual(len(seq.candles),20)

    def test_empty_frame_is_safe(self):
        from qxvision.domain import ImageFrame
        f=ImageFrame(20,20,tuple(tuple((0,0,0) for _ in range(20)) for _ in range(20)))
        seq=analyze(f)
        self.assertEqual(seq.candles,[])
        self.assertFalse(seq.chart.quality.is_supported)

if __name__ == '__main__': unittest.main()

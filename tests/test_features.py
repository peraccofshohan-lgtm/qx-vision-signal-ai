import unittest
from qxvision.synthetic import OHLC, render
from qxvision.vision import analyze
from qxvision.features import build_state, build_feature_vector, FEATURE_SCHEMA_VERSION

class FeatureTests(unittest.TestCase):
    def _state(self):
        rows=[]
        for i in range(30):
            base=100+i*.4
            rows.append(OHLC(base,base+1.0,base-.15,base+.75))
        seq=analyze(render(rows)); return seq
    def test_structure_trend_and_vector(self):
        seq=self._state(); state=build_state(seq.candles); v=build_feature_vector(seq.candles,seq.chart.quality,state)
        self.assertEqual(v.schema_version,FEATURE_SCHEMA_VERSION)
        self.assertEqual(len(v.values),len(v.names)); self.assertEqual(len(v.names),57); self.assertTrue(all(x==x for x in v.values))
        self.assertIn(state.regime.value, ["TRENDING_BULL","BREAKOUT","TRANSITION","RANGING","VOLATILITY_COMPRESSION"])
    def test_running_values_are_explicit(self):
        seq=self._state(); state=build_state(seq.candles); v=build_feature_vector(seq.candles,seq.chart.quality,state)
        d=v.as_dict(); self.assertEqual(d["running_present"],1.0); self.assertEqual(v.metadata["running_candle_excluded_from_target"],"true")

if __name__ == '__main__': unittest.main()

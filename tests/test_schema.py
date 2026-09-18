import json
import unittest
from pathlib import Path
from qxvision.schema import feature_schema, validate_feature_schema
from qxvision.synthetic import OHLC, render
from qxvision.vision import analyze
from qxvision.features import build_feature_vector, build_state

class SchemaTests(unittest.TestCase):
    def test_json_schema_matches_feature_vector_order(self):
        rows=[OHLC(100+i*.1,101+i*.1,99+i*.1,100.5+i*.1) for i in range(12)]
        sequence=analyze(render(rows)); state=build_state(sequence.candles); vector=build_feature_vector(sequence.candles,sequence.chart.quality,state)
        validate_feature_schema(feature_schema(),vector.names)
        self.assertEqual(vector.schema_version, "qxvision.features.v1")
        self.assertEqual(len(vector.names),57)
    def test_checked_in_schema_is_valid_json(self):
        document=json.loads(Path("feature_schema.json").read_text())
        self.assertEqual(document["schema_version"],"qxvision.features.v1")
        self.assertEqual(len(document["features"]),57)

if __name__=='__main__': unittest.main()

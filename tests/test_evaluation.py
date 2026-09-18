import unittest
from qxvision.evaluate import baseline_predictions, evaluate, risk_coverage_curve, reliability_curve

class EvaluationTests(unittest.TestCase):
    def test_risk_coverage_and_reliability_are_explicit(self):
        labels=[1,1,0,0]; probabilities=[.9,.7,.2,.4]; predictions=[1,1,0,1]
        curve=risk_coverage_curve(predictions,probabilities,labels,[.5,.8])
        self.assertEqual(curve[0]["sample_count"],4); self.assertEqual(curve[1]["sample_count"],2)
        self.assertEqual(len(reliability_curve(probabilities,labels)),10)
    def test_baselines_are_reproducible(self):
        values=baseline_predictions([0,1,0,1],previous=[0,0,1,1],momentum=[1,1,0,0],seed=7)
        self.assertEqual(values,baseline_predictions([0,1,0,1],previous=[0,0,1,1],momentum=[1,1,0,0],seed=7))
        self.assertEqual(set(values),{"random_50_50","majority_class","previous_direction","simple_momentum"})

if __name__=='__main__': unittest.main()

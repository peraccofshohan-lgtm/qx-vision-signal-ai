import unittest
from qxvision.model import fit_logistic, fit_gradient_boosting, fit_random_forest, PlattCalibrator, FeatureOOD
from qxvision.evaluate import evaluate
from qxvision.signal import SignalPolicy

class ModelTests(unittest.TestCase):
    def test_logistic_probability_bounds_and_calibration(self):
        x=[[0.,0.],[1.,1.],[2.,1.],[-1.,-1.]]; y=[0,1,1,0]
        m=fit_logistic(x,y,epochs=100); probs=[m.probability(a) for a in x]
        self.assertTrue(all(0 <= p <= 1 for p in probs))
        c=PlattCalibrator().fit(probs,y); self.assertTrue(all(0 <= c.transform(p) <= 1 for p in probs))
    def test_candidate_model_families_return_bounded_probabilities(self):
        x=[[0.,0.],[1.,1.],[2.,1.],[-1.,-1.],[1.,0.]]; y=[0,1,1,0,1]
        for model in (fit_gradient_boosting(x,y,rounds=4),fit_random_forest(x,y,trees=5,seed=3)):
            self.assertTrue(all(0 <= model.probability(row) <= 1 for row in x))

    def test_ood_is_fit_without_target(self):
        o=FeatureOOD.fit([[0,0],[1,1],[2,2]])
        self.assertFalse(o.is_ood([1,1])); self.assertTrue(o.is_ood([100,100]))
    def test_abstention_metrics(self):
        r=evaluate([1,None,0],[.8,None,.3],[1,0,1],["TRENDING_BULL"]*3)
        self.assertEqual(r.abstentions,1); self.assertAlmostEqual(r.coverage,2/3)

if __name__ == '__main__': unittest.main()

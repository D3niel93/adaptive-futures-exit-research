import unittest
from report import metrics
class MetricsTests(unittest.TestCase):
    def test_initial_loss_is_drawdown(self):
        self.assertEqual(metrics([-10,20,-5])["max_closed_trade_drawdown_usd"],10)
    def test_net_expectancy_pf(self):
        m=metrics([10,-5,0])
        self.assertEqual(m["profit_factor"],2)
        self.assertAlmostEqual(m["win_rate_pct"],100/3)
        self.assertAlmostEqual(m["expectancy_usd"],5/3)
    def test_reject_invalid(self):
        with self.assertRaises(ValueError): metrics([float("nan")])

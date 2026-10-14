import unittest
from datetime import datetime,timedelta,timezone
from learner import select_exit
class LearnerTests(unittest.TestCase):
    def test_pending_outcomes_cannot_leak(self):
        now=datetime(2025,1,20,tzinfo=timezone.utc)
        history=[dict(entry=now-timedelta(days=1),exit=now+timedelta(seconds=1),
            regime="NORMAL",tp=20,sl=40,net_points=1000) for _ in range(8)]
        self.assertEqual(select_exit(history,now,"NORMAL",[(20,40)]),(10,30))
    def test_known_history(self):
        now=datetime(2025,1,20,tzinfo=timezone.utc)
        history=[dict(entry=now-timedelta(days=1),exit=now-timedelta(seconds=1),
            regime="NORMAL",tp=20,sl=40,net_points=3) for _ in range(8)]
        self.assertEqual(select_exit(history,now,"NORMAL",[(20,40)]),(20,40))

import unittest
import pandas as pd

from scanner_core import _normalize_common_columns


class JointEventDefinitionTests(unittest.TestCase):
    def test_all_three_selections_share_exact_joint_outcome(self):
        # 1X AND 12 AND Over 1.5 all win only for a home win with 2+ goals.
        scores = pd.DataFrame({
            "HomeTeam": ["H"] * 4,
            "AwayTeam": ["A"] * 4,
            "FTHG": [2, 1, 1, 0],
            "FTAG": [0, 0, 1, 2],
        })
        home_win = scores["FTHG"] > scores["FTAG"]
        over_15 = (scores["FTHG"] + scores["FTAG"]) >= 2
        all3 = home_win & over_15
        self.assertEqual(all3.tolist(), [True, False, False, False])

    def test_argentina_schema_is_normalized(self):
        source = pd.DataFrame({
            "League": ["ARG"],
            "Home": ["Home FC"],
            "Away": ["Away FC"],
            "HG": [2],
            "AG": [1],
            "AvgCH": [1.8],
            "AvgCD": [3.4],
            "AvgCA": [4.2],
        })
        normalized = _normalize_common_columns(source)
        self.assertEqual(normalized.loc[0, "HomeTeam"], "Home FC")
        self.assertEqual(normalized.loc[0, "AwayTeam"], "Away FC")
        self.assertEqual(normalized.loc[0, "FTHG"], 2)
        self.assertEqual(normalized.loc[0, "FTAG"], 1)
        self.assertAlmostEqual(normalized.loc[0, "odd_H"], 1.8)


if __name__ == "__main__":
    unittest.main()

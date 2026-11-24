# Changelog

## Joint-event correction
- Added a dedicated ALL3 optimizer trained on the exact joint outcome.
- Defined the joint outcome as home win AND total goals >= 2.
- Changed qualification to require P_ALL3 > 0.60.
- Added actual_ALL3/Hit_ALL3 labels and joint-event hit-rate reporting.
- Included the supplied European 2025/26, European 2026/27 and Argentina historical workbooks.
- Normalized Argentina Home/Away/HG/AG fields and isolated team histories by league.
- Removed synthetic settlement odds and misleading single-leg P&L from the joint-event report.

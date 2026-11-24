# Walk-forward validation record

Validation was run on the three supplied workbooks using the scanner's chronological processing order.

- Combined historical rows loaded: 16,520. One 2025/26 European row had no final score and was excluded from settled-match training.
- 2026/27 workbook rows: 2,399.
- 2026/27 qualification threshold: `P_ALL3 > 0.60`.
- Qualified settled fixtures: 150.
- Joint-event successes: 95.
- Observed hit rate: **63.33%**.

This is a chronological walk-forward result: each prediction is made before its own result is added to training history, and the optimizer is recalibrated using earlier settled matches. It is not a guarantee of future performance. The 2026/27 workbook ends on 9 November 2026, so it does not provide current fixtures after that date. Re-run validation as new results become available and monitor calibration, sample size, and league-level performance.

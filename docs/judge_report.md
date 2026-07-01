# Judge Report

## Before mitigation
- Heuristic judges can be biased by ordering and verbosity.

## After mitigation
- The pipeline uses rubric-based prompts, order-swapped evaluation, self-enhancement mitigation, sycophancy probes, and calibration anchors.

## Bias statistics
- Position bias is measured using flip-rate experiments.
- Verbosity bias is measured by comparing baseline and padded responses.
- Self-enhancement mitigation is active when generator and judge families differ.
- Sycophancy probes report how often the judge is swayed by overconfident but unsupported output.

## Validation
- Agreement and consistency metrics are stored in [evaluation/judge_validation.json](../evaluation/judge_validation.json).
- Bias results are stored in [evaluation/bias_results.json](../evaluation/bias_results.json).

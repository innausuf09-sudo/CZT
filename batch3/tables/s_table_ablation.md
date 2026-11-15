| Configuration (ρ = 1.2, c = 4) | CV | ZCV | Mean zero-day wait (min) | SLA violation | Alerts per shift |
|---|---|---|---|---|---|
| Risk baseline (p x catalogue severity) | 0.635 ± 0.011 | 0.569 ± 0.025 | 30.2 ± 6.7 | 0.043 ± 0.011 | 225 |
| CZT-D (default weights) | 0.624 ± 0.012 | 0.522 ± 0.024 | 57.0 ± 8.5 | 0.022 ± 0.005 | 225 |
|   D, without calibration | 0.623 ± 0.012 | 0.514 ± 0.023 | 60.5 ± 8.9 | 0.023 ± 0.005 | 225 |
|   D, without evidence decay | 0.625 ± 0.013 | 0.545 ± 0.028 | 35.2 ± 7.8 | 0.003 ± 0.002 | 225 |
|   D, without deadline term | 0.639 ± 0.010 | 0.547 ± 0.023 | 59.3 ± 9.2 | 0.069 ± 0.012 | 225 |
|   D, without detector score (gamma = 0) | 0.616 ± 0.012 | 0.506 ± 0.025 | 63.3 ± 8.6 | 0.024 ± 0.005 | 225 |
| CZT-TE (tuned index + escalation) | 0.617 ± 0.009 | 0.589 ± 0.034 | 3.6 ± 1.1 | 0.012 ± 0.005 | 170 |
|   TE, without calibration | 0.616 ± 0.010 | 0.585 ± 0.034 | 4.2 ± 1.3 | 0.012 ± 0.004 | 170 |
|   TE with escalation reset to q = 0.99 (= CZT-T) | 0.628 ± 0.011 | 0.670 ± 0.020 | 7.0 ± 1.8 | 0.048 ± 0.010 | 225 |
|   TE, imputed severity s_u = 0.5 | 0.618 ± 0.009 | 0.530 ± 0.025 | 21.3 ± 7.2 | 0.009 ± 0.005 | 170 |
|   TE, without evidence decay | 0.613 ± 0.009 | 0.589 ± 0.035 | 3.1 ± 0.7 | 0.007 ± 0.003 | 170 |
|   TE, without detector score (gamma = 0) | 0.614 ± 0.010 | 0.586 ± 0.034 | 3.9 ± 1.3 | 0.013 ± 0.005 | 170 |

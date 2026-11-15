| model       | scores                         |   ece_deploy |   logloss_deploy |   brier_deploy |   mean_p_deploy |   mean_p_benign |   missing_p_attack |   ece_indep_valmix |
|:------------|:-------------------------------|-------------:|-----------------:|---------------:|----------------:|----------------:|-------------------:|-------------------:|
| B-ProtoMeta | corrected, assumed prior 0.003 |       0.0232 |           0.1012 |         0.0229 |          0.0071 |          0.0025 |             0.8451 |           nan      |
| B-ProtoMeta | corrected, assumed prior 0.10  |       0.0275 |           0.0832 |         0.0227 |          0.0562 |          0.0395 |             0.4033 |           nan      |
| B-ProtoMeta | prior-corrected                |       0.0080 |           0.0686 |         0.0172 |          0.0291 |          0.0165 |             0.5626 |           nan      |
| B-ProtoMeta | recalibrated                   |       0.0084 |           0.0693 |         0.0172 |          0.0283 |          0.0158 |             0.5654 |           nan      |
| B-ProtoMeta | uncorrected                    |       0.1123 |           0.2106 |         0.0629 |          0.1423 |          0.1223 |             0.2088 |             0.0472 |
| CLAD        | corrected, assumed prior 0.003 |       0.0233 |           0.0970 |         0.0235 |          0.0067 |          0.0031 |             0.8742 |           nan      |
| CLAD        | corrected, assumed prior 0.10  |       0.0255 |           0.0834 |         0.0232 |          0.0544 |          0.0355 |             0.3360 |           nan      |
| CLAD        | prior-corrected                |       0.0126 |           0.0675 |         0.0171 |          0.0317 |          0.0175 |             0.5098 |           nan      |
| CLAD        | recalibrated                   |       0.0132 |           0.0675 |         0.0168 |          0.0290 |          0.0158 |             0.5451 |           nan      |
| CLAD        | uncorrected                    |       0.1018 |           0.1948 |         0.0451 |          0.1318 |          0.1109 |             0.1940 |             0.0633 |
| CZT-Hybrid  | corrected, assumed prior 0.003 |       0.0218 |           0.0889 |         0.0201 |          0.0082 |          0.0016 |             0.7756 |           nan      |
| CZT-Hybrid  | corrected, assumed prior 0.10  |       0.0210 |           0.0642 |         0.0166 |          0.0482 |          0.0296 |             0.3488 |           nan      |
| CZT-Hybrid  | prior-corrected                |       0.0070 |           0.0561 |         0.0135 |          0.0265 |          0.0116 |             0.4895 |           nan      |
| CZT-Hybrid  | recalibrated                   |       0.0073 |           0.0572 |         0.0134 |          0.0274 |          0.0114 |             0.4567 |           nan      |
| CZT-Hybrid  | uncorrected                    |       0.0890 |           0.1620 |         0.0514 |          0.1177 |          0.0960 |             0.1802 |             0.0280 |
| DNN         | corrected, assumed prior 0.003 |       0.0186 |           0.0874 |         0.0183 |          0.0115 |          0.0019 |             0.6810 |           nan      |
| DNN         | corrected, assumed prior 0.10  |       0.0216 |           0.0699 |         0.0181 |          0.0500 |          0.0318 |             0.3618 |           nan      |
| DNN         | prior-corrected                |       0.0060 |           0.0600 |         0.0146 |          0.0280 |          0.0127 |             0.4774 |           nan      |
| DNN         | recalibrated                   |       0.0059 |           0.0599 |         0.0146 |          0.0278 |          0.0128 |             0.4872 |           nan      |
| DNN         | uncorrected                    |       0.0963 |           0.1755 |         0.0557 |          0.1249 |          0.1038 |             0.1947 |             0.0224 |
| LightGBM    | corrected, assumed prior 0.003 |       0.0201 |           0.0835 |         0.0182 |          0.0099 |          0.0014 |             0.7151 |           nan      |
| LightGBM    | corrected, assumed prior 0.10  |       0.0176 |           0.0588 |         0.0146 |          0.0451 |          0.0256 |             0.3262 |           nan      |
| LightGBM    | prior-corrected                |       0.0058 |           0.0529 |         0.0124 |          0.0261 |          0.0099 |             0.4504 |           nan      |
| LightGBM    | recalibrated                   |       0.0057 |           0.0542 |         0.0124 |          0.0270 |          0.0100 |             0.4247 |           nan      |
| LightGBM    | uncorrected                    |       0.0808 |           0.1447 |         0.0451 |          0.1090 |          0.0868 |             0.1736 |             0.0226 |
| ResSE-CNN1D | corrected, assumed prior 0.003 |       0.0228 |           0.0974 |         0.0221 |          0.0074 |          0.0020 |             0.8207 |           nan      |
| ResSE-CNN1D | corrected, assumed prior 0.10  |       0.0244 |           0.0749 |         0.0202 |          0.0522 |          0.0348 |             0.3854 |           nan      |
| ResSE-CNN1D | prior-corrected                |       0.0068 |           0.0638 |         0.0158 |          0.0275 |          0.0143 |             0.5449 |           nan      |
| ResSE-CNN1D | recalibrated                   |       0.0073 |           0.0644 |         0.0158 |          0.0282 |          0.0142 |             0.5193 |           nan      |
| ResSE-CNN1D | uncorrected                    |       0.0980 |           0.1859 |         0.0590 |          0.1268 |          0.1057 |             0.1913 |             0.0260 |
| XGBoost     | corrected, assumed prior 0.003 |       0.0199 |           0.0841 |         0.0182 |          0.0101 |          0.0014 |             0.7099 |           nan      |
| XGBoost     | corrected, assumed prior 0.10  |       0.0182 |           0.0595 |         0.0147 |          0.0454 |          0.0261 |             0.3301 |           nan      |
| XGBoost     | prior-corrected                |       0.0061 |           0.0535 |         0.0126 |          0.0261 |          0.0100 |             0.4550 |           nan      |
| XGBoost     | recalibrated                   |       0.0058 |           0.0550 |         0.0125 |          0.0274 |          0.0103 |             0.4191 |           nan      |
| XGBoost     | uncorrected                    |       0.0819 |           0.1473 |         0.0464 |          0.1102 |          0.0881 |             0.1743 |             0.0224 |
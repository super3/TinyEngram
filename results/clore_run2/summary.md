| suite | task | metric | base | sel_table_Nov05_15-34-26 | sel_random_table_Nov05_16-06-08 |
|---|---|---|---|---|---|
| domain | mmlu_clinical_knowledge (±0.030) | acc | 0.6000 | 0.5623 (-0.0377) | 0.5736 (-0.0264) |
| domain | mmlu_medical_genetics (±0.050) | acc | 0.5800 | 0.5700 (-0.0100) | 0.5400 (-0.0400) |
| domain | mmlu_professional_medicine (±0.030) | acc | 0.5000 | 0.4779 (-0.0221) | 0.4816 (-0.0184) |
| domain | mmlu_anatomy (±0.043) | acc | 0.4889 | 0.4667 (-0.0222) | 0.4519 (-0.0370) |
| domain | mmlu_college_medicine (±0.038) | acc | 0.5087 | 0.5202 (+0.0116) | 0.5145 (+0.0058) |
| domain | mmlu_college_biology (±0.041) | acc | 0.5694 | 0.5694 (+0.0000) | 0.5833 (+0.0139) |
| domain | pubmedqa | None | - | - | - |
| domain | medmcqa | None | - | - | - |
| domain | medqa_4options (±0.014) | acc | 0.3849 | 0.3723 (-0.0126) | 0.3731 (-0.0118) |
| ood | mmlu_non_biomed | acc | 0.5177 | 0.5189 (+0.0013) | 0.5172 (-0.0005) |
| ood | truthfulqa_mc1 (±0.016) | acc | 0.2938 | 0.2974 (+0.0037) | 0.2974 (+0.0037) |
| ood | truthfulqa_mc2 (±0.015) | acc | 0.4587 | 0.4596 (+0.0008) | 0.4592 (+0.0005) |
| ood | arc_challenge (±0.014) | acc_norm | 0.3874 | 0.3925 (+0.0051) | 0.3857 (-0.0017) |
| ood | hellaswag (±0.005) | acc_norm | 0.5380 | 0.5371 (-0.0009) | 0.5366 (-0.0014) |
| ood | winogrande (±0.014) | acc | 0.5912 | 0.5817 (-0.0095) | 0.5841 (-0.0071) |
| **domain avg** | | | 0.5188 | 0.5056 (-0.0133) | 0.5026 (-0.0163) |
| **ood avg** | | | 0.4645 | 0.4645 (+0.0001) | 0.4634 (-0.0011) |

Trained minus random-frozen table (sel_table_Nov05_15-34-26 vs sel_random_table_Nov05_16-06-08): domain: +0.0030, ood: +0.0012

| run | task | fire rate | mean gate (fired) | inject ratio (fired) |
|---|---|---|---|---|
| sel_table_Nov05_15-34-26 | mmlu_clinical_knowledge | 0.033 | 0.495 | 0.1101 |
| sel_table_Nov05_15-34-26 | mmlu_medical_genetics | 0.093 | 0.524 | 0.1045 |
| sel_table_Nov05_15-34-26 | mmlu_professional_medicine | 0.118 | 0.450 | 0.1013 |
| sel_table_Nov05_15-34-26 | mmlu_anatomy | 0.117 | 0.480 | 0.1087 |
| sel_table_Nov05_15-34-26 | mmlu_college_medicine | 0.033 | 0.472 | 0.1074 |
| sel_table_Nov05_15-34-26 | mmlu_college_biology | 0.078 | 0.449 | 0.0987 |
| sel_table_Nov05_15-34-26 | medqa_4options | 0.145 | 0.445 | 0.1002 |
| sel_table_Nov05_15-34-26 | mmlu_non_biomed | 0.012 | 0.517 | 0.1161 |
| sel_table_Nov05_15-34-26 | truthfulqa_mc1 | 0.001 | 0.513 | 0.1057 |
| sel_table_Nov05_15-34-26 | truthfulqa_mc2 | 0.001 | 0.509 | 0.1051 |
| sel_table_Nov05_15-34-26 | arc_challenge | 0.033 | 0.617 | 0.1223 |
| sel_table_Nov05_15-34-26 | hellaswag | 0.017 | 0.547 | 0.1228 |
| sel_table_Nov05_15-34-26 | winogrande | 0.012 | 0.728 | 0.1340 |
| sel_random_table_Nov05_16-06-08 | mmlu_clinical_knowledge | 0.033 | 0.370 | 0.1030 |
| sel_random_table_Nov05_16-06-08 | mmlu_medical_genetics | 0.093 | 0.427 | 0.1035 |
| sel_random_table_Nov05_16-06-08 | mmlu_professional_medicine | 0.118 | 0.340 | 0.0951 |
| sel_random_table_Nov05_16-06-08 | mmlu_anatomy | 0.117 | 0.398 | 0.1030 |
| sel_random_table_Nov05_16-06-08 | mmlu_college_medicine | 0.033 | 0.358 | 0.1010 |
| sel_random_table_Nov05_16-06-08 | mmlu_college_biology | 0.078 | 0.366 | 0.0966 |
| sel_random_table_Nov05_16-06-08 | medqa_4options | 0.145 | 0.342 | 0.0946 |
| sel_random_table_Nov05_16-06-08 | mmlu_non_biomed | 0.012 | 0.402 | 0.1134 |
| sel_random_table_Nov05_16-06-08 | truthfulqa_mc1 | 0.001 | 0.391 | 0.1056 |
| sel_random_table_Nov05_16-06-08 | truthfulqa_mc2 | 0.001 | 0.388 | 0.1047 |
| sel_random_table_Nov05_16-06-08 | arc_challenge | 0.033 | 0.534 | 0.1286 |
| sel_random_table_Nov05_16-06-08 | hellaswag | 0.017 | 0.446 | 0.1203 |
| sel_random_table_Nov05_16-06-08 | winogrande | 0.012 | 0.668 | 0.1444 |

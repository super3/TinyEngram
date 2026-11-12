| suite | task | metric | chat | table_Nov11_23-31-10 | random_table_Nov12_00-01-19 | zero_table_Nov11_22-55-01 |
|---|---|---|---|---|---|---|
| domain | mmlu_clinical_knowledge (±0.029) | acc | 0.3358 | 0.4340 (+0.0981) | 0.4113 (+0.0755) | 0.4566 (+0.1208) |
| domain | mmlu_medical_genetics (±0.049) | acc | 0.3700 | 0.4300 (+0.0600) | 0.4400 (+0.0700) | 0.4000 (+0.0300) |
| domain | mmlu_professional_medicine (±0.028) | acc | 0.3199 | 0.4301 (+0.1103) | 0.4301 (+0.1103) | 0.4301 (+0.1103) |
| domain | mmlu_anatomy (±0.042) | acc | 0.3704 | 0.4444 (+0.0741) | 0.4148 (+0.0444) | 0.4222 (+0.0519) |
| domain | mmlu_college_medicine (±0.035) | acc | 0.2948 | 0.3873 (+0.0925) | 0.3988 (+0.1040) | 0.3699 (+0.0751) |
| domain | mmlu_college_biology (±0.042) | acc | 0.4653 | 0.5139 (+0.0486) | 0.4722 (+0.0069) | 0.4931 (+0.0278) |
| domain | pubmedqa | None | - | - | - | - |
| domain | medmcqa | None | - | - | - | - |
| domain | medqa_4options (±0.013) | acc | 0.3236 | 0.3354 (+0.0118) | 0.3236 (+0.0000) | 0.3268 (+0.0031) |
| ood | mmlu_non_biomed | acc | 0.4221 | 0.4367 (+0.0146) | 0.4556 (+0.0335) | 0.4720 (+0.0499) |
| ood | truthfulqa_mc1 (±0.016) | acc | 0.2729 | 0.3048 (+0.0318) | 0.2889 (+0.0159) | 0.2950 (+0.0220) |
| ood | truthfulqa_mc2 (±0.015) | acc | 0.4284 | 0.4687 (+0.0402) | 0.4482 (+0.0197) | 0.4436 (+0.0152) |
| ood | arc_challenge (±0.014) | acc_norm | 0.3370 | 0.3456 (+0.0085) | 0.3498 (+0.0128) | 0.3652 (+0.0282) |
| ood | hellaswag (±0.005) | acc_norm | 0.4730 | 0.4740 (+0.0010) | 0.4771 (+0.0041) | 0.4815 (+0.0085) |
| ood | winogrande (±0.014) | acc | 0.5627 | 0.5620 (-0.0008) | 0.5627 (+0.0000) | 0.5588 (-0.0039) |
| **domain avg** | | | 0.3543 | 0.4250 (+0.0708) | 0.4130 (+0.0587) | 0.4141 (+0.0599) |
| **ood avg** | | | 0.4160 | 0.4319 (+0.0159) | 0.4304 (+0.0143) | 0.4360 (+0.0200) |

Trained minus random-frozen table (table_Nov11_23-31-10 vs random_table_Nov12_00-01-19): domain: +0.0120, ood: +0.0016

| run | task | fire rate | mean gate (fired) | inject ratio (fired) |
|---|---|---|---|---|
| table_Nov11_23-31-10 | mmlu_clinical_knowledge | 1.000 | 0.681 | 0.0490 |
| table_Nov11_23-31-10 | mmlu_medical_genetics | 1.000 | 0.470 | 0.1414 |
| table_Nov11_23-31-10 | mmlu_professional_medicine | 1.000 | 0.430 | 0.1341 |
| table_Nov11_23-31-10 | mmlu_anatomy | 1.000 | 0.450 | 0.1376 |
| table_Nov11_23-31-10 | mmlu_college_medicine | 1.000 | 0.395 | 0.1260 |
| table_Nov11_23-31-10 | mmlu_college_biology | 1.000 | 0.423 | 0.1337 |
| table_Nov11_23-31-10 | medqa_4options | 1.000 | 0.427 | 0.1340 |
| table_Nov11_23-31-10 | mmlu_non_biomed | 1.000 | 0.431 | 0.1425 |
| table_Nov11_23-31-10 | truthfulqa_mc1 | 1.000 | 0.488 | 0.1514 |
| table_Nov11_23-31-10 | truthfulqa_mc2 | 1.000 | 0.488 | 0.1514 |
| table_Nov11_23-31-10 | arc_challenge | 1.000 | 0.524 | 0.1567 |
| table_Nov11_23-31-10 | hellaswag | 1.000 | 0.410 | 0.1318 |
| table_Nov11_23-31-10 | winogrande | 1.000 | 0.543 | 0.1537 |
| random_table_Nov12_00-01-19 | mmlu_clinical_knowledge | 1.000 | 0.474 | 0.0501 |
| random_table_Nov12_00-01-19 | mmlu_medical_genetics | 1.000 | 0.442 | 0.1498 |
| random_table_Nov12_00-01-19 | mmlu_professional_medicine | 1.000 | 0.405 | 0.1472 |
| random_table_Nov12_00-01-19 | mmlu_anatomy | 1.000 | 0.420 | 0.1485 |
| random_table_Nov12_00-01-19 | mmlu_college_medicine | 1.000 | 0.352 | 0.1520 |
| random_table_Nov12_00-01-19 | mmlu_college_biology | 1.000 | 0.394 | 0.1482 |
| random_table_Nov12_00-01-19 | medqa_4options | 1.000 | 0.403 | 0.1470 |
| random_table_Nov12_00-01-19 | mmlu_non_biomed | 1.000 | 0.409 | 0.1510 |
| random_table_Nov12_00-01-19 | truthfulqa_mc1 | 1.000 | 0.429 | 0.1537 |
| random_table_Nov12_00-01-19 | truthfulqa_mc2 | 1.000 | 0.429 | 0.1537 |
| random_table_Nov12_00-01-19 | arc_challenge | 1.000 | 0.513 | 0.1496 |
| random_table_Nov12_00-01-19 | hellaswag | 1.000 | 0.364 | 0.1432 |
| random_table_Nov12_00-01-19 | winogrande | 1.000 | 0.503 | 0.1436 |
| zero_table_Nov11_22-55-01 | mmlu_clinical_knowledge | 1.000 | 0.500 | 0.0261 |
| zero_table_Nov11_22-55-01 | mmlu_medical_genetics | 1.000 | 0.500 | 0.0785 |
| zero_table_Nov11_22-55-01 | mmlu_professional_medicine | 1.000 | 0.500 | 0.0808 |
| zero_table_Nov11_22-55-01 | mmlu_anatomy | 1.000 | 0.500 | 0.0785 |
| zero_table_Nov11_22-55-01 | mmlu_college_medicine | 1.000 | 0.500 | 0.0833 |
| zero_table_Nov11_22-55-01 | mmlu_college_biology | 1.000 | 0.500 | 0.0798 |
| zero_table_Nov11_22-55-01 | medqa_4options | 1.000 | 0.500 | 0.0814 |
| zero_table_Nov11_22-55-01 | mmlu_non_biomed | 1.000 | 0.500 | 0.0812 |
| zero_table_Nov11_22-55-01 | truthfulqa_mc1 | 1.000 | 0.500 | 0.0808 |
| zero_table_Nov11_22-55-01 | truthfulqa_mc2 | 1.000 | 0.500 | 0.0808 |
| zero_table_Nov11_22-55-01 | arc_challenge | 1.000 | 0.500 | 0.0735 |
| zero_table_Nov11_22-55-01 | hellaswag | 1.000 | 0.500 | 0.0766 |
| zero_table_Nov11_22-55-01 | winogrande | 1.000 | 0.500 | 0.0665 |

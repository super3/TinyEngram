| suite | task | metric | base | table_Nov05_13-50-10 | random_table_Nov05_14-21-11 |
|---|---|---|---|---|---|
| domain | mmlu_clinical_knowledge (±0.030) | acc | 0.6000 | 0.5774 (-0.0226) | 0.5849 (-0.0151) |
| domain | mmlu_medical_genetics (±0.050) | acc | 0.5800 | 0.5500 (-0.0300) | 0.5500 (-0.0300) |
| domain | mmlu_professional_medicine (±0.030) | acc | 0.5000 | 0.4816 (-0.0184) | 0.4816 (-0.0184) |
| domain | mmlu_anatomy (±0.043) | acc | 0.4889 | 0.4519 (-0.0370) | 0.4444 (-0.0444) |
| domain | mmlu_college_medicine (±0.038) | acc | 0.5087 | 0.5145 (+0.0058) | 0.5029 (-0.0058) |
| domain | mmlu_college_biology (±0.041) | acc | 0.5694 | 0.5486 (-0.0208) | 0.5556 (-0.0139) |
| domain | pubmedqa | None | - | - | - |
| domain | medmcqa | None | - | - | - |
| domain | medqa_4options (±0.014) | acc | 0.3849 | 0.3692 (-0.0157) | 0.3763 (-0.0086) |
| ood | mmlu_non_biomed | acc | 0.5177 | 0.5089 (-0.0088) | 0.5118 (-0.0059) |
| ood | truthfulqa_mc1 (±0.016) | acc | 0.2938 | 0.2962 (+0.0024) | 0.2938 (+0.0000) |
| ood | truthfulqa_mc2 (±0.015) | acc | 0.4587 | 0.4530 (-0.0057) | 0.4510 (-0.0077) |
| ood | arc_challenge (±0.014) | acc_norm | 0.3874 | 0.3985 (+0.0111) | 0.3908 (+0.0034) |
| ood | hellaswag (±0.005) | acc_norm | 0.5380 | 0.5351 (-0.0030) | 0.5338 (-0.0043) |
| ood | winogrande (±0.014) | acc | 0.5912 | 0.5864 (-0.0047) | 0.5817 (-0.0095) |
| **domain avg** | | | 0.5188 | 0.4990 (-0.0198) | 0.4994 (-0.0195) |
| **ood avg** | | | 0.4645 | 0.4630 (-0.0014) | 0.4605 (-0.0040) |

Trained minus random-frozen table (table_Nov05_13-50-10 vs random_table_Nov05_14-21-11): domain: -0.0004, ood: +0.0025

| run | task | fire rate | mean gate (fired) | inject ratio (fired) |
|---|---|---|---|---|
| table_Nov05_13-50-10 | mmlu_clinical_knowledge | 1.000 | 0.908 | 0.0174 |
| table_Nov05_13-50-10 | mmlu_medical_genetics | 1.000 | 0.895 | 0.0520 |
| table_Nov05_13-50-10 | mmlu_professional_medicine | 1.000 | 0.890 | 0.0533 |
| table_Nov05_13-50-10 | mmlu_anatomy | 1.000 | 0.892 | 0.0522 |
| table_Nov05_13-50-10 | mmlu_college_medicine | 1.000 | 0.892 | 0.0574 |
| table_Nov05_13-50-10 | mmlu_college_biology | 1.000 | 0.892 | 0.0529 |
| table_Nov05_13-50-10 | medqa_4options | 1.000 | 0.886 | 0.0537 |
| table_Nov05_13-50-10 | mmlu_non_biomed | 1.000 | 0.916 | 0.0537 |
| table_Nov05_13-50-10 | truthfulqa_mc1 | 1.000 | 0.923 | 0.0550 |
| table_Nov05_13-50-10 | truthfulqa_mc2 | 1.000 | 0.923 | 0.0549 |
| table_Nov05_13-50-10 | arc_challenge | 1.000 | 0.925 | 0.0501 |
| table_Nov05_13-50-10 | hellaswag | 1.000 | 0.910 | 0.0519 |
| table_Nov05_13-50-10 | winogrande | 1.000 | 0.930 | 0.0467 |
| random_table_Nov05_14-21-11 | mmlu_clinical_knowledge | 1.000 | 0.767 | 0.0165 |
| random_table_Nov05_14-21-11 | mmlu_medical_genetics | 1.000 | 0.845 | 0.0495 |
| random_table_Nov05_14-21-11 | mmlu_professional_medicine | 1.000 | 0.828 | 0.0503 |
| random_table_Nov05_14-21-11 | mmlu_anatomy | 1.000 | 0.840 | 0.0495 |
| random_table_Nov05_14-21-11 | mmlu_college_medicine | 1.000 | 0.815 | 0.0545 |
| random_table_Nov05_14-21-11 | mmlu_college_biology | 1.000 | 0.831 | 0.0502 |
| random_table_Nov05_14-21-11 | medqa_4options | 1.000 | 0.823 | 0.0507 |
| random_table_Nov05_14-21-11 | mmlu_non_biomed | 1.000 | 0.837 | 0.0504 |
| random_table_Nov05_14-21-11 | truthfulqa_mc1 | 1.000 | 0.846 | 0.0518 |
| random_table_Nov05_14-21-11 | truthfulqa_mc2 | 1.000 | 0.846 | 0.0518 |
| random_table_Nov05_14-21-11 | arc_challenge | 1.000 | 0.859 | 0.0472 |
| random_table_Nov05_14-21-11 | hellaswag | 1.000 | 0.824 | 0.0485 |
| random_table_Nov05_14-21-11 | winogrande | 1.000 | 0.867 | 0.0440 |

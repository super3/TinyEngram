| suite | task | metric | chat | chat_table_Nov05_16-59-02 | chat_random_table_Nov05_17-29-49 |
|---|---|---|---|---|---|
| domain | mmlu_clinical_knowledge (±0.029) | acc | 0.3358 | 0.4377 (+0.1019) | 0.4491 (+0.1132) |
| domain | mmlu_medical_genetics (±0.049) | acc | 0.3700 | 0.4100 (+0.0400) | 0.4300 (+0.0600) |
| domain | mmlu_professional_medicine (±0.028) | acc | 0.3199 | 0.4375 (+0.1176) | 0.4191 (+0.0993) |
| domain | mmlu_anatomy (±0.042) | acc | 0.3704 | 0.3852 (+0.0148) | 0.4222 (+0.0519) |
| domain | mmlu_college_medicine (±0.035) | acc | 0.2948 | 0.3815 (+0.0867) | 0.3757 (+0.0809) |
| domain | mmlu_college_biology (±0.042) | acc | 0.4653 | 0.4722 (+0.0069) | 0.5139 (+0.0486) |
| domain | pubmedqa | None | - | - | - |
| domain | medmcqa | None | - | - | - |
| domain | medqa_4options (±0.013) | acc | 0.3236 | 0.3323 (+0.0086) | 0.3299 (+0.0063) |
| ood | mmlu_non_biomed | acc | 0.4221 | 0.4582 (+0.0361) | 0.4731 (+0.0511) |
| ood | truthfulqa_mc1 (±0.016) | acc | 0.2729 | 0.2889 (+0.0159) | 0.2913 (+0.0184) |
| ood | truthfulqa_mc2 (±0.015) | acc | 0.4284 | 0.4436 (+0.0151) | 0.4417 (+0.0133) |
| ood | arc_challenge (±0.014) | acc_norm | 0.3370 | 0.3618 (+0.0247) | 0.3660 (+0.0290) |
| ood | hellaswag (±0.005) | acc_norm | 0.4730 | 0.4816 (+0.0086) | 0.4820 (+0.0090) |
| ood | winogrande (±0.014) | acc | 0.5627 | 0.5627 (+0.0000) | 0.5604 (-0.0024) |
| **domain avg** | | | 0.3543 | 0.4081 (+0.0538) | 0.4200 (+0.0657) |
| **ood avg** | | | 0.4160 | 0.4328 (+0.0167) | 0.4358 (+0.0197) |

Trained minus random-frozen table (chat_table_Nov05_16-59-02 vs chat_random_table_Nov05_17-29-49): domain: -0.0119, ood: -0.0030

| run | task | fire rate | mean gate (fired) | inject ratio (fired) |
|---|---|---|---|---|
| chat_table_Nov05_16-59-02 | mmlu_clinical_knowledge | 1.000 | 0.942 | 0.0363 |
| chat_table_Nov05_16-59-02 | mmlu_medical_genetics | 1.000 | 0.914 | 0.1085 |
| chat_table_Nov05_16-59-02 | mmlu_professional_medicine | 1.000 | 0.902 | 0.1097 |
| chat_table_Nov05_16-59-02 | mmlu_anatomy | 1.000 | 0.911 | 0.1084 |
| chat_table_Nov05_16-59-02 | mmlu_college_medicine | 1.000 | 0.918 | 0.1148 |
| chat_table_Nov05_16-59-02 | mmlu_college_biology | 1.000 | 0.911 | 0.1097 |
| chat_table_Nov05_16-59-02 | medqa_4options | 1.000 | 0.894 | 0.1100 |
| chat_table_Nov05_16-59-02 | mmlu_non_biomed | 1.000 | 0.930 | 0.1117 |
| chat_table_Nov05_16-59-02 | truthfulqa_mc1 | 1.000 | 0.938 | 0.1112 |
| chat_table_Nov05_16-59-02 | truthfulqa_mc2 | 1.000 | 0.938 | 0.1112 |
| chat_table_Nov05_16-59-02 | arc_challenge | 1.000 | 0.938 | 0.1038 |
| chat_table_Nov05_16-59-02 | hellaswag | 1.000 | 0.923 | 0.1048 |
| chat_table_Nov05_16-59-02 | winogrande | 1.000 | 0.945 | 0.0969 |
| chat_random_table_Nov05_17-29-49 | mmlu_clinical_knowledge | 1.000 | 0.848 | 0.0275 |
| chat_random_table_Nov05_17-29-49 | mmlu_medical_genetics | 1.000 | 0.937 | 0.0827 |
| chat_random_table_Nov05_17-29-49 | mmlu_professional_medicine | 1.000 | 0.936 | 0.0850 |
| chat_random_table_Nov05_17-29-49 | mmlu_anatomy | 1.000 | 0.934 | 0.0828 |
| chat_random_table_Nov05_17-29-49 | mmlu_college_medicine | 1.000 | 0.933 | 0.0888 |
| chat_random_table_Nov05_17-29-49 | mmlu_college_biology | 1.000 | 0.934 | 0.0844 |
| chat_random_table_Nov05_17-29-49 | medqa_4options | 1.000 | 0.936 | 0.0856 |
| chat_random_table_Nov05_17-29-49 | mmlu_non_biomed | 1.000 | 0.932 | 0.0852 |
| chat_random_table_Nov05_17-29-49 | truthfulqa_mc1 | 1.000 | 0.928 | 0.0848 |
| chat_random_table_Nov05_17-29-49 | truthfulqa_mc2 | 1.000 | 0.928 | 0.0848 |
| chat_random_table_Nov05_17-29-49 | arc_challenge | 1.000 | 0.931 | 0.0774 |
| chat_random_table_Nov05_17-29-49 | hellaswag | 1.000 | 0.928 | 0.0803 |
| chat_random_table_Nov05_17-29-49 | winogrande | 1.000 | 0.931 | 0.0701 |

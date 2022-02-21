Note that on both the train and dev splits, macro results are lower than test split results reported in the paper.

# Train

|Action| prec  | rec   | f1    |Actor | prec  | rec   | f1    |Cert  | prec  | rec   | f1    |
|------|-------|-------|-------|------|-------|-------|-------|------|-------|-------|-------|
|MICRO | 0.395 | 0.395 | 0.395 |MICRO | 0.910 | 0.910 | 0.910 |MICRO | 0.840 | 0.840 | 0.840 |
|MACRO | 0.056 | 0.143 | 0.081 |MACRO | 0.303 | 0.333 | 0.318 |MACRO | 0.210 | 0.250 | 0.228 |


|Neg   | prec  | rec   | f1    |Temp  | prec  | rec   | f1    |
|------|-------|-------|-------|------|-------|-------|-------|
|MICRO | 0.976 | 0.976 | 0.976 |MICRO | 0.515 | 0.515 | 0.515 |
|MACRO | 0.488 | 0.500 | 0.494 |MACRO | 0.129 | 0.250 | 0.170 |



# Dev

|Action| prec  | rec   | f1    |Actor | prec  | rec   | f1    |Cert  | prec  | rec   | f1    |
|------|-------|-------|-------|------|-------|-------|-------|------|-------|-------|-------|
|MICRO | 0.439 | 0.439 | 0.439 |MICRO | 0.878 | 0.878 | 0.878 |MICRO | 0.792 | 0.792 | 0.792 |
|MACRO | 0.073 | 0.167 | 0.102 |MACRO | 0.293 | 0.333 | 0.312 |MACRO | 0.264 | 0.333 | 0.295 |

|Neg   | prec  | rec   | f1    |Temp  | prec  | rec   | f1    |
|------|-------|-------|-------|------|-------|-------|-------|
|MICRO | 0.982 | 0.982 | 0.982 |MICRO | 0.593 | 0.593 | 0.593 |
|MACRO | 0.491 | 0.500 | 0.495 |MACRO | 0.148 | 0.250 | 0.186 |


# Test (Reported in Table 3 of CMED.pdf)
**N.B.** These exclude OtherChange for Action dimension and Unknown for all dimensions.

|Action| prec  | rec   | f1    |Actor | prec  | rec   | f1    |Cert  | prec  | rec   | f1    |
|------|-------|-------|-------|------|-------|-------|-------|------|-------|-------|-------|
|MICRO | 0.41  | 0.41  | 0.41  |MICRO | 0.92  | 0.92  | 0.92  |MICRO | 0.83  | 0.83  | 0.83  |
|MACRO | 0.08  | 0.20  | 0.12  |MACRO | 0.46  | 0.50  | 0.48  |MACRO | 0.28  | 0.33  | 0.30  |


|Neg   | prec  | rec   | f1    |Temp  | prec  | rec   | f1    |
|------|-------|-------|-------|------|-------|-------|-------|
|MICRO | NR    | NR    | NR    |MICRO | 0.54  | 0.54  | 0.54  |
|MACRO | NR    | NR    | NR    |MACRO | 0.18  | 0.33  | 0.23  |



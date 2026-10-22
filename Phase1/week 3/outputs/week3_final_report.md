# Week 3 EDA + Data Strategy Finalisation


## 1. Class Imbalance


The following table reports imbalance at volume, slice and pixel levels.


|   class |   volume_count |   slice_count |      pixel_count |   pixel_fraction |
|--------:|---------------:|--------------:|-----------------:|-----------------:|
|       1 |             23 |           181 | 301856           |        0.0560379 |
|       2 |            127 |           888 |      2.36379e+06 |        0.438825  |
|       3 |            104 |           713 | 709872           |        0.131784  |
|       4 |            109 |           976 | 787323           |        0.146162  |
|       5 |             70 |           765 |      1.2238e+06  |        0.227192  |


## 2. Low-Volume Classes


The following classes contain fewer than 50 training volumes and should be considered high-risk for unstable segmentation performance.


|   class |   volume_count |   slice_count |   pixel_count |   pixel_fraction | high_risk_low_dice   |
|--------:|---------------:|--------------:|--------------:|-----------------:|:---------------------|
|       1 |             23 |           181 |        301856 |        0.0560379 | True                 |


## 3. Dataset Split


| split           |   volume_count |
|:----------------|---------------:|
| train           |            134 |
| validation      |             28 |
| active_learning |             30 |


## 4. Zero Overlap Verification


```text
ZERO OVERLAP CHECK
==================

Train ∩ Validation: 0
Train ∩ AL: 0
Validation ∩ AL: 0

ZERO OVERLAP: True

```


## 5. Data Strategy



The analysis distinguishes three levels of class imbalance:

1. Volume-level imbalance
2. Slice-level imbalance
3. Pixel-level imbalance

These measurements are treated separately because they affect
different modelling decisions.

Volume-level imbalance affects whether the model has sufficient
independent patient/volume examples.

Slice-level imbalance affects the number of training examples
containing each class.

Pixel-level imbalance reflects the spatial dominance of large
background or hemorrhage regions.

Training, validation and active-learning pools are split at
volume level to prevent slices from the same volume appearing
in different partitions.


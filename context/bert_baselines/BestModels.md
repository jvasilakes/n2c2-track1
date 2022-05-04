# Action

## Micro F1
Tuned to micro F1
```
bluebert/pl_markers/mdl/n2c2action_i2b2event/version_1/eval_dev.txt << Submission 1
0.837 | 0.837 | 0.837 |

avg_micro_f1/bluebert/mdl/n2c2action_i2b2event/version_1/eval_dev.txt
0.833 | 0.833 | 0.833

clinical_bert/mdl/n2c2action_i2b2event/version_1/eval_dev.txt
0.833 | 0.833 | 0.833

CV
bluebert/pl_markers/mdl/n2c2action_i2b2event/version_2/avg_micro_f1/version_1  << Submission 3
0.8868 | 0.8868 | 0.8868

avg_micro_f1/bluebert/mdl/n2c2action_i2b2event/version_1
0.8848 | 0.8848 | 0.8848

bluebert/pl_markers/mdl/n2c2action_i2b2event/version_1/avg_micro_f1/version_1
0.8838 | 0.8838 | 0.8838

avg_micro_f1/bluebert/action_entity_marker/version_1
0.8784 | 0.8784 | 0.8784

bluebert/action_entity_marker/version_1
0.8736 | 0.8736 | 0.8736

avg_micro_f1/clinical_bert/mtl/action_negation/version_1
0.8608 | 0.8608 | 0.8608
```

## Macro F1
```
bluebert/pl_markers/mdl/n2c2action_i2b2event/version_1/eval_dev.txt  << Submission 2
0.849 | 0.815 | 0.829 |

bluebert/mdl/n2c2action_i2b2event/version_1/eval_dev.txt
0.834 | 0.830 | 0.824 |

clinical_bert/mdl/n2c2action_i2b2event/version_1/eval_dev.txt
0.821 | 0.819 | 0.818 |

clinical_bert/mdl/n2c2action_i2b2event/version_3/eval_dev.txt
0.840 | 0.786 | 0.809 |


CV
bluebert/pl_markers/mdl/n2c2action_i2b2event/version_2 << Submission 3
0.8066 | 0.8024 | 0.7956

bluebert/pl_markers/mdl/n2c2action_i2b2event/version_1
0.8318 | 0.7798 | 0.7896

clinical_bert/pl_markers/action_entity_marked/version_1
0.7962 | 0.796 | 0.7842

clinical_bert/action_entity_marked.orig/version_4
0.8056 | 0.767 | 0.776

clinical_bert/mdl/n2c2action_i2b2event/version_1
0.789 | 0.7806 | 0.7758

clinical_bert/mdl/n2c2action_negation_i2b2event/version_1
0.7914 | 0.7766 | 0.7726

bluebert/action_entity_marker/version_1
0.8014 | 0.774 | 0.7716
```


# Actor

## Micro F1
```
bert/actor_entity_marked/version_1/eval_dev.txt
0.923 | 0.923 | 0.923
Retuned to micro f1
avg_micro_f1/bert/actor_entity_marked/version_1/eval_dev.txt  << Submission 1
0.923 | 0.923 | 0.923

CV
avg_micro_f1/bert/actor_entity_marked/version_1 << Submission 3
0.9322 | 0.9322 | 0.9322

bert/actor_entity_marked/version_1
0.9184 | 0.9184 | 0.9184
```

## Macro F1
```
bert/actor_entity_marked/version_1/eval_dev.txt << Submission 2
0.762 | 0.656 | 0.700 |

CV
bert/actor_entity_marked/version_1 << Submission 3
0.7538 | 0.7322 | 0.7206
```


# Certainty

## Micro F1
```
clinical_bert/mdl/n2c2cert_i2b2cert/version_2/eval_dev.txt  << Submission 1
0.914 | 0.914 | 0.914

avg_micro_f1/clinical_bert/mdl/n2c2cert_i2b2cert/version_1/eval_dev.txt
0.914 | 0.914 | 0.914

bert/certainty_entity_marked/version_3/eval_dev.txt
0.910 | 0.910 | 0.910 |

bert/certainty_entity_marked/version_2/eval_dev.txt
0.905 | 0.905 | 0.905

bert/mdl/n2c2cert_i2b2cert/version_1/eval_dev.txt
0.905 | 0.905 | 0.905

CV
avg_micro_f1/clinical_bert/mdl/n2c2cert_i2b2cert/version_1 << Submission 3
0.926 | 0.926 | 0.926

clinical_bert/mdl/n2c2cert_i2b2cert/version_2
0.9184 | 0.9184 | 0.9184

bert/certainty_entity_marked/version_3
0.9082 | 0.9082 | 0.9082
```

## Macro F1
```
clinical_bert/mdl/n2c2cert_i2b2cert/version_2/eval_dev.txt  << Submission 2
0.884 | 0.789 | 0.830 |

bert/certainty_entity_marked/version_3/eval_dev.txt
0.854 | 0.779 | 0.808 |

bert/certainty_entity_marked/version_2/eval_dev.txt
0.835 | 0.769 | 0.795

bert/mdl/n2c2cert_i2b2cert/version_1/eval_dev.txt
0.901 | 0.695 | 0.768


CV
clinical_bert/mdl/n2c2cert_i2b2cert/version_2 << Submission 3
0.698 | 0.6934 | 0.678

bert/certainty_entity_marked/version_3/
0.6852 | 0.6718 | 0.6654
```


# Negation

## Micro F1
```
clinical_bert/mtl/action_negation/version_3/eval_dev.txt << Submission 1
0.986 | 0.986 | 0.986 |
Retuned to micro f1
avg_micro_f1/clinical_bert/mtl/action_negation/version_1/eval_dev.txt
0.982 | 0.982 | 0.982

clinical_bert/mtl/action_negation/version_1/eval_dev.txt
0.986 | 0.986 | 0.986 |

clinical_bert/mdl/n2c2action_negation_i2b2event/version_1/eval_dev.txt
0.986 | 0.986 | 0.986 |

clinical_bert/mdl/n2c2action_negation_i2b2certainty/version_1/eval_dev.txt
0.986 | 0.986 | 0.986 |

CV
clinical_bert/mtl/action_negation/version_3  << Submission 3
0.9782 | 0.9782 | 0.9782

clinical_bert/mdl/n2c2action_negation_i2b2event/version_1
0.9778 | 0.9778 | 0.9778

clinical_bert/mdl/n2c2action_negation_i2b2certainty/version_1
0.9774 | 0.9774 | 0.9774

bert/negation/version_1
0.977 | 0.977 | 0.977
```

## Macro F1
```
clinical_bert/mtl/action_negation/version_3/eval_dev.txt
0.993 | 0.625 | 0.697 |

clinical_bert/mtl/action_negation/version_1/eval_dev.txt
0.993 | 0.625 | 0.697 |

clinical_bert/mdl/n2c2action_negation_i2b2event/version_1/eval_dev.txt
0.993 | 0.625 | 0.697 |

clinical_bert/mdl/n2c2action_negation_i2b2certainty/version_1/eval_dev.txt << Submission 2
0.993 | 0.625 | 0.697 |

CV
clinical_bert/pl_markers/mdl/n2c2action_negation_i2b2certainty/version_1  << Submission 3
0.8484 | 0.715 | 0.7302

clinical_bert/mdl/n2c2action_negation_i2b2certainty/version_1
0.8076 | 0.6844 | 0.7172

clinical_bert/mdl/n2c2action_negation_i2b2event/version_1
0.7396 | 0.6494 | 0.6694

clinical_bert/mtl/action_negation/version_3
0.6564 | 0.566 | 0.589

bert/negation/version_1
0.5884 | 0.5166 | 0.5226
```


# Temporality

## Micro F1
```
clinical_bert/mdl/n2c2_i2b2_temporality/version_3/eval_dev.txt  << Submission 1
0.864 | 0.864 | 0.864 |

clinical_bert/pl_markers/n2c2_i2b2_temporality/version_2/eval_dev.txt
0.855 | 0.855 | 0.855 |

clinical_bert/pl_markers/n2c2_i2b2_temporality/version_1/eval_dev.txt
0.855 | 0.855 | 0.855 |

bluebert/temporality_entity_marked/version_1/eval_dev.txt
0.851 | 0.851 | 0.851 |

CV
avg_micro_f1/clinical_bert/temporality_entity_marked/version_1  << Submission 3
0.8722 | 0.8722 | 0.8722

avg_micro_f1/clinical_bert/mdl/n2c2_i2b2_temporality/version_1
0.8662 | 0.8662 | 0.8662 

clinical_bert/temporality_entity_marked/version_3
0.8626 | 0.8626 | 0.8626
```

## Macro F1
```
clinical_bert/pl_markers/n2c2_i2b2_temporality/version_4/eval_dev.txt << Submission 2
0.729 | 0.751 | 0.738 |

clinical_bert/pl_markers/n2c2_i2b2_temporality/version_1/eval_dev.txt
0.863 | 0.687 | 0.730

clinical_bert/mdl/n2c2_i2b2_temporality/version_3/eval_dev.txt
0.865 | 0.682 | 0.727

CV
clinical_bert/pl_markers/n2c2_i2b2_temporality/version_2  << Submission 3
0.8066 | 0.814 | 0.7968

clinical_bert/temporality_entity_marked/version_3
0.7986 | 0.8136 | 0.7846

clinical_bert/mdl/n2c2_i2b2_temporality/version_3
0.8236 | 0.7724 | 0.7602
```

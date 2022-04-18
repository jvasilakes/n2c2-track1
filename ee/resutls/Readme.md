## Lastest version: Packed levitated markers with 10 verbs

```
************************ Event Classification ************************
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
         Disposition  0.9727  0.8856  0.9271    0.9727  0.8856  0.9271
       Nodisposition  0.9569  0.9793  0.9680    0.9569  0.9793  0.9680
        Undetermined  0.8471  0.8276  0.8372    0.8471  0.8276  0.8372
                      ------------------------------------------------
     Overall (micro)  0.9505  0.9477  0.9491    0.9505  0.9477  0.9491
     Overall (macro)  0.9255  0.8975  0.9108    0.9255  0.8975  0.9108


*********************** Context Classification ***********************
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
              Action  0.8796  0.7602  0.8155    0.8796  0.7602  0.8155
```

Compared with mean-pool and also different embs (2) for start and end, but 1 emb with max performs best.<br>

Jake configuration performance:
```
Events : Macro_Pr = 0.9075 | Macro_Re = 0.8653 | Macro_F1  = 0.8851 | Micro_F1 = 0.9380 <<<
Actions: Macro_Pr = 0.5429 | Macro_Re = 0.6297 | Macro_F1  = 0.4502 | Micro_F1 = 0.2855
Actions: Macro_Pr = 0.7692 | Macro_Re = 0.6297 | Macro_F1  = 0.6180 | Micro_F1 = 0.7736
```

Across all splits:
```
                                Default: 
Events : Macro_Pr = 0.9255 | Macro_Re = 0.8975 | Macro_F1  = 0.9108 | Micro_F1 = 0.9491 <<<
Actions: Macro_Pr = 0.7780 | Macro_Re = 0.7665 | Macro_F1  = 0.7675 | Micro_F1 = 0.8322
                                Split0:
Events : Macro_Pr = 0.8157 | Macro_Re = 0.7143 | Macro_F1  = 0.7371 | Micro_F1 = 0.9249 <<<
Actions: Macro_Pr = 0.8172 | Macro_Re = 0.7503 | Macro_F1  = 0.7083 | Micro_F1 = 0.7805
                                Split1:
Events : Macro_Pr = 0.8788 | Macro_Re = 0.8459 | Macro_F1  = 0.8617 | Micro_F1 = 0.9389 <<<
Actions: Macro_Pr = 0.8745 | Macro_Re = 0.6728 | Macro_F1  = 0.6757 | Micro_F1 = 0.7821
                                Split2:
Events : Macro_Pr = 0.9096 | Macro_Re = 0.8885 | Macro_F1  = 0.8987 | Micro_F1 = 0.9504 <<<
Actions: Macro_Pr = 0.8613 | Macro_Re = 0.9238 | Macro_F1  = 0.8882 | Micro_F1 = 0.8746
                                Split3:
Events : Macro_Pr = 0.8744 | Macro_Re = 0.8858 | Macro_F1  = 0.8799 | Micro_F1 = 0.9312 <<<
Actions: Macro_Pr = 0.8420 | Macro_Re = 0.4580 | Macro_F1  = 0.4338 | Micro_F1 = 0.7929
                                Split4:
Events : Macro_Pr = 0.8416 | Macro_Re = 0.8357 | Macro_F1  = 0.8385 | Micro_F1 = 0.9010 <<<
Actions: Macro_Pr = 0.7004 | Macro_Re = 0.7651 | Macro_F1  = 0.7267 | Micro_F1 = 0.7946
                                Average: 
Events : Micro_F1 = 0.9326 | Actions: Micro_F1 = 0.8095
```
Jake average: ```Events : Micro_F1 = 0.9329 | Actions: Micro_F1 = 0.79905```

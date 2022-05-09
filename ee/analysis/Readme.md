# Performance on the test sets
## Release 2 (events + context)
### Submission 1
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
         Disposition  0.8333  0.8517  0.8424    0.8333  0.8517  0.8424
       Nodisposition  0.9554  0.9532  0.9543    0.9554  0.9532  0.9543
        Undetermined  0.5915  0.6885  0.6364    0.5915  0.6885  0.6364
                      ------------------------------------------------
     Overall (micro)  0.9044  0.9167  0.9105    0.9044  0.9167  0.9105
     Overall (macro)  0.7934  0.8312  0.8110    0.7934  0.8312  0.8110
### Submission 2

                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
         Disposition  0.8615  0.8044  0.8320    0.8615  0.8044  0.8320
       Nodisposition  0.9604  0.9698  0.9651    0.9604  0.9698  0.9651
        Undetermined  0.6418  0.7049  0.6719    0.6418  0.7049  0.6719
                      ------------------------------------------------
     Overall (micro)  0.9197  0.9218  0.9208    0.9197  0.9218  0.9208
     Overall (macro)  0.8212  0.8264  0.8230    0.8212  0.8264  0.8230
### Submission 3
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
         Disposition  0.8526  0.8391  0.8458    0.8526  0.8391  0.8458
       Nodisposition  0.9538  0.9502  0.9520    0.9538  0.9502  0.9520
        Undetermined  0.6031  0.6475  0.6245    0.6031  0.6475  0.6245
                      ------------------------------------------------
     Overall (micro)  0.9099  0.9093  0.9096    0.9099  0.9093  0.9096
     Overall (macro)  0.8031  0.8123  0.8074    0.8031  0.8123  0.8074

     














# Performance on the dev sets/splits

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
Blue average: ```Events : Micro_F1 = 0.9327 | Actions: Micro_F1 = 0.8096```
Blue results:
```
default:
Events : Macro_Pr = 0.9217 | Macro_Re = 0.9001 | Macro_F1  = 0.9104 | Micro_F1 = 0.9421 <<<
Actions: Macro_Pr = 0.5546 | Macro_Re = 0.7612 | Macro_F1  = 0.6003 | Micro_F1 = 0.4513
Actions: Macro_Pr = 0.7951 | Macro_Re = 0.7612 | Macro_F1  = 0.7716 | Micro_F1 = 0.8037
Split0:
Events : Macro_Pr = 0.8520 | Macro_Re = 0.8038 | Macro_F1  = 0.8232 | Micro_F1 = 0.9337 <<<
Actions: Macro_Pr = 0.5707 | Macro_Re = 0.8634 | Macro_F1  = 0.6491 | Micro_F1 = 0.4398
Actions: Macro_Pr = 0.8005 | Macro_Re = 0.8634 | Macro_F1  = 0.8206 | Micro_F1 = 0.8253
split1:
Events : Macro_Pr = 0.8313 | Macro_Re = 0.8094 | Macro_F1  = 0.8193 | Micro_F1 = 0.9188 <<<
Actions: Macro_Pr = 0.6843 | Macro_Re = 0.6504 | Macro_F1  = 0.5023 | Micro_F1 = 0.2428
Actions: Macro_Pr = 0.8352 | Macro_Re = 0.6504 | Macro_F1  = 0.6315 | Micro_F1 = 0.7410
split2:
Events : Macro_Pr = 0.9265 | Macro_Re = 0.8819 | Macro_F1  = 0.9024 | Micro_F1 = 0.9549 <<<
Actions: Macro_Pr = 0.5779 | Macro_Re = 0.9380 | Macro_F1  = 0.6341 | Micro_F1 = 0.3384
Actions: Macro_Pr = 0.8608 | Macro_Re = 0.9380 | Macro_F1  = 0.8855 | Micro_F1 = 0.8837
split3:
Events : Macro_Pr = 0.9007 | Macro_Re = 0.9126 | Macro_F1  = 0.9065 | Micro_F1 = 0.9447 <<<
Actions: Macro_Pr = 0.5303 | Macro_Re = 0.6498 | Macro_F1  = 0.4214 | Micro_F1 = 0.2688
Actions: Macro_Pr = 0.7928 | Macro_Re = 0.6498 | Macro_F1  = 0.6446 | Micro_F1 = 0.8109
split4:
Events : Macro_Pr = 0.8403 | Macro_Re = 0.7931 | Macro_F1  = 0.8099 | Micro_F1 = 0.9019 <<<
Actions: Macro_Pr = 0.5220 | Macro_Re = 0.7962 | Macro_F1  = 0.5861 | Micro_F1 = 0.4368
Actions: Macro_Pr = 0.7240 | Macro_Re = 0.7962 | Macro_F1  = 0.7535 | Micro_F1 = 0.7929
Average
Events: Micro_F1: 0.93268 | Actions: Micro_F1: 0.80958
```

# Saved models folder
In this folder the best trained bert model are saved, according to early stopping.

## Our results on test set using the default train/dev split and golden labels:
### blue bert - types
```
************************ Event Classification ************************
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
         Disposition  0.8558  0.8612  0.8585    0.8558  0.8612  0.8585
       Nodisposition  0.9617  0.9661  0.9639    0.9617  0.9661  0.9639
        Undetermined  0.7069  0.6721  0.6891    0.7069  0.6721  0.6891
                      ------------------------------------------------
     Overall (micro)  0.9259  0.9269  0.9264    0.9259  0.9269  0.9264
     Overall (macro)  0.8415  0.8331  0.8371    0.8415  0.8331  0.8371
```  
### blue bert - LCM
```
************************ Event Classification ************************
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
         Disposition  0.8190  0.8707  0.8440    0.8190  0.8707  0.8440
       Nodisposition  0.9759  0.9465  0.9609    0.9759  0.9465  0.9609
        Undetermined  0.6565  0.7049  0.6798    0.6565  0.7049  0.6798
                      ------------------------------------------------
     Overall (micro)  0.9219  0.9161  0.9190    0.9219  0.9161  0.9190
     Overall (macro)  0.8171  0.8407  0.8283    0.8171  0.8407  0.8283
```     
### blue bert - baseline_mtl
```
************************ Event Classification ************************
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
         Disposition  0.8365  0.8233  0.8299    0.8365  0.8233  0.8299
       Nodisposition  0.9623  0.9623  0.9623    0.9623  0.9623  0.9623
        Undetermined  0.6957  0.6557  0.6751    0.6957  0.6557  0.6751
                      ------------------------------------------------
     Overall (micro)  0.9224  0.9161  0.9193    0.9224  0.9161  0.9193
     Overall (macro)  0.8315  0.8138  0.8224    0.8315  0.8138  0.8224
```     
## Our results on test set using the default train/dev split and NER predicted medication spans:
### blue bert - types
```
************************ Event Classification ************************
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
         Disposition  0.8012  0.8265  0.8137    0.8165  0.8423  0.8292
       Nodisposition  0.9302  0.9351  0.9327    0.9385  0.9434  0.9410
        Undetermined  0.6508  0.6721  0.6613    0.6587  0.6803  0.6694
                      ------------------------------------------------
     Overall (micro)  0.8869  0.8975  0.8921    0.8964  0.9071  0.9017
     Overall (macro)  0.7941  0.8113  0.8025    0.8046  0.8220  0.8132
```     

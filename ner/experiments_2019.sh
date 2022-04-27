for i in {0..4}
do  
    mkdir $i
    # mkdir $i/baseline+i2b2_2019
    # mkdir $i/baseline_roberta+i2b2_2019
    # mkdir $i/clinical_bert+i2b2_2019
    mv ../experiments/$i/*i2b2_2019 $i    
done

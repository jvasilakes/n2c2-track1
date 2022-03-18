Downloaded from https://portal.dbmi.hms.harvard.edu/projects/download_dataset/?file_uuid=7f03c3f7-10a1-491b-aff7-cc4d02881f70 on 10/3/2022.

To convert data to brat format, please use the following command line:

python scripts/process_n2c2_2010.py \
        --textdir auxiliary_data/n2c2_2010_concept_assertion_relation/beth/txt \
        --anndir auxiliary_data/n2c2_2010_concept_assertion_relation/beth/ast \
        --bratdir auxiliary_data/n2c2_2010_concept_assertion_relation/beth/ast_brat

To run sentence segmentation

python scripts/run_biomedicus_sentences.py \
        --biomedicus_data_dir ~/.biomedicus/data/sentences \
        --indir auxiliary_data/n2c2_2010_concept_assertion_relation/beth/txt \
        --outdir auxiliary_data/n2c2_2010_concept_assertion_relation/beth/segmented/
# Likewise for partners/


To generate the train/dev splits

ls -1 beth/ast_brat/*.ann | shuf -n 50 > combined/beth_train.txt
find beth/ast_brat/ -name '*.ann' -print | grep -Fxvf combined/beth_train.txt > combined/beth_dev.txt
ls -1 partners/ast_brat/*.ann | shuf -n 70 > combined/partners_train.txt
find partners/ast_brat/ -name '*.ann' -print | grep -Fxvf combined/partners_train.txt > combined/partners_dev.txt

cp $(cat combined/beth_train.txt | xargs) combined/ast_brat/train
cp $(cat combined/beth_dev.txt | xargs) combined/ast_brat/dev
# Likewise for partners

for line in $(cat combined/beth_train.txt); do
bn=$(basename $line)
sent_file=${bn/.ann/.txt.json}
cp beth/segmented/${sent_file} combined/segmented/train
done
# Likewise for beth_dev and partners_{train,dev}

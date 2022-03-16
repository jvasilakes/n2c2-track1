Downloaded from https://portal.dbmi.hms.harvard.edu/projects/download_dataset/?file_uuid=7f03c3f7-10a1-491b-aff7-cc4d02881f70 on 10/3/2022.

To convert data to brat format, please use the following command line:

python scripts/process_n2c2_2010.py \
        --textdir auxiliary_data/n2c2_2010_concept_assertion_relation/beth/txt \
        --anndir auxiliary_data/n2c2_2010_concept_assertion_relation/beth/ast \
        --bratdir auxiliary_data/n2c2_2010_concept_assertion_relation/beth/ast_brat

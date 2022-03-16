The original data was downloaded from: https://portal.dbmi.hms.harvard.edu/projects/download_dataset/?file_uuid=f348a196-6d1e-422f-ba7a-ef68fdea922c

For the format of the data, please check: https://portal.dbmi.hms.harvard.edu/projects/download_dataset/?file_uuid=875a190f-d9ae-44ac-8c93-ea376525ed0e

To convert the data to brat format, please use the following command line

```bash
python scripts/process_i2b2_2009.py --textdir auxiliary_data/i2b2_2009/training.sets.released \
        --anndir auxiliary_data/i2b2_2009/training.ground.truth \
        --bratdir auxiliary_data/i2b2_2009/training.sets/training.brat
```

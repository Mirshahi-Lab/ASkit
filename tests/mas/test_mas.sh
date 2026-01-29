askit mas \
-i /home/irdinsmore1/projects/ASkit/tests/mas/data/phewas_example_5000_samples_5_covariates.csv \
-o /home/irdinsmore1/projects/ASkit/tests/mas/results/phewas_example_5000_samples_5_covariates_mas.parquet \
-p rsEXAMPLE \
-d i:7- \
-c age,sex,race_1,bmi,smoking_status \
-m firth \
-n 16 \
-t 1 \
--dry-run

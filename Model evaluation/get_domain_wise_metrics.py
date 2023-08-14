base_path = '/mnt/hdd/jawaril/full_pipeline/Evaluation_Metric'

image_to_domain_map = r"data/eval_218/image_domain_mapping.csv"
v1_metric_path = r"_reconstruction_v1_metric.csv"
v1_output_summary = r"_reconstruction_v1_metric_domain.csv"
v2_metric_path = r"_reconstruction_v2_metric.csv"
v2_output_summary = r"_reconstruction_v2_metric_domain.csv"

import pandas as pd
import os

img_to_domain_df = pd.read_csv(os.path.join(base_path, image_to_domain_map))
v1_df = pd.read_csv(os.path.join(base_path, v1_metric_path))
v2_df = pd.read_csv(os.path.join(base_path, v2_metric_path))
print(img_to_domain_df, v1_df, v2_df)

v1_df['average_NED']=v1_df['average_NED'].astype(float)
v1_df['average_accuracy']=v1_df['average_accuracy'].astype(float)

# import pdb; pdb.set_trace()

# v1_df.join(img_to_domain_df, on='file_stem')

v1_domain = v1_df.merge(img_to_domain_df, on="file_stem")
v1_domain_summary = v1_domain.groupby('Domain_Name')[['average_NED', 'average_accuracy', 'average_precision','average_f1']].mean()
v1_domain_summary.round(2).to_csv(os.path.join(base_path, v1_output_summary), index=True)

v2_domain = v2_df.merge(img_to_domain_df, on="file_stem")

v2_domain_summary = v2_domain.groupby('Domain_Name')[['doc_cer', 'doc_wer']].mean()
v2_domain_summary.round(2).to_csv(os.path.join(base_path, v2_output_summary), index=True)







OPERATORS = """
<TRANSFORMATION OPERATORS>
1. Read each source data file. For each source data file, according to <SOURCE DATA DESCRIPTION>,if Sepator: SEP,Encoding: ENC,then it can be read with "sep=SEP,encoding=ENC". If the data is in pandas.Dataframe,rename each column when reading the data file.
2. Select rows/columns/dimensions in the source data files according to Semantic information and Content of target data files.
4. If the Semantic information of a particular row/column/dimension in the target file does not exist in any row/column/dimension of the source file, the corresponding row/column/dimension in the transformed data file should be set to empty.
5. Reorder columns to match target data format
6. Rename columns to match target data format
7. Save transformed data in '{output_file}'.Do not use to_excel function of panda.
"""

PROMPT_HEAD = """
You are a Data Transforming Agent. Your job is to transform data from a given format, to the same format as the target data. Following are the detailed instructions for the same:
1. Read the <SOURCE DATA DESCRIPTION> and <SOURCE DATA SAMPLE> to get a full understand of source data files.Read the <TARGET DATA DESCRIPTION> and <TARGET DATA SAMPLE> to get a full understand of target data files.
2. Path of source data files:'{path}',please join the path with name of source data file when reading the source data file;
   Names of each source data file: According to <SOURCE DATA DESCRIPTION>, if File num: n,then source data file names: source1,source2,....sourcen.
3. According to Content and Semantic information of each source data file in <SOURCE DATA DESCRIPTION> and Content and Semantic information of each target data file in <TARGET DATA DESCRIPTION>,you should convert the source data into the same format as the target data.Files with the same Content in the transformed and target files should have the same Semantic information in each dimension (or rows and columns).
4. Choose proper transformation strategies combination in <TRANSFORMATION OPERATORS>,and make a data transformation plan considering instructions above and transformation strategies you have chosen.
5. Read the data transformation plan carefully and implement each step using Python.
6. Add comments for each step.
7. End your response with a plan and a python code.
{operator}
-------------------------------------------------
Here are some examples for your reference.
Incontext Example 1
<SOURCE DATA DESCRIPTION>
{source_example}
<TARGET DATA DESCRIPTION>
{target_example}
{operator}
Response:
<TRANSFORMATION PLAN>
1.Read the source data files in path '{path}' separately using the specified separators and encodings.
2.Select relevant columns/dimensions from the source data files
3.Generate a new DataFrame for the distance.csv file, use the detector IDs from source2.npz as the first column,set the second and third columns to empty (NaN) values for latitude and longitude.
4.Reorder columns to match the target data format.
5.Rename columns to match the target data format.
6.Save transformed data in the target format.
<TRANSFORMATION CODE>
import pandas as pd
import numpy as np
# Read source data files
source1_df = pd.read_csv(os.path.join('{path}','source1.csv'), 
                         sep=',', encoding='utf-8', 
                         names=['Starting detector ID', 'End detector ID', 'Distance'])
source2_data = np.load(os.path.join('{path}',source2.npz'))['data']
# Select relevant columns/dimensions from the source data files.The column of transformed feature file and target feature file should be detector ID,the index should be timestamp.
source2_df = pd.DataFrame(source2_data[:, :, 2],  # Select dimension 3 (Traffic speed)
                          index=source2_data[:, 0, 0],  # Use timestamp of source data as index
                          columns=range(source2_data.shape[1]))  # Use detector ID of source data as columns
# Rename columns to match target data format.
source2_df.columns = [f'Detector ID {{i}}' for i in range(source2_data.shape[1])]
# Convert timestamp to the same format as the target file
timestamps = np.arange(0, 5 * 60 * source2_data.shape[0], 5 * 60, dtype='datetime64[s]')
timestamps = pd.to_datetime(timestamps, unit='s')
source2_df.index = timestamps
# Because latitude and longitude in the target file does not exist in any row/column/dimension of the source file, they should be set to empty.
distance_df = pd.DataFrame({{
    'Detector ID': source2_df.columns,
    'Latitude': [np.nan] * len(source2_df.columns),
    'Longitude': [np.nan] * len(source2_df.columns)
}})
# Save transformed data in target format
distance_df.to_csv(os.path.join('{output_file}','distance.csv')), 
                   index=False)
source1_df.to_csv(os.path.join('{output_file}','target_distance.csv'), 
                  index=False)
source2_df.to_hdf(os.path.join('{output_file}',target_feature.h5'), 
                  key='data', mode='w')
------------------------------------------------
Now do the following transformation task: 
"""


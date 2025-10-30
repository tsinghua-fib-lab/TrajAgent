OPERATORS = """
<TRANSFORMATION OPERATORS>
1. Read each source data file. For each source data file, according to <SOURCE DATA DESCRIPTION>,if Sepator: SEP,Encoding: ENC,then it can be read with "sep=SEP,encoding=ENC". You can set header=0 to prevent reading any row as headers, then add column names via names.
2. Select rows/columns/dimensions in the source data files according to Semantic information and Content of target data files.
3. Convert the format of time column in the source data to match the format of the time column in the target data.
4. If target data has columns not exist in source data,fill these columns with the same data type as those in target data.
5. Reorder columns to match target data format
6. Rename columns to match target data format
7. Save transformed data in '{output_path}/target_{dataset}_{city}.csv'.Do not use to_excel function of panda.
"""

PROMPT_HEAD = """
You are a Data Transforming Agent. Your job is to transform data from a given format, to the same format as the target data. Following are the detailed instructions for the same:
1. Read the <SOURCE DATA DESCRIPTION> and <SOURCE DATA SAMPLE> to get a full understand of source data files.Read the <TARGET DATA DESCRIPTION> and <TARGET DATA SAMPLE> to get a full understand of target data files.
2. Path of source data files:'{path}/{dataset}',please join the path with name of source data file when reading the source data file;
   Names of each source data file: According to <SOURCE DATA DESCRIPTION>, if there are <POIs> and <Checkins>, then source data file names are POIs_{city} and Checkins_{city}.If there is <Checkins>, then source data file name is Checkins_{city}.
3. Transformed data file: '{output_path}/target_{dataset}_{city}.csv'.Save transformed data in '{output_path}/target_{dataset}_{city}.csv'
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
<SOURCE DATA SAMPLE>
{source_sample}
<TARGET DATA SAMPLE>
{target_sample}
Response:
<TRANSFORMATION PLAN>
1. Read the source data Checkins_Unknown.txt.According to SOURCE DATA DESCRIPTION,the file is in txt format,seperated by '\t',The encoding of the text file is utf-8,so it can be read with "'sep='\t',encoding='utf-8'".
2. Select first five columns in source data.
3. Transform ISO 8601 time format(the second column) in source data to UTC time format (the third column) in target data.
4. 'Time Offset' and 'Venue Category Name' columns only exist in target data according to SOURCE DATA DESCRIPTION and TARGET DATA DESCRIPTION.The data type of values in 'Time Offset' column is int,so add a new 'Time Offset' column to transformed data, and fill it with int values, like "1". \
   The data type of values in 'Venue Category Name' column is string,so add a new 'Venue Category Name' column to transformed data, and fill it with int values, like "Unknown".
5. Reorder columns to match target data format.
6. Rename columns to match target data format.
7. Write the transformed data to a new CSV file:'{output_path}/target_{dataset}_{city}.csv'.
<TRANSFORMATION CODE>
# Read the source data
source_data = pd.read_csv(os.path.join('{path}/gowalla','Checkins_Unknown.txt'), sep='\t', header=0, 
                          names=['User ID', 'check-in time', 'Latitude', 'Longitude', 'Location ID'],encoding='utf-8')

# Select the first five columns
source_data = source_data[['User ID', 'check-in time', 'Latitude', 'Longitude', 'Location ID']]

# Transform ISO 8601 time format to UTC time format
source_data['check-in time'] = pd.to_datetime(source_data['check-in time'])
source_data['UTC Time'] = source_data['check-in time'].dt.strftime('%a %b %d %H:%M:%S %z %Y')

source_data['check-in time'] = pd.to_datetime(source_data['check-in time'])
source_data['UTC Time'] = source_data

# Reorder columns to match target data format
source_data = source_data[['User ID', 'Location ID', 'UTC Time', 'Longitude', 'Latitude']]

# Fill the columns in transformed data with certain data type of values
source_data['Time Offset'] = 1
source_data['Venue Category Name'] = 'Unknown'

# Rename columns to match target data format
source_data = source_data.rename(columns={{'Location ID': 'Venue ID'\}})

# Write the transformed data to a new CSV file
source_data.to_csv('{output_path}/target_gowalla_Unknown.csv', index=False)

Now do the following transformation task: 
"""

# Incontext Example 2
# <SOURCE DATA DESCRIPTION>
# {source_example2}
# <TARGET DATA DESCRIPTION>
# {target_example}
# <SOURCE DATA SAMPLE>
# {source_sample2}
# <TARGET DATA SAMPLE>
# {target_sample}
# Response:
# <TRANSFORMATION PLAN>
# 1. Read Source Files Checkins_Unknown.txt and POIs_Unknown.txt.
# 2. Join the check-in and POI data on Venue ID to combine geographic and category details with check-in records.
# 3. Add Missing Columns: Target requires City, which is absent in the source. Fill with "Unknown" (string placeholder).
# 4. Time Format Handling: Source UTC Time already matches the target format (Tue Apr 03 18:00:06 +0000 2012), so no conversion needed.
# 5. Reorder Columns: Target order: City, User ID, Time Offset, Venue ID, UTC Time, Longitude, Latitude, Venue Category Name.
# 6. Rename Columns.
# 7. Save as CSV.
# <TRANSFORMATION CODE>
# import pandas as pd
# import os

# # Step 1: Read source files with proper separators and encoding
# # ------------------------------------------------------------
# # Read POI data (POIs_Unknown.txt)
# poi_path = os.path.join('{path}/foursquare', 'POIs_Unknown.txt')
# poi_df = pd.read_csv(
#     poi_path,
#     sep='\t',
#     encoding='ISO-8859-1',
#     header=None,
#     names=['Venue ID', 'Latitude', 'Longitude', 'Venue Category Name', 'Country Code']
# )

# # Read check-in data (source2.txt)
# checkin_path = os.path.join('{path}/foursquare', 'Checkins_Unknown.txt')
# checkin_df = pd.read_csv(
#     checkin_path,
#     sep='\t',
#     encoding='ISO-8859-1',
#     header=None,
#     names=['User ID', 'Venue ID', 'UTC Time', 'Time Offset']
# )

# # Step 2: Merge POI and check-in data on Venue ID
# merged_df = pd.merge(
#     checkin_df,
#     poi_df,
#     on='Venue ID',
#     how='left'  # Retain all check-ins even if POI data is missing
# )

# # Step 3: Add missing 'City' column (target-specific)
# merged_df['City'] = 'Unknown'  # Fill with placeholder string

# # Step 4: Reorder columns to match target format
# target_columns = [
#     'City', 
#     'User ID', 
#     'Time Offset', 
#     'Venue ID', 
#     'UTC Time', 
#     'Longitude', 
#     'Latitude', 
#     'Venue Category Name'
# ]

# transformed_df = merged_df[target_columns]

# # Step 5: Rename columns to match target naming(['City','User ID', 'Time Offset', 'Venue ID', 'UTC Time', 'Longitude', 'Latitude', 'Venue Category Name'])

# # Step 6: Save transformed data to CSV
# output_file = '{output_path}/target_foursquare_Unknown.csv'
# transformed_df.to_csv(
#     output_file,
#     sep=',',
#     index=False,
#     encoding='utf-8'
# )
# ------------------------------------------------
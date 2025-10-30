import pandas as pd
import os

# Step 1: Read the source data
source_data = pd.read_csv(os.path.join('/data3/duyuwei/AugMove/dataset/input_format/source', 'source1.txt'), 
                           sep='\t', header=None, 
                           names=['User ID', 'check-in time', 'Latitude', 'Longitude', 'Location ID'], 
                           encoding='utf-8')

# Step 2: Select the first five columns
source_data = source_data[['User ID', 'check-in time', 'Latitude', 'Longitude', 'Location ID']]

# Step 3: Transform ISO 8601 time format to UTC time format
source_data['check-in time'] = pd.to_datetime(source_data['check-in time'])
source_data['UTC Time'] = source_data['check-in time'].dt.strftime('%a %b %d %H:%M:%S +0000 %Y')

# Step 4: Fill in the additional columns
source_data['City'] = 'Beijing'  # Placeholder value
source_data['Time Offset'] = 0    # Placeholder integer value
source_data['Venue Category Name'] = 'Unknown'  # Placeholder string

# Step 5: Reorder columns to match target data format
source_data = source_data[['City', 'User ID', 'Time Offset', 'Location ID', 'UTC Time', 'Longitude', 'Latitude', 'Venue Category Name']]

# Step 6: Rename columns to match target data format
source_data = source_data.rename(columns={'Location ID': 'Venue ID'})

# Step 7: Write the transformed data to a new CSV file
source_data.to_csv('/data3/duyuwei/AugMove/dataset/input_format/target/target.csv', index=False)
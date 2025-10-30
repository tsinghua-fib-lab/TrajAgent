import pandas as pd
import os

# Read the source data
source_data = pd.read_csv(os.path.join('/data3/duyuwei/data/dataset/input_format/agentmove', 'Checkins_NewYork.csv'), 
                          sep=',', header=0, 
                          names=['City', 'User', 'Time', 'Venue ID', 'UTC Time', 'Longitude', 'Latitude', 'Venue Category Name'], 
                          encoding='utf-8')

# Select all columns
source_data = source_data.copy()

# Convert the format of the utc_time column to match the format of the UTC Time column in the target data
source_data['UTC Time'] = pd.to_datetime(source_data['UTC Time'])

# Add Time Offset column and fill with int values
source_data['Time Offset'] = 1

# Add Latitude column and fill with float values (assuming float is the correct type)
source_data['Latitude'] = 0.0

# Reorder columns to match the target data format
source_data = source_data[['City', 'User', 'Time Offset', 'Venue ID', 'UTC Time', 'Longitude', 'Latitude', 'Venue Category Name']]

# Rename columns to match the target data format
source_data = source_data.rename(columns={'User': 'User ID', 'City': 'City'})

# Save the transformed data to a new CSV file
source_data.to_csv('/data3/duyuwei/data/dataset/input_format/target/target_agentmove_NewYork.csv', index=False)
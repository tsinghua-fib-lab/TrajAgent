DATA_DESCRIPTION = {
    "foursquare":
        """
This dataset includes long-term (about 18 months from April 2012 to September 2013) global-scale check-in data collected from Foursquare webset.
It contains 33,278,683 checkins by 266,909 users on 3,680,126 venues (in 415 cities in 77 countries). 
<FILE1>
Content: Check-in location
Format: txt.
Seperator:'\t'.
Encoding:ISO-8859-1.
Semantic information:
column1-Venue ID (Foursquare) 
column2-Latitude
column3-Longitude
column4-Venue category name (Foursquare)
<FILE2>
Content: User ID
Format: txt.
Seperator:'\t'.
Encoding:ISO-8859-1.
Semantic information:
column1-User ID (anonymized)
column2-Venue ID (Foursquare)
column3-UTC time
column4-Timezone offset in minutes (The offset in minutes between when this check-in occurred and the same time in UTC, i.e., UTC time + offset is the local time)
""",
    "gowalla":
        """
This dataset includes long-term (about 18 months from Feb. 2009 to Oct. 2010) global-scale check-in data collected from Gowalla webset.
It contains 6,442,890 check-ins of users on 196,591 venues.
<FILE1>
Content: Check-in location, User ID
Format: txt.
Seperator:'\t'.
Encoding:utf-8.
Semantic information:
column1-User ID 
column2-Check-in time(ISO 8601)
column3-Latitude
column4-Lontitude
column5-Location ID(Gowalla)
""",
    "brightkite":
        """
This dataset contains long-term (from Apr. 2008 to Oct. 2010) global-scale check-in data collected from BrightKite webset. It contains 4,491,143 check-ins by 58228 users.
<FILE1>
Content: Check-in location, User ID
Format: txt.
Seperator:'\t'.
Encoding:utf-8.
Semantic information:
column1-User ID	
column2-check-in time(ISO 8601)		
column3-Latitude	
column4-Longitude	
column5-Location ID(Gowalla)
""",
    "standard":
        """
This dataset contains check-ins in NYC and Tokyo collected for about 10 month (from 12 April 2012 to 16 February 2013). It contains 227,428 check-ins in New York city.
<FILE1>
Content: Check-in location, User ID
Format: csv.
Seperator:','.
Encoding:utf-8.
Semantic information:
column1-User ID
column2-Venue ID(Foursquare)
column3-UTC Time
column4-Longitude
column5-Latitude
column6-Venue category name (Foursquare)
""",
    "pemsd8":
        """ 
This dataset contains the traffic data in San Bernardino Jul. 1, 2016 - Aug. 31, 2016. It contains 170 detectors. Each detector has \
17856 records with a time interval of 5 minutes.There is no location data.
File num:2
<FILE1>
Content: Distance
Format: csv
Seperator:','
Encoding:utf-8
Semantic information:
column1-Starting detector ID.Integer type(PEMSD8)
column2-End detector ID.Integer type(PEMSD8)
column3-Distance.Float type.Corresponds to distance between starting detector in the first column and end detector in the second column.
<FILE2>
Content: Feature
Format: npz
Key: data
Shape: (num of timestamps, num of detectors, num of features of each record)
Semantic information:
dimension1-All timestamps of each detector.Each dector has one record per timestamp, recorded every 5 minutes starting at Jul. 1, 2016,with a total of 17856 records.
dimension2-All detector IDs.Integer type(PEMSD8). Corresponds to detector ID,with a total of 170 detectors.
dimension3-[Traffic flow ,Traffic occupancy, Traffic speed] of each record.Each element in the list is of float type.Select the element that is related with target data files. 
""",
"metr-la":
    """
This dataset contains the traffic data in Los Angeles, USA from Mar. 1, 2012 to Jun. 27, 2012. It contains 207 detectors. Each detector has \
34272 records with a time interval of 5 minutes.
File num:3
<FILE1>
Content: Distance
Format: csv.
Seperator:','.
Encoding:utf-8.
Semantic information:
column1-Starting Detector ID.Integer type(METR LA)
column2-End Detector ID.Integer type(METR LA)
column3-Distance.Float type, representing the distance between the starting detector and the end detector
<FILE2>
Content: Feature
Format: h5(*.h5 files store the data in panads.DataFrame using the HDF5 file format.)
Shape: (num of timestamps, num of detectors)
Semantic information:
Index (row labels)-All timestamps of each detector. Formatted as YYYY-MM-DD HH:MM:SS, recorded every 5 minutes starting at Mar. 1, 2012,with a total of 34272 records.
Columns (column labels)-All detector IDs. Integer type(METR LA),with a total of 207 detectors.
Values-Traffic speed of each record. Float type.Corresponds to speed value for each detector at each timestamp.
<FILE3>
Content: Location
Format: csv
Seperator:','
Encoding:utf-8
Semantic information:
column1-Index.Integers starting from zero.
column2-Detector IDs.Integer type(METR LA), unique items in column labels of FILE2
column3-Latitude.Float type.
column4-Longitude.Float type.   
""",
"pems-bay":
"""
This dataset contains the traffic data in San Francisco Bay Area from Jan. 1, 2017 to Jun. 30, 2017. It contains 325 detectors. Each detector has \
52116 records with a time interval of 5 minutes.
File num:3
<FILE1>
Content: Distance
Format: csv.
Seperator:','.
Encoding:utf-8.
Semantic information:
column1-Starting Detector ID.Integer type(PEMS BAY)
column2-End Detector ID.Integer type(PEMS BAY)
column3-Distance.Float type, representing the distance between the starting detector and the end detector
<FILE2>
Content: Feature
Format: h5(*.h5 files store the data in panads.DataFrame using the HDF5 file format.)
Shape: (num of timestamps, num of detectors)
Semantic information:
Index (row labels)-All timestamps of each detector. Formatted as YYYY-MM-DD HH:MM:SS, recorded every 5 minutes starting at Jan. 1, 2017,with a total of 52116 records.
Columns (column labels)-All detector IDs. Integer type(PEMS BAY),with a total of 325 detectors.
Values-Traffic speed of each record. Float type.Corresponds to speed value for each detector at each timestamp.
<FILE3>
Content: Location
Format: csv
Seperator:','
Encoding:utf-8
Semantic information:
column1-Detector IDs.Integer type(PEMS BAY), unique items in column labels of FILE2
column2-Latitude.Float type.
column3-Longitude.Float type.
""",
"porto":"""
This dataset contains a full year (01/07/2013 to 30/06/2014) of taxi trajectories for 442 taxis operating in Porto, Portugal (i.e. one CSV file named "porto.csv"). Each trip, represented as a row in CSV file, includes the following details:
TRIP_ID: Unique identifier for each trip
CALL_TYPE: Service request type:
‘A’ for dispatch center
‘B’ for taxi stand
‘C’ for street hail
ORIGIN_CALL: ID for the phone number requesting a trip (only for CALL_TYPE 'A')
ORIGIN_STAND: ID for the taxi stand starting point (only for CALL_TYPE 'B')
TAXI_ID: Unique ID for the taxi driver
TIMESTAMP: Trip start time as a Unix timestamp
DAYTYPE: Day type:‘B’ for holidays/special days,‘C’ for the day before a holiday,‘A’ for normal days
MISSING_DATA: Boolean indicating if GPS data is complete
POLYLINE: List of GPS coordinates (WGS84 format) recorded every 15 seconds, marking the trip route from start to destination.
""",
"chengdu":"""
This dataset contains GPS trajectory records of Chengdu from 01/11/2016 to 30/11/2016. Each record includes taxiID, timestamp, longitude and latitude, collected and released by Didi Chuxing.
"""
}

DATA_SAMPLE = {
    "foursquare":"<FILE1>\n3fd66200f964a52000e71ee3,40.733596,-74.003139,Jazz Club\n<FILE2>\n50756,4f5e3a72e4b053fd6a4313f6,Tue Apr 03 18:00:06 +0000 2012,240",
    "gowalla":"<FILE1>\n0,2010-10-19T23:55:27Z,30.2359091167,-97.7951395833,22847",
    "standard":"<FILE1>\n221021,4a85b1b3f964a520eefe1fe3,Tue Apr 03 18:00:08 +0000 2012,-73.99228,40.748939,Coffee Shop",
    "brightkite":"<FILE1>\n0,2010-10-17T01:48:53Z,39.747652,-104.99251,88c46bf20db295831bd2d1718ad7e6f5",
    "pemsd8":
        """
        <FILE1>\nfrom,to,cost\n9,153,310.6\n<FILE2>\n
        array([[[1.330e+02, 6.030e-02, 6.580e+01],
        [2.100e+02, 5.890e-02, 6.960e+01],
        ...,
        [9.400e+01, 2.260e-02, 6.800e+01],
        [6.000e+00, 3.100e-03, 6.500e+01]],
        ...
       [[1.140e+02, 5.320e-02, 6.690e+01],
        [1.850e+02, 5.500e-02, 6.850e+01],
        ...,
        [8.400e+01, 1.890e-02, 6.870e+01],
        [4.000e+00, 1.800e-03, 6.500e+01]]]
        )  
        """,
    "pems-bay":
        """
    <FILE1>\n400001,400001,0.0\n400017,400017,0.0\n400030,400030,0.0\n<FILE2>\n
    sensor_id            400001  400017  400030  400040  400045  ...  413845  413877  413878  414284  414694
    2017-01-01 00:00:00    71.4    67.8    70.5    67.4    68.8  ...    68.9    70.4    68.8    71.1    68.0\n
    <FILE3>\n400001,37.364085,-121.901149\n400017,37.253303,-121.945440
"""
}

# FORMAT_DICT = {
#     "FOURSQUARE":FOURSQUARE,
#     "GOWALLA": GOWALLA,
#     "LIBCITY": LIBCITY,
#     "BRIGHTKITE": BRIGHTKITE,
#     "FOURSQUARE": FOURSQUARE,
#     "PEMSD8": PEMSD8,
#     "PEMS_BAY": PEMS_BAY
# }
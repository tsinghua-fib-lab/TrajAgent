DATA_DESCRIPTION ={
"foursquare": """
This dataset includes long-term (about 18 months from April 2012 to September 2013) global-scale check-in data collected from Foursquare webset.
It contains 33,278,683 checkins by 266,909 users on 3,680,126 venues (in 415 cities in 77 countries). 
Format: txt.
Seperator:'\t'.
Encoding:ISO-8859-1.
<Checkins>
The POI file contains all venue data with 5 columns, which are:
1. Venue ID (Foursquare) 
2. Latitude
3. Longitude
4. Venue category name (Foursquare)
5. Country code (ISO 3166-1 alpha-2 two-letter country codes)
<POIs>
The checkin file contains all check-ins with 4 columns, which are:
1. User ID (anonymized)
2. Venue ID (Foursquare)
3. UTC time
4. Timezone offset in minutes (The offset in minutes between when this check-sin occurred and the same time in UTC, i.e., UTC time + offset is the local time)
""",
"gowalla": """
This dataset includes long-term (about 18 months from Feb. 2009 to Oct. 2010) global-scale check-in data collected from Gowalla webset.
It contains 6,442,890 check-ins of users on 196,591 venues.
Format: txt.
Seperator:'\t'.
Encoding:utf-8.
<Checkins>
The trajectory file contains all check-ins and user IDs with 5 columns, which are:
1. User ID 
2. Check-in time(ISO 8601)
3. Latitude
4. Lontitude
5. Location ID(Gowalla)
""",
"agentmove":"""
This dataset contains check-ins in different cities collected for about 10 month (from 12 April 2012 to 16 February 2013).
Format: csv.
Seperator:','.
Encoding:utf-8.

<Checkins>
The trajectory file contains all check-ins and user IDs with 8 columns, which are:
1. city
2. user
3. time
4. venue_id
5. utc_time
6. lon
7. lat
8. venue_cat_name
""",

"standard": """
This dataset contains check-ins in certain city collected for about 10 month (from 12 April 2012 to 16 February 2013).
Format: csv.
Seperator:','.
Encoding:utf-8.

<Checkins>
The trajectory file contains all check-ins and user IDs with 8 columns, which are:
1. City
2. User ID
3. Time Offset
4. Venue ID
5. UTC Time
6. Longitude
7. Latitude
8. Venue Category Name
""",
"brightkite": """
This dataset contains long-term (from Apr. 2008 to Oct. 2010) global-scale check-in data collected from BrightKite webset. It contains 4,491,143 check-ins by 58228 users.
Format: txt.
Seperator:'\t'.
Encoding:utf-8.
<Checkins>
The trajectory file contains all check-ins and user IDs with 5 columns, which are:
1. User ID	
2. check-in time(ISO 8601)		
3. Latitude	
4. Longitude	
5. Location ID(Gowalla)
"""
}
DATA_SAMPLE = {
    "foursquare":"<Checkins>\n3fd66200f964a52000e71ee3,40.733596,-74.003139,Jazz Club\n<POIs>\n50756,4f5e3a72e4b053fd6a4313f6,Tue Apr 03 18:00:06 +0000 2012,240",
    "gowalla":"<Checkins>\n0,2010-10-19T23:55:27Z,30.2359091167,-97.7951395833,22847",
    "standard":"<Checkins>\nBeijing,83132,480,4d67ecb5052ea1cd2b5aa049,Tue Apr 03 18:28:06 +0000 2012,116.437258,39.918656,Lounge",
    "brightkite":"<Checkins>\n0,2010-10-17T01:48:53Z,39.747652,-104.99251,88c46bf20db295831bd2d1718ad7e6f5",
    "agentmove":"<Checkins>\nCape Town,30830,120,4f620a43e4b0ed0157e8b19f,Tue Apr 03 18:05:20 +0000 2012,18.408495,-33.899721,Residential Building (Apartment / Condo)"
}

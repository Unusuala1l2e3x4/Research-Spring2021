nClimDiv Station Inventories
March 2014

Divisional inventory files are located in the main CIRS 
nClimDiv directory:  ftp://ftp.ncdc.noaa/gov/pub/data/cirs/climdiv
and have the following naming conventions:

climdiv-prcp-inv-recent-vX.Y.Z-YYYMMDD
climdiv-tmax-inv-recent-vX.Y.Z-YYYMMDD
climdiv-tmin-inv-recent-vX.Y.Z-YYYMMDD
climdiv-prcp-inv-vX.Y.Z-YYYYMMDD
climdiv-tmax-inv-vX.Y.Z-YYYYMMDD
climdiv-tmin-inv-vX.Y.Z-YYYYMMDD

where:
X,Y,Z represents a version number (i.e. 1.0.0 for the initial release)
YYYMMDD represents the year/month/day data were processed 

Inventory files which include *recent* in the name are comprised of 
stations used to compute ClimDiv values for the most recent two calendar 
years.  Data values and station information prior to this timeframe will 
not change unless the dataset undergoes a version change.

For a station to be included in the divisional mean temperature data set,
it must be included in both tmax and tmin inventories.

Sample Data:

GHCN-ID      LAT      LON       STDV YR/MO  NETWORK

USC00010402  31.1819  -87.4389   107 201301 COOP
USC00010425  32.5992  -85.4653   105 201301 COOP
USC00010505  33.4528  -87.3572   103 201301 COOP
USC00010583  30.8839  -87.7853   108 201301 COOP
USC00010655  34.6908  -86.8825   101 201301 COOP

Format:

 c1-11:  GHCN ID
c14-20:  Lat
c22-30:  Lon
c33-36:  State/Division  (1-48 and 01-10, 9999 if not in CONUS)
c38-43:  4-digit year/2-digit month station was used in divisional calculation
c45-50:  Network (BUOY, CANADA, CIMIS, COOP, MEXICO, RAWS, SNOTEL, WBAN)

Example:

The top line indicates that Atmore, Alabama (GHCN-Daily station USC00010402)
was included in the minimum temperture calculation for January 2013. The
Atmore station is located at latitude 31.1819N and longitude 87.4389W, within
Alabama Climate Division 7 (107, state number 1, division number 7). 

Note that climate division values are influenced by information from stations in
neighboring areas, and are not a straight average of the station within them.

For information on the methodology used to convert station data to gridded 
divisional, statewide, regional and CONUS values, please see the following 
journal article: 

http://journals.ametsoc.org/doi/abs/10.1175/JAMC-D-13-0248.1

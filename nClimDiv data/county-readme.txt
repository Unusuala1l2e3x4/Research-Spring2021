This documentation describes the record format for the county files on 
/pub/data/cirs/climdiv that have the filenames:

climdiv-pcpncy-vx.y.z-YYYYMMDD
climdiv-tmaxcy-vx.y.z-YYYYMMDD
climdiv-tmincy-vx.y.z-YYYYMMDD
climdiv-tmpccy-vx.y.z-YYYYMMDD


                                    nClimDiv
                                     COUNTY
                            TEMPERATURE-PRECIPITATION

                                 OCTOBER 2018

The major parameters in this file are sequential climatic county  monthly 
maximum, minimum and average temperature (deg. F. to 10ths) and precipitation (inches to 100ths). 
Period of record is 1895 through latest month available, updated 
monthly.

Values from the most recent two calendar years will be updated on a monthly 
basis.  Period of record updates will occur when the underlying data set 
undergoes a version change.

METHODOLOGY:

County values in nClimDiv were derived from area-weighted averages of 
grid-point estimates interpolated from station data.  A nominal grid resolution
of 5 km was used to ensure that all divisions had sufficient spatial sampling 
(only four small divisions had less than 100 points) and because the impact of 
elevation on precipitation is minimal below 5 km.  Station data were gridded 
via climatologically aided interpolation to minimize biases from topographic 
and network variability.

The Global Historical Climatology Network (GHCN)  Daily dataset is the source 
of station data for nClimDiv.  GHCN-Daily contains several major observing 
networks in North America, five of which are used here.  The primary network 
is the National Weather Service (NWS) Cooperative Observing (COOP) program, 
which consists of stations operated by volunteers as well as by agencies such 
as the Federal Aviation Administration.  To improve coverage in western states 
and along international borders, nClimDiv also includes the National 
Interagency Fire Center (NIFC) Remote Automatic Weather Station (RAWS) network, 
the USDA Snow Telemetry (SNOTEL) network, the Environment Canada (EC) 
network (south of 52°N), and part of Mexicos Servicio Meteorologico Nacional 
(SMN) network (north of 24°N).  Note that nClimDiv does not incorporate 
precipitation data from RAWS because that networks tipping-bucket gauges are 
unheated, leading to suspect cold-weather data.

All GHCN-Daily stations are routinely processed through a suite of logical, 
serial, and spatial quality assurance reviews to identify erroneous 
observations.  For nClimDiv, all such data were set to missing before 
computing monthly values, which in turn were subjected to additional serial 
and spatial checks to eliminate residual outliers. Stations having at least 
10 years of valid monthly data since 1950 were used in nClimDiv.

For temperature, bias adjustments were computed to account for historical 
changes in observation time, station location, temperature instrumentation, 
and siting conditions.  Changes in observation time are only problematic for 
the COOP network whereas changes in station location and instrumentation occur 
in almost all surface networks.   As in the U.S. Historical Climatology Network
version 2.5, the method of Karl et al. (1986) was applied to remove the 
observation time bias from the COOP network, and the pairwise method of Menne 
and Williams (2009) was used to address changes in station location and 
instrumentation in all networks.  Because the pairwise method also largely 
accounts for local, unrepresentative trends that arise from changes in siting 
conditions, nClimDiv contains no separate adjustment in that regard.

For additional information on how nClimDiv is constructed, please see:
http://journals.ametsoc.org/doi/abs/10.1175/JAMC-D-13-0248.1

STATE CODE TABLE: 
                             Range of values of 01-48.

                             01 Alabama                 28 New Jersey
                             02 Arizona                 29 New Mexico
                             03 Arkansas                30 New York
                             04 California              31 North Carolina
                             05 Colorado                32 North Dakota
                             06 Connecticut             33 Ohio
                             07 Delaware                34 Oklahoma
                             08 Florida                 35 Oregon
                             09 Georgia                 36 Pennsylvania
                             10 Idaho                   37 Rhode Island
                             11 Illinois                38 South Carolina
                             12 Indiana                 39 South Dakota
                             13 Iowa                    40 Tennessee
                             14 Kansas                  41 Texas
                             15 Kentucky                42 Utah
                             16 Louisiana               43 Vermont
                             17 Maine                   44 Virginia
                             18 Maryland                45 Washington
                             19 Massachusetts           46 West Virginia
                             20 Michigan                47 Wisconsin
                             21 Minnesota               48 Wyoming
                             22 Mississippi
                             23 Missouri   
                             24 Montana   
                             25 Nebraska 
                             26 Nevada  
                             27 New Hampshire


FILE FORMAT:

IMPORTANT NOTE: 

The format of the county data is slightly different than the other data files. To accomadate
the 2 digit state code and the 3 digit county FIPS code, the first field contains 11 columns. 
The other data files still contain 10 columns. 

Element          Record
Name             Position    Element Description

STATE-CODE          1-2      STATE-CODE as indicated in State Code Table as
                             described in FILE 1.  Range of values is 01-48.

DIVISION-NUMBER     3-5      COUNTY FIPS - Range of values 001-999.

ELEMENT CODE        6-7      01 = Precipitation
                             02 = Average Temperature
                             27 = Maximum Temperature
                             28 = Minimum Temperature
			     
YEAR                8-11     This is the year of record.  Range is 1895 to
                             current year processed.

(all data values are right justified):

JAN-VALUE          12-18     

                             Monthly Divisional Temperature format (f7.2)
                             Range of values -50.00 to 140.00 degrees Fahrenheit.
                             Decimals retain a position in the 7-character
                             field.  Missing values in the latest year are
                             indicated by -99.99.

                             Monthly Divisional Precipitation format (f7.2)
                             Range of values 00.00 to 99.99.  Decimal point
                             retains a position in the 7-character field.
                             Missing values in the latest year are indicated
                             by -9.99.

                             

FEB-VALUE          19-25     

MAR-VALUE          26-32    

APR-VALUE          33-39   

MAY-VALUE          40-46  

JUNE-VALUE         47-53     

JULY-VALUE         54-60     

AUG-VALUE          61-67     

SEPT-VALUE         68-74     

OCT-VALUE          75-81     

NOV-VALUE          82-88     

DEC-VALUE          89-95     

This documentation describes the record format for the divisional,
statewide, regional and national drought files on 
/pub/data/cirs/climdiv that have the filenames:

climdiv-pdsist-vx.y.z-YYYYMMDD
climdiv-phdist-vx.y.z-YYYYMMDD
climdiv-pmdist-vx.y.z-YYYYMMDD
climdiv-sp01st-vx.y.z-YYYYMMDD
climdiv-sp02st-vx.y.z-YYYYMMDD
climdiv-sp03st-vx.y.z-YYYYMMDD
climdiv-sp06st-vx.y.z-YYYYMMDD
climdiv-sp09st-vx.y.z-YYYYMMDD
climdiv-sp12st-vx.y.z-YYYYMMDD
climdiv-sp24st-vx.y.z-YYYYMMDD
climdiv-zndxst-vx.y.z-YYYYMMDD

climdiv-pdsidv-vx.y.z-YYYYMMDD
climdiv-phdidv-vx.y.z-YYYYMMDD
climdiv-pmdidv-vx.y.z-YYYYMMDD
climdiv-sp01dv-vx.y.z-YYYYMMDD
climdiv-sp02dv-vx.y.z-YYYYMMDD
climdiv-sp03dv-vx.y.z-YYYYMMDD
climdiv-sp06dv-vx.y.z-YYYYMMDD
climdiv-sp09dv-vx.y.z-YYYYMMDD
climdiv-sp12dv-vx.y.z-YYYYMMDD
climdiv-sp24dv-vx.y.z-YYYYMMDD
climdiv-zndxdv-vx.y.z-YYYYMMDD


                                    nClimDiv
                           STATEWIDE-REGIONAL-NATIONAL
                                    DROUGHT

                                   MARCH 2014

The major parameters in this file are sequential climatic divisional,
statewide, regional and national monthly Standardized Precipitation 
Index (SPI), and Palmer Drought Indices (PDSI, PHDI, PMDI, and ZNDX). 
Period of record is 1895 through latest month available, updated monthly.

Values from the most recent two calendar years will be updated on a monthly
basis.  Period of record updates will occur when the underlying data set
undergoes a version change.

METHODOLOGY:

Divisional, statewide, regional and national values in nClimDiv were derived 
from area-weighted averages of grid-point estimates interpolated from station 
data.  A nominal grid resolution of 5 km was used to ensure that all divisions 
had sufficient spatial sampling (only four small divisions had less than 100 
points) and because the impact of elevation on precipitation is minimal below 
5 km.  Station data were gridded via climatologically aided interpolation to 
minimize biases from topographic and network variability.

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
and spatial checks to eliminate residual outliers.  Overall, the quality
assurance reviews deemed less than 0.25% of the monthly data as being
erroneous.  Stations having at least 10 years of valid monthly data since
1950 were used in nClimDiv.

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

Historical drought data have been added to this file for the period 1895 to
present.  The file is updated monthly.  All drought data are calibrated using
the period 1931-1990 (cf. Karl, 1986; Journal of Climate and Applied
Meteorology, Vol. 25, No. 1, January 1986).  Drought data include:

1.  Palmer Drought Severity Index (PDSI)

 This is the monthly value (index) that is generated indicating the severity
 of a wet or dry spell.  This index is based on the principles of a balance
 between moisture supply and demand.  Man-made changes were not considered in
 this calculation.  The index generally ranges from -6 to +6, with negative
 values denoting dry spells and positive values indicating wet spells.  There
 are a few values in the magnitude of +7 or -7.  PDSI values 0 to -.5 =
 normal; -0.5 to -1.0 = incipient drought; -1.0 to -2.0 = mild drought; -2.0
 to -3.0 = moderate drought; -3.0 to -4.0 = severe drought; and greater than -
 4.0 = extreme drought.  Similar adjectives are attached to positive values of
 wet spells.  This is a meteorological drought index used to assess the
 severity of dry or wet spells of weather.

2.  Palmer Hydrological Drought Index (PHDI)

 This is the monthly value (index) generated monthly that indicates the
 severity of a wet or dry spell.  This index is based on the principles of a
 balance between moisture supply and demand.  Man-made changes such as
 increased irrigation, new reservoirs, and added industrial water use were not
 included in the computation of this index.  The index generally ranges from -
 6 to +6, with negative values denoting dry spells, and positive values
 indicating wet spells.  There are a few values in the magnitude of +7 or -7. 
 PHDI values 0 to -0.5 = normal; -0.5 to -1.0 = incipient drought; -1.0 to -
 2.0 = mild drought; -2.0 to -3.0 = moderate drought; -3.0 to -4.0 = severe
 drought; and greater than -4.0 = extreme drought.  Similar adjectives are
 attached to positive values of wet spells.  This is a hydrological drought
 index used to assess long-term moisture supply.

3.  Palmer "Z" Index (ZNDX)

 This is the generated monthly Z values, and they can be expressed as the
 "Moisture Anomaly Index."  Each monthly Z value is a measure of the departure
 from normal of the moisture climate for that month.  This index can respond
 to a month of above-normal precipitation, even during periods of drought. 
 Table 1 contains expected values of the Z index and other drought parameters. 
 See Historical Climatology Series 3-6 through 3-9 for a detailed description
 of the drought indices.

4.  Modified Palmer Drought Severity Index (PMDI)

 This is a modification of the Palmer Drought Severity Index.  The
 modification was made by the National Weather Service Climate Analysis Center
 for operational meteorological purposes.  The Palmer drought program
 calculates three intermediate parallel index values each month.  Only one
 value is selected as the PDSI drought index for the month.  This selection is
 made internally by the program on the basis of probabilities.  If the
 probability that a drought is over is 100%, then one index is used.  If the
 probability that a wet spell is over is 100%, then another index is used.  If
 the probability is between 0% and 100%, the third index is assigned to the
 PDSI.  The modification (PMDI) incorporates a weighted average of the wet and
 dry index terms, using the probability as the weighting factor.  (Thomas R.
 Heddinghause and Paul Sabol, 1991; "A Review of the Palmer Drought Severity
 Index and Where Do We Go From Here?," Proceedings of the Seventh Conference
 on Applied Climatology, pp. 242-246, American Meteorological Society, Boston,
 MA).  The PMDI and PDSI will have the same value during an established
 drought or wet spell (i.e., when the probability is 100%), but they will have
 different values during transition periods.

5.  Standardized Precipitation Index (SPxx)

This is a transformation of the probability of observing a given amount of
precipitation in xx months.  A zero index value reflects the median of the
distribution of precipitation, a -3 indicates a very extreme dry spell, and a
+3 indicates a very extreme wet spell.  The more the index value departs from
zero, the drier or wetter an event lasting xx months is when compared to the
long-term climatology of the location.  The index allows for comparison of
precipitation observations at different locations with markedly different
climates; an index value at one location expresses the same relative departure
from median conditions at one location as at another location.  It is
calculated for different time scales since it is possible to experience dry
conditions over one time scale while simultaneously experiencing wet conditions
over a different time scale.


               Table 1    Classes for Wet and Dry Periods


Approximate 
Cumulative                                                           
Frequency               Range                                     Range
    %                   PHDI                  Category              Z         

  > 96                > 4.00                Extreme wetness      > 3.50

    90-95               3.00,  3.99         Severe wetness         2.50,  3.49

    73-89               1.50,  2.99         Mild to moderate       1.00,  2.49
                                                    wetness

    28-72              -1.49,  1.49         Near normal           -1.24,  0.99

    11-27              -1.50, -2.99         Mild to moderate      -1.25, -1.99
                                                    drought

     5-10              -3.00, -3.99         Severe drought        -2.00, -2.74

  <  4                <-4.00                Extreme drought      <-2.75


STATE CODE TABLE:

     Range of values for the states, regions, and nation is 001-110.

          001 Alabama         030 New York
          002 Arizona         031 North Carolina
          003 Arkansas        032 North Dakota
          004 California      033 Ohio
          005 Colorado        034 Oklahoma
          006 Connecticut     035 Oregon
          007 Delaware        036 Pennsylvania
          008 Florida         037 Rhode Island
          009 Georgia         038 South Carolina
          010 Idaho           039 South Dakota
          011 Illinois        040 Tennessee
          012 Indiana         041 Texas
          013 Iowa            042 Utah
          014 Kansas          043 Vermont
          015 Kentucky        044 Virginia
          016 Louisiana       045 Washington
          017 Maine           046 West Virginia
          018 Maryland        047 Wisconsin
          019 Massachusetts   048 Wyoming
          020 Michigan        101 Northeast Region
          021 Minnesota       102 East North Central Region
          022 Mississippi     103 Central Region
          023 Missouri        104 Southeast Region
          024 Montana         105 West North Central Region
          025 Nebraska        106 South Region
          026 Nevada          107 Southwest Region
          027 New Hampshire   108 Northwest Region
          028 New Jersey      109 West Region
          029 New Mexico      110 National (contiguous 48 States)

The following are the range of code values for the National Weather 
Service Regions, river basins, and agricultural regions:

111 Great Plains
115 Southern Plains and Gulf Coast
120 US Rockies and Westward
121 NWS Eastern Region
122 NWS Southern Region
123 NWS Central Region
124 NWS Western Region
201 Pacific Northwest Basin
202 California River Basin
203 Great Basin
204 Lower Colorado River Basin
205 Upper Colorado River Basin
206 Rio Grande River Basin
207 Texas Gulf Coast River Basin
208 Arkansas-White-Red Basin
209 Lower Mississippi River Basin
210 Missouri River Basin
211 Souris-Red-Rainy Basin
212 Upper Mississippi River Basin
213 Great Lakes Basin
214 Tennessee River Basin
215 Ohio River Basin
216 South Atlantic-Gulf Basin
217 Mid-Atlantic Basin
218 New England Basin
220 Mississippi River Basin & Tributaties (N. of Memphis, TN)

250 Spring Wheat Belt (area weighted)
255 Primary Hard Red Winter Wheat Belt (area weighted)
256 Winter Wheat Belt (area weighted)
260 Primary Corn and Soybean Belt (area weighted)
261 Corn Belt (area weighted)
262 Soybean Belt (area weighted)
265 Cotton Belt (area weighted)

350 Spring Wheat Belt (productivity weighted)
356 Winter Wheat Belt(productivity weighted)
361 Corn Belt (productivity weighted)
362 Soybean Belt (productivity weighted)
365 Cotton Belt (productivity weighted)

450 Spring Wheat Belt (% productivity in the Palmer Z Index)
456 Winter Wheat Belt (% productivity in the Palmer Z Index)
461 Corn Belt (% productivity in the Palmer Z Index)
462 Soybean Belt (% productivity in the Palmer Z Index)
465 Cotton Belt (% productivity in the Palmer Z Index)


DIVISIONAL FILE FORMAT:

Element          Record
Name             Position    Element Description

STATE-CODE          1-2      STATE-CODE as indicated in State Code Table as
                             described in FILE 1.  Range of values is 01-91.

DIVISION-NUMBER     3-4      DIVISION NUMBER - Assigned by NCDC.  Range of
                             values 01-10.


STATE/REGIONAL/NATIONAL FILE FORMAT:

STATE-CODE          1-3     STATE-CODE as indicated in State Code Table above.
                            Range of values is 001-110 for standard states,
                            regions and national,  111-465 for special regions.

DIVISION-NUMBER       4     DIVISION NUMBER.  Value is 0 which indicates an area
                            -averaged element.


REMAINING FILE FORMAT FOR DIVISIONAL/STATE/REGIONAL/NATIONAL:

ELEMENT CODE        5-6      05 = PDSI
                             06 = PHDI
                             07 = ZNDX
                             08 = PMDI
			     71 = 1-month Standardized Precipitation Index
			     72 = 2-month Standardized Precipitation Index
                             73 = 3-month Standardized Precipitation Index
                             74 = 6-month Standardized Precipitation Index
                             75 = 9-month Standardized Precipitation Index
                             76 = 12-month Standardized Precipitation Index
                             77 = 24-month Standardized Precipitation Index

YEAR                7-10     This is the year of record.  Range is 1895 to
                             current year processed.

(all data values are right justified):

JAN-VALUE          11-17     Palmer Drought Index format (f7.2)
			     Range of values -20.00 to 20.00. Decimal point
			     retains a position in 7-character field.  
                             Missing values in the latest year are indicated
                             by -99.99.

			     Standardized Precipitation Index format (f7.2).
		             Range of values -4.00 to 4.00.  Decimal
                             point retains a position in 7-character field. 
                             Missing values in the latest year are indicated
                             by -99.99.

FEB-VALUE          18-24     

MAR-VALUE          25-31    

APR-VALUE          32-38   

MAY-VALUE          39-45  

JUNE-VALUE         46-52     

JULY-VALUE         53-59     

AUG-VALUE          60-66     

SEPT-VALUE         67-73     

OCT-VALUE          74-80     

NOV-VALUE          81-87     

DEC-VALUE          88-94     



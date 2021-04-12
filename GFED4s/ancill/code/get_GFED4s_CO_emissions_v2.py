import numpy as np
import h5py # if this creates an error please make sure you have the h5py library
import os
import pathlib

import time





# in this example we will calculate annual CO emissions for the 14 GFED 
# basisregions over 1997-2014. Please adjust the code to calculate emissions
# for your own specie, region, and time period of interest. Please
# first download the GFED4.1s files and the GFED4_Emission_Factors.txt
# to your computer and adjust the directory where you placed them below

# (All fire types combined)


ancillDir = str(pathlib.Path(__file__).parent.parent.absolute())
GFED4sDir = str(pathlib.Path(__file__).parent.parent.parent.absolute())
# print(ancillDir)


months_all       = '01','02','03','04','05','06','07','08','09','10','11','12'
sources_all      = 'SAVA','BORF','TEMF','DEFO','PEAT','AGRI'



if __name__ == "__main__":
    """
    Read in emission factors
    """
    species = [] # names of the different gas and aerosol species
    EFs     = np.zeros((41, 6)) # 41 species, 6 sources

    k = 0
    f = open(ancillDir+'/GFED4_Emission_Factors.txt')
    while 1:
        line = f.readline()
        if line == "":
            break
            
        if line[0] != '#':
            contents = line.split()
            species.append(contents[0])
            EFs[k,:] = contents[1:]
            k += 1
                    
    f.close()

    # print(species)
    # print(EFs)


    # params
    
    months = months_all     # ['01','02','03','04','05','06','07','08','09','10','11','12']
    sources = sources_all   # ['SAVA','BORF','TEMF','DEFO','PEAT','AGRI']
    start_year = 1997
    end_year   = 2016
    
    EF_CO = EFs[3,:]                # we are interested in CO for this example (4th row):
    # (0)   DM      C       CO2             CO              CH4
    # (5)   NMHC    H2      NOx             N2O             PM2.5
    # (10)  TPM     TPC     OC              BC              SO2
    # (15)  C2H6    CH3OH   C2H5OH          C3H8            C2H2
    # (20)  C2H4    C3H6    C5H8            C10H16          C7H8
    # (25)  C6H6    C8H10   Toluene_lump    Higher_Alkenes  Higher_Alkanes
    # (30)  CH2O    C2H4O   C3H6O           NH3             C2H6S
    # (35)  HCN     HCOOH   CH3COOH         MEK             CH3COCHO
    # (40)  HOCH2CHO

    # end params


    # print(EF_CO)


    # t0 = timer_start()

    """
    make table with summed DM emissions for each region, year, and source
    """
    CO_table = np.zeros((15, end_year - start_year + 1)) # region, year

    for year in range(start_year, end_year+1):
        path = GFED4sDir+'/GFED4.1s_'+str(year)+'.hdf5'
        f = h5py.File(path, 'r')
        
        
        if year == start_year: # these are time invariable    
            basis_regions = f['/ancill/basis_regions'][:]
            grid_area     = f['/ancill/grid_cell_area'][:]
        
        
        CO_emissions = np.zeros((720, 1440))
        for month in range(len(months)):
            # read in DM emissions
            path = '/emissions/'+months[month]+'/DM'
            DM_emissions = f[path][:]
            for source in range(len(sources)):
                # read in the fractional contribution of each source
                path = '/emissions/'+months[month]+'/partitioning/DM_'+sources[source]
                contribution = f[path][:]
                # calculate CO emissions as the product of DM emissions (kg DM per 
                # m2 per month), the fraction the specific source contributes to 
                # this (unitless), and the emission factor (g CO per kg DM burned)
                CO_emissions += DM_emissions * contribution * EF_CO[source]
        
        
        # fill table with total values for the globe (row 15) or basisregion (1-14)
        for region in range(15):
            if region == 14:
                mask = np.ones((720, 1440))
            else:
                mask = basis_regions == (region + 1)            
        
            CO_table[region, year-start_year] = np.sum(grid_area * mask * CO_emissions)
            
        # print(year)
            
    # convert to Tg CO     (tera grams; g * 10^12)
    CO_table = CO_table / 1E12


    # t0 = timer_restart(t0)
    # 28.388230323791504 sec (1997-2016, all sources)
    # 9.57642650604248 sec (1997-2016, SAVA)



    print(CO_table)

    # please compare this to http://www.falw.vu/~gwerf/GFED/GFED4/tables/GFED4.1s_CO.txt
    # All fire types combined
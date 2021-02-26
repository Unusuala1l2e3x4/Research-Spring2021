prefix='GFED4.0s_';
postfix='.hdf5';
path='/Users/wolfgang/Data/Fire/GFED4s/';
years=1997:2014;
nyears=length(years);
% load table of emission factors:
input=importdata([path 'ancil/GFED4_Emission_Factors.txt']);
EF=input.data;
[nspec ncat]=size(EF);
if ncat~=6; disp('Error: There should be six columns for different categories of emissions. Returning ...'); return; end
% get names of chemical species in emissions table:
species=input.textdata(size(input.textdata,1)-nspec+1:end,1);
% display various information out of HDF5 files
% h5disp([path prefix '2014' postfix]);
% h5disp([path prefix '2014' postfix],'/ancill');
% h5disp([path prefix '2014' postfix],'/lon');
% h5disp([path prefix '2014' postfix],'/lat');
% h5disp([path prefix '2014' postfix],'/emissions/01/DM');
% h5disp([path prefix '2014' postfix],'/emissions/01/partitioning/DM_BORF');
% h5disp([path prefix '2014' postfix],'/emissions/01/partitioning/DM_SAVA');
% h5disp([path prefix '2014' postfix],'/emissions/01/partitioning/DM_TEMF');
% h5disp([path prefix '2014' postfix],'/emissions/01/partitioning/DM_DEFO');
% h5disp([path prefix '2014' postfix],'/emissions/01/partitioning/DM_PEAT');
% h5disp([path prefix '2014' postfix],'/emissions/01/partitioning/DM_AGRI');
% get land mask and area per grid cell; transform to more Matlab friendly format (rows running E-W, columns S-N)
mask25=flipdim(h5read([path prefix '2014' postfix],'/ancill/basis_regions')',1);
area25=flipdim(h5read([path prefix '2014' postfix],'/ancill/grid_cell_area')',1).*(mask25>0); % flipdim is actually unnecessary here
[ny25 nx25]=size(mask25); % 0.25 x 0.25 degree grid
% transform to new grid (can be any dimensions, but the grid cell borders must extend 180W-180E, 90S-90N
ny=180; nx=360; % target 1x1 degree grid
area=rebin(rebin(area25,ny,1),nx,2); % rebin is additive, so total area is conserved
mask=rebin(rebin(mask25>0,ny,1),nx,2); % number of land grid cells contributing in original resolution (can be fractional)
% set up arrays to store targetted arrays:
emit=zeros(ny,nx,12,'single'); % total biomass burning emissions climatology[kg DM / m^2 / month]
emit_wf=zeros(ny,nx,12,'single'); % wildfire emissions climatology[kg DM / m^2 / month]
emit_defo=zeros(ny,nx,12,'single'); % deforestation emissions climatology [kg DM / m^2 / month]
aemit=zeros(ny,nx,nspec,'single'); % mean annual total biomass burning emissions per scpecies [g / m^2 / month]
aemit_wf=zeros(ny,nx,nspec,'single'); % mean annual widlfire biomass burning emissions per scpecies [g / m^2 / month]
aemit_defo=zeros(ny,nx,nspec,'single'); % mean annual total deforestation burning emissions per scpecies [g / m^2 / month]
% check global sum in original resolution:
% fname=([prefix '2014' postfix]);
% emit2014=zeros(size(mask25),'single');
% for m=1:12
%     emit2014=emit2014+flipdim(h5read([path fname],sprintf('/emissions/%2.2i/DM',m))',1);
% end
% sum(sum(emit2014.*area25)) % should be 4.2 Pg DM / yr
% clear emit2014
% load the data and generate the climatologies for DM and mean emission fields by species
for year=years
    fname=sprintf('%s%4.4i%s',prefix,year,postfix);
    fprintf('Loading from %s ...\n',fname);
    for m=1:12
        % read total emissions and contributions from different fire types
        DM=rebin(rebin(flipdim( ... % in dry matter
            h5read([path fname],sprintf('/emissions/%2.2i/DM',m))',1).*...
            area25,ny,1),nx,2)./area;
        f_BORF=rebin(rebin(flipdim(h5read([path fname],sprintf('/emissions/%2.2i/partitioning/DM_BORF',m))',1).*area25,ny,1),nx,2)./area;
        f_SAVA=rebin(rebin(flipdim(h5read([path fname],sprintf('/emissions/%2.2i/partitioning/DM_SAVA',m))',1).*area25,ny,1),nx,2)./area;
        f_TEMF=rebin(rebin(flipdim(h5read([path fname],sprintf('/emissions/%2.2i/partitioning/DM_TEMF',m))',1).*area25,ny,1),nx,2)./area;
        f_DEFO=rebin(rebin(flipdim(h5read([path fname],sprintf('/emissions/%2.2i/partitioning/DM_DEFO',m))',1).*area25,ny,1),nx,2)./area;
        f_PEAT=rebin(rebin(flipdim(h5read([path fname],sprintf('/emissions/%2.2i/partitioning/DM_PEAT',m))',1).*area25,ny,1),nx,2)./area;
        f_AGRI=rebin(rebin(flipdim(h5read([path fname],sprintf('/emissions/%2.2i/partitioning/DM_AGRI',m))',1).*area25,ny,1),nx,2)./area;
        emit(:,:,m)=emit(:,:,m)+DM/nyears;
        emit_wf(:,:,m)=emit_wf(:,:,m)+DM.*(f_BORF+f_SAVA+f_TEMF)/nyears;
        emit_defo(:,:,m)=emit_defo(:,:,m)+DM.*f_DEFO/nyears;
        for k=1:nspec
            aemit(:,:,k)=aemit(:,:,k)+DM.*(EF(k,1)*f_BORF+EF(k,2)*f_SAVA+EF(k,3)*f_TEMF+EF(k,4)*f_DEFO+EF(k,5)*f_PEAT+EF(k,6)*f_AGRI)/nyears/12;
            aemit_wf(:,:,k)=aemit_wf(:,:,k)+DM.*(EF(k,1)*f_BORF+EF(k,2)*f_SAVA+EF(k,3)*f_TEMF)/nyears/12;
            aemit_defo(:,:,k)=aemit_defo(:,:,k)+DM*EF(k,4).*f_DEFO/nyears/12;
        end
    end
end
% check sums to be sure they are OK
% nansum(nansum(sum(emit,3).*area)) % all fires, should be 4.4 Pg DM / year
% nansum(nansum(sum(emit_wf,3).*area)) % wildfires, should be 3.1 Pg DM / year
% clean up:
clear input DM f_* year m k path prefix postfix *25 fname n*
% save the output:
area=single(area);
mask=single(mask);
save GFED4s_species
save GFED4s_dm emit* area mask years

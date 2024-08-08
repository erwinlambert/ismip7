import xarray as xr
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import scipy.signal as sig
import numpy as np
#import cmocean as cmo

#Days per month
def dpm(years):
    return np.tile([31,28,31,30,31,30,31,31,30,31,30,31],years)

class Run:
    def __init__(self,model,region,run,y0,y1,k0=0,k1=6000):
        self.model = model
        self.region = region
        self.run = run
        self.y0 = y0
        self.y1 = y1
        self.k0 = k0
        self.k1 = k1

        self.get_coordinates()
        self.get_volume()
        self.get_TS()
        self.get_bins()
        print(f'Got run {self.model} {self.region} {self.run} {self.y0}-{self.y1}')

    def get_coordinates(self):
        if self.region == 'Amundsen':
            if self.model == 'CESM2':
                self.i0,self.i1,self.j0,self.j1 = 250,270,8,18
            elif self.model == 'UKESM1-0-LL':
                self.i0,self.i1,self.j0,self.j1 = 170,190,50,70
            elif self.model == 'EC-Earth3':
                self.i0,self.i1,self.j0,self.j1 = 170,190,12,32
            elif self.model == 'WOA23':
                self.i0,self.i1,self.j0,self.j1 = -116,-96,-76,-70
        elif self.region == 'Ross':
            if self.model == 'CESM2':
                 self.i0,self.i1,self.j0,self.j1 = 180,220,2,7
            elif self.model == 'UKESM1-0-LL':
                self.i0,self.i1,self.j0,self.j1 = 85,140,38,48
            elif self.model == 'EC-Earth3':
                self.i0,self.i1,self.j0,self.j1 = 85,140,0,10           
            elif self.model == 'WOA23':
                self.i0,self.i1,self.j0,self.j1 = -180,-150,-79,-76
        elif self.region == 'Weddell':
            if self.model == 'CESM2':
                 self.i0,self.i1,self.j0,self.j1 = 300,360,3,9
            elif self.model == 'UKESM1-0-LL':
                self.i0,self.i1,self.j0,self.j1 = 224,246,38,55
            elif self.model == 'EC-Earth3':
                self.i0,self.i1,self.j0,self.j1 = 226,248,0,17
            elif self.model == 'WOA23':
                self.i0,self.i1,self.j0,self.j1 = -65,-40,-78.5,-74.5
        elif self.region == 'Totten':
            if self.model == 'CESM2':
                 self.i0,self.i1,self.j0,self.j1 = 137,146,24,28
            elif self.model == 'UKESM1-0-LL':
                self.i0,self.i1,self.j0,self.j1 = 40,50,78,84
            elif self.model == 'EC-Earth3':
                self.i0,self.i1,self.j0,self.j1 = 42,52,40,46  
            elif self.model == 'WOA23':
                self.i0,self.i1,self.j0,self.j1 = 113,124,-67.5,-64.5

    def get_volume(self):
        if self.model in ['CESM2']:
            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/volcello*.nc')
            
            ds = ds.sel(nlat=slice(self.j0,self.j1),nlon=slice(self.i0,self.i1),lev=slice(self.k0*100.,self.k1*100.))

            self.lev = ds.lev_bnds.values
            self.depth = -np.average(self.lev,axis=1)
            self.lon = ds.lon.values
            self.lat = ds.lat.values
            self.V = ds.volcello.values
            ds.close()

            #Mask
            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/historical/thetao/thetao*.nc')
            ds = ds.sel(nlat=slice(self.j0,self.j1),nlon=slice(self.i0,self.i1),lev=slice(self.k0*100.,self.k1*100.))
            ds = ds.isel(time=0)
            self.V = np.where(np.isnan(ds.thetao),0,self.V)
            ds.close()

        elif self.model in ['EC-Earth3']:

            if self.y1-self.y0==1:
                ds = xr.open_dataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/thkcello/thkcello_Omon_{self.model}_{self.run}_r1i1p1f1_gn_{self.y0}01-{self.y0}12.nc')
            else:
                ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/thkcello/*.nc',combine='by_coords')
            ds = ds.sel(time=slice(f'{self.y0}-01-01',f'{self.y1}-01-01'),j=slice(self.j0,self.j1),i=slice(self.i0,self.i1),lev=slice(self.k0,self.k1))
            if self.y1-self.y0==1:
                self.lev = ds.lev_bnds.values
            else:
                self.lev = np.average(ds.lev_bnds.values,axis=0,weights=dpm(self.y1-self.y0))
            self.depth = -np.average(self.lev,axis=1)
            self.lon = ds.longitude.values
            self.lat = ds.latitude.values
            self.thkcello = np.average(ds.thkcello,axis=0,weights=dpm(self.y1-self.y0))
            ds.close()

            ds = xr.open_dataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/areacello_Ofx_EC-Earth3_historical_r1i1p1f1_gn.nc')
            ds = ds.sel(j=slice(self.j0,self.j1),i=slice(self.i0,self.i1))

            self.areacello = ds.areacello.values
            ds.close()

            self.V = self.thkcello*self.areacello
            self.V = np.where(np.isnan(self.V),0,self.V)
        
        elif self.model in ['UKESM1-0-LL']:

            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/thkcello/*.nc')
            ds = ds.sel(time=slice(f'{self.y0}-01-01',f'{self.y1}-01-01'),j=slice(self.j0,self.j1),i=slice(self.i0,self.i1),lev=slice(self.k0,self.k1))
            self.lev = np.average(ds.lev_bnds.values,axis=0,weights=dpm(self.y1-self.y0))
            self.depth = -np.average(self.lev,axis=1)
            self.lon = ds.longitude.values
            self.lat = ds.latitude.values
            self.thkcello = np.average(ds.thkcello,axis=0,weights=dpm(self.y1-self.y0))
            ds.close()

            ds = xr.open_dataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/areacello_Ofx_UKESM1-0-LL_piControl_r1i1p1f2_gn.nc')
            ds = ds.sel(j=slice(self.j0,self.j1),i=slice(self.i0,self.i1))

            self.areacello = ds.areacello.values
            ds.close()

            self.V = self.thkcello*self.areacello
            self.V = np.where(np.isnan(self.V),0,self.V)
        
        elif self.model == 'WOA23':
            ds = xr.open_dataset('/usr/people/lambert/work/projects/data/woa23/woa23_decav91C0_t00_04.nc',decode_cf=False)
            ds = ds.isel(time=0)
            ds = ds.sel(lon=slice(self.i0,self.i1),lat=slice(self.j0,self.j1),depth=slice(self.k0,self.k1))

            self.T = np.where(ds.t_an<1e30,ds.t_an,np.nan)

            self.lon = ds.lon.values
            self.lat = ds.lat.values
            self.depth = -ds.depth

            self.V = np.zeros((len(ds.depth),len(ds.lat),len(ds.lon)))
            R = 6.371e6
            A = np.ones((len(ds.lat),len(ds.lon)))
            for l in range(len(ds.lat)):
                dy = 2*np.pi*R*(ds.lat_bnds[l,1]-ds.lat_bnds[l,0])/360
                dx = 2*np.pi*R*(ds.lon_bnds[:,1]-ds.lon_bnds[:,0])/360 * np.cos(2*np.pi*ds.lat[l]/360)
                A[l,:] = dx*dy

            for d in range(len(ds.depth)):
                D = np.where(np.isnan(self.T[d,:,:]),0,ds.depth_bnds[d,1]-ds.depth_bnds[d,0])
                self.V[d,:,:] = D*A
            ds.close()
            
    def get_TS(self):
        if self.model in ['CESM2']:
            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/thetao/*.nc',combine='by_coords')
            ds = ds.sel(time=slice(f'{self.y0}-01-01',f'{self.y1}-01-01'),nlat=slice(self.j0,self.j1),nlon=slice(self.i0,self.i1),lev=slice(self.k0*100.,self.k1*100.))
            self.T = np.average(ds.thetao,axis=0,weights=dpm(self.y1-self.y0))
            ds.close()

            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/so/*.nc',combine='by_coords')
            ds = ds.sel(time=slice(f'{self.y0}-01-01',f'{self.y1}-01-01'),nlat=slice(self.j0,self.j1),nlon=slice(self.i0,self.i1),lev=slice(self.k0*100.,self.k1*100.))
            self.S = np.average(ds.so,axis=0,weights=dpm(self.y1-self.y0))
            ds.close()

        elif self.model in ['EC-Earth3']:
            if self.y1-self.y0==1:
                ds = xr.open_dataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/thetao/thetao_Omon_{self.model}_{self.run}_r1i1p1f1_gn_{self.y0}01-{self.y0}12.nc')
            else:
                ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/thetao/*.nc',combine='by_coords')
            ds = ds.sel(time=slice(f'{self.y0}-01-01',f'{self.y1}-01-01'),j=slice(self.j0,self.j1),i=slice(self.i0,self.i1),lev=slice(self.k0,self.k1))
            self.T = np.average(ds.thetao,axis=0,weights=dpm(self.y1-self.y0))
            ds.close()

            if self.y1-self.y0==1:
                ds = xr.open_dataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/so/so_Omon_{self.model}_{self.run}_r1i1p1f1_gn_{self.y0}01-{self.y0}12.nc')
            else:
                ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/so/*.nc',combine='by_coords')
            ds = ds.sel(time=slice(f'{self.y0}-01-01',f'{self.y1}-01-01'),j=slice(self.j0,self.j1),i=slice(self.i0,self.i1),lev=slice(self.k0,self.k1))
            self.S = np.average(ds.so,axis=0,weights=dpm(self.y1-self.y0))
            ds.close()

        elif self.model in ['UKESM1-0-LL']:
            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/thetao/*.nc',combine='by_coords')
            ds = ds.sel(time=slice(f'{self.y0}-01-01',f'{self.y1}-01-01'),j=slice(self.j0,self.j1),i=slice(self.i0,self.i1),lev=slice(self.k0,self.k1))
            self.T = np.average(ds.thetao,axis=0,weights=dpm(self.y1-self.y0))
            ds.close()

            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/so/*.nc',combine='by_coords')
            ds = ds.sel(time=slice(f'{self.y0}-01-01',f'{self.y1}-01-01'),j=slice(self.j0,self.j1),i=slice(self.i0,self.i1),lev=slice(self.k0,self.k1))
            self.S = np.average(ds.so,axis=0,weights=dpm(self.y1-self.y0))
            ds.close()            

        elif self.model == 'WOA23':
            #Already got T
            ds = xr.open_dataset('/usr/people/lambert/work/projects/data/woa23/woa23_decav91C0_s00_04.nc',decode_cf=False)
            ds = ds.isel(time=0)
            ds = ds.sel(lon=slice(self.i0,self.i1),lat=slice(self.j0,self.j1),depth=slice(self.k0,self.k1))
            self.S = np.where(ds.s_an<1e30,ds.s_an,np.nan)
            ds.close()

    def get_bins(self):
        self.Vb,self.Sb,self.Tb = np.histogram2d(self.S.flatten()[self.V.flatten()>0],self.T.flatten()[self.V.flatten()>0],bins=100,weights=self.V.flatten()[self.V.flatten()>0])

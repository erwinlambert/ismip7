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
        elif self.region == 'Ross':
            if self.model == 'CESM2':
                 self.i0,self.i1,self.j0,self.j1 = 180,220,2,7
            elif self.model == 'UKESM1-0-LL':
                self.i0,self.i1,self.j0,self.j1 = 85,140,38,48
            elif self.model == 'EC-Earth3':
                self.i0,self.i1,self.j0,self.j1 = 85,140,0,10           
            
    def get_volume(self):
        if self.model in ['CESM2']:
            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/volcello*.nc')
            
            ds = ds.sel(nlat=slice(self.j0,self.j1),nlon=slice(self.i0,self.i1),lev=slice(self.k0*100.,self.k1*100.))

            self.lev = ds.lev_bnds.values#/100.
            self.lon = ds.lon.values
            self.lat = ds.lat.values
            #print('longitude:',np.min(self.lon),np.max(self.lon))
            #print('latitude:',np.min(self.lat),np.max(self.lat))
            self.V = ds.volcello.values
            ds.close()

            #Mask
            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/historical/thetao/thetao*.nc')
            ds = ds.sel(nlat=slice(self.j0,self.j1),nlon=slice(self.i0,self.i1))
            ds = ds.isel(time=0)
            self.V = np.where(np.isnan(ds.thetao),0,self.V)
            ds.close()

        elif self.model in ['EC-Earth3','UKESM1-0-LL']:

            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/thkcello/*.nc')
            ds = ds.sel(time=slice(f'{self.y0}-01-01',f'{self.y1}-01-01'),j=slice(self.j0,self.j1),i=slice(self.i0,self.i1),lev=slice(self.k0,self.k1))
            #print(ds)
            self.lev = np.average(ds.lev_bnds.values,axis=0,weights=dpm(self.y1-self.y0))
            self.lon = ds.longitude.values
            self.lat = ds.latitude.values
            #print('longitude:',np.min(self.lon),np.max(self.lon))
            #print('latitude:',np.min(self.lat),np.max(self.lat))
            #self.thkcello = ds.thkcello
            self.thkcello = np.average(ds.thkcello,axis=0,weights=dpm(self.y1-self.y0))
            #self.thkcello = np.sum(ds.thkcello*dpm(self.y1-self.y0)[:,np.newaxis,np.newaxis,np.newaxis],axis=0)/np.sum(dpm(self.y1-self.y0))
            ds.close()

            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/areacello*.nc')
            ds = ds.sel(j=slice(self.j0,self.j1),i=slice(self.i0,self.i1))

            self.areacello = ds.areacello.values
            ds.close()

            self.V = self.thkcello*self.areacello
            self.V = np.where(np.isnan(self.V),0,self.V)
            
        #print('got volume')
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
        elif self.model in ['EC-Earth3','UKESM1-0-LL']:
            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/thetao/*.nc',combine='by_coords')
            ds = ds.sel(time=slice(f'{self.y0}-01-01',f'{self.y1}-01-01'),j=slice(self.j0,self.j1),i=slice(self.i0,self.i1),lev=slice(self.k0,self.k1))
            self.T = np.average(ds.thetao,axis=0,weights=dpm(self.y1-self.y0))
            ds.close()

            ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{self.run}/so/*.nc',combine='by_coords')
            ds = ds.sel(time=slice(f'{self.y0}-01-01',f'{self.y1}-01-01'),j=slice(self.j0,self.j1),i=slice(self.i0,self.i1),lev=slice(self.k0,self.k1))
            self.S = np.average(ds.so,axis=0,weights=dpm(self.y1-self.y0))
            ds.close()

    def get_bins(self):
        self.Vb,self.Sb,self.Tb = np.histogram2d(self.S.flatten()[self.V.flatten()>0],self.T.flatten()[self.V.flatten()>0],bins=100,weights=self.V.flatten()[self.V.flatten()>0])

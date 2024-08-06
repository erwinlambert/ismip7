import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
#import scipy.signal as sig
import numpy as np
import cmocean as cmo

from run import Run
from run import dpm

class Region:
    def __init__(self, region,model,refmodel='UKESM1-0-LL'):
        # Get volume CMIP
        self.region = region
        self.model = model
        self.refmodel = refmodel

        self.get_historical()

    def get_historical(self):
        #self.get_cmip_fixed()
        #self.get_woa()
        #self.get_woa()
        #self.ref = self.woa
        self.ref = Run(self.refmodel,self.region,'historical',1995,2015)
        self.prd = Run(self.model,self.region,'historical',1995,2015)
        self.get_bias()

    def plot_region_cmip(self,ii,jj):
        plt.pcolormesh(self.lonc,self.latc,np.nansum(self.thkcello,axis=0),cmap='Blues',norm=mpl.colors.LogNorm(vmin=10,vmax=1e4))
        plt.scatter(self.lonc[jj,ii],self.latc[jj,ii],50,c='tab:red')

    def plot_volumes(self):
        fig,ax = plt.subplots(1,3,figsize=(3*5,6),sharex=True,sharey=True)

        ax[0].pcolormesh(self.ref.Sb,self.ref.Tb,self.prd.Vb.T,norm=mpl.colors.LogNorm(vmin=1e9,vmax=1e13),cmap='Greys')
        plt.colorbar(im,ax=ax[0],orientation='horizontal')
        ax[0].set_title(f'Reference ({self.ref.model} {self.ref.y0}-{self.ref.y1})')
        ax[0].set_ylabel(self.region)

        im = ax[1].pcolormesh(self.prd.Sb,self.prd.Tb,self.prd.Vb.T,norm=mpl.colors.LogNorm(vmin=1e9,vmax=1e13))
        plt.colorbar(im,ax=ax[1],orientation='horizontal')
        ax[1].set_title(f'Present day {self.model} ({self.prd.y0}-{self.prd.y1})')
        #ax[1].plot(self.prd.S[:,jj,ii],self.prd.T[:,jj,ii],c='tab:red')

        im = ax[2].pcolormesh(self.fut.Sb,self.fut.Tb,self.fut.Vb.T,norm=mpl.colors.LogNorm(vmin=1e9,vmax=1e13),cmap='Oranges')
        plt.colorbar(im,ax=ax[2],orientation='horizontal')
        ax[2].set_title(f'{self.fut.run} ({self.fut.y0}-{self.fut.y1})')
        #ax[2].plot(self.fut.S[:,jj,ii],self.fut.T[:,jj,ii],c='tab:red')

    def plot_delta(self):
        fig,ax = plt.subplots(1,4,figsize=(4*4,5),sharex=True,sharey=True)

        im = ax[0].pcolormesh(self.ref.Sb,self.ref.Tb,self.ref.Vb.T,norm=mpl.colors.LogNorm(vmin=1e9,vmax=1e13))
        plt.colorbar(im,ax=ax[0],orientation='horizontal')
        ax[0].set_title(f'Reference ({self.ref.model} {self.ref.y0}-{self.ref.y1})')
        ax[0].set_ylabel(self.region)

        im = ax[1].pcolormesh(self.prd.Sb,self.prd.Tb,self.prd.Vb.T,norm=mpl.colors.LogNorm(vmin=1e9,vmax=1e13))
        plt.colorbar(im,ax=ax[1],orientation='horizontal')
        ax[1].set_title(f'present day {self.model} ({self.prd.y0}-{self.prd.y1})')
        #ax[0].plot(self.prd.S[:,jj,ii],self.prd.T[:,jj,ii],c='tab:red')

        #im = ax[2].pcolormesh(self.prd.Sc,self.prd.Tc,self.fut.Vb.T,norm=mpl.colors.LogNorm(vmin=1e9,vmax=1e13),cmap='Greys')
        im = ax[2].pcolormesh(self.prd.Sb,self.prd.Tb,self.fut.deltaT.T,cmap='cmo.balance',vmin=-3,vmax=3)
        #im = ax[2].pcolormesh(self.prd.Sc,self.prd.Tc,self.fut.deltaTf.T,cmap='cmo.balance',vmin=-2,vmax=2)

        plt.colorbar(im,ax=ax[2],orientation='horizontal')
        ax[2].set_title(f'{self.fut.run} ({self.fut.y0}-{self.fut.y1})')
        #ax[1].plot(self.fut.S[:,jj,ii],self.fut.T[:,jj,ii],c='tab:red')

        im = ax[3].pcolormesh(self.ref.Sb,self.ref.Tb,self.ref.dTb.T,cmap='cmo.balance',vmin=-3,vmax=3)
        plt.colorbar(im,ax=ax[3],orientation='horizontal')
        #ax[3].set_title(f'{self.fut.run} ({self.fut.y0}-{self.fut.y1})')

    def plot_profiles(self,vlim=-10):
        fig,ax = plt.subplots(1,3,figsize=(10,5),sharey=True)

        Tav = np.average(np.where(np.isnan(self.ref.T[:vlim,:,:]),0,self.ref.T[:vlim,:,:]),axis=(1,2),weights=self.ref.V[:vlim,:,:])
        ax[1].plot(Tav,self.ref.lev[:vlim,0],label='optimised',c='tab:red',ls='--')

        Tav = np.average(np.where(np.isnan(self.prd.T[:vlim,:,:]),0,self.prd.T[:vlim,:,:]),axis=(1,2),weights=self.prd.V[:vlim,:,:])
        ax[0].plot(Tav,self.prd.lev[:vlim,0],label='raw',c='.5',ls='--')

        Tav = np.average(np.where(np.isnan(self.ref.T[:vlim,:,:]),0,self.ref.dT[:vlim,:,:]),axis=(1,2),weights=self.ref.V[:vlim,:,:])
        ax[2].plot(Tav,self.ref.lev[:vlim,0],c='tab:red')

        Tav = np.average(np.where(np.isnan(self.prd.T[:vlim,:,:]),0,(self.fut.T-self.prd.T)[:vlim,:,:]),axis=(1,2),weights=self.prd.V[:vlim,:,:])
        ax[2].plot(Tav,self.prd.lev[:vlim,0],c='.5')

        Tav = np.average(np.where(np.isnan(self.ref.T[:vlim,:,:]),0,(self.ref.T+self.ref.dT)[:vlim,:,:]),axis=(1,2),weights=self.ref.V[:vlim,:,:])
        ax[1].plot(Tav,self.ref.lev[:vlim,0],c='tab:red')

        Tav = np.average(np.where(np.isnan(self.prd.T[:vlim,:,:]),0,self.fut.T[:vlim,:,:]),axis=(1,2),weights=self.prd.V[:vlim,:,:])
        ax[0].plot(Tav,self.prd.lev[:vlim,0],c='.5')

        ax[0].set_title('Raw model')
        ax[1].set_title('Optimised')
        ax[2].set_xlabel('Warming T')
        #ax[0].legend()

        for Ax in ax:
            Ax.set_ylim([0,1500])
            Ax.invert_yaxis()

    def get_woa(self):
        self.woa = Run('woa',1991,2020)

        ds = xr.open_dataset('/usr/people/lambert/work/projects/data/woa23/woa23_decav91C0_t00_04.nc',decode_cf=False)
        ds = ds.isel(time=0)
        ds = ds.sel(lon=slice(self.iw0,self.iw1),lat=slice(self.jw0,self.jw1))
        self.woa.T = np.where(ds.t_an<1e30,ds.t_an,np.nan)
        ds.close()

        ds = xr.open_dataset('/usr/people/lambert/work/projects/data/woa23/woa23_decav91C0_s00_04.nc',decode_cf=False)
        ds = ds.isel(time=0)
        ds = ds.sel(lon=slice(self.iw0,self.iw1),lat=slice(self.jw0,self.jw1))
        self.woa.S = np.where(ds.s_an<1e30,ds.s_an,np.nan)
        ds.close()

        #Get volume
        self.woa.V = np.zeros((len(ds.depth),len(ds.lat),len(ds.lon)))
        R = 6.371e6
        A = np.ones((len(ds.lat),len(ds.lon)))
        for l in range(len(ds.lat)):
            dy = 2*np.pi*R*(ds.lat_bnds[l,1]-ds.lat_bnds[l,0])/360
            dx = 2*np.pi*R*(ds.lon_bnds[:,1]-ds.lon_bnds[:,0])/360 * np.cos(2*np.pi*ds.lat[l]/360)
            A[l,:] = dx*dy

        for d in range(len(ds.depth)):
            D = np.where(np.isnan(self.woa.T[d,:,:]),0,ds.depth_bnds[d,1]-ds.depth_bnds[d,0])
            self.woa.V[d,:,:] = D*A

        self.woa.Vb,self.woa.Sb,self.woa.Tb = np.histogram2d(self.woa.S.flatten()[self.woa.V.flatten()>0],self.woa.T.flatten()[self.woa.V.flatten()>0],bins=100,weights=self.woa.V.flatten()[self.woa.V.flatten()>0])

    def get_cmip_fixed(self,model):
        'Get fixed CMIP variables from historical run'
        ds2 = xr.open_dataset(f'/usr/people/lambert/work2/data/cmip6/{model}/historical/thkcello/thkcello_Omon_{model}_historical_r1i1p1f1_gn_185001-185012.nc')
        ds2 = ds2.sel(j=slice(self.jc0,self.jc1),i=slice(self.ic0,self.ic1),lev=slice(self.k0,self.k1))
        ds2 = ds2.isel(time=0)

        self.lev = ds2.lev_bnds.values
        self.lonc = ds2.longitude
        self.latc = ds2.latitude
        self.thkcello = ds2.thkcello.values
        ds2.close()

        ds3 = xr.open_dataset(f'/usr/people/lambert/work2/data/cmip6/{model}/areacello_Ofx_{model}_historical_r1i1p1f1_gn.nc')
        ds3 = ds3.sel(j=slice(self.jc0,self.jc1),i=slice(self.ic0,self.ic1))

        self.Vc = self.thkcello*ds3.areacello.values
        ds3.close()
        print('got fixed variables')

    def get_cmip_run(self,run,y0,y1):
        'Get CMIP variables from prescribed run over prescribed period'
        out = Run(run,y0,y1)
        out.V = self.Vc
        
        ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{run}/thetao/*.nc',combine='by_coords')
        ds = ds.sel(time=slice(f'{y0}-01-01',f'{y1}-01-01'),j=slice(self.jc0,self.jc1),i=slice(self.ic0,self.ic1),lev=slice(self.k0,self.k1))
        out.T = np.average(ds.thetao,axis=0,weights=dpm(y1-y0))
        ds.close()

        ds = xr.open_mfdataset(f'/usr/people/lambert/work2/data/cmip6/{self.model}/{run}/so/*.nc',combine='by_coords')
        ds = ds.sel(time=slice(f'{y0}-01-01',f'{y1}-01-01'),j=slice(self.jc0,self.jc1),i=slice(self.ic0,self.ic1),lev=slice(self.k0,self.k1))
        out.S = np.average(ds.so,axis=0,weights=dpm(y1-y0))
        ds.close()

        out.Vb,out.Sb,out.Tb = np.histogram2d(out.S.flatten()[out.V.flatten()>0],out.T.flatten()[out.V.flatten()>0],bins=100,weights=out.V.flatten()[out.V.flatten()>0])

        print(f'got variables of {run} run from year {y0} to {y1}')
        
        return out
    
    def get_delta(self):

        VTdel,Sdel,Tdel = np.histogram2d(self.prd.S.flatten()[self.prd.V.flatten()>0],self.prd.T.flatten()[self.prd.V.flatten()>0],bins=100,weights=(self.prd.V*(self.fut.T-self.prd.T)).flatten()[self.prd.V.flatten()>0])
        self.fut.deltaT = VTdel/self.prd.Vb
        VSdel,Sdel,Tdel = np.histogram2d(self.prd.S.flatten()[self.prd.V.flatten()>0],self.prd.T.flatten()[self.prd.V.flatten()>0],bins=100,weights=(self.prd.V*(self.fut.S-self.prd.S)).flatten()[self.prd.V.flatten()>0])
        self.fut.deltaS = VSdel/self.prd.Vb

        self.fut.deltaTf = self.fill_delta(self.fut.deltaT)
        self.fut.deltaSf = self.fill_delta(self.fut.deltaS)

    def get_bias(self,Tperc=.99,Sperc=.99):

        # S-bias constrain 95th percentile
        A,B = np.histogram((self.ref.S).flatten()[self.ref.V.flatten()>0],weights=self.ref.V.flatten()[self.ref.V.flatten()>0],bins=1000,density=True)
        AA = np.cumsum(A)/np.cumsum(A)[-1]
        aa = np.argmin((AA-Sperc)**2)
        self.ref.S95 = B[aa]

        A,B = np.histogram((self.prd.S).flatten()[self.prd.V.flatten()>0],weights=self.prd.V.flatten()[self.prd.V.flatten()>0],bins=1000,density=True)
        AA = np.cumsum(A)/np.cumsum(A)[-1]
        aa = np.argmin((AA-Sperc)**2)
        self.prd.S95 = B[aa]

        # S-bias
        a = np.arange(.5,1.5,.01)
        ref,dum = np.histogram(self.ref.S,bins=self.ref.Sb,weights=self.ref.V)
        rmse = np.zeros((len(a)))
        for A,AA in enumerate(a):
            prd,dum = np.histogram(AA*(self.prd.S-self.prd.S95)+self.ref.S95,bins=self.ref.Sb,weights=self.prd.V)
            rmse[A] = np.sum(np.where(prd==0,0,(np.log10(prd/np.sum(prd))-np.log10(ref/np.sum(ref))))**2)**.5 / np.sum(np.where(prd==0,0,1))
            #print(AA,rmse[A])
        out = np.unravel_index(np.nanargmin(rmse, axis=None), rmse.shape)
        self.dSa = a[out[0]]
        self.prd.Sc = self.dSa*(self.prd.Sb-self.prd.S95)+self.ref.S95 #Corrected S
        #self.fut.Sc = self.dSa*(self.fut.Sb-self.prd.S95)+self.ref.S95 #Corrected S

        # T-bias (scale 95th percentile of T-Tmin)
        A,B = np.histogram((self.ref.T-self.ref.Tb[0]).flatten()[self.ref.V.flatten()>0],weights=self.ref.V.flatten()[self.ref.V.flatten()>0],bins=1000,density=True)
        AA = np.cumsum(A)/np.cumsum(A)[-1]
        aa = np.argmin((AA-Tperc)**2)
        self.ref.TF95 = B[aa]

        A,B = np.histogram((self.prd.T-self.prd.Tb[0]).flatten()[self.prd.V.flatten()>0],weights=self.prd.V.flatten()[self.prd.V.flatten()>0],bins=1000,density=True)
        AA = np.cumsum(A)/np.cumsum(A)[-1]
        aa = np.argmin((AA-Tperc)**2)
        self.prd.TF95 = B[aa]
        self.dTa = self.ref.TF95/self.prd.TF95
        self.prd.Tc = self.dTa*(self.prd.Tb-self.prd.Tb[0])+self.ref.Tb[0]
        #self.fut.Tc = self.dTa*(self.fut.Tb-self.fut.Tb[0])+self.ref.Tb[0]
        
    def fill_delta(self,deltavar):
        '''Routine to fill deltaT and deltaS in full normalised T,S space'''

        newval = np.nan*np.ones((len(self.prd.Sc),len(self.prd.Tc)))

        newval[1:,1:] = deltavar
        Nleft = np.sum(np.isnan(newval[1:,1:]))
        while Nleft>0:
            mask = np.where(np.isnan(newval),0,1)
            newval2 = np.where(np.isnan(newval),0,newval)
            AA = newval2*mask
            AAm1 = np.roll(AA,-1,axis=0)
            m1 = np.roll(mask,-1,axis=0)
            AAp1 = np.roll(AA,1,axis=0)
            p1 = np.roll(mask,1,axis=0)
            newval3 = (np.roll(AAm1,-1,axis=1)+AAm1+np.roll(AAm1,1,axis=1) + np.roll(AA,-1,axis=1)+np.roll(AA,1,axis=1) + np.roll(AAp1,-1,axis=1)+AAp1+np.roll(AAp1,1,axis=1)) / (np.roll(m1,-1,axis=1)+m1+np.roll(m1,1,axis=1) + np.roll(mask,-1,axis=1)+np.roll(mask,1,axis=1) + np.roll(p1,-1,axis=1)+p1+np.roll(p1,1,axis=1))
            newval4 = np.where(mask,newval,newval3)
            newval = newval4
            newval[0,:] = np.nan
            newval[:,0] = np.nan
            Nleft = np.sum(np.isnan(newval[1:,1:]))

        return newval[1:,1:]
    
    def get_anom(self):
        '''Get anomalies of reference T,S values'''

        self.ref.dT = np.zeros(self.ref.T.shape)
        self.ref.dS = np.zeros(self.ref.S.shape)

        for i in range(self.ref.dT.shape[0]):
            for j in range(self.ref.dT.shape[1]):
                for k in range(self.ref.dT.shape[2]):
                    if self.ref.V[i,j,k]==0:
                        continue
                    else:
                        isref = min(99,max(0,int(99*(self.ref.S[i,j,k]-self.prd.Sc[0])/(self.prd.Sc[-1]-self.prd.Sc[0]))))
                        jtref = min(99,max(0,int(99*(self.ref.T[i,j,k]-self.prd.Tc[0])/(self.prd.Tc[-1]-self.prd.Tc[0]))))
                        self.ref.dS[i,j,k] = self.fut.deltaSf[isref,jtref]
                        self.ref.dT[i,j,k] = self.fut.deltaTf[isref,jtref]

        #Distributions of dS and dT, only for plotting purposes
        VTdel,Sdel,Tdel = np.histogram2d(self.ref.S.flatten()[self.ref.V.flatten()>0],self.ref.T.flatten()[self.ref.V.flatten()>0],bins=100,range=[[self.ref.Sb[0],self.ref.Sb[-1]],[self.ref.Tb[0],self.ref.Tb[-1]]],weights=(self.ref.V*self.ref.dT).flatten()[self.ref.V.flatten()>0])
        self.ref.dTb = VTdel/self.ref.Vb
        VSdel,Sdel,Tdel = np.histogram2d(self.ref.S.flatten()[self.ref.V.flatten()>0],self.ref.S.flatten()[self.ref.V.flatten()>0],bins=100,range=[[self.ref.Sb[0],self.ref.Sb[-1]],[self.ref.Tb[0],self.ref.Tb[-1]]],weights=(self.ref.V*self.ref.dS).flatten()[self.ref.V.flatten()>0])
        self.ref.dSb = VSdel/self.ref.Vb

    def get_future(self,run,y0,y1):
        self.fut = Run(self.model,self.region,run,y0,y1)#'ssp585',2081,2101)
        self.get_delta()
        self.get_anom()
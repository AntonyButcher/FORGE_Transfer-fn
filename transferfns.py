import os
import matplotlib.pyplot as plt
import numpy as np
from obspy import Trace, Stream, read
import scipy
from scipy.integrate import cumtrapz
from scipy.fftpack import fft, ifft, fft2, ifft2, fftshift, ifftshift, fftfreq
from scipy import signal
from scipy import ndimage
import math
from math import ceil

def segy2stream(path, network, channel_spacing=1, units='Strain rate'):
    """
    Reads in SEGY file and outputs and obspy stream with correct header information.
    
    Arguments:
    Required:
    st - obspy stream
    path - path of segy file 
    id - unique file name

    Returns:
    obspy stream
    7-Nov-2021, read trace header
    """
    
    st=read(path, format='SEGY', unpack_trace_headers=True, headonly=False)

    for i in range(len(st)):
        
#         st[i].stats.distance=(tr.stats.segy.trace_header.x_coordinate_of_ensemble_position_of_this_trace)/1000
        st[i].stats.distance=(i*channel_spacing)
        st[i].stats.network=network
        st[i].stats.channel='Z'
        st[i].stats.units = units
    
    return st

def stream2array(st):
    """
    Populates a 2D np.array that is the traces as rows by the samples as cols.
    
    Arguments:
    Required:
    st - obspy stream

    Returns:
    nparray - numpy array
    """
    
    nparray=np.empty((len(st),len(st[0].data)),dtype=float) # initialize
    for index,trace in enumerate(st):
        check=len(nparray[index,:])-len(trace.data)
        if check != 0:
            trace.data=np.resize(trace.data, len(nparray[index,:]))
        
        nparray[index,:]=trace.data

    return nparray

def segy2stream(path, network, channel_spacing=1.02, units='Strain rate'):
    """
    Reads in SEGY file and outputs and obspy stream with correct header information.
    
    Arguments:
    Required:
    st - obspy stream
    path - path of segy file 
    id - unique file name

    Returns:
    obspy stream
    7-Nov-2021, read trace header
    """
    
    st=read(path, format='SEGY', unpack_trace_headers=True, headonly=False)

    for i in range(len(st)):
        
#         st[i].stats.distance=(tr.stats.segy.trace_header.x_coordinate_of_ensemble_position_of_this_trace)/1000
        st[i].stats.distance=(i*channel_spacing)
        st[i].stats.network=network
        st[i].stats.channel='Z'
        st[i].stats.units = units
    
    return st

def image(st,style=1,skip=10,clim=[0],tmin=0,tmax=None,
        physicalFiberLocations=False, picks=None,
        distance_unit='km', xlabel='Offset',fig=None):
    """
    Simple image plot of DAS Stream, adapted from IRIS DAS workshop function.
    #skip=10 is default to skip every 10 ch for speed
    #style=1 is a raw plot, or 2 is a trace normalized plot
    #clim=[min,max] will clip the colormap to [min,max], deactivated by default
    
    Arguments:
    Required:
    st - The stream containing the DAS data to plot.
    Optional:
    style - Type of plot. Default is raw plot (style=1). style=2 is a trace normalized plot. 
    skip - The decimation in the spatioal domain. Default is 10. (int)
    clim - If specified, it is a list containing the lower and upper 
            limits of the colormap. Default is [], which specifies 
            that python defaults should be used. (list of 2 ints)
    tmin - Plot start time in seconds.
    tmax - Plot end time in seconds.
    physicalFiberLocations - Defines distance from header information.
    picks - DASpy event object. If specified, will plot phase picks 
            on the figure. Default is None, which specifies it is 
            unused. (DASpy detect.detect.event object)

    Returns:
    fig - A python figure object.
    """
    
    
    if fig==None:
        fig = plt.figure(figsize=(8,7))
#     fig = plt.figure()
    if style==1:
        img = stream2array(st[::skip]) # raw amplitudes
        clabel = st[0].stats.units
    if style==2:
        img = stream2array(st[::skip].copy().normalize()) # trace normalize
        clabel = st[0].stats.units+' (trace normalized)'

    t_ = st[0].stats.endtime-st[0].stats.starttime
    if physicalFiberLocations==True:
        extent = [st[0].stats.distance/1e3,st[-1].stats.distance/1e3,0,t_]
        if distance_unit=='m':
            extent = [st[0].stats.distance,st[-1].stats.distance,0,t_]
        xlabel = '{} [{}]'.format(xlabel,distance_unit) # relative to wellhead
    else:
        dx_ = st[1].stats.distance - st[0].stats.distance
        extent = [0,len(st)*dx_/1e3,0,t_]
        if distance_unit=='m':
            extent = [0,len(st)*dx_,0,t_]
        xlabel = 'Linear Fiber Length [{}]'.format(distance_unit)
    if len(clim)>1:
        plt.imshow(img.T,aspect='auto',interpolation='gaussian',alpha=0.7,
                   origin='lower',extent=extent,vmin=clim[0],vmax=clim[1],cmap="seismic");
    else:
        plt.imshow(img.T,aspect='auto',interpolation='gaussian',alpha=0.7,
                   origin='lower',extent=extent,cmap="seismic");
        
        
    try: 
        if distance_unit=='km':
            plt.scatter(picks.x/1000,picks.t,marker='_',c='k')
        if distance_unit=='m':
            plt.scatter(picks.x,picks.t,marker='_',c='k')
    except:
        pass
        
 


    h=plt.colorbar(pad=0.01);
    h.set_label(clabel,fontsize=12)
#     plt.ylim(np.max(extent),0)
    plt.ylabel('Time [s]',fontsize=14);
    plt.xlabel(xlabel,fontsize=14);
    
    if tmax:
        plt.ylim(tmax,tmin)
    else:
        plt.ylim(np.max(extent),tmin)

    # plt.gca().set_title(str(st[0].stats.starttime.datetime)+' - '+str(st[0].stats.endtime.datetime.strftime('%H:%M:%S'))+' (UTC)');

    plt.tight_layout();

    return fig

def median_remove(st):
    '''
    Remove median of all channels for each time step. Compared with removing mean, removing median does not generate zeros at zero wavenumber, which keeps f-k plot smooth.
    
    Arguments:
    st - Stream of DAS data to apply notch filter to (obspy stream)
    Returns:
    st_shp - median removed Stream.
    '''
    
    image=stream2array(st)
    
    image2=image - np.median(image, axis=0)
    
    st_shp=st.copy()
    for channel in range(len(image2)):
        st_shp[channel].data=image2[channel]
        
    return st_shp

def taper_cos_1D(nt_tot, ntap, nzeros):
        # if nt_tot is even, nzeros must be even and at least 2
        # if nt_tot is odd, nzeros must be odd and at least 1 
        n_half = ceil(nt_tot/2)
        tap_tmp = np.ones(n_half-(ceil(nzeros/2)-1))
        for i in range(0, ntap):
            val = 0.5*(1-np.cos((np.pi*i)/(ntap))) 
            tap_tmp[i] = val        
        tap_tmp = np.concatenate((np.zeros(ceil(nzeros/2)-1), tap_tmp))        
        
        if nt_tot%2 != 0:
            tap1d_out = np.concatenate((tap_tmp[:0:-1], tap_tmp))
        else:
            tap1d_out = np.concatenate((tap_tmp[::-1], tap_tmp))
        
        return tap1d_out

def fk_filter(st, wavenumber_limits=(-0.5, 0.5), freq_limits=(-1000, 1000), taper=0,resample=False,eta=1e-2,plot=True):
    '''
    FK filter for a 2D DAS numpy array. Returns a filtered image.
    Arguments:
    st - Stream of DAS data to apply notch filter to (obspy stream)
    wavenumber - maximum value for the filter  
    max_freq - maximum value for the filter  
    Returns:
    st_fk - FK filtered time series.
    17/March/2022: add tapering function before fk
    '''
    data=stream2array(st).T
    ns_t, ns_x = data.shape
    print(ns_t, ns_x)
    
    if taper>0:
        data = data * signal.tukey(ns_x, alpha=taper)
        data = (data.T * signal.tukey(ns_t, alpha=taper)).T
    
    fs=st[0].stats.sampling_rate
    ch_space=abs(st[1].stats.distance-st[0].stats.distance)

    # Detrend by removing the mean 
    data=data-np.mean(data)
    
    # Apply a 2D fft transform
    fftdata=np.fft.fftshift(np.fft.fft2(data.T))
    
    freqs=np.fft.fftfreq(ns_t,d=(1./fs))
    wavenums=2*np.pi*np.fft.fftfreq(ns_x,d=ch_space)
    
    freqs=np.fft.fftshift(freqs) 
    wavenums=np.fft.fftshift(wavenums)

    freqsgrid=np.broadcast_to(freqs,fftdata.shape)   
    print(freqsgrid.shape)

    wavenumsgrid=np.broadcast_to(wavenums,fftdata.T.shape).T
    print(wavenumsgrid.shape)
    
    mask=np.logical_and(np.logical_and(wavenumsgrid>=wavenumber_limits[0], wavenumsgrid<=wavenumber_limits[1]),
                            np.logical_and(freqsgrid>=freq_limits[0], freqsgrid<=freq_limits[1]))
    x=mask*1.

    blurred_mask = ndimage.gaussian_filter(x, sigma=3)
    
    if resample==True:
        # Convert from strain to velocity. Assumes DAS units are strain in m/m, ie *1e-9 and integrated.
        data_resample=np.zeros(fftdata.shape,dtype=np.complex_)
        tap1d = taper_cos_1D(len(wavenums), 10, 1)  

        for i in range(len(wavenums)):
            # data_resample[i]=(11.6*10**-9)*fftdata[i]*((-1*wavenums[i]+eta)/(2*np.pi*(freqs)+eta))
            # data_resample[i]=fftdata[i]*((-1*wavenums[i]+eta)/(2*np.pi*(freqs)+eta))
            data_resample[i]=fftdata[i]*((-2*np.pi*(freqs)+eta)/(wavenums[i]+eta))

        tap1d_2 = taper_cos_1D(len(wavenums), 10, 1)   
        tap2d_2 = np.repeat([tap1d_2],len(freqs),axis=0).T 
        
        data_resample = data_resample * tap2d_2
           
        ftimagep_tmp = data_resample * blurred_mask    
        ftimagep = np.fft.ifftshift(ftimagep_tmp)
        
        for tr in st:
            tr.stats.units="Velocity"
            
    else:
        # Apply the mask to the data and rettransform back to the time-domain
        ftimagep_tmp = fftdata * blurred_mask
    
        ftimagep = np.fft.ifftshift(ftimagep_tmp)
    
    if plot==True:
        # Plots the filter, with area remove greyed out
        max_val=np.max(abs(ftimagep_tmp)) 
        fig = plt.figure(figsize=[6,6])
        img1 = plt.imshow(abs(ftimagep_tmp), interpolation='bilinear',extent=[-fs/2,fs/2,-1/(2*ch_space),1/(2*ch_space)],aspect='auto')
        img1.set_clim(-1*max_val,max_val)
        img1 = plt.imshow(abs(blurred_mask-1),cmap='Greys',extent=[-fs/2,fs/2,-1/(2*ch_space),1/(2*ch_space)],alpha=0.2,aspect='auto')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Wavenumber (1/m)')
        xrange=300
        yrange=0.1
        plt.xlim(-1*xrange,xrange)
        plt.ylim(-1*yrange,yrange)
        plt.show()
    
    # Finally, take the inverse transform 
    imagep = np.fft.ifft2(ftimagep)
    imagep = imagep.real
    
    # Convert back to a stream        
    st_fk=st.copy()
    for channel in range(len(imagep)):
        st_fk[channel].data=imagep[channel]
        
        
    return st_fk
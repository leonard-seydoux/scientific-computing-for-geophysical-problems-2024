import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.util as cutil
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

def init_conf_plt(): 

    plt.rc('legend', fontsize=16)
    plt.rc('axes', labelsize=16,facecolor='None')
    plt.rc('savefig', facecolor='None', edgecolor='None')

def hammer_plot(SVZ, nlat, nlon, colat, longi, Title=None, obs_coords=None, central_long=None, filename=None, debug=False, unit=None):
    if debug is True: 
        print(np.shape(obs_coords))
    init_conf_plt()
    latit = 90 - colat
    lons, lats = np.meshgrid(longi,latit)
    svz2d =  np.reshape(SVZ, (nlat,nlon) )
    maxsvz = max(np.abs(SVZ))
    clevs = np.linspace(-maxsvz, maxsvz, 11, endpoint=True)
    fig = plt.figure()#figsize=(10,10))
    if central_long is None:
        central_long = 180
    ax = plt.axes( projection = ccrs.Hammer(central_longitude=central_long) )
    ax.set_global()
    ax.coastlines()
    m = ax.contourf(lons, lats, svz2d, levels=clevs, transform = ccrs.PlateCarree(), cmap="RdBu_r", extend='both')
    c = ax.contour(lons, lats,  svz2d, clevs, colors='k', linewidths=0.5,  transform = ccrs.PlateCarree())
    if obs_coords is not None:
        ax.scatter( obs_coords[0,:], obs_coords[1,:], marker="+", color='k', transform = ccrs.PlateCarree())
    cax = fig.add_axes([0.05, 0.08, .9, 0.05])
    cm = fig.colorbar(m, cax=cax, orientation='horizontal')
    cax.xaxis.set_label_position('bottom')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_tick_params(pad=0)
    if unit is not None:
        cm.set_label(unit)
    else: # default unit is SV unit
        cm.set_label("nT/yr")
    #font_size = 14 # Adjust as appropriate.
    #cm.ax.tick_params(labelsize=font_size)
    cm.ax.tick_params()
    if Title is not None:
        ax.set_title(Title)
#   
    if filename is not None: 
        plt.savefig(filename + ".png")
        plt.savefig(filename + ".pdf")
    plt.show()

def init_conf_plt_3(): 

   plt.rc('legend', fontsize=14)
   plt.rc('axes', labelsize=14,facecolor='None')
   plt.rc('savefig', facecolor='None', edgecolor='None')

def hammer_plot_3(SVX, SVY, SVZ, nlat, nlon, colat, longi, central_long=None, outfile_name=None):   

    init_conf_plt_3()

    fig = plt.figure()
    latit = 90 - colat
    lons, lats = np.meshgrid(longi,latit)
    
    gs = gridspec.GridSpec(3, 1)
    if central_long is None:
        central_long = 180. 
    ax1 = plt.subplot(gs[0, 0], projection = ccrs.Hammer(central_longitude=central_long) )
    ax2 = plt.subplot(gs[1, 0], projection = ccrs.Hammer(central_longitude=central_long) )
    ax3 = plt.subplot(gs[2, 0], projection = ccrs.Hammer(central_longitude=central_long) )

    iplot = 0
    SV = [SVX, SVY, SVZ]
    title = ["dX/dt", "dY/dt", "dZ/dt"]
    for iplot, ax in enumerate([ax1, ax2, ax3]):
        sv2d =  np.reshape(SV[iplot], (nlat,nlon) )
        maxsv = max(np.abs(SV[iplot]))
        clevs = np.linspace(-maxsv, maxsv, 11, endpoint=True)
        ax.set_global()
        ax.coastlines()
        try: 
            m = ax.contourf(lons, lats, sv2d, levels=clevs, transform = ccrs.PlateCarree(), cmap="RdBu_r", extend='both')
            c = ax.contour(lons, lats,  sv2d, clevs, colors='k', transform = ccrs.PlateCarree())
            cm = fig.colorbar(m, ax=ax)
            cm.set_label("nT/yr")
            font_size = 10 # Adjust as appropriate.
            cm.ax.tick_params(labelsize=font_size)
        except:
            #add a piece of text to indicate that the plot is empty
            ax.text(0.5, 0.5, 'No contour', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title[iplot], fontsize=font_size)
#
    plt.tight_layout()
    if outfile_name is not None:
        plt.savefig(outfile_name+'.png')
        plt.savefig(outfile_name+'.pdf')
    plt.show()

def plot_residuals(resid, nbins=20, Title=None, show=True, outfile_name=None, unit=None):
#   nbins is the number of bins used to represent the histogram

    fig = plt.figure()
    fig.subplots_adjust(top=0.98,left=0.08,right=0.9)
    ax1 = fig.add_subplot(111)
    ax1.hist(resid, bins=nbins, rwidth=0.9, edgecolor='None', facecolor='#348ABD')
    if unit is not None:
        ax1.set_xlabel(unit)
    else:
        ax1.set_xlabel(r'nT/yr')
    ax1.set_ylabel(r'counts')
    if Title is not None:
        ax1.set_title(Title)
    plt.tight_layout()
    if outfile_name is not None:
        plt.savefig(outfile_name+'.pdf')
        plt.savefig(outfile_name+'.png')
    if show is True:
        plt.show()

def plot_spectrum(matD,ll,mygh, show=True, outfile_name=None):

    init_conf_plt()
    mynorm = np.zeros(ll+1)
    lm = -1 
    for il in range(1,ll+1):
        for im in range (0,il): 
            if ( im == 0 ): 
                lm = lm + 1 
                mynorm[il] = mynorm[il] + matD[lm,lm] * mygh[lm]**2
            elif ( im > 0 ): 
                lm = lm + 1 
                mynorm[il] = mynorm[il] + matD[lm,lm] * mygh[lm]**2
                lm = lm + 1 
                mynorm[il] = mynorm[il] + matD[lm,lm] * mygh[lm]**2
 
    fig = plt.figure()
    fig.subplots_adjust(top=0.98,left=0.08,right=0.9)
    ax1 = fig.add_subplot(111)
    lrange = range(0,ll+1)
    print(np.shape(lrange))
    print(np.shape(mynorm))
    ax1.semilogy(lrange, mynorm, linestyle="-", marker="o", color='b')
    ax1.set_xlabel(r'spherical harmonic degree $\ell$')
    ax1.set_ylabel(r'squared mynorm (nT/yr)^2')
    ax1.xaxis.set_major_locator(MultipleLocator(1))  # Graduation majeure tous les 1 degré
    ax1.grid(which='both')
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    if outfile_name is not None:
        plt.savefig(outfile_name+'.pdf')
        plt.savefig(outfile_name+'.png')
    if show is True:
        plt.show()

def compute_spectrum(matD,ll,mygh):
#
    mynorm = np.zeros(ll+1)
    lm = -1 
    for il in range(1,ll+1):
        for im in range (0,il): 
            if ( im == 0 ): 
                lm = lm + 1 
                mynorm[il] = mynorm[il] + matD[lm,lm] * mygh[lm]**2
            elif ( im > 0 ): 
                lm = lm + 1 
                mynorm[il] = mynorm[il] + matD[lm,lm] * mygh[lm]**2
                lm = lm + 1 
                mynorm[il] = mynorm[il] + matD[lm,lm] * mygh[lm]**2
    return mynorm

if __name__ == "__main__":
    # Exemple d'utilisation
    ll = 9  # Exemple de limite supérieure
    ngh = ll * (ll + 2)  # Nombre de coefficients de Gauss
    matD = np.random.rand(ngh, ngh)  # Exemple de matrice D
    mygh = np.random.rand(ngh)  # Exemple de coefficients de Gauss
    plot_spectrum(matD, ll, mygh, show=True, outfile_name=None)

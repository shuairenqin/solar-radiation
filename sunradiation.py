#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 18:21:34 2018

@author: shuairenqin
"""

# this python procedure is used to compute incoming shortwave solar radiation
# based on Dr. Yang's hybrid model.
# reference: mproving estimation of hourly, daily, and monthly solar radiation
# by importing global data sets, Agric. For. Meteorol., 137(1-2), 43-55


import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from  sunposition import sunpos

#+++++++++++++++++++++++++ bilinear interpolation +++++++++++++++++++++++++++++

def bilinear_interpolation(x, y, points):
    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

#+++++++++++++++++++++++ load ozone climatology +++++++++++++++++++++++++++++++

with open("./parameter/ozone_monthly_nasa.txt", "r") as f:
    ozone_toms = []
    for row in f:
        ozone_toms.append(map(float,row.strip().split()[2:]))
ozone_toms = np.array(ozone_toms)

with open("./parameter/ozone_monthly_sunya.txt", "r") as f:
    ozone_sunya = []
    for row in f:
        ozone_sunya.append(map(float,row.strip().split()))
ozone_sunya = np.array(ozone_sunya)
    
ozone = {"toms":ozone_toms,"sunya":ozone_sunya}

#+++++++++++++++++++++++ load aerosol climatology +++++++++++++++++++++++++++++
    
beta_modis = []
for i in xrange(1,13):
    file_name = "./parameter/beta_modis"+str(i).zfill(2)+".txt"
    with open(file_name, "r") as f:
        beta_month = []
        for row in f:
            beta_month.append(map(float,row.strip().split()))
    beta_modis.append(beta_month)
beta_modis = np.array(beta_modis)

beta_gocart = []
for i in xrange(1,13):
    file_name = "./parameter/beta_gocart"+str(i).zfill(2)+".txt"
    with open(file_name, "r") as f:
        beta_month = []
        for row in f:
            beta_month.append(map(float,row.strip().split()))
    beta_gocart.append(beta_month)
beta_gocart = np.array(beta_gocart)

with open("./parameter/beta_mpi.txt", "r") as f:
    beta_gads = []
    for i, row in enumerate(f):
        if i<=1: continue
        beta_gads.append(map(float,row.strip().split()[3:]))
    beta_gads = np.moveaxis(np.array(beta_gads),(0,1),(1,0)).reshape((-1,37,72))
    new_dim = beta_gads[:,:,0][:,:,np.newaxis]
    beta_gads = np.concatenate((beta_gads,new_dim),axis=2)
    beta_gads ={"winter":beta_gads[0:8,:,:],"summer":beta_gads[8:,:,:]}
    
beta = {"modis":beta_modis,"gocart":beta_gocart,"gads":beta_gads}

 #+++++++++ model for computing surface incoming shortwave radiation ++++++++++   

class hybrid(object):
    
    p0 = 1.013e5 # air pressure at sea level [pascal]
    s0 = 1367.0 # solar constant at the mean sun-earth distance [w/m2]
    
    def __init__(self, lon, lat, z, lt, Ta, rh, sd, ps=None, ltz=0, step=20,
                 ozone ="SUNYA", aerosol="MODIS"):
        super(hybrid, self).__init__()
        
        self.lon = lon   # longitude [degrees]
        self.lat = lat   # latitude  [degrees]
        self.z = z       # elevation [meters]
        self.Ta = Ta     # surface air temperature [K]
        self.rh = rh     # surface relative humidity [%]
        self.sd = sd     # surface sunshine duration [hours]
        self.ps = ps     # surface air pressure [pascal]
        
        self.ltz = ltz           # local time zone from -11 to 14
        self.lt  = lt            # local time, datetime.datetime object
        
        self.step = timedelta(minutes = step) # calculas step [minutes]
        
        self.ozone   = ozone
        self.aerosol = aerosol
        
        self.utc = self.lt - relativedelta(hours = self.ltz)
        self._surface_pressure()
        self._pwv()
        self._ozone()
        self._beta()
        
    def _surface_pressure(self):
        # z [m]: elevation above sea level
        if not self.ps:
            self.ps = hybrid.p0*np.exp(-self.z/8430.)
        return None
    
    def _pwv(self):
        # rh [-]: relative humidiy
        # Ta [K]: surface air temperature
        self.w = 0.00493*self.rh/self.Ta*np.exp(26.23-5416.0/self.Ta)
        return None
    
    def _ozone(self):
        # empirical formula for ozone. consult Yang2005WRR       
        if self.ozone == "empirical":
            jd = self.utc.timetuple().tm_yday
            if jd<300:
                d = jd
            else:
                d = jd-366.
                self.l = 0.44 - 0.16*np.sqrt(((np.abs(self.lat)-80.0)/60.)**2
                                                 +((d-120.)/(263.-np.abs(self.lat)))**2)
        elif self.ozone == "TOMS":
            r = np.int(np.fix(np.abs(self.lat-90)/5.))
            t = self.utc.month-1
            self.l = ozone["toms"][r,t]
            
            
        elif self.ozone == "SUNYA":
            r = np.int(np.fix(np.abs(self.lat-90)))
            t = self.utc.month-1
            self.l = ozone["sunya"][r,t]
        
        return None
    
    def _beta(self):
        # empirical formula for beta. consult Yang2005WRR
        if self.aerosol == "empirical":
            self.beta = (0.025+0.1*np.cos(self.lat*np.pi/180)**2)*np.exp(-0.7*self.z/1000.) 
        elif self.aerosol == "MODIS":
            r = np.int(np.fix(np.abs(self.lat-90)/1.))
            c = np.int(np.fix(np.abs(self.lon-(-180))/1.))
            t = self.utc.month-1
            self.beta = beta["modis"][t,r,c]
        elif self.aerosol == "GOCART":
            r = np.int(np.fix(np.abs(self.lat-90)/1.))
            c = np.int(np.fix(np.abs(self.lon-(-180))/1.))
            t = self.utc.month-1
            self.beta = beta["gocart"][t,r,c]
        elif self.aerosol == "GADS":
            r = np.int(np.fix(np.abs(self.lat-90)/5.))
            c = np.int(np.fix(np.abs(self.lon-(-180))/5.))
            t = self.utc.month-1
            if t in {9, 10, 11, 12, 1, 2}:
                aerosol=beta["gads"]["winter"]
            else:
                aerosol=beta["gads"]["summary"]
                
            x, y = [], []
            for k, h in enumerate([0.,50.,70.,80.,90.,95.,98.,99.]):
                grids = []
                for i in [r, r+1]:
                    for j in [c, c+1]:
                        grids.append((90.0-i*5,-180.0+j*5,aerosol[k,i,c]))
                y.append(bilinear_interpolation(self.lat, self.lon, grids))
                x.append(h)
                
            if self.rh<x[1]:
                self.beta = (y[1]-y[0])/(x[1]-x[0])*(self.rh-x[0])+y[0]
            elif x[1]<self.rh<=x[2]:
                self.beta = (y[2]-y[1])/(x[2]-x[1])*(self.rh-x[1])+y[1]
            elif x[2]<self.rh<=x[3]:
                self.beta = (y[3]-y[2])/(x[3]-x[2])*(self.rh-x[2])+y[2]
            elif x[3]<self.rh<=x[4]:
                self.beta = (y[4]-y[3])/(x[4]-x[3])*(self.rh-x[3])+y[3]
            elif x[4]<self.rh<=x[5]:
                self.beta = (y[5]-y[4])/(x[5]-x[4])*(self.rh-x[4])+y[4]
            elif x[5]<self.rh<=x[6]:
                self.beta = (y[6]-y[5])/(x[6]-x[5])*(self.rh-x[5])+y[5]
            elif self.rh>x[6]:
                self.beta = (y[7]-y[6])/(x[7]-x[6])*(self.rh-x[6])+y[6]
                
                                
        return None
    
    def _air_mass(self):
        self.m=1.0/(np.cos(self.theta)+0.15*(57.296*(np.pi/2.-self.theta)+3.885)**-1.253)
        return None 
    
    def _corrected_air_mass(self):
        self.mc = self.m*self.ps/hybrid.p0
        return None
 
    def _taug(self):
        # extinction by gas absorption
        self.taug=np.exp(-0.0117*self.mc**0.3139)
        return None
    
    def _taur(self):
        # extinction by Rayleigh scattering
        self.taur = np.exp(-8.735e-3*self.mc*(0.547+0.014*self.mc-3.8e-4 \
                                              *self.mc**2+4.6e-6*self.mc**3)**-4.08)
        return None

    def _tauw(self):
        # extinction by water absorption
        self.tauw = min(1.0, 0.909-0.036*np.log(self.m*self.w))
        return None
    
    def _tauo(self):
        # extinction by ozone absorption
        self.tauo = np.exp(-3.65e-2*(self.m*self.l)**0.7136)
        return None
    
    def _taua(self):
        # extinction by aerosol absorption and scattering
        t = self.m*self.beta
        self.taua = np.exp(-t*(0.6777+0.1464*t-6.26e-3*t**2.0)**-1.3)
        return None
    
    def _tau_clear_beam(self):
        t = self.tauo*self.tauw*self.taug*self.taur*self.taua 
        self.tau_clear_beam = max(0.0, t-0.013)
        return None
    
    def _tau_clear_diff(self):
        t = self.tauo*self.taug*self.tauw*(1.0-self.taua*self.taur)
        self.tau_clear_diff = max(0.0, 0.5*t)
        return None
    
    
    def rad_clear_instant(self):
        
        self.utc = self.lt - relativedelta(hours=self.ltz)
        self.theta = sunpos(self.utc, self.lat, self.lon, self.z, radians=True)[1]
        if self.theta>=np.pi/2.0:
            rad_total, rad_beam, rad_diff, rad_norm = 0.0, 0.0, 0.0, 0.0
        else:
            self.jd = self.utc.timetuple().tm_yday       
            sun2earth = 1.0-0.01672*np.cos(0.9856*(self.jd-4.0)*np.pi/180.) # relative sun-earth distance [-]
            s  = hybrid.s0*sun2earth**2
            I0 = s*np.cos(self.theta)
            self._air_mass()
            self._corrected_air_mass()
            self._taug()
            self._taur()
            self._tauw()
            self._tauo()
            self._taua()
            self._tau_clear_beam()
            self._tau_clear_diff()
            rad_beam  = I0*self.tau_clear_beam
            rad_diff  = I0*self.tau_clear_diff
            rad_norm  = s *self.tau_clear_beam
            rad_total = rad_beam + rad_diff

        return rad_total, rad_beam, rad_diff, rad_norm
    
    def rad_clear_hourly(self):
        year, month, day, hour = self.lt.year, self.lt.month, self.lt.day, self.lt.hour
        start = datetime(year, month, day, hour)
        end   = start + relativedelta(hours=1)
        rad, counter = 0.0, 0
        while start<end:
            self.lt = start
            rad_total,_,_,rad_norm = self.rad_clear_instant()
            rad += rad_total*self.step.total_seconds()*1.0e-6
            start += self.step
            if rad_norm>120.0: counter += 1
        
        return rad, counter
    
    def rad_clear_daily(self):
        year, month, day, hour = self.lt.year, self.lt.month, self.lt.day, 0
        start = datetime(year, month, day, hour)
        end   = start + relativedelta(days=1)
        rad, counter = 0.0, 0
        while start<end:
            self.lt = start
            rad_total,_,_,rad_norm = self.rad_clear_instant()
            rad += rad_total*self.step.total_seconds()*1.0e-6
            start += self.step
            if rad_norm>120.0: counter += 1
            
        return rad, counter
    
    def rad_clear_monthly(self):
        year, month, day, hour = self.lt.year, self.lt.month, 1, 0
        start = datetime(year, month, day, hour)
        end   = start + relativedelta(months=1)
        rad, counter = 0.0, 0
        while start<end:
            self.lt = start
            rad_total,_,_,rad_norm = self.rad_clear_instant()
            rad += rad_total*self.step.total_seconds()*1.0e-6
            start += self.step
            if rad_norm>120.0: counter += 1
            
        return rad, counter
    
    def rad_allsky_hourly(self):
        rad, counter = self.rad_clear_hourly()
        if self.sd>0:
            rsd = self.sd/(counter*self.step.total_seconds()/3600.0)
            rad = rad*(0.4435 + 0.3976*rsd + 0.1589*rsd**2.0)
        else:
            rad = rad*0.2560
            
        return rad
    
    def rad_allsky_daily(self):
        rad, counter = self.rad_clear_daily()
        rsd = self.sd/(counter*self.step.total_seconds()/3600.0)        
        rad = rad*(0.2505 + 1.1468*rsd - 0.3974*rsd**2.0)
        
        return rad
    
    def rad_allsky_monthly(self):
        year, month, day, hour = self.lt.year, self.lt.month, 1, 0
        start = datetime(year, month, day, hour)
        end   = start + relativedelta(months=1)
        days  = (end - start).days
        rad, counter = self.rad_clear_monthly()
        rsd = self.sd*days/(counter*self.step.total_seconds()/3600.0)        
        rad = rad*(0.2777 + 0.8636*rsd - 0.1413*rsd**2.0)/days
        
        return rad
    
if __name__ == "__main__":
    
    lon, lat, z, lt, Ta, rh, sd = 137,30,20,datetime.now(), 273, 100, 6
    model = hybrid(lon, lat, z, lt, Ta, rh, sd, ltz=8, aerosol="GADS")
    print model.rad_clear_instant()
    
    
    
    
    
    
    
    
    
    

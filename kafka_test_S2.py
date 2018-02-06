#!/usr/bin/env python

import logging
logging.basicConfig( 
    level=logging.getLevelName(logging.DEBUG), 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="the_log.log")
import os
from datetime import datetime, timedelta
import numpy as np


import numpy as np

import gdal

import osr

import scipy.sparse as sp

# from multiply.inference-engine blah blah blah
try:
    from multiply_prior_engine import PriorEngine
except ImportError:
    pass


import kafka
from kafka.input_output import Sentinel2Observations, KafkaOutput
from kafka import LinearKalman
from kafka.inference import block_diag
from kafka.inference import propagate_information_filter_LAI
from kafka.inference import no_propagation
from kafka.inference import create_prosail_observation_operator



# Probably should be imported from somewhere else, but I can't see
# where from ATM... No biggy

def reproject_image(source_img, target_img, dstSRSs=None):
    """Reprojects/Warps an image to fit exactly another image.
    Additionally, you can set the destination SRS if you want
    to or if it isn't defined in the source image."""
    g = gdal.Open(target_img)
    geo_t = g.GetGeoTransform()
    x_size, y_size = g.RasterXSize, g.RasterYSize
    xmin = min(geo_t[0], geo_t[0] + x_size * geo_t[1])
    xmax = max(geo_t[0], geo_t[0] + x_size * geo_t[1])
    ymin = min(geo_t[3], geo_t[3] + y_size * geo_t[5])
    ymax = max(geo_t[3], geo_t[3] + y_size * geo_t[5])
    xRes, yRes = abs(geo_t[1]), abs(geo_t[5])
    if dstSRSs is None:
        dstSRS = osr.SpatialReference()
        raster_wkt = g.GetProjection()
        dstSRS.ImportFromWkt(raster_wkt)
    else:
        dstSRS = dstSRSs
    g = gdal.Warp('', source_img, format='MEM',
                  outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                  dstSRS=dstSRS)
    return g



###class DummyInferencePrior(_WrappingInferencePrior):
    ###"""
    ###This class is merely a dummy.
    ###"""

    ###def process_prior(self, parameters: List[str], time: Union[str, datetime], state_grid: np.array,

class SAILPrior(object):
    def __init__ (self, parameter_list, state_mask):
        self.parameter_list = parameter_list
        if isinstance(state_mask, (np.ndarray, np.generic) ):
            self.state_mask = state_mask
        else:
            self.state_mask = self._read_mask(state_mask)
    #parameter_list = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm',
     #                 'lai', 'ala', 'bsoil', 'psoil']
            self.mean = np.array([1.19, np.exp(-14.4/100.),
                                 np.exp(-4.0/100.), 0,
                                 np.exp(-50*0.68), np.exp(-100./21.0),
                                 np.exp(-3.97/2.),70./90., 0.5, 0.9])
            sigma = np.array([0.69, -0.016,
                                 -0.0086, 0,
                                 -1.71e-15, 0.017,
                                 0.20, 0.5, 0.5, 0.5])
            self.covar = np.diag(sigma**2).astype(np.float32)
            self.inv_covar = 1.0/self.covar
            self.inv_covar[3,3]=0
        ########self.mean = 
        ########self.variance = 
    ########lai_m2_m2: [3.1733 1.7940]

    ########cab_ug_cm2: [14.4369 1.8284]

     ########car_u_cm2: [3.9650 0.8948]

     ########cdm_g_cm2: [20.9675 6.4302]

         ########cw_cm: [0.6781 0.6749]

            ########N_: [1.1937 0.6902]    
        
    def _read_mask(self, fname):
        """Tries to read the mask as a GDAL dataset"""
        if not os.path.exists(fname):
            raise IOError("State mask is neither an array or a file that exists!")
        g = gdal.Open(fname)
        if g is None:
            raise IOError("{:s} can't be opened with GDAL!".format(fname))
        mask = g.ReadAsArray()
        return mask

    def process_prior ( self, time, inv_cov=True):
        # Presumably, self._inference_prior has some method to retrieve 
        # a bunch of files for a given date...
        n_pixels = self.state_mask.sum()
        x0 = np.array([self.mean for i in xrange(n_pixels)]).flatten()
        if inv_cov:
            inv_covar_list = [self.inv_covar for m in xrange(n_pixels)]
            inv_covar = block_diag(inv_covar_list, dtype=np.float32)
            return x0, inv_covar
        else:
            covar_list = [self.covar for m in xrange(n_pixels)]
            covar = block_diag(covar_list, dtype=np.float32)
            return x0, covar
        
        
        

class KafkaOutputMemory(object):
    """A very simple class to output the state."""
    def __init__(self, parameter_list):
        self.parameter_list = parameter_list
        self.output = {}
    def dump_data(self, timestep, x_analysis, P_analysis, P_analysis_inv,
                state_mask):
        solution = {}
        for ii, param in enumerate(self.parameter_list):
            solution[param] = x_analysis[ii::7]
        self.output[timestep] = solution


if __name__ == "__main__":
    
    parameter_list = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm',
                      'lai', 'ala', 'bsoil', 'psoil']
    
    start_time = "2017001"
    
    emulator_folder = "/home/ucfafyi/DATA/Multiply/emus/sail/"
    
    data_folder = "/data/nemesis/S2_data/30/S/WJ/"

    state_mask = "./Barrax_pivots.tif"
    
    s2_observations = Sentinel2Observations(data_folder,
                                            emulator_folder, 
                                            state_mask)

    projection, geotransform = s2_observations.define_output()

    output = KafkaOutput(parameter_list, geotransform,
                         projection, "S2out_test")



    the_prior = SAILPrior(parameter_list, state_mask)

    #output = []
    g = gdal.Open(state_mask)
    mask = g.ReadAsArray().astype(np.bool)

    class oot(object):
        def __init__ (self):
            pass
        
        def dump_data(*opts):
            pass
        
    ooot = oot()


    kf = LinearKalman(s2_observations, output, mask,
                      create_prosail_observation_operator,
                      parameter_list,
                      state_propagation=None,
                      prior=the_prior,
                      linear=False)
    

    # Get starting state... We can request the prior object for this
    x_forecast, P_forecast_inv = the_prior.process_prior(None)
    
    Q = np.zeros_like(x_forecast)
    ###Q[6::7] = 0.025
    
    kf.set_trajectory_model()
    kf.set_trajectory_uncertainty(Q)
    
    base = datetime(2017,1,1)
    num_days = 60
    time_grid = list((base + timedelta(days=x) 
                  for x in range(0, num_days, 16)))
    kf.run(time_grid, x_forecast, None, P_forecast_inv,
            iter_obs_op=True)
    


    

#!/usr/bin/env python
import _pickle as cPickle
import datetime
import glob
import os
import sys

import numpy as np
import scipy.sparse as sp # Required for unc
import gdal
import osr

import xml.etree.ElementTree as ET
from collections import namedtuple

def parse_xml(filename):
    """Parses the XML metadata file to extract view/incidence 
    angles. The file has grids and all sorts of stuff, but
    here we just average everything, and you get 
    1. SZA
    2. SAA 
    3. VZA
    4. VAA.
    """
    with open(filename, 'r') as f:
        tree = ET.parse(filename)
        root = tree.getroot()

        vza = []
        vaa = []
        for child in root:
            for x in child.findall("Tile_Angles"):
                for y in x.find("Mean_Sun_Angle"):
                    if y.tag == "ZENITH_ANGLE":
                        sza = float(y.text)
                    elif y.tag == "AZIMUTH_ANGLE":
                        saa = float(y.text)
                for s in x.find("Mean_Viewing_Incidence_Angle_List"):
                    for r in s:
                        if r.tag == "ZENITH_ANGLE":
                            vza.append(float(r.text))
                            
                        elif r.tag == "AZIMUTH_ANGLE":
                            vaa.append(float(r.text))
                            
    return sza, saa, np.mean(vza), np.mean(vaa)


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
    if g is None:
        raise ValueError("Something failed with GDAL!")
    return g


# NP the observation operator returns a named tuple so this is needed
planetlabs_data = namedtuple('planetlabs_data',
                             'observations uncertainty mask metadata emulator')


class PlanetlabsObservations(object):
    # NP This is the class that needs modifying/writing for planetlabs
    # it needs an _init_ and it needs get_band_data(...). The signature of
    # get_band_data(..) is fixed but you can change the input values of
    # the init to contain anything you need. An other example of an
    # observation operator that has a different signature is the class
    # BHRObservations in the file observations.py

    def __init__(self, parent_folder, emulator_folder, state_mask):

        # This is tailored to the native S2 file structure with
        # parent folder being the base folder where all the data is.
        if not os.path.exists(parent_folder):
            raise IOError("S2 data folder doesn't exist")
        self.parent = parent_folder


        self.emulator_folder = emulator_folder

        # NP This is a static mask, i.e. the same for all time-steps
        # and independent of the cloud mask or any time varying mask
        self.state_mask = state_mask

        # NP What the function below does is set:
        # self.dates < a list of datetime objects
        # self.date_data < a dictionary whose keys are the dates above
        #           and values (for S2) are the folders where the relevent
        #           band data is. You might need filenames or some other thing
        # self.bands_per_observation < dictionary whose keys are the dates
        #           and values are the number of bands for that observation.
        #           for S2 this is fixed at 10 for every observation. That might
        #           make sense for you too.
        self._find_granules(self.parent)

        # NP band_map is used to map internal band numbers on to
        # values in the data file names. Internally the bands have
        # sequential integer numbers. For sentinel 2 these are the
        # strings that are found in the file names of the bands that
        # we use. Can be ditched or modified as needed.
        self.band_map = ['02', '03', '04', '05', '06', '07',
                         '08', '8A', '09', '12']

        # NP Locating emulators. Shouldn't need changing.
        emulators = glob.glob(os.path.join(self.emulator_folder, "*.pkl"))
        emulators.sort()
        self.emulator_files = emulators

    def define_output(self):
        # NP I can't see that this is used anywhere at the moment...
        g = gdal.Open(self.state_mask)
        proj = g.GetProjection()
        geoT = np.array(g.GetGeoTransform())
        #new_geoT = geoT*1.
        #new_geoT[0] = new_geoT[0] + self.ulx*new_geoT[1]
        #new_geoT[3] = new_geoT[3] + self.uly*new_geoT[5]
        return proj, geoT.tolist() #new_geoT.tolist()


    def _find_granules(self, parent_folder):
        """Finds granules. Currently does so by checking for
        Feng's AOT file."""
        #NP
        self.dates = []
        self.date_data = {}
        for root, dirs, files in os.walk(parent_folder):
            for fich in files:
                if fich.find("aot.tif") >= 0:
                    this_date = datetime.datetime(*[int(i) 
                                for i in root.split("/")[-4:-1]])
                    self.dates.append(this_date)
                    self.date_data[this_date] = root
        self.bands_per_observation = {}
        for the_date in self.dates:
            self.bands_per_observation[the_date] = 10 # 10 bands


    def _find_emulator(self, sza, saa, vza, vaa):
        raa = vaa - saa
        vzas = np.array([float(s.split("_")[-3]) 
                         for s in self.emulator_files])
        szas = np.array([float(s.split("_")[-2]) 
                         for s in self.emulator_files])
        raas = np.array([float(s.split("_")[-1].split(".")[0]) 
                         for s in self.emulator_files])        
        e1 = szas == szas[np.argmin(np.abs(szas - sza))]
        e2 = vzas == vzas[np.argmin(np.abs(vzas - vza))]
        e3 = raas == raas[np.argmin(np.abs(raas - raa))]
        iloc = np.where(e1*e2*e3)[0][0]
        return self.emulator_files[iloc]


    def get_band_data(self, timestep, band):
        # NP This is the only function that is needed by the rest of
        # the KaFKA code and must have this signature. All other
        # functions here are called by this one and can be changed
        # or renamed as appropriate although some can be reused
        # as they are.
        # band is an integer between 0 and (Nbands-1)
        # timestep is a date time object


        # NP The next 4 lines are just about getting the angles from
        # the meta data. The angles are used by the emulators
        # and in the returned object so best to keep the dictionary format.
        # For S2 each observation date has it's own folder and
        # that is what is saved in self.date_data. You might need
        # a different system
        current_folder = self.date_data[timestep]
        meta_file = os.path.join(current_folder, "metadata.xml")
        sza, saa, vza, vaa = parse_xml(meta_file)
        metadata = dict(zip(["sza", "saa", "vza", "vaa"],
                            [sza, saa, vza, vaa]))


        # NP This is reading in the emulators (stored as pickles)
        # It shouldn't need changing except for if we generate the
        # emulators using python 3 (the encoding variable is there
        # because existing emulator pickles were written in python 2.
        emulator_file = self._find_emulator(sza, saa, vza, vaa)
        emulator = cPickle.load( open (emulator_file, 'rb'),
                                 encoding='latin1' )


        # Read and reproject S2 surface reflectance

        # NP get the file name for this band for this timestep.
        # band_map maps the internal band number to a string that
        # identifies the file name.
        the_band = self.band_map[band]
        original_s2_file = os.path.join ( current_folder, 
                                         "B{}_sur.tif".format(the_band))
        print(original_s2_file)


        # NP this is here because in the future we may have observations
        # on a different grid to the grid we retrieve biophysical parameters
        # on. Jose currently uses the state_mask to define that grid and
        # reprojects on to that. state_mask is the static grid that is the
        # same for every timestep (i.e. your field mask)
        # Probably this function will work on the planet labs data (just
        # uses gdal) but worth checking
        g = reproject_image( original_s2_file, self.state_mask)

        # NP Yay! lets read in the reflectance data!
        rho_surface = g.ReadAsArray()

        # NP Apply you wonderful new cloud/good obs mask here!
        mask = rho_surface > 0
        rho_surface = np.where(mask, rho_surface/10000., 0)


        # Read and reproject S2 angles

        # NP I presume that this is to map internal band numbering onto
        # band numbering in the emulator list. When we've made the emulators we
        # need to make this consistent.
        emulator_band_map = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]
        
        # NP Your uncertainties go here. Looks like set to zero where masked
        R_mat = rho_surface*0.05
        R_mat[np.logical_not(mask)] = 0.

        # This is calculating the inverse of the covariance matrix and shouldn't
        # need modifying
        N = mask.ravel().shape[0]
        R_mat_sp = sp.lil_matrix((N, N))
        R_mat_sp.setdiag(1./(R_mat.ravel())**2)
        R_mat_sp = R_mat_sp.tocsr()

        # NP I'll have to get back to you on this... Hopefully all becomes clear
        # when we've got the emulators. I think it will stay about the same though.
        s2_band = bytes("S2A_MSI_{:02d}".format(emulator_band_map[band]), 'latin1' )

        # Create the named tuple (see top of file) to be returned
        s2data = planetlabs_data(rho_surface, R_mat_sp, mask, metadata, emulator[s2_band])

        return s2data


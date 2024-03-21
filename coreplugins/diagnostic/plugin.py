import os

from io import BytesIO

from app.plugins import PluginBase, Menu, MountPoint
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.utils.translation import gettext as _
# import cv2
import json, shutil
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import os
import tempfile
from django import forms
import shutil
import numpy as np
import matplotlib.pyplot as plt
import argparse
import textwrap
from PIL import Image
import matplotlib as mpl
from osgeo import gdal, osr
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from app.api.imageuploads import normalize
from django.core.files.uploadhandler import TemporaryFileUploadHandler

from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST
# from .scripts import calculate_vegetative_indices  # Import the script function


class CustomTemporaryFileUploadHandler(TemporaryFileUploadHandler):
    def file_complete(self, file_size):
        self.file.seek(0)
        self.file.size = file_size
        self.file.close()  # Close the file as not to hog the number of open files descriptors
        return self.file


class VegetativeIndicesForm(forms.Form):
    input_image = forms.ImageField(label='Upload Image')
    output_directory = forms.CharField(label='Output Directory')
    indices = forms.MultipleChoiceField(
        label='Indices to Calculate',
        choices=[('VARI', 'VARI'), ('GLI', 'GLI'), ('NGRDI', 'NGRDI'), ('NGBDI', 'NGBDI')],
        widget=forms.CheckboxSelectMultiple,
        required=False
    )


class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='Select an image')
    # output_directory = forms.CharField(label='Output directory', max_length=100)


class Plugin(PluginBase):
    def main_menu(self):
        return [Menu(_("Problematic Areas Detection"), self.public_url(""), "fa fa-chart-pie fa-fw")]

    def app_mount_points(self):
        @login_required
        def display_form(request):
            form = VegetativeIndicesForm()
            return render(request, self.template_path("vegetative_indices.html"), {'form': form})

        @login_required
        def process_vegetative_indices(request):
            if request.method == 'POST':
                form = ImageUploadForm(request.POST, request.FILES)
                if form.is_valid():
                    input_image = request.FILES['image']
                    # indexes = calculate_vegetative_indices(input_image)
                    return render(request, self.template_path('vegetative_indices_results.html'), {})
            else:
                form = ImageUploadForm()
            return render(request, self.template_path('vegetative_indices_results.html'), {'form': form})

        return [
            MountPoint('', display_form),

            MountPoint('$', process_vegetative_indices)
            ]

    


# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib as mpl
# from osgeo import gdal, osr

# class Indexes:
#     def __init__(self, img):
#         self.img = img
#         self.R = self.img[:, :, 2].astype(np.float32)
#         self.G = self.img[:, :, 1].astype(np.float32)
#         self.B = self.img[:, :, 0].astype(np.float32)

#     def VARI(self):
#         vari = np.divide((self.G - self.R), (self.G + self.R - self.B + 0.00001))
#         return np.clip(vari, -1, 1)

#     def GLI(self):
#         gli = np.divide((2 * self.G - self.R - self.B), (2 * self.G + self.R + self.B + 0.00001))
#         return np.clip(gli, -1, 1)

#     def NGRDI(self):  
#         v_ndvi = np.divide((self.G - self.R), (self.G + self.R + 0.00001))
#         return np.clip(v_ndvi, -1, 1)

#     def NGBDI(self): 
#         ngbdi = (self.G - self.B) / (self.G + self.B + 0.00001) 
#         return np.clip(ngbdi, -1, +1)

#     def get_index(self, index_name):
#         if index_name == 'VARI':
#             return self.VARI()
#         elif index_name == 'GLI':
#             return self.GLI()
#         elif index_name == 'NGRDI':
#             return self.NGRDI()
#         elif index_name == 'NGBDI':
#             return self.NGBDI()
#         else:
#             print('Unknown index')

# def find_real_min_max(perc, edges, index_clear):
#     mask = perc > (0.05 * len(index_clear))
#     edges = edges[:-1]
#     min_v = edges[mask].min()
#     max_v = edges[mask].max()
#     return min_v, max_v

# def array_to_raster(output_path, ds_reference, array, name1, name2):
#     rows, cols, band_num = array.shape

#     driver = gdal.GetDriverByName("GTiff")

#     outRaster = driver.Create(os.path.join(output_path, name1 + '_' + name2 + '.tif'), cols, rows, band_num, gdal.GDT_Byte, options=["COMPRESS=DEFLATE"])
#     originX, pixelWidth, b, originY, d, pixelHeight = ds_reference.GetGeoTransform()
#     outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

#     descriptions = ['Red Band', 'Green Band', 'Blue Band', 'Alpha Band', 'Index Array']
#     for b in range(band_num):
#         outband = outRaster.GetRasterBand(b + 1)
#         outband.WriteArray(array[:, :, b])
#         outband.SetDescription(descriptions[b])
#         if b + 1 == 1:
#             outRaster.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
#         elif b + 1 == 2:
#             outRaster.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
#         elif b + 1 == 3:
#             outRaster.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
#         elif b + 1 == 4:
#             outRaster.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
        
#     outRasterSRS = osr.SpatialReference(wkt=prj)
#     outRaster.SetProjection(outRasterSRS.ExportToWkt())
#     driver = None
#     outband.FlushCache()

#     print('Georeferenced {} map was extracted!'.format(index_name))

#     return outRaster

# def calculate_vegetative_indices(input_image, output_path=None, indices=None):
#     print("entered into calcucation")
#     if output_path is None: 
#         os.makedirs('results', exist_ok=True)
#         save_dir = os.path.join(os.getcwd(), 'results')
#     else:
#         save_dir = os.path.abspath(output_path)

#     if indices is None:
#         indices = ['VARI', 'GLI', 'NGRDI', 'NGBDI']
#         print('All VIs will be calculated!')
#     else:
#         indices = [elem.upper() for elem in indices]

#     img_path = os.path.abspath(input_image)
#     img_name = os.path.basename(img_path)
#     name, ext = os.path.splitext(img_name)

#     os.chdir(os.path.dirname(img_path))

#     img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
#     h, w, ch = img.shape

#     if ch > 3:
#         image = img[:, :, :3].astype(float)
#         image[img[:, :, 3] == 0] = np.nan
#         empty_space = img[:, :, 3] == 0
#     else:
#         image = img

#     print('Processing image with shape {} x {}'.format(img.shape[0], img.shape[1]))

#     Idx = Indexes(image)

#     for index_name in indices:
#         idx = Idx.get_index(index_name)

#         index_clear = idx[~np.isnan(idx)]

#         perc, edges, _ = plt.hist(index_clear, bins=100, range=(-1, 1), color='darkcyan', edgecolor='black')
#         plt.close()

#         lower, upper = find_real_min_max(perc, edges, index_clear)
#         index_clipped = np.clip(idx, lower, upper)
        
#         cm = plt.get_cmap('RdYlGn')
#         cNorm = mpl.colors.Normalize(vmax=upper, vmin=lower)
#         colored_image = cm(cNorm(index_clipped))

#         img = Image.fromarray(np.uint8(colored_image * 255), mode='RGBA')
        
#         rgba = np.array(img, dtype=np.float32)

#         ds = gdal.Open(img_path, gdal.GA_ReadOnly)
#         prj = ds.GetProjection()

#         if prj: 
#             array_to_raster(save_dir, ds, rgba, name, index_name)    
#         else:
#             img.save('{}/{}_{}.tif'.format(save_dir, name, index_name))
#             print('Non georeferrenced {} map was extracted!'.format(index_name))

#         np.save('{}/{}_{}.npy'.format(save_dir, name, index_name), index_clipped)
        
#     print('Done!')

# # Example usage:
# # calculate_vegetative_indices(input_image='input_image.tif', output_path='output_folder', indices=['VARI', 'GLI'])

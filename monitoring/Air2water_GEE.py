import ee
import folium
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from django.conf import settings

def MNDWICalculator(green, SWIR1, NIR, red, blue, image):

    MNDWIimage = green.expression('(green-swir1)/(green+swir1)', {
        'green': green,
        'swir1': SWIR1
    })
    NDVIimage = NIR.expression('(NIR-red)/(NIR+red)', {
        'NIR': NIR,
        'red': red
    })
    EVIimage = NIR.expression('2.5 * (NIR - red) / (1 + NIR + 6 * red - 7.5 * blue)', {
        'NIR': NIR,
        'red': red,
        'blue': blue
    })
    water = (MNDWIimage.gt(NDVIimage).Or(MNDWIimage.gt(EVIimage))).And(EVIimage.lt(0.1))
    waterMasked = water.updateMask(water.gt(0))
    return image.mask(waterMasked)

def mapMNDWIlandsat(image):
    green = image.select(1)
    SWIR1 = image.select(4)
    NIR = image.select(3)
    red = image.select(2)
    blue = image.select(0)
    return MNDWICalculator(green, SWIR1, NIR, red, blue, image)


def mapMNDWIsentinel(image):
    green = image.select(1)
    SWIR1 = image.select(8)
    NIR = image.select(6)
    red = image.select(2)
    blue = image.select(0)
    return MNDWICalculator(green, SWIR1, NIR, red, blue, image)

def CIcalc(red, green):
    redsubgreen = red.subtract(green)
    redaddgreen = red.add(green)
    return redsubgreen.divide(redaddgreen)

class Air2water_monit:

    def __init__(self, start_date=None, end_date=None, long=None, lat=None, cc=None, satellite=None, variable=None, service_account=None, service_key=None, user_id=0, group_id=0, sim_id=0):
        self.start_date= start_date
        self.end_date= end_date
        self.long= long
        self.lat= lat
        self.cc= cc
        self.satellite=satellite
        self.variable=variable
        self.service_account=service_account
        self.service_key=service_key
        self.user_id=user_id
        self.group_id=group_id
        self.sim_id=sim_id

    def atmosphericcorrection_landsat(self, L8, geometry, start_date, end_date, cc):
        filter = L8.filterBounds(geometry).filterDate(start_date, end_date).filterMetadata('CLOUD_COVER', 'less_than',
                                                                                         cc);  # .select(ee.List.sequence(0,10));#selecting first 11 bands only as the 12th band is quality assessment band which has no application in this atmospheric correction code

        def maskclouds(image):
            qa = image.select('QA_PIXEL')

            # Bits 10 and 11 are clouds and cirrus, respectively.
            cloudBitMask = 1 << 2
            cirrusBitMask = 1 << 3

            # Both flags should be set to zero, indicating clear conditions.
            mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
                .And(qa.bitwiseAnd(cirrusBitMask).eq(0))

            return image.updateMask(mask)

        filter = filter.map(maskclouds)
        filter = filter.select(ee.List.sequence(0, 10))
        print('Original Image:', filter)
        count = filter.size()
        # If 0 elements error
        errorgen = ee.Number(count)
        print('Number of images to be processed:', errorgen)

        if (errorgen.getInfo()):
            # TOA Reflectance
            def TOAReflectanceCalc(i):
                i = ee.Number(i).toInt()
                image = ee.Image(filterlist.get(i))
                return ee.Algorithms.Landsat.TOA(image)

            def divide10000(image):
                image = image.toFloat()
                return image.divide(10000)

            TOAReflectance = filter.map(divide10000)
            TOAcount = TOAReflectance.size()
            TOAReflectancelist = ee.List(TOAReflectance.toList(TOAcount))

            print('TOAReflectance:', TOAReflectance)

            listTOAref = ee.List(TOAReflectance.toList(TOAReflectance.size()))
            listTOArefsize = listTOAref.size()
            iterator = ee.List.sequence(0, listTOArefsize.subtract(1))

            bluereflectance = TOAReflectance.select(1).toList(TOAReflectance.size())
            greenreflectance = TOAReflectance.select(2).toList(TOAReflectance.size())
            redreflectance = TOAReflectance.select(3).toList(TOAReflectance.size())
            NIRreflectance = TOAReflectance.select(4).toList(TOAReflectance.size())

            def bluesurfacereflectancecalculator(l):
                l = ee.Number(l).toInt()
                blueimage = ee.Image(bluereflectance.get(l))
                # bluescattervalue= ee.Number(bluescatter.get(l))
                # bluescatterimage= ee.Image.constant(ee.List.repeat(bluescattervalue,blueimage.bandNames().length()))
                return blueimage;  # .subtract(bluescatterimage)

            def greensurfacereflectancecalculator(l):
                l = ee.Number(l).toInt()
                greenimage = ee.Image(greenreflectance.get(l))
                # greenscattervalue= ee.Number(greenscatter.get(l))
                # greenscatterimage= ee.Image.constant(ee.List.repeat(greenscattervalue,greenimage.bandNames().length()))
                return greenimage;  # .subtract(greenscatterimage)

            def redsurfacereflectancecalculator(l):
                l = ee.Number(l).toInt()
                redimage = ee.Image(redreflectance.get(l))
                # redscattervalue= ee.Number(redscatter.get(l))
                # redscatterimage= ee.Image.constant(ee.List.repeat(redscattervalue,redimage.bandNames().length()))
                return redimage;  # .subtract(redscatterimage)

            def NIRsurfacereflectancecalculator(l):
                l = ee.Number(l).toInt()
                NIRimage = ee.Image(NIRreflectance.get(l))
                # NIRscattervalue= ee.Number(NIRscatter.get(l))
                # NIRscatterimage= ee.Image.constant(ee.List.repeat(NIRscattervalue,NIRimage.bandNames().length()))
                return NIRimage;  # .subtract(NIRscatterimage)

            blueSR = iterator.map(bluesurfacereflectancecalculator)
            greenSR = iterator.map(greensurfacereflectancecalculator)
            redSR = iterator.map(redsurfacereflectancecalculator)
            NIRSR = iterator.map(NIRsurfacereflectancecalculator)

            def compositeSRmaker(i):
                blueSRi = ee.Image(blueSR.get(i)).rename('B2')
                greenSRi = ee.Image(greenSR.get(i)).rename('B3')
                redSRi = ee.Image(redSR.get(i)).rename('B4')
                NIRSRi = ee.Image(NIRSR.get(i)).rename('B5')
                SWIR1i = ee.Image(TOAReflectancelist.get(i)).select(5).rename('B6')
                SWIR2i = ee.Image(TOAReflectancelist.get(i)).select(6).rename('B7')
                return ee.Image([blueSRi, greenSRi, redSRi, NIRSRi, SWIR1i, SWIR2i])

            compositeSR = ee.ImageCollection(iterator.map(compositeSRmaker))
            print('Surface Reflectance for Blue, Green, Red, NIR, SWIR 1 and SWIR2:', compositeSR)
        return compositeSR

    def atmosphericcorrection_sentinel(self, S2, geometry, startdate, enddate, cc):
        # Filter the image collection
        filter = (S2.filterBounds(geometry)
                  .filterDate(startdate, enddate)
                  .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cc))

        def maskS2clouds(image):
            qa = image.select('QA60')
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11
            mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            return image.updateMask(mask)

        filter = filter.map(maskS2clouds)
        filter = filter.select([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        count = filter.size().getInfo()
        print(f'Number of images to be processed: {count}')

        def divide10000(image):
            return image.toFloat().divide(10000)

        TOAReflectance = filter.map(divide10000)

        # List of TOA Reflectance Images
        TOAReflectance_list = TOAReflectance.toList(TOAReflectance.size())

        def create_composite(i):
            blueSRi = ee.Image(TOAReflectance_list.get(i)).select('B2').rename('B2')
            greenSRi = ee.Image(TOAReflectance_list.get(i)).select('B3').rename('B3')
            redSRi = ee.Image(TOAReflectance_list.get(i)).select('B4').rename('B4')
            rededge1SRi = ee.Image(TOAReflectance_list.get(i)).select('B5').rename('B5')
            rededge2SRi = ee.Image(TOAReflectance_list.get(i)).select('B6').rename('B6')
            rededge3SRi = ee.Image(TOAReflectance_list.get(i)).select('B7').rename('B7')
            NIRSRi = ee.Image(TOAReflectance_list.get(i)).select('B8').rename('B8')
            rededge4SRi = ee.Image(TOAReflectance_list.get(i)).select('B8A').rename('B8A')
            SWIR1SRi = ee.Image(TOAReflectance_list.get(i)).select('B11').rename('B11')
            SWIR2SRi = ee.Image(TOAReflectance_list.get(i)).select('B12').rename('B12')
            return ee.Image(
                [blueSRi, greenSRi, redSRi, rededge1SRi, rededge2SRi, rededge3SRi, NIRSRi, rededge4SRi, SWIR1SRi,
                 SWIR2SRi])

        compositeSR = ee.ImageCollection(ee.List.sequence(0, count - 1).map(create_composite))

        print(
            'Surface Reflectance for Blue, Green, Red, Red Edge (Band 5), Red Edge (Band 6), Red Edge (Band 7), NIR, Red Edge (Band 8a), SWIR (Band 11) and SWIR (Band 12):',
            compositeSR.getInfo())
        return compositeSR

    def CI_Landsat(self, imageCollection):

        def mapCIlandsat(image):
            red = image.select(2)
            green = image.select(1)
            return CIcalc(red,green)


        MNDWI = imageCollection.map(mapMNDWIlandsat)
        MNDWIextract = MNDWI.select('B[2-8]')
        print('MNDWI Images', MNDWIextract)

        NDCI = MNDWI.map(mapCIlandsat).select(['B4'], ['Chlorophyllindex'])

        return ee.Feature(None, {
            'MNDWIimage': MNDWIextract,
            'Chlorophyllindex': NDCI
        })

    def CI_Sentinel(self, imageCollection):
        def mapCIsentinel(image):
            rededge = image.select(3)
            red = image.select(2)
            return CIcalc(rededge,red)


        MNDWI = imageCollection.map(mapMNDWIsentinel)
        MNDWIextract = MNDWI.select('B[1-9]')
        print('MNDWI Images', MNDWIextract)

        NDCI = MNDWI.map(mapCIsentinel).select(['B5'], ['Chlorophyllindex'])

        return ee.Feature(None, {
            'MNDWIimage': MNDWIextract,
            'Chlorophyllindex': NDCI
        })

    def TI_Sentinel(self, imageCollection):
        def mapCIsentinel(image):
            green = image.select(1)
            red = image.select(2)
            return CIcalc(red,green)


        MNDWI = imageCollection.map(mapMNDWIsentinel)
        MNDWIextract = MNDWI.select('B[1-9]')
        print('MNDWI Images', MNDWIextract)

        NDTI = MNDWI.map(mapCIsentinel).select(['B4'], ['Turbidityindex'])

        return ee.Feature(None, {
            'MNDWIimage': MNDWIextract,
            'Turbidityindex': NDTI
        })

    def DO_Landsat(self,imageCollection):
        def DOcalc(image):
            blue = image.select(0)
            green = image.select(1)
            NIR = image.select(3)

            bluebyNIR = blue.divide(NIR)
            greenbyNIR = green.divide(NIR)

            constant = ee.Image.constant(8.2)
            DO = (constant.subtract(bluebyNIR.multiply(ee.Number(0.15)))
                  .add(greenbyNIR.multiply(ee.Number(0.32))))

            return DO.rename('Dissolvedoxygen')

        MNDWI = imageCollection.map(mapMNDWIlandsat)
        MNDWIextract = MNDWI.select('B[2-8]')
        print('MNDWI Images', MNDWIextract)

        DO = MNDWI.map(DOcalc).select(['Dissolvedoxygen'])

        return ee.Feature(None, {'Dissolvedoxygen': DO})

    def DO_Sentinel(self,imageCollection):
        def DOcalc(image):
            rededge1 = image.select(3)
            narrowNIR = image.select(7)
            red = image.select(2)
            rededge3 = image.select(5)
            NIR = image.select(6)

            NIRsubred = NIR.subtract(red)
            NIRaddred = NIR.add(red)

            fraction1 = rededge1.divide(red.add(narrowNIR))
            fraction2 = red.divide(rededge3.subtract(NIR))

            constant = ee.Image.constant(1.687)
            DO = (constant.add(fraction1.multiply(ee.Number(13.65)))
                  .subtract(fraction2.multiply(ee.Number(0.3714))))

            return DO.rename('Dissolvedoxygen')

        MNDWI = imageCollection.map(mapMNDWIsentinel)
        MNDWIextract = MNDWI.select('B[1-9]')
        print('MNDWI Images', MNDWIextract)

        DO = MNDWI.map(DOcalc).select(['Dissolvedoxygen'])

        return ee.Feature(None, {'Dissolvedoxygen': DO})

    def find_nearest_feature(self, fc, point):
        # Compute distances from the point to each feature in the collection
        distances = fc.map(lambda feature: feature.set('distance', feature.geometry().distance(point)))

        # Sort by distance and get the nearest feature
        nearest_feature = distances.sort('distance').first()
        return nearest_feature

    def map_clip_function(self,image_collection, buffered_shapefile):
        return image_collection.map(lambda image: image.clip(buffered_shapefile))

    def run(self):
        owd = settings.MEDIA_ROOT
        # Initialize Google Earth Engine
        try:
            service_account = self.service_account
            credentials = ee.ServiceAccountCredentials(service_account, self.service_key)
            ee.Initialize(credentials)
        except ee.EEException as e:
            print(str(e))

        self.cc = ee.Number(self.cc)
        self.geometry = ee.Geometry.Point([self.lat, self.long], 'EPSG:4326')
        self.start_date = ee.Date(str(self.start_date))
        self.end_date = ee.Date(str(self.end_date))

        # Create base folium map centered on the point of interest
        m = folium.Map(
            location=[self.long, self.lat],
            zoom_start=10
        )

        # Get and process lake data
        self.table = ee.FeatureCollection("projects/sat-io/open-datasets/HydroLakes/lake_poly_v10")
        filtered_table = self.table.filterBounds(self.geometry.buffer(300000))
        nearest_feature = self.find_nearest_feature(filtered_table, self.geometry)
        shapefile = ee.Feature(nearest_feature)
        buffered_shapefile = shapefile.buffer(1000)

        # Add lake boundary to map
        lake_coords = ee.Feature(nearest_feature).geometry().coordinates().getInfo()
        folium.GeoJson(
            data={
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": lake_coords
                }
            },
            style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 2, "fillOpacity": 0.1}
        ).add_to(m)

        # Get and process satellite data
        if self.satellite == 1:
            tile = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            tile = self.map_clip_function(tile, buffered_shapefile)
            Reflectance = self.atmosphericcorrection_landsat(tile, self.geometry, self.start_date, self.end_date,
                                                             self.cc)

            if self.variable == 1:
                result = ee.ImageCollection(self.CI_Landsat(Reflectance).get('Chlorophyllindex'))
                layer_name = 'Chlorophyll Index'
            elif self.variable == 2:
                result = ee.ImageCollection(self.CI_Landsat(Reflectance).get('Chlorophyllindex'))
                layer_name = 'Turbidity Index'
            elif self.variable == 3:
                result = ee.ImageCollection(self.DO_Landsat(Reflectance).get('Dissolvedoxygen'))
                layer_name = 'Dissolved Oxygen'

        elif self.satellite == 2:
            tile = ee.ImageCollection("COPERNICUS/S2_SR")
            tile = self.map_clip_function(tile, buffered_shapefile)
            Reflectance = self.atmosphericcorrection_sentinel(tile, self.geometry, self.start_date, self.end_date,
                                                              self.cc)

            if self.variable == 1:
                result = ee.ImageCollection(self.CI_Sentinel(Reflectance).get('Chlorophyllindex'))
                layer_name = 'Chlorophyll Index'
            elif self.variable == 2:
                result = ee.ImageCollection(self.TI_Sentinel(Reflectance).get('Turbidityindex'))
                layer_name = 'Turbidity Index'
            elif self.variable == 3:
                result = ee.ImageCollection(self.DO_Sentinel(Reflectance).get('Dissolvedoxygen'))
                layer_name = 'Dissolved Oxygen'

        thumb_url = result.first().getThumbUrl({'min':0, 'max': 0.1, 'dimensions': 1024, 'format': 'png', 'palette': ['000000','FFFFFF']})
        response = requests.get(thumb_url)
        img = Image.open(BytesIO(response.content)).convert("RGBA")

        # Convert the image to a numpy array
        array = np.array(img)

        # Extract the red channel and the alpha channel
        red_channel = array[:, :, 0]
        alpha_channel = array[:, :, 3]

        # Mask out the transparent pixels
        masked_red_channel = np.ma.masked_where(alpha_channel == 0, red_channel)

        # Plot the masked array
        plt.figure(figsize=[8, 8])
        maxm = masked_red_channel.max()
        minm = masked_red_channel.min()
        plt.imshow(masked_red_channel, cmap="jet", clim=(minm, maxm), origin="upper", vmin=minm, vmax=maxm)
        plt.colorbar(shrink=0.8)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{owd}/monitoring_results/{self.user_id}_{self.group_id}/{self.satellite}_{self.variable}_{self.sim_id}.png", dpi=100)

        # Add result as a tile layer
        map_id_dict = result.first().getMapId({
            'min': 0,
            'max': 0.1,
            'palette': ['000000', 'FFFFFF']
        })

        folium.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name=layer_name,
            overlay=True,
            control=True
        ).add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Return the HTML representation of the map
        return m._repr_html_()

if __name__ == "__main__":
    Run = Air2water_monit(start_date='2022-03-01',end_date='2022-12-15',lat=10.683,long=45.667, cc=7, satellite=2, variable=2, service_account='hydrosquas-monitoring@ee-hydrosquas-riddick.iam.gserviceaccount.com' , service_key='/home/riddick/Downloads/ee-hydrosquas-riddick-a7cc9060a474.json')
    folium=Run.run()
    print(thumb)
import numpy as np
import PIL 
import noise
import cv2
from PIL import Image, ImageEnhance
from io import BytesIO
from io import StringIO

def _to_cv_image(image):
    return (image*255.).astype(np.uint8) #[:, :, ::-1] #rgb -> bgr

def _cv_to_array(image):
    return image.astype(np.float32)/255.0 #[:, :, ::1] bgr -> rgb

def _to_pil_image(image):
    if image.shape[2] == 1:
        image = np.resize(image, (*image.shape[:2],))
        return PIL.Image.fromarray((image * 255.0).astype(np.uint8), 'L')
    return PIL.Image.fromarray((image * 255.0).astype(np.uint8))

def _pil_to_array(image):
    if image.mode == 'L':
        npimage = np.array(image).astype(np.float32) / 255.0
        return  np.resize(npimage, (*npimage.shape,1))
    return np.array(image).astype(np.float32) / 255.0

def _file_to_cv(image):
    # read as bytes
    image = image.read()
    # convert byte image image to numpy array
    image = np.frombuffer(image, np.uint8)
    # decode bytes as a cv image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def _file_to_array(image):
    # read as bytes
    image = image.read()
    # convert byte image image to numpy array
    image = np.frombuffer(image, np.uint8)
    # decode bytes as a cv image
    image = cv2.imdecode(np, cv2.IMREAD_COLOR)
    return _cv_to_array(image)

def _perlin_array(shape = (200, 200),
                  scale = 10.,
                  octaves = 2, 
                  persistence = 0.5, 
                  lacunarity = 2.0, 
                  seed = 12345):
    arr = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            arr[i][j] = noise.pnoise2(float(i) / scale, 
                                      float(j) / scale,
                                      octaves=octaves,
                                      persistence=persistence,
                                      lacunarity=lacunarity)
    max_arr = np.max(arr)
    min_arr = np.min(arr)
    norm_me = lambda x: (x-min_arr)/(max_arr - min_arr)
    norm_me = np.vectorize(norm_me)
    arr = norm_me(arr)
    return arr

def show_image(image):
    img = _to_pil_image(image)
    img.show()

def gamma_correction(image, gamma=2.2):
  invgamma = 1.0 / gamma
  imin, imax = image.min(), image.max()
  newimg = image.copy()
  newimg = ((newimg - imin) / (imax - imin)) ** invgamma
  newimg = newimg * (imax - imin) + imin
  return newimg

def edge_enhance(image, alpha=0.5):
    pil_image = _to_pil_image(image)
    converter = PIL.ImageEnhance.Color(pil_image)
    pil_image = converter.enhance(alpha)
    return _pil_to_array(pil_image)

def brightness (image, alpha = 1.0):
    pil_image = _to_pil_image(image)
    converter = PIL.ImageEnhance.Brightness(pil_image)
    pil_image = converter.enhance(alpha)
    return _pil_to_array(pil_image)

def contrast(image, alpha=1.0):
    pil_image = _to_pil_image(image)
    converter = PIL.ImageEnhance.Contrast(pil_image)
    pil_image = converter.enhance(alpha)
    return _pil_to_array(pil_image)

def sharpness(image, alpha = 1.0):
    pil_image = _to_pil_image(image)
    converter = PIL.ImageEnhance.Sharpness(pil_image)
    pil_image = converter.enhance(alpha)
    return _pil_to_array(pil_image)

def scale(image, factor=2.0):
    pil_image = _to_pil_image(image)
    target_size = pil_image.size
    pil_image = pil_image.resize((int(pil_image.width * factor), int(pil_image.height * factor)))
    left = (pil_image.size[0] - target_size[0])/2
    top = (pil_image.size[1] - target_size[1])/2
    right = (pil_image.size[0] + target_size[0])/2
    bottom = (pil_image.size[1] + target_size[1])/2
    pil_image = pil_image.crop((left, top, right, bottom))
    return _pil_to_array(pil_image)

def rotate(image, angle = 0.0):
    pil_image = _to_pil_image(image)
    pil_image = pil_image.rotate(angle)
    return _pil_to_array(pil_image)

def vintage(image, factor = 1.0):
    im = _to_cv_image(image)
    rows, cols = im.shape[:2]# Create a Gaussian filter
    kernel_x = cv2.getGaussianKernel(cols,200)
    kernel_y = cv2.getGaussianKernel(rows,200)
    kernel = kernel_y * kernel_x.T
    _filter = (255 * kernel / np.linalg.norm(kernel)) * factor
    im = _cv_to_array(im)
    for i in range(3):
        im[:,:,i] *= _filter 
    return im

'''
def cartoonize(image):
    from cartoon import cartoonize   
    cv_image = _to_cv_image(image)
    cv_image = cartoonize(cv_image)
    return _cv_to_array(cv_image)
'''

def sharpen(image):
    pil_image = _to_pil_image(image)
    pil_image.filter(PIL.ImageFilter.SHARPEN)
    return _pil_to_array(pil_image)

def smooth_more(image):
    pil_image = _to_pil_image(image)
    pil_image.filter(PIL.ImageFilter.SMOOTH_MORE)
    return _pil_to_array(pil_image)

def gaussian_blur(image, radius=2):
    pil_image = _to_pil_image(image)
    pil_image.filter(PIL.ImageFilter.GaussianBlur(radius=radius))
    return _pil_to_array(pil_image)

def jpeg_compression(image, quality = 90):
    pil_image = _to_pil_image(image)  
    buffer = BytesIO()
    pil_image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    pil_image = PIL.Image.open(buffer)
    return _pil_to_array(pil_image)
 
def perlin_noise(image, octaves = 6, 
                        scale = 10, 
                        alpha = 0.1):
    c_img = image.copy()
    p_nois = _perlin_array(shape=(c_img.shape[0], c_img.shape[1]),
                           octaves=octaves,
                           scale=scale)
    for c in range(image.shape[2]):
        c_img[:,:,c] += (p_nois * alpha)
    c_img /= c_img.max()
    return c_img 

#instragram help 
from scipy.interpolate import UnivariateSpline

#calcola le lookuptable in hue utilizzando una spline
def spread_lookup_table(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))

def hue(image, red = 1, green = 1, blue = 1):
    
    #converte i valori rgb dell'immagine a interi di 8 bit
    image = _to_cv_image(image)

    # liste di 4 valori per costruire le lookup tables
    base = np.array([0, 64, 128, 255]) 
    redValues = np.clip(base * red, 0, 255)
    greenValues = np.clip(base * green, 0, 255)
    blueValues = np.clip(base * blue, 0, 255)
    
    # fa in modo che l'ultimo valore (il più alto), rimanga pari a 255. Questo impedisce di modificare i bianchi
    redValues[-1] = \
    greenValues[-1] = \
    blueValues[-1] = 255

    # calcola le lookuptable di lunghezza 256 a partire da quelle di lunghezza 4
    # le tabelle conterranno i nuovi valori per i canali
    redLookupTable = spread_lookup_table(base, redValues)
    greenLookupTable = spread_lookup_table(base, greenValues)
    blueLookupTable = spread_lookup_table(base, blueValues)

    # opencv utilizza il formato BGR
    # LUT effettua una trasformazione di un array basata su una lookup table:
    # ogni vecchio valore viene sostituito da quello cui è mappato nella lookup table
    red_channel = cv2.LUT(image[:,:, 2], redLookupTable).astype(np.uint8)
    green_channel = cv2.LUT(image[:,:, 1], greenLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(image[:,:, 0], blueLookupTable).astype(np.uint8)

    #assegna i nuovi canali all'immagine
    image[:,:, 0] = blue_channel
    image[:,:, 1] = green_channel
    image[:,:, 2] = red_channel 

    np_image = _cv_to_array(image)

    return np_image

def select_by_hsv(image, lower_bound = (90, 50, 30), upper_bound = (130,255,230), color = None):
    
    if(color is not None):
        lower_bound = (color.value[0], 0, 0)
        upper_bound = (color.value[1], 255, 255)
    
    image = _to_cv_image(image)
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array( lower_bound , dtype = np.uint8, ndmin = 1)
    upper_bound = np.array( upper_bound , dtype = np.uint8, ndmin = 1)

    #ricava la maschera di pixel selezionati. la maschera sarà in scala di grigio, con valore 0(nero) sui pixel scartati e 255(bianco) su quelli selezionati
    mask = cv2.inRange(temp, lower_bound , upper_bound)

    #la maschera va convertita a bgr per poterle riapplicare il colore
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    #assegna i colori originali ai pixel selezionati
    temp = _cv_to_array(image & mask_bgr)

    return temp

# sostituzione: sostituisce i pixel di im2 != (0, 0, 0) ai pixel di im1
def img_replace(im1, im2):
    im1 = np.where(im2 == 0, im1 , im2)
    return im1

# intersezione: ritorna i pixel di im1 che sono presenti in entrambe le immagini
def img_intersection(im1, im2):
    im1 = np.where(im2 != 0, im1 , 0)
    return im1

# unisce i pixel di entrambe le immagini. Per i pixel presenti in entrambi, quelli di im2 sovrascriveranno quelli di im1
def img_union(im1, im2):
    im1 = np.where(im2 != 0, im2 , im1)

# fonde le immagini in percentuale data dal parametro alpha, riferito alla seconda immagine
def interpolate(im1, im2, alpha):
    return ((1.0 - alpha) * im1 + alpha * im2)

#filtri di instagram
def clarendon(in_image, intensity = 1, alpha = 1):
    image = contrast(in_image, 1.2 * alpha)
    image = edge_enhance(image, 2.0 * alpha)
    image = hue(image, 0.6 * alpha, 1.0 * alpha, 1.2 * alpha)
    out_image = interpolate(in_image, image, intensity)

    return out_image
    
def gingham(in_image, intensity = 1 , alpha = 1):
    image = brightness(in_image, 1.1 * alpha)
    image = edge_enhance(image, 1.1 * alpha)
    image = contrast(image, 0.7 * alpha)
    out_image = interpolate(in_image, image, intensity)

    return out_image

def juno(in_image, intensity = 1, alpha = 1):
    image = contrast(in_image, 1.15 * alpha)
    image = edge_enhance(image, 1.1 * alpha)
    image = gamma_correction(image, 1.3 * alpha)
    out_image = interpolate(in_image, image, intensity)

    return out_image

def reyes(in_image, intensity = 1 ,alpha = 1):
    image = contrast(in_image, 0.8 * alpha)
    image = edge_enhance(image, 0.7 * alpha)
    image = brightness(image, 1.2 * alpha)
    image = gamma_correction(image, 1.2 * alpha)
    image = hue(image, 1.1 * alpha, 1.1 * alpha, 1 * alpha)
    out_image = interpolate(in_image, image, intensity)

    return out_image

def lark_hsv(in_image, intensity = 1, alpha = 1):
    image = gamma_correction(in_image, 0.8 * alpha)

    #seleziona il blu. la maschera deve solo selezionare i pixel da modificare e non deve essere modificata dall'alpha
    mask = gamma_correction(image, 1.3)
    mask = select_by_hsv(mask, lower_bound = (90, 50, 30), upper_bound = (130,255,230))

    #applica filtro ai pixel selezionati
    filtered = hue(mask, 0.8 * alpha, 0.8 * alpha, 1.2 * alpha)
    filtered = img_intersection(filtered, mask)
    image = img_replace(image, filtered)
    out_image = interpolate(in_image, image, intensity)

    return out_image

import sys
import os
import random
from filters import *
#set modules path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..', ".."))

class ParameterType(object):
    def __init__(self, strtype):
        self.ptype = eval(strtype) if strtype != 'class' else int
        self.is_class_type = strtype == 'class'

    def test(self, x, domain):
        if type(x) == self.ptype:
            if self.is_class:
                return x in domain
            else:
                return  domain[0] <= x and x <= domain[1]
        return False

    def is_float(self):
        return self.ptype == float

    def is_int(self):
        return self.ptype == int and not self.is_class_type

    def is_class(self):
        return self.is_class_type

class ParameterDomain(object):
    def __init__(self, name, strtype, domain, vinit):
        self.name = name 
        self.ptype = ParameterType(strtype)
        self.domain = domain 
        self.value = vinit
    
    def test(self,x):
        return self.ptype.test(x, self.domain) 

    def random(self):
        if self.ptype.is_float():
            return random.uniform(*self.domain)
        elif self.ptype.is_int():
            return random.randint(*self.domain)
        elif self.ptype.is_class():
            return random.choice(self.domain)

class ImageFilter(object):
    def __init__(self, name, parameters_domains, fun):
        self.name = name
        self.domains = parameters_domains
        self.fun = fun  

    def nparams(self):
        return len(self.domains)

    def __call__(self, image, *params):
        return self.fun(image, *params)

# First test
# _Filters = [
#     ImageFilter('gamma correction', [ 
#         ParameterDomain('gamma', 'float', [0.1,3.], 1.92) 
#     ], gamma_correction),
#     ImageFilter('edge enhance', [], edge_enhance),
#     ImageFilter('gaussian blur', [ 
#         ParameterDomain('radius', 'int', [2,100], 2) 
#     ], gaussian_blur),
#     ImageFilter('jpeg compression', [ 
#         ParameterDomain('quality', 'int', [50,100], 90) 
#     ], jpeg_compression),
#     ImageFilter('perlin noise', [ 
#         ParameterDomain('octaves', 'int', [2,10], 6),
#         ParameterDomain('scale', 'float', [1.,15.], 10.),
#         ParameterDomain('alpha', 'float', [0.05,0.25], 0.1) 
#     ], perlin_noise),
#     ImageFilter('sharpen', [], sharpen),
#     ImageFilter('smooth more', [], smooth_more),
# ]

# Runs at ~40-30
# _Filters = [
#     ImageFilter('gamma correction', [ 
#         ParameterDomain('gamma', 'float', [0.9,1.92], 1.0) 
#     ], gamma_correction),
#     ImageFilter('edge enhance', [ 
#         ParameterDomain('gamma', 'float', [0.9,1.1], 1.) 
#     ], gamma_correction),
#     ImageFilter('contrast', [ 
#         ParameterDomain('gamma', 'float', [0.9,1.1], 1.) 
#     ], contrast),
#     ImageFilter('gaussian blur', [ 
#         ParameterDomain('radius', 'int', [2,5], 3) 
#     ], gaussian_blur),
#     ImageFilter('jpeg compression', [ 
#         ParameterDomain('quality', 'int', [60,100], 90) 
#     ], jpeg_compression),
#     ImageFilter('scale', [ 
#         ParameterDomain('factor', 'float', [1.,2.], 1.5),
#     ], scale),
#     ImageFilter('rotate', [ 
#         ParameterDomain('angle', 'float', [-180.,180.], 0.0),
#     ], rotate),
#     #ImageFilter('vintage', [ 
#     #    ParameterDomain('factor', 'float', [0.1,1.0], 1.0),
#     #], vintage),
#     ImageFilter('perlin noise', [ 
#         ParameterDomain('octaves', 'int', [2,8], 6),
#         ParameterDomain('scale', 'float', [1.,12.], 8.),
#         ParameterDomain('alpha', 'float', [0.05,0.2], 0.1) 
#     ], perlin_noise)
# ]

# TEST PARETO VS NO PARETO
# Filters = [
#     ImageFilter('gamma correction', [ 
#         ParameterDomain('gamma', 'float', [1.0,1.92], 1.2) 
#     ], gamma_correction),
#     ImageFilter('edge enhance', [ 
#         ParameterDomain('gamma', 'float', [0.9,1.1], 1.) 
#     ], edge_enhance),
#     ImageFilter('contrast', [ 
#         ParameterDomain('gamma', 'float', [0.9,1.1], 1.) 
#     ], contrast),
#     ImageFilter('gaussian blur', [ 
#         ParameterDomain('radius', 'int', [2, 4], 3) 
#     ], gaussian_blur),
#     ImageFilter('jpeg compression', [ 
#         ParameterDomain('quality', 'int', [70, 90], 80) 
#     ], jpeg_compression),
#     ImageFilter('scale', [ 
#         ParameterDomain('factor', 'float', [0.95, 1.05], 1.0),
#     ], scale),
#     ImageFilter('rotate', [ 
#         ParameterDomain('angle', 'class', [0.0, 90.0, 180.0, 270.0], 0.0),
#     ], rotate),
#     #ImageFilter('vintage', [ 
#     #    ParameterDomain('factor', 'float', [0.1,1.0], 1.0),
#     #], vintage),
#     ImageFilter('perlin noise', [ 
#         ParameterDomain('octaves', 'int', [2,8], 6),
#         ParameterDomain('scale', 'float', [1.,12.], 8.),
#         ParameterDomain('alpha', 'float', [0.01,0.05], 0.035) 
#     ], perlin_noise)
# ]

# INSTAGRAM
Filters = [
    ImageFilter('clarendon', [ 
        ParameterDomain('intensity', 'float', [0.1,1.0], 1.0),
        ParameterDomain('alpha', 'float', [0.8,1.3], 1.0) 
    ], clarendon),
    ImageFilter('gingham', [ 
        ParameterDomain('intensity', 'float', [0.1,1.0], 1.0),
        ParameterDomain('alpha', 'float', [0.8,1.3], 1.0) 
    ], gingham),
    ImageFilter('juno', [ 
        ParameterDomain('intensity', 'float', [0.1,1.0], 1.0),
        ParameterDomain('alpha', 'float', [0.5,1.5], 1.0) 
    ], juno),
    ImageFilter('reyes', [ 
        ParameterDomain('intensity', 'float', [0.1,1.0], 1.0),
        ParameterDomain('alpha', 'float', [0.8,1.1], 1.0) 
    ], reyes),
    ImageFilter('lark', [ 
        ParameterDomain('intensity', 'float', [0.1,1.0], 1.0),
        ParameterDomain('alpha', 'float', [0.9,1.1], 1.0) 
    ], lark_hsv)
]

if __name__ == "__main__":
    #import dataset
    from datasets import CIFAR10Dataset
    #set modules path
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
    sys.path.append(os.path.join(os.path.dirname(__file__),'..', ".."))
    #test
    S = 55
    X, Y = CIFAR10Dataset().get_test_dataset()
    print(X.shape, Y.shape)
    print(X[S].shape)
    show_image(X[S])
    for f in Filters:
        show_image(f(X[S]))

import json
import pickle
import copy
import base64
import random
import cv2, base64
import PIL
import numpy 
import os

from filters import _to_cv_image, _cv_to_array

def attack(in_img, model_path):

    # load the model and apply
    model = ModelLoader().load(model_path)
    in_img = _cv_to_array(in_img)
    mod_img_np = model.apply(in_img)

    # convert modified image to jpg and decode in base64
    buffer = cv2.imencode('.jpg', _to_cv_image(mod_img_np))
    mod_image_b64 = base64.b64encode(buffer[1]).decode()

    # return modified np image and base64 encoding
    return (mod_img_np, mod_image_b64)

class ModelLoader(object):

    def __init__(self):
        self.model = None 

    def load(self,path):
        with open(path, 'r') as jfile:
            jmodel = json.load(jfile)
            self.model = copy.copy(jmodel)
            filters_data_byte = base64.b64decode(jmodel["filters_data"])
            self.model["filters_data"] = pickle.loads(filters_data_byte)
        return self

    def save(self,path):
        with open(path, 'w') as jfile:
            jmodel = copy.copy(self.model)
            filters_data_b64 = base64.b64encode(pickle.dumps(self.model["filters_data"]))
            jmodel["filters_data"] = filters_data_b64.decode("ascii")
            json.dump(jmodel, jfile, indent=4)
        return self

    def apply(self,X):
        ilast = 0 
        image = X
        for fid in self.model["filters"]:
            ifilter = self.model["filters_data"][fid]
            image = ifilter(image,*self.model["params"][ilast:ilast+ifilter.nparams()])
            ilast += ifilter.nparams()
        return image

    def to_individual(self):
        indv = Individual(0,0,float("inf"))
        indv.genotype = self.model["filters"]
        indv.params = self.model["params"]
        indv.fitness = self.model["fitness"]
        indv.filters = self.model["filters_data"]
        return indv

    def get_filters(self):
        list_filter = []
        ilast = 0 
        for fid in self.model["filters"]:
            ifilter = self.model["filters_data"][fid]
            list_filter.append((ifilter, *self.model["params"][ilast:ilast+ifilter.nparams()]))
            ilast += ifilter.nparams()
        return list_filter

    def from_individual(self, individual, metainfo = None):
        self.model = {
            "filters" : individual.genotype,
            "params" : individual.params,
            "fitness" : individual.fitness
        }
        if metainfo is not None:
            self.model = { **self.model,  **metainfo }
        #serialize functions
        self.model["filters_data"] = individual.filters
        #retult self
        return self

class Individual(object):

    def __init__(self, 
                 Nf,
                 filters, 
                 fitness_max):
        self.genotype = [random.randrange(0, len(filters)) for _ in range(Nf)]
        self.params = []
        self.filters = filters
        for fid in self.genotype:
            self.params += [d.value for d in self.filters[fid].domains]
        self.fitness_max = fitness_max
        self.fitness = fitness_max
    
    def apply(self, image, params = None):
        if params is None:
            params = self.params
        ilast = 0
        for fid in self.genotype:
            ifilter = self.filters[fid]
            image = ifilter(image,*params[ilast:ilast+ifilter.nparams()])
            ilast += ifilter.nparams()
        return image

    def change(self, i, j, rand_params = False):
        p_i = 0
        for p in range(i):
            p_i += len(self.filters[self.genotype[p]].domains)
        e_i = p_i + len(self.filters[self.genotype[i]].domains)
        if rand_params == False:
            self.params = self.params[:p_i] + [d.value for d in self.filters[j].domains] + self.params[e_i:]
        else:
            self.params = self.params[:p_i] + [d.random() for d in self.filters[j].domains] + self.params[e_i:]
        self.genotype[i] = j 

    def pice(self, s=0, e=None):
        if e is None:
            e = len(self.genotype)
        new = copy.copy(self)
        new.fitness = new.fitness_max
        new.genotype = self.genotype[s:e]
        p_s = 0
        for i in range(s):
            p_s += len(self.filters[self.genotype[i]].domains)
        p_e = p_s
        for i in range(s,e):
            p_e += len(self.filters[self.genotype[i]].domains)
        new.params = self.params[p_s:p_e]
        return new

    def __add__(self, other):
        new = copy.copy(self)
        new.fitness = new.fitness_max
        new.genotype += other.genotype
        new.params += other.params
        return new
    
    def __len__(self):
        return len(self.genotype)

    def nparams(self):
        return len(self.params)

if __name__ == "__main__":
    run_attack()
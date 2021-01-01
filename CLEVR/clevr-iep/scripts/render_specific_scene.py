import random, argparse, torch, os, pyro, math, uuid, json
import numpy as np
import csv
from multiprocessing import set_start_method
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, HMC, NUTS

# local imports
import run_model
from iep.data import ClevrDataset, ClevrDataLoader
from iep.preprocess import tokenize, encode
import iep.utils as utils
import iep.programs

AVAILABLE_MATERIALS, AVAILABLE_OBJECTS, AVAILABLE_SIZES, AVAILABLE_COLOURS = [], [], [], []
NUM_AVAILABLE_OBJECTS, NUM_AVAILABLE_MATERIALS, NUM_AVAILABLE_SIZES, NUM_AVAILABLE_COLOURS = None, None, None, None
obj_probs, material_probs, colour_probs, size_probs = None, None, None, None

save_dir = None

parser = argparse.ArgumentParser()

parser.add_argument('--file_name')
parser.add_argument('--save_dir', default = '')


# Available properties (objects, sizes, colours, etc)
parser.add_argument('--properties_json', default='../clevr-dataset-gen/image_generation/data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")

# Inference-related properties



def latent_to_image(z, save_dir):
    """
    Given a latent representation of a CLEVR image, construct and save the image.

    Params
    ------
        z : dictionary
            z is a latent describing a clevr scene.

    Returns
    -------
        latent_id : int
            a psuedo-unique id detailing the saved location of the generated image
    """
    # give image a unique id
    latent_id = uuid.uuid1().int>>64

    # write z to json file
    latent_filename = '../output/tmp_latent.json'
    with open(latent_filename, 'w') as outfile:
        json.dump(z, outfile)


    output_image_dir = '../output/demo/saliency_map_comparisons/images/' + save_dir
    output_scene_dir = '../output/demo/saliency_map_comparisons/scenes/' + save_dir

    # call blender to generate image
    blender_request = "~/blender/blender \
                        --background \
                        --python ../clevr-dataset-gen/image_generation/render_images.py -- \
                        --start_idx %s \
                        --base_scene_blendfile ../clevr-dataset-gen/image_generation/data/base\_scene.blend \
                        --properties_json ../clevr-dataset-gen/image_generation/data/properties.json \
                        --shape_dir ../clevr-dataset-gen/image_generation/data/shapes \
                        --material_dir ../clevr-dataset-gen/image_generation/data/materials \
                        --obj_properties %s\
                        --output_image_dir %s \
                        --output_scene_dir %s \
                        --num_images 1 \
                        --use_gpu 1" % (latent_id, latent_filename, output_image_dir, output_scene_dir)
    os.system(blender_request)

    # tell us where to find the image
    return latent_id


def read_scene(file, save_dir):
    print (AVAILABLE_MATERIALS)
    latent = {}

    with open(file) as f:
        latent_data = json.load(f)
        latent["num_objects"] = len(latent_data['objects'])

        for idx, object in enumerate(latent_data['objects']):
            for shape_entry in AVAILABLE_OBJECTS:
                if object['shape'] == shape_entry[1]:
                    shape = shape_entry

            for color_entry in AVAILABLE_COLOURS:
                if object['color'] == color_entry[0]:
                    colour = color_entry

            for material_entry in AVAILABLE_MATERIALS:
                if object['material'] == material_entry[1]:
                    material = material_entry

            for size_entry in AVAILABLE_SIZES:
                if object['size'] == size_entry[0]:
                    size = size_entry

            x, y, orientation = object['3d_coords']

            print (x, y, orientation)

            latent[idx] = {
                    "object_type": shape,
                    "colour": colour,
                    "material": material,
                    "size": size,
                    "x": x,
                    "y": y,
                    "orientation": orientation,
                    }

        print (latent)
        print (latent.keys())
        print ("Which object should I remove?")
        object_to_remove = int(input())

        if object_to_remove != -1:
            del latent[object_to_remove]

            new_latent = {}
            new_latent["num_objects"] = latent["num_objects"] - 1
            del latent["num_objects"]
            for idx, entry in enumerate(latent.keys()):
                if entry == "num_objects":
                    continue
                new_latent[idx] = latent[entry]
            print (new_latent.keys())

            img_file = latent_to_image(new_latent, save_dir)
        else:
            img_file = latent_to_image(latent, save_dir)
        # print ("Which object should I change?")
        # object_to_change = int(input())
        #
        # if object_to_change != -1:
        #     # del latent[object_to_remove]
        #     for shape_entry in AVAILABLE_OBJECTS:
        #         if "sphere" == shape_entry[1]:
        #             latent[object_to_change]["object_type"] = shape_entry
        #     img_file = latent_to_image(latent, save_dir)
        # else:
        #     img_file = latent_to_image(latent, save_dir)

def main(args):
    global AVAILABLE_OBJECTS, AVAILABLE_MATERIALS, AVAILABLE_SIZES, AVAILABLE_COLOURS
    global NUM_AVAILABLE_OBJECTS, NUM_AVAILABLE_MATERIALS, NUM_AVAILABLE_SIZES, NUM_AVAILABLE_COLOURS
    global obj_probs, material_probs, colour_probs, size_probs

    try:
        with open(args.properties_json, 'r') as f:
            properties = json.load(f)
            for name, rgb in properties['colors'].items():
                rgba = [float(c) / 255.0 for c in rgb] + [1.0]
                AVAILABLE_COLOURS.append((name, rgba))
            AVAILABLE_MATERIALS = [(v, k) for k, v in properties['materials'].items()]
            AVAILABLE_OBJECTS = [(v, k) for k, v in properties['shapes'].items()]
            AVAILABLE_SIZES = list(properties['sizes'].items())
            AVAILABLE_OBJECTS.append(('Cone', 'cone'))
            AVAILABLE_OBJECTS.append(('Corgi', 'corgi'))

        print (AVAILABLE_OBJECTS)
    except:
        print ("Unable to open properties file (properties_json argument)")
        exit()
    read_scene(args.file_name, args.save_dir)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args)

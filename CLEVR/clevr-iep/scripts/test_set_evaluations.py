import random, argparse, torch, os, pyro, math, uuid, json, tqdm
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

parser = argparse.ArgumentParser()
# Parameters for calling run_model.py

parser.add_argument('--program_generator', default="models/CLEVR/program_generator_18k.pt")
parser.add_argument('--execution_engine', default="models/CLEVR/execution_engine_18k.pt")
parser.add_argument('--baseline_model', default=None)
parser.add_argument('--vocab_json', default=None)
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--cnn_model', default='resnet101')
parser.add_argument('--cnn_model_stage', default=3, type=int)
parser.add_argument('--image_width', default=224, type=int)
parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--sample_argmax', type=int, default=1)
parser.add_argument('--temperature', default=1.0, type=float)
parser.add_argument('--question', default = None)

# Available properties (objects, sizes, colours, etc)
parser.add_argument('--properties_json', default='../clevr-dataset-gen/image_generation/data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")


questions = ["How many blue spheres?", "How many spheres?", "How many cubes?", "How many cylinders?"]

def call_model(args, model):

    high_conf_5_spheres = 0
    high_conf_2_blue_spheres = 0


    num_adv_ex_1_cube = 0
    num_adv_ex_1_sph = 0
    num_adv_ex_1_cyl = 0
    num_adv_ex_2_cyl = 0

    with open('../CLEVR_v1.0/scenes/CLEVR_val_scenes.json') as f:
        data = json.load(f)
        print (data.keys())

        # 1076 fails - idk why.
        for i in tqdm(range(0, len(data["scenes"]))):
            scene = data["scenes"][i]

            # classifier says
            img_id = scene["image_index"]
            img_id = "{:06d}".format(img_id)
            # exit()


            cnn_classified_objects = {
                                    "sphere": 0,
                                    "sphere_high_conf": False,
                                    "cube": 0,
                                    "cube_high_conf": False,
                                    "cylinder": 0,
                                    "cyl_high_conf": False,
                                    "blue_sphere": 0,
                                    "blue_sphere_high_conf": False,
                                    }

            for question in questions:
                args.question = question
                probs = run_model.run_single_example(args, model, "../CLEVR_v1.0/images/val/CLEVR_val_" + img_id + ".png", test_eval = True, verbose=False)

                if probs[1] == 'no':
                    continue
                print (probs)

                if "blue sphere" in question:
                    cnn_classified_objects["blue_sphere"] = int(probs[1])

                    if float(probs[0]) > 0.9:
                        cnn_classified_objects["blue_sphere_high_conf"] = True
                elif "sphere" in question:
                    if float(probs[0]) > 0.9:
                        cnn_classified_objects["sphere_high_conf"] = True
                    cnn_classified_objects["sphere"] = int(probs[1])
                elif "cube" in question:
                    if float(probs[0]) > 0.9:
                        cnn_classified_objects["cube_high_conf"] = True
                    cnn_classified_objects["cube"] = int(probs[1])
                elif "cylinder" in question:
                    cnn_classified_objects["cylinder"] = int(probs[1])

                    if float(probs[0]) > 0.9:
                        cnn_classified_objects["cyl_high_conf"] = True

            ground_truth_objects = {
                                    "sphere": 0,
                                    "cube": 0,
                                    "cylinder": 0,
                                    "blue_sphere": 0
                                    }
            for object in scene["objects"]:
                obj_shape = object["shape"]
                ground_truth_objects[obj_shape] += 1

                if obj_shape == "sphere" and object["color"] == "blue":
                    ground_truth_objects["blue_sphere"] += 1

            print (cnn_classified_objects)
            print (ground_truth_objects)

            if cnn_classified_objects["cyl_high_conf"] and cnn_classified_objects["cylinder"] == 2 and ground_truth_objects["cylinder"] == 0:
                num_adv_ex_2_cyl += 1
                with open("../test_set_evals/adv_2cyl.txt", "a+") as file_object:
                    file_object.write(str(img_id) + "\n")
            elif cnn_classified_objects["cyl_high_conf"] and cnn_classified_objects["cylinder"] == 1 and ground_truth_objects["cylinder"] == 0:
                num_adv_ex_1_cyl += 1
                with open("../test_set_evals/adv_1cyl.txt", "a+") as file_object:
                    file_object.write(str(img_id) + "\n")
            elif cnn_classified_objects["cube_high_conf"] and cnn_classified_objects["cube"] == 1 and ground_truth_objects["cube"] == 0:
                num_adv_ex_1_cube += 1
                with open("../test_set_evals/adv_1cube.txt", "a+") as file_object:
                    file_object.write(str(img_id) + "\n")
            elif cnn_classified_objects["sphere_high_conf"] and cnn_classified_objects["sphere"] == 1 and ground_truth_objects["sphere"] == 0:
                num_adv_ex_1_sph += 1
                with open("../test_set_evals/adv_1sph.txt", "a+") as file_object:
                    file_object.write(str(img_id) + "\n")
            if cnn_classified_objects["sphere_high_conf"] and  cnn_classified_objects["sphere"] == 5:
                high_conf_5_spheres += 1
                with open("../test_set_evals/hc_5_spheres", "a+") as file_object:
                    file_object.write(str(img_id) + "\n")
            if cnn_classified_objects["blue_sphere_high_conf"] and cnn_classified_objects["blue_sphere"] == 2:
                high_conf_2_blue_spheres += 1
                with open("../test_set_evals/hc_2_blue_spheres", "a+") as file_object:
                    file_object.write(str(img_id) + "\n")

            print ("Adv - 2 cyl: " + str(num_adv_ex_2_cyl))
            print ("Adv - 1 cyl: " + str(num_adv_ex_1_cyl))
            print ("Adv - 1 cube: " + str(num_adv_ex_1_cube))
            print ("Adv - 1 sphere: " + str(num_adv_ex_1_sph))

            print ("HC - 5 sphere: " + str(high_conf_5_spheres))
            print ("HC - 2 Blue sphere: " + str(high_conf_2_blue_spheres))

def main(args):
    model = None

    try:
        with open(args.properties_json, 'r') as f:
            properties = json.load(f)
    except:
        print ("Unable to open properties file (properties_json argument)")
        exit()

    if args.baseline_model is not None:
        print('Loading baseline model from ', args.baseline_model)
        model, _ = utils.load_baseline(args.baseline_model)
        if args.vocab_json is not None:
          new_vocab = utils.load_vocab(args.vocab_json)
          model.rnn.expand_vocab(new_vocab['question_token_to_idx'])
    elif args.program_generator is not None and args.execution_engine is not None:
        print('Loading program generator from ', args.program_generator)
        program_generator, _ = utils.load_program_generator(args.program_generator)
        print('Loading execution engine from ', args.execution_engine)
        execution_engine, _ = utils.load_execution_engine(args.execution_engine, verbose=False)
        if args.vocab_json is not None:
            new_vocab = utils.load_vocab(args.vocab_json)
            program_generator.expand_encoder_vocab(new_vocab['question_token_to_idx'])
        model = (program_generator, execution_engine)
    else:
        print('Must give either --baseline_model or --program_generator and --execution_engine')
        return

    call_model(args, model)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

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

# global properties, including which objects can be rendered
AVAILABLE_MATERIALS, AVAILABLE_OBJECTS, AVAILABLE_SIZES, AVAILABLE_COLOURS = [], [], [], []
NUM_AVAILABLE_OBJECTS, NUM_AVAILABLE_MATERIALS, NUM_AVAILABLE_SIZES, NUM_AVAILABLE_COLOURS = None, None, None, None
obj_probs, material_probs, colour_probs, size_probs = None, None, None, None

parser = argparse.ArgumentParser()

# Parameters for calling run_model.py
parser.add_argument('--program_generator', default=None)
parser.add_argument('--execution_engine', default=None)
parser.add_argument('--baseline_model', default=None)
parser.add_argument('--vocab_json', default=None)
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--cnn_model', default='resnet101')
parser.add_argument('--cnn_model_stage', default=3, type=int)
parser.add_argument('--image_width', default=224, type=int)
parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--sample_argmax', type=int, default=1)
parser.add_argument('--temperature', default=1.0, type=float)
parser.add_argument('--question', default=None)

# Available properties (objects, sizes, colours, etc)
parser.add_argument('--properties_json', default='../clevr-dataset-gen/image_generation/data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")

# Inference-related properties
parser.add_argument('--prob_test', type=int, default=0)
parser.add_argument('--out_of_distribution', type=int, default=0)
parser.add_argument('--remove_object_type', default=None)
parser.add_argument('--class_a', type=int, default=5)
parser.add_argument('--class_b', type=int, default=-1)
parser.add_argument('--save_dir', default=None)
parser.add_argument('--num_iters', default=200)
parser.add_argument('--num-objects', default=5)
parser.add_argument('--output_csv' , default=None)
parser.add_argument('--test_name' , default=None)

def latent_to_image(z, output_dir = "../output/"):
    """
    Given a latent representation of a CLEVR image, construct and save the image.

    Params
    ------
        z : dictionary
            z is a latent describing a clevr scene.
        output_dir : string
            where to save the resultant images and scenes

    Returns
    -------
        latent_id : int
            a psuedo-unique id detailing the saved location of the generated image
    """
    # check if the directory exists; if not, make it
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # give image a unique id
    latent_id = uuid.uuid1().int>>64

    # write z to json file
    latent_filename = output_dir + 'tmp_latent.json'
    with open(latent_filename, 'w') as outfile:
        json.dump(z, outfile)

    # store the output in a sensible location
    if save_directory != None:
        output_image_dir = output_dir + save_directory + "/images/"
        output_scene_dir =  output_dir + save_directory + "/scenes/"
    else:
        output_image_dir =  output_dir + 'images/'
        output_scene_dir =  output_dir + 'scenes/'

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
                        --use_gpu 1 " % (latent_id, latent_filename, output_image_dir, output_scene_dir)
    os.system(blender_request)

    # tell us where to find the image
    return latent_id

def generate_random_object():
    """
    Select random object properties.
    Each object consists of a type, colour, material, size, orientation, and location

    Params
    ------
        None
    Returns
    -------
        object : dictionary
            A dictionary with all object-associated properties set (e.g. "size : 0")
    """
    object_type = random.randint(0, NUM_AVAILABLE_OBJECTS-1)
    colour = random.randint(0, NUM_AVAILABLE_COLOURS-1)

    material = random.randint(0, NUM_AVAILABLE_MATERIALS-1)

    selected_material = AVAILABLE_MATERIALS[material]

    # only add the Corgi texture if it's a corgi
    if (AVAILABLE_OBJECTS[object_type][0] == "Corgi"):
        selected_material = ("CorgiMaterial", "CorgiMaterial")

    size = random.randint(0, NUM_AVAILABLE_SIZES-1)
    object = {
        "object_type": AVAILABLE_OBJECTS[object_type],
        "colour": AVAILABLE_COLOURS[colour],
        "material": selected_material,
        "size": AVAILABLE_SIZES[size],
        "x": random.uniform(-3.0,3.0),
        "y": random.uniform(-3.0,3.0),
        "orientation": random.uniform(0,1)
        }
    return object

def generate_random_latent(num_objects = 5):
    """
    Construct a viable random latent.
    We must be able to render this scene with CLEVR (e.g. no object collisions)

    Params
    ------
        num_objects : int
            The number of objects to add to the scene.
            Train distribution bounds: [3, 10].
    Returns
    -------
        z : dictionary
            A dictionary representing the latent
        img_filepath : string
            A string storing the save location of the generated image
    """
    while True:
        z = {}
        z["num_objects"] = num_objects
        for i in range(0, num_objects):
            z[str(i)] = generate_random_object()

        img_file = latent_to_image(z)
        if save_directory != None:
            img_filepath = "../output/" + save_directory + "/images/CLEVR_new_" + str(img_file) + ".png"
        else:
            img_filepath = "../output/images/CLEVR_new_" + str(img_file) + ".png"
        # Create a truncated distribution (retry if the file does not exist)
        if (os.path.exists(img_filepath)):
            break
    return z, img_filepath

def peturb_latent(old_latent, std=0.05,  output_dir = "../output/"):
    """
    New latent proposal (for MH inference).
    Must return a viable CLEVR scene.

    Params
    ------
        old_latent : tuple
            [0] A dictionary describing a valid CLEVR scene
            [1] A string corresponding to the filepath of the generated image
        std : float
            A parameter for controlling peturbation variance
        output_dir : string
            Directory to save peturbed latents to
    Returns
    -------
        new_latent : dictionary
            A dictionary representing the peturbed latent
        img_filepath : string
            A string storing the save location of the generated image
    """
    old_latent = old_latent[0]
    while True:
        new_latent = {}
        new_latent["num_objects"] = old_latent["num_objects"]

        for i in range(0, new_latent["num_objects"]):
            if str(i) in old_latent.keys():
                x = np.random.normal(old_latent[str(i)]["x"], 6*std)
                y = np.random.normal(old_latent[str(i)]["y"], 6*std)
                orientation = np.random.normal(old_latent[str(i)]["orientation"], std)

                object_type = old_latent[str(i)]["object_type"]
                colour = old_latent[str(i)]["colour"]
                material = old_latent[str(i)]["material"]
                size = old_latent[str(i)]["size"]

                x = (x + 3) % 6 - 3
                y = (y + 3) % 6 - 3
                orientation %= 1
                if random.random() < 0.2:
                    object_type = random.randint(0, NUM_AVAILABLE_OBJECTS-1)
                    object_type = AVAILABLE_OBJECTS[object_type]
                if random.random() < 0.2:
                    colour = random.randint(0, NUM_AVAILABLE_COLOURS-1)
                    colour = AVAILABLE_COLOURS[colour]
                if random.random() < 0.2:
                    material = random.randint(0, NUM_AVAILABLE_MATERIALS-1)
                    material = AVAILABLE_MATERIALS[material]
                if random.random() < 0.2:
                    size = random.randint(0, NUM_AVAILABLE_SIZES-1)
                    size = AVAILABLE_SIZES[size]

                if object_type[0] == "Corgi":
                    material = ("CorgiMaterial", "CorgiMaterial")

                new_latent[str(i)] = {
                        "object_type": object_type,
                        "colour": colour,
                        "material": material,
                        "size": size,
                        "x": x,
                        "y": y,
                        "orientation": orientation,
                        }
            else:
                new_latent[str(i)] = generate_random_object()

        img_file = latent_to_image(new_latent)
        if save_directory != None:
            img_filepath = output_dir + save_directory + "/images/CLEVR_new_" + str(img_file) + ".png"
        else:
            img_filepath = output_dir + "images/CLEVR_new_" + str(img_file) + ".png"

        # Create a truncated distribution.
        # Make sure the image was writing was successful;
        # if not, make another proposal
        if (os.path.exists(img_filepath)):
            break
    return new_latent, img_filepath

def normpdf(x, mean, sd):
    """
    stackoverflow.com/questions/12412895
    Some strange bug in scipy -- versions conflict, etc.
    We instead compute the PDF for the normal distribution directly.
    """
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def metropolis_hastings(initial_proposal, num_iters, std, args, model, target_class, num_objects = 5, output_csv = None, test_name = None):
    """
    Metropolis hastings to optimize a single constraint. For example, "1 blue ball"
    Writes accepted images and probabilities to ../output/save_directory/accepted_latents.txt

    Params:
    ------
        initial_proposal : tuple
            latent dictionary + filepath pair for the starting proposal
        num_iters : int
            How many inference steps to take
        std : float
            hyperparameter for acceptance criteria
        args : list
            a list of arguments required by CLEVR's run_model.py
        model : tuple
            required by CLEVR's run_model - (program_generator, execution_engine)
        target_class : int
            index of target class
        num_objects : int
            the number of objects to render in the scene

    Returns
    -------
        None
    """
    accepted_proposals = []
    accepted_probabilities = []
    old_latent = initial_proposal
    prob_old = run_model.run_single_example(args, model, old_latent[1])[target_class]
    for _ in tqdm(range(num_iters)):
        new_latent = peturb_latent(old_latent)
        prob_new = run_model.run_single_example(args, model, new_latent[1])[target_class]
        alpha = normpdf(1, prob_new, std) / normpdf(1, prob_old, std)
        if random.uniform(0,1) <= alpha:
            accepted_proposals.append(new_latent[1])
            accepted_probabilities.append(prob_new)
            old_latent = new_latent
            prob_old = prob_new
        else:
            old_latent = old_latent
    if save_directory != None:
        accepted_latents = "../output/" + save_directory + "/accepted_latents.txt"
    else:
        accepted_latents = "../output/accepted_latents.txt"
    with open(accepted_latents, "w") as text_file:
        text_file.write("Prob (last): " + str(prob_old.item()) + " \n")
        text_file.write("Prob array: " + str(accepted_probabilities) + " \n")
        text_file.write(str(accepted_proposals))

    file_exists = os.path.isfile(output_csv)
    if file_exists:
        write_mode = 'a'
    else:
        write_mode = 'w'

    with open(output_csv, mode=write_mode) as output_csv:
        fieldnames = ['Test Name', 'Output Probability', 'Image Location', 'Num Iterations']
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({'Test Name': test_name,
                         'Output Probability': prob_old,
                         'Image Location': accepted_proposals[-1],
                         'Num Iterations': num_iters})


def rejection_sampling(initial_proposal, num_iters, args, model, target_class, num_objects = 5, output_csv = None, test_name = None):
    """
    Rejection sampling to optimize a single constraint. For example, "1 blue ball"
    Writes accepted images and probabilities to ../output/save_directory/accepted_latents.txt

    Params:
    ------
        initial_proposal : tuple
            latent dictionary + filepath pair for the starting proposal
        num_iters : int
            How many inference steps to take
        std : float
            hyperparameter for acceptance criteria
        args : list
            a list of arguments required by CLEVR's run_model.py
        model : tuple
            required by CLEVR's run_model - (program_generator, execution_engine)
        target_class : int
            index of target class
        num_objects : int
            the number of objects to render in the scene

    Returns
    -------
        None
    """
    accepted_proposals = []
    accepted_probabilities = []
    old_latent = initial_proposal
    prob_old = run_model.run_single_example(args, model, old_latent[1])[target_class]
    for _ in tqdm(range(num_iters)):
        new_latent = generate_random_latent(num_objects = num_objects)
        prob_new = run_model.run_single_example(args, model, new_latent[1])[target_class]

        if prob_new > prob_old:
            accepted_proposals.append(new_latent[1])
            accepted_probabilities.append(prob_new)
            old_latent = new_latent
            prob_old = prob_new

    if save_directory != None:
        accepted_latents = "../output/" + save_directory + "/accepted_latents.txt"
    else:
        accepted_latents = "../output/accepted_latents.txt"
    with open(accepted_latents, "w") as text_file:
        text_file.write("Prob (last): " + str(prob_old.item()) + " \n")
        text_file.write("Prob array: " + str(accepted_probabilities) + " \n")
        text_file.write(str(accepted_proposals))

    file_exists = os.path.isfile(output_csv)
    if file_exists:
        write_mode = 'a'
    else:
        write_mode = 'w'

    with open(output_csv, mode=write_mode) as output_csv:
        fieldnames = ['Test Name', 'Output Probability', 'Image Location', 'Num Iterations']
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({'Test Name': test_name,
                         'Output Probability': prob_old,
                         'Image Location': accepted_proposals[-1],
                         'Num Iterations': num_iters})

def metropolis_hastings_two_classes(initial_proposal, num_iters, std, args, model, target_classes, num_objects = 5):
    """
    Metropolis hastings to optimize two constraints. For example, "1 blue ball, 1 green cylinder"
    Used for finding ambiguous examples, etc.
    Writes accepted images and probabilities to ../output/save_directory/accepted_latents.txt

    Params:
    ------
        initial_proposal : tuple
            latent dictionary + filepath pair for the starting proposal
        num_iters : int
            How many inference steps to take
        std : float
            hyperparameter for acceptance criteria
        args : list
            a list of arguments required by CLEVR's run_model.py
        model : tuple
            required by CLEVR's run_model - (program_generator, execution_engine)
        target_classes : list
            list of integers corresponding to indexes of target class
        num_objects : int
            the number of objects to render in the scene

    Returns
    -------
        None
    """
    assert(len(target_classes) == 2)

    accepted_proposals = []
    old_latent = initial_proposal

    prob_old = run_model.run_single_example(args, model, old_latent[1])

    non_target_classes = np.arange(len(prob_old))
    non_target_classes = np.delete(non_target_classes, target_classes)

    prob_old_target_classes = [prob_old[i] for i in target_classes]
    prob_old_non_target_classes = [prob_old[i] for i in range(len(prob_old)) ]

    target_class_metric = abs(prob_old_target_classes[0] - prob_old_target_classes[1])
    non_target_class_metric = np.min(prob_old_target_classes) - np.max(prob_old_non_target_classes)


    q_t0 = normpdf(0, target_class_metric, std) * normpdf(0.45, target_class_metric, std)
    for _ in tqdm(range(num_iters)):
        new_latent = peturb_latent(old_latent)

        prob_new = run_model.run_single_example(args, model, new_latent[1])
        prob_new_target_classes = [prob_new[i] for i in target_classes]
        prob_new_non_target_classes = [prob_new[i] for i in non_target_classes]

        target_class_metric = abs(prob_new_target_classes[0] - prob_new_target_classes[1])
        non_target_class_metric = np.min(prob_new_target_classes) - np.max(prob_new_non_target_classes)

        q_t1 = normpdf(0, target_class_metric, std) * normpdf(0.4, target_class_metric, std)
        print ("Prob Old: " + str(prob_old))
        print ("Prob New: " + str(prob_new))
        alpha = q_t1 / q_t0
        if random.uniform(0,1) <= alpha:
            print ("Accepted latent!\n")
            accepted_proposals.append(new_latent[1])
            prob_old = prob_new
            old_latent = new_latent
            q_t0 = q_t1
        else:
            old_latent = old_latent
    print (accepted_proposals)
    if save_directory != None:
        accepted_latents = "../output/" + save_directory + "/accepted_latents.txt"
    else:
        accepted_latents = "../output/accepted_latents.txt"
    with open(accepted_latents, "w") as text_file:
        text_file.write("Prob: " + str(prob_old) + " \n")
        text_file.write(str(accepted_proposals))

def main(args):
    global AVAILABLE_OBJECTS, AVAILABLE_MATERIALS, AVAILABLE_SIZES, AVAILABLE_COLOURS
    global NUM_AVAILABLE_OBJECTS, NUM_AVAILABLE_MATERIALS, NUM_AVAILABLE_SIZES, NUM_AVAILABLE_COLOURS
    global obj_probs, material_probs, colour_probs, size_probs
    global save_directory
    model = None

    try:
        with open(args.properties_json, 'r') as f:
            properties = json.load(f)
            for name, rgb in properties['colors'].items():
                rgba = [float(c) / 255.0 for c in rgb] + [1.0]
                AVAILABLE_COLOURS.append((name, rgba))
            AVAILABLE_MATERIALS = [(v, k) for k, v in properties['materials'].items()]
            AVAILABLE_OBJECTS = [(v, k) for k, v in properties['shapes'].items()]
            AVAILABLE_SIZES = list(properties['sizes'].items())

            NUM_AVAILABLE_OBJECTS = len(AVAILABLE_OBJECTS)
            NUM_AVAILABLE_MATERIALS = len(AVAILABLE_MATERIALS)
            NUM_AVAILABLE_SIZES = len(AVAILABLE_SIZES)
            NUM_AVAILABLE_COLOURS = len(AVAILABLE_COLOURS)

            # categorical probabilities
            obj_probs = torch.ones(NUM_AVAILABLE_OBJECTS) / NUM_AVAILABLE_OBJECTS
            material_probs = torch.ones(NUM_AVAILABLE_MATERIALS) / NUM_AVAILABLE_MATERIALS
            colour_probs = torch.ones(NUM_AVAILABLE_COLOURS) / NUM_AVAILABLE_COLOURS
            size_probs = torch.ones(NUM_AVAILABLE_SIZES) / NUM_AVAILABLE_SIZES
    except:
        print ("Unable to open properties file (properties_json argument)")
        exit()

    # OOD extrapolation: add object (out of training set)
    if args.out_of_distribution == 1:
        AVAILABLE_OBJECTS.append(('Cone', 'cone'))
        NUM_AVAILABLE_OBJECTS += 1
        obj_probs = torch.ones(NUM_AVAILABLE_OBJECTS) / NUM_AVAILABLE_OBJECTS
    elif args.out_of_distribution == 2:
        AVAILABLE_OBJECTS.append(('Corgi', 'corgi'))
        NUM_AVAILABLE_OBJECTS += 1
        obj_probs = torch.ones(NUM_AVAILABLE_OBJECTS) / NUM_AVAILABLE_OBJECTS
    # adversarial or OOD extrapolation: remove object
    if args.remove_object_type != None:
        NEW_AVAILABLE_OBJECTS = []
        for i in range(len(AVAILABLE_OBJECTS)):
            _, object_name = AVAILABLE_OBJECTS[i]
            if object_name != args.remove_object_type:
                NEW_AVAILABLE_OBJECTS.append(AVAILABLE_OBJECTS[i])
        AVAILABLE_OBJECTS = NEW_AVAILABLE_OBJECTS
        NUM_AVAILABLE_OBJECTS = len(AVAILABLE_OBJECTS)
        obj_probs = torch.ones(NUM_AVAILABLE_OBJECTS) / NUM_AVAILABLE_OBJECTS

    if args.save_dir != None:
        save_directory = args.save_dir
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

    print ("Calling inference!")
    random_latent = generate_random_latent(num_objects = args.num_objects)
    print (random_latent)
    if args.prob_test == 0:
        print ("Running Metropolis Hastings (one constraint)")
        metropolis_hastings(initial_proposal=random_latent,
                                num_iters=int(args.num_iters),
                                std=0.05,
                                args=args,
                                model=model,
                                target_class=args.class_a,
                                num_objects = args.num_objects,
                                output_csv = args.output_csv,
                                test_name = args.test_name)
    if args.prob_test == 1:
        print ("Running Metropolis Hastings (two constraints)")
        target_classes = [args.class_a, args.class_b]
        metropolis_hastings_two_classes(initial_proposal=random_latent,
                            num_iters=int(args.num_iters),
                            std=0.05,
                            args=args,
                            model=model,
                            target_classes=target_classes,
                            num_objects = args.num_objects)
    if args.prob_test == 2:
        print ("Running Rejection Sampling (one constraint)")
        rejection_sampling(initial_proposal=random_latent,
                                num_iters=int(args.num_iters),
                                args=args,
                                model=model,
                                target_class=args.class_a,
                                num_objects = args.num_objects,
                                output_csv = args.output_csv,
                                test_name = args.test_name)
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

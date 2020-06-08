import os

"""
example question: "How many <x>?"
prob_test: 0 - Metropolis Hastings (one constraint), 1 - Metropolis Hastings (two constraints)
out_of_distribution: 0 - false, 1 - true
class_a: metropolis constraint. If class_a = 5 and prob_test = 0,
         will find scenes containing 5 of the target class
class_b: optional, only used for prob_test = 1.
save_dir: string, where to save examples
remove_object_type: cube | sphere | cylinder
num_iters: number of iterations of inference. default 500.
test_name: save this test to a spreadsheet


"""

request_template = 'python3 scripts/analyze_level_sets.py \
                    --program_generator models/CLEVR/program_generator_18k.pt \
                    --execution_engine models/CLEVR/execution_engine_18k.pt \
                     --question "%s" \
                     --prob_test %s \
                     --out_of_distribution %s \
                     --class_a %s \
                     --class_b %s \
                     --save_dir %s \
                     --remove_object_type %s \
                     --num_iters %s \
                     --output_csv %s \
                     --test_name %s'

num_iters = 10

"""
Example 1: Highly-confident scenes
"""

# confident scenes - 5 spheres
num_spheres = 5

request = request_template % (
                "How many spheres?",                               # question
                "0",                                               # prob_test
                "0",                                             # out_of_distribution
                str(num_spheres),                                  # class_a
                "-1",                                            # class_b
                "demo/confident/" + str(num_spheres) + "_spheres_confident_" + str(num_iters) + "iters",      # save_dir
                "none",                                          # remove_object_type
                str(num_iters),                                           # num_iters
                "../output/output.csv",                          # csv for plotting
                "confident"
         )
os.system(request)


"""
Example 2: In-distribution adversarial scenes

Set remove_object_type
"""

# find scenes which contain NO CUBES, but the classifier has high confidence contain 1 cube
num_cubes = 1

request = request_template % (
                "How many cubes?",                               # question
                "0",                                             # prob_test
                "0",                                             # out_of_distribution
                str(num_cubes),                                  # class_a
                "-1",                                            # class_b
                "demo/in_dist_adv/" + str(num_cubes) + "_cubes_" + str(num_iters) + "iters",      # save_dir
                "cube",                                          # remove_object_type
                str(num_iters),                                  # num_iters
                "../output/output.csv",                          # csv for plotting
                "in_dist_adv"
         )
os.system(request)

"""
Example 3: Novel class extrapolation scenes

Set OOD = 1
Set remove_object_type
"""

# find scenes which contain NO CUBES, but the classifier has high confidence contain 1 cube
num_cylinders = 1

request = request_template % (
                "How many cylinders?",                               # question
                "0",                                             # prob_test
                "1",                                             # out_of_distribution
                str(num_cylinders),                                  # class_a
                "-1",                                            # class_b
                "demo/in_dist_adv/" + str(num_cylinders) + "_cubes_" + str(num_iters) + "iters",      # save_dir
                "cylinder",                                          # remove_object_type
                str(num_iters),                                  # num_iters
                "../output/output.csv",                          # csv for plotting
                "in_dist_adv"
         )
os.system(request)


"""
Example 4 - saliency maps
"""

request = "python3 scripts/run_model.py \
                --program_generator models/CLEVR/program_generator_18k.pt \
                --execution_engine models/CLEVR/execution_engine_18k.pt \
                --saliency_map 1 \
                --image img/5cones.png \
                --question 'How many cubes?'"
os.system(request)

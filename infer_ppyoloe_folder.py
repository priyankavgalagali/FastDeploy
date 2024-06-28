import cv2
import os
import fastdeploy as fd
import argparse
import ast

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Path of PaddleDetection model directory")
    parser.add_argument(
        "--image_folder", 
        default=None, 
        help="Path of the folder containing test images.")
    parser.add_argument(
        "--output_folder", 
        default=None, 
        help="Path of the folder to save output images.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'kunlunxin', 'cpu' or 'gpu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="Whether to use TensorRT.")
    return parser.parse_args()

def build_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "kunlunxin":
        option.use_kunlunxin()

    if args.device.lower() == "ascend":
        option.use_ascend()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_trt_backend()
    return option

args = parse_arguments()

# Download the model if the specified model_dir is invalid
def check_model_files(model_dir):
    required_files = ["model.pdmodel", "model.pdiparams", "infer_cfg.yml"]
    for f in required_files:
        if not os.path.exists(os.path.join(model_dir, f)):
            return False
    return True

if args.model_dir is None or not check_model_files(args.model_dir):
    print("Downloading the model because the specified model directory is invalid or missing files.")
    model_dir = fd.download_model(name='ppyoloe_crn_l_300e_coco')
else:
    model_dir = args.model_dir

model_file = os.path.join(model_dir, "model.pdmodel")
params_file = os.path.join(model_dir, "model.pdiparams")
config_file = os.path.join(model_dir, "infer_cfg.yml")

# Configure runtime and load model
runtime_option = build_option(args)
model = fd.vision.detection.PPYOLOE(
    model_file, params_file, config_file, runtime_option=runtime_option)

# Check if the image folder is provided
if args.image_folder is None:
    raise ValueError("Please provide a folder containing images using --image_folder argument.")

image_folder = args.image_folder
output_folder = args.output_folder if args.output_folder else "visualized_results"
os.makedirs(output_folder, exist_ok=True)

# Loop through all images in the folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue  # skip non-image files

    im = cv2.imread(image_path)
    result = model.predict(im)
    print(f"Results for {image_name}:")
    print(result)

    # Visualize and save the result
    vis_im = fd.vision.vis_detection(im, result, score_threshold=0.5)
    output_path = os.path.join(output_folder, f"visualized_{image_name}")
    cv2.imwrite(output_path, vis_im)
    print(f"Visualized result saved in {output_path}")

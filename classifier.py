# built-in modules
import json
import os

# web dev modules
from flask import Flask, flash, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename

# deep learning/image modules
from PIL import Image
import torch
from torchvision import models, transforms

# store the model as a global
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.eval()

# flask constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.secret_key = 'ax9o4klasi-0oakdn'  # random secret key (needed for flashing)


def transform_image(image_file):
    """Transforms an input image_file to a standardized Pytorch Tensor.

    Args:
        image_file (str): the filename of the image

    Returns:
        Tensor: 4D tensor (batch_size=1, RGB=3, height, width) representing the
                input image
    """
    img_transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    # remove the alpha channel if it is present
    image = Image.open(image_file).convert('RGB')
    img_tensor = img_transform(image).unsqueeze(0)
    return img_tensor


def get_prediction(img_tensor, model):
    """
    Get the model's prediction of the img_tensor.

    Args:
        img_tensor (Tensor): the 4D (batch_size=1, RGB=3, height, width)
                             input image sensor
        model: a Torch model that can evaluate images

    Returns:
        int: the integer index of the predicted class
    """
    return int(torch.argmax(model(img_tensor), axis=1).item())


def is_bird(torch_label):
    """
    Returns whether the given PyTorch class label corresponds to a bird class.

    Args:
        torch_label (int): PyTorch class label, 0-999
    Returns:
        bool: whether the label is a bird class according to bird_synset.txt
    """

    with open('bird_synset.txt') as bird:
        birds = bird.read().splitlines()

    with open('imagenet_class_index.json') as f:
        data = json.load(f)

    return data[str(torch_label)][0] in birds


@app.route('/')
def home():
    """
    Redirect the user from the root URL to the /upload URL.

    Args:
        None

    Returns:
        The required return by Flask so the user is redirected to the /upload
        URL
    """
    return redirect(url_for('handle_upload'))


@app.route('/upload', methods=['GET', 'POST'])
def handle_upload():
    """
    Method that handles the /upload route.

    Args:
        None

    Returns:
        The required returns by Flask for redirect/file upload behavior
    """
    if request.method == 'GET':
        return render_template('upload_template.html')
    else:
        filename = request.files['file'].filename
        if not filename:
            flash('No file uploaded')
            return redirect(url_for('handle_upload'))
        elif filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
            flash('Please upload an image file')
            return redirect(url_for('handle_upload'))
        else:
            request.files['file'].save(os.path.join('static', filename))
            return redirect(url_for('handle_result', image_name=filename))


@app.route('/result')
def handle_result():
    """
    Method that handles the /result route.

    Args:
        None

    Returns:
        The required returns by Flask for redirect/rendering behavior
    """
    if not request.args:
        flash('Please upload an image first')
        return redirect(url_for('handle_upload'))
    else:
        image_name = os.path.join('static', request.args['image_name'])
        birdtrue = is_bird(get_prediction(
            transform_image(image_name), mobilenet))
        return render_template('result_template.html', image_name=image_name.replace(' ', '%20'), is_bird=birdtrue)


if __name__ == "__main__":
    app.run(port=5000, debug=True)

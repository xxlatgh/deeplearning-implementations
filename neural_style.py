import importlib

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from keras import metrics

from vgg16_avg import VGG16_Avg

import myutils
from myutils import *
import argparse
from keras.models import Model

def rand_img(shape):
    '''
    returns a random image with given shape
    '''
    return np.random.uniform(-2.5, 2.5, shape)

def style_loss(x, targ):
    '''
    returns style loss by comparing the input image and target image
    the factor 1/(4*(N**N)* (M*M)) is calculated in the gram_matrix simplify style_loss
    '''
    return metrics.mse(gram_matrix(x), gram_matrix(targ))

def gram_matrix(x):
    '''
    returns the gram matrix of an input image
    moved 1/(2*N*M) in the gram_matrix function to simplify style_loss
    '''
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features, K.transpose(features)) / (x.get_shape().num_elements()*2)

def blurify(x):
    '''
    returns an image by blurring the input image with a gaussian filter
    '''
    x = x.reshape((1, 254, 320, 3))
    for k in range(3):
        x[:,:,:,k] = scipy.ndimage.filters.gaussian_filter(x[:,:,:,k], 1)
    return x

def total_variation_loss(x):
    '''
    returns the total variation loss to account for noise in the input image
    '''
    img_rows = 254
    img_cols =320
    assert K.ndim(x) == 4
    a = K.square(x[:, :img_rows - 1, :img_cols - 1, :] - x[:, 1:, :img_cols - 1, :])
    b = K.square(x[:, :img_rows - 1, :img_cols - 1, :] - x[:, :img_rows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def solve_image(eval_obj, niter, x, path=None):
    '''
    returns an optimized image base ond l-bfgs optimizer, also returns the loss history
    '''
    last_min_val = 1000 # start from an arbitrary number
    loss_history = {}
    shp = x.shape
    imagenet_mean = [123.68, 116.779, 103.939]
    rn_mean = np.array((imagenet_mean), dtype=np.float32)
    deproc = lambda x,s: np.clip(x.reshape(s)[:, :, :, ::-1] + rn_mean, 0, 255)
    for i in range(niter):
        x, min_val, info = fmin_l_bfgs_b(eval_obj.loss, x.flatten(),
                                         fprime=eval_obj.grads, maxfun=20)
        x = np.clip(x, -127,127)
        if abs(last_min_val - min_val)< 0.1:
            x = blurify(x)
        last_min_val = min_val
        loss_history[i] = min_val
        img =(deproc(x.copy(), shp)[0])
        imsave(path+'/res_at_iteration_%d.png' %(i), img)
    return x, loss_history


def style_recreate():
    '''
    returns an image of recreated style
    '''
    input_shape = style_arr.shape[1:]
    model = VGG16_Avg(include_top=False, input_shape=input_shape)
    outputs = {l.name: l.output for l in model.layers}
    layers = [outputs['block{}_conv1'.format(o)] for o in range(1,4)]
    layers_model = Model(model.input, layers)
    targs = [K.variable(o) for o in layers_model.predict(style_arr)]
    loss = sum(style_loss(l1[0], l2[0]) for l1,l2 in zip(layers, targs))
    grads = K.gradients(loss, model.input)

    function_input = [model.input]
    function_output = ([loss]+grads)
    style_fn = K.function(function_input, function_output)
    evaluator = Evaluator(style_fn, style_arr.shape)
    style_iterations=10
    x = rand_img(shp)
    x, style_loss_history = solve_image(evaluator, style_iterations, x, style_result_path)
    s_path = style_result_path + '/res_at_iteration_9.png'
    return s_path

def content_recreate():
    '''
    returns an image of recreated content
    '''
    model = VGG16_Avg(include_top=False)
    layer = model.get_layer('block5_conv1').output
    layer_model = Model(model.input, layer)
    targ = K.variable(layer_model.predict(img_arr))

    loss = metrics.mse(layer, targ)
    grads = K.gradients(loss, model.input)

    function_input = [model.input]
    function_output = ([loss]+grads)
    fn = K.function(function_input, function_output)
    evaluator = Evaluator(fn, img_arr.shape)

    x = rand_img(img_arr.shape)
    content_iterations=10
    x_final, content_loss_history = solve_image(evaluator, content_iterations, x, path = content_result_path)
    c_path = content_result_path + '/res_at_iteration_9.png'
    return c_path

def merge():
    '''
    returns an image of the neural style transfer
    '''

    input_shape = style_arr.shape[1:]
    model = VGG16_Avg(include_top=False, input_shape=input_shape)
    outputs = {l.name: l.output for l in model.layers}

    style_layers = [outputs['block{}_conv1'.format(o)] for o in range(1,6)]
    content_layer = outputs['block4_conv2']

    style_model = Model(model.input, style_layers)
    style_targs = [K.variable(o) for o in style_model.predict(style_arr)]
    content_model = Model(model.input, content_layer)
    content_targ = K.variable(content_model.predict(img_arr))

    alpha = 0.1
    beta = 0.00001
    gama = 0.000001

    st_loss = sum(style_loss(l1[0], l2[0]) for l1,l2 in zip(style_layers, style_targs))
    loss = alpha*metrics.mse(content_layer, content_targ) +beta* st_loss + gama*total_variation_loss(model.input)
    grads = K.gradients(loss, model.input)
    transfer_fn = K.function([model.input], [loss]+grads)
    evaluator = Evaluator(transfer_fn, shp)
    merge_iterations=10
    x = rand_img(shp)
    x, merge_loss_history = solve_image(evaluator, merge_iterations, x, merge_result_path)

parser = argparse.ArgumentParser(description='Neural style transfer with Keras')
parser.add_argument('content_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('merge_results_path', metavar='res_prefix', type=str,
                    help='Path to the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')

args = parser.parse_args()
content_path = args.content_path
style_path = args.style_path
merge_results_path = args.merge_results_path
iterations = args.iter

total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# limit tensorflow memory allocation
from keras import backend as K
K.get_session().close()
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))


image_size = (320, 254)
img_arr = get_content(image_size, content_path)
shp = img_arr.shape
style_arr = get_style(image_size, style_path)
merge()

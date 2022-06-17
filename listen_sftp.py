import os
import argparse
from ml import call_TF
from time import sleep
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='resnet50', 
    choices=['resnet50', 'vgg16', 'mobilenet'])
args = vars(parser.parse_args())

img_dict = {
    'paper_towel': 'paper',
    'toilet_tissue': 'paper',
    'bath_towel': 'paper',
    'gown': 'paper',
    'carton': 'paper',
    'envelope': 'paper',
    'packet': 'paper',
    'menu': 'paper',
    'medicine_chest': 'paper',
    'plastic_bag': 'plastic',
    'shower_cap': 'plastic',
    'mosquito_net': 'plastic',
    'bassinet': 'plastic',
    'measuring_cup': 'plastic',
    'beaker': 'plastic',
    'sleeping_bag': 'plastic',
    'beer_glass': 'plastic',
    'pop_bottle': 'plastic',
    'shower_curtain': 'plastic',
    'cocktail_shaker': 'plastic',
    'water_bottle': 'plastic',
    'water_jug': 'plastic'
}

while True:
    if os.path.exists('img.jpg'):
        if os.path.exists('cmpl_ml'):
            os.remove('cmpl_ml')
        while not os.path.exists('cmpl_upload'):
            pass
        res, prob = call_TF(args['model'], 'img.jpg')
        print(res, prob)
        solution = 'residual'
        if res in img_dict:
            solution = img_dict[res]
        file = open('solution.txt', 'w')
        file.write(solution)
        file.close()
        os.system('touch cmpl_ml')
        os.rename('img.jpg', f"history/{datetime.now()}.jpg")
        os.remove('cmpl_upload')

from PIL import Image
import numpy as np
import subprocess
import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import importlib
import random
import json
import glob
import os
import cv2
def set_device(args):
  if torch.cuda.is_available():
    if isinstance(args, list):
      return (item.cuda() for item in args)
    else:
      return args.cuda()
  return args
class ui_model():
    mask = None 
    img = None
    result = None
    mask_tensor = None
    img_tensor = None
    config = None
    config_name_list = ['celebahq_gconv', 'celebahq_center', 'places2_gconv']
    model_name_list = ['tfpnnet', 'tfpnnet', 'tfpnnet_simple']
    index = 0
    def __init__(self):
        self.load_model()
    def change_model(self, image_type, mask_mode):
        model = '{}_{}'.format(image_type, mask_mode)
        # print('Model change to ', model)
        if model=='celebahq_gconv' and self.index!=0:
            self.index = 0
            self.load_model()
        elif model=='celebahq_center' and self.index!=1:
            self.index = 1
            self.load_model()
        elif model=='places2_gconv' and self.index!=2:
            self.index = 2
            self.load_model()
    def load_model(self):
        """Load different kind models for different datasets and mask types"""
        config_file_name = './configs/{}{}'.format(self.config_name_list[self.index], '.json')
        self.config = json.load(open(config_file_name))
        net = importlib.import_module('model.'+self.model_name_list[self.index])
        self.config['model_name'] = self.model_name_list[self.index]
        self.h = self.config['data_loader']['h']
        self.w = self.config['data_loader']['w']
        self.model = set_device(net.InpaintGenerator())
        self.config['model_dir'] = os.path.join(self.config['model_dir'], '{}_{}_{}_{}{}'.format(self.config['model_val_sub_dir'], self.config['model_name'], 
    self.config['data_loader']['name'], self.config['data_loader']['mask'], self.config['data_loader']['w']))
        latest_epoch = open(os.path.join(self.config['model_dir'], 'latest.ckpt'), 'r').read().splitlines()[-1]
        if self.index<2:
            path = os.path.join(self.config['model_dir'], '{}.pth'.format(latest_epoch))
        else:
            path = os.path.join(self.config['model_dir'], 'gen_{}.pth'.format(latest_epoch))
        data = torch.load(path, map_location = lambda storage, loc: set_device(storage)) 
        self.model.load_state_dict(data['netG'])
        self.model.eval()    
        self.img_file_dir = './datasets/{}'.format(self.config['data_loader']['name'])
    

    def postprocess(self, img):
        img = (img+1)/2*255
        img = img.permute(1,2,0)
        img = img.int().detach().cpu().numpy().astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def set_img(self, img_arr=None):
        if img_arr is not None:
            # array list to Image
            img = np.array(img_arr).reshape((self.h, self.w, 4)) # list array -> h,w,c
            # img = img.transpose([2,0,1]) # h,w,c -> c, h, w
            img = img.astype(np.uint8)
            self.img =  Image.fromarray(img).convert('RGB')
            self.img_tensor = set_device(F.to_tensor(self.img)*2-1).unsqueeze(0)

    def set_mask(self, mask_arr=None):
        """draw the mask"""
        if mask_arr is not None:
            # array list to Image
            m = np.array(mask_arr).reshape((self.h, self.w))
            m[m!=0] = 255
            self.mask = Image.fromarray(m).convert('L')

        self.mask_tensor = set_device(F.to_tensor(self.mask)).unsqueeze(0)
        self.masked_img_tensor = self.img_tensor*(1.-self.mask_tensor) + self.mask_tensor

    
    def random_ff_mask(self):
        config = {
        'img_shape': (256, 256),
        'mv' : 5,
        'ma' : 4.0,
        'ml' : 40,
        'mbw' : 10
        }
        h, w = config['img_shape']
        mask = np.zeros((h,w))
        num_v = 12+np.random.randint(config['mv'])

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1+np.random.randint(5)):
                angle = 0.01+np.random.randint(config['ma'])
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10+np.random.randint(config['ml'])
                brush_w = 10+np.random.randint(config['mbw'])
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        
        mask = mask.astype(np.float32)*255
        return Image.fromarray(mask).convert('L')
    def fill_mask(self):
        """Forward to get the generation results"""
        masks = self.mask_tensor
        images = self.img_tensor
        images_masked = self.masked_img_tensor
        with torch.no_grad():
            _, output = self.model(torch.cat((images_masked, masks), dim=1), masks)
            comp_imgs = (1-masks)*images + masks*output
            self.result = self.postprocess(comp_imgs[0])
        self.result.save('result.png')

    def inpaint(self, img_arr, mask_arr):
        self.set_img(img_arr)
        self.set_mask(mask_arr)
        self.fill_mask()
        ret = self.result.convert("RGBA")
        ret = np.array(ret).reshape(-1)
        return ret.tolist()

    def randomImage(self, image_type, mask_mode):
        self.change_model(image_type, mask_mode)
        self.img_names = list(glob.glob('{}/*.jpg'.format(self.img_file_dir)))
        image_num = len(self.img_names)   
        item = random.randint(0, image_num-1)
        self.fname = self.img_names[item]
        self.img = Image.open(self.fname).convert("RGB")
        self.img = self.img.resize((self.w, self.h))
        self.img_tensor = set_device(F.to_tensor(self.img)*2-1).unsqueeze(0)
        if mask_mode == 'gconv':
            self.mask = self.random_ff_mask()
        elif mask_mode == 'center':
            m = np.zeros((self.h, self.w)).astype(np.uint8)
            m[self.h//4:self.h*3//4, self.w//4:self.w*3//4] = 255
            self.mask = Image.fromarray(m).convert('L')
            self.mask = self.mask.resize((self.w, self.h), Image.NEAREST)
        self.mask_tensor = set_device(F.to_tensor(self.mask)).unsqueeze(0)
        self.masked_img_tensor = self.img_tensor*(1.-self.mask_tensor) + self.mask_tensor

        self.fill_mask()
        image = np.array(self.img.convert("RGBA")).reshape(-1)
        mask = np.array(self.mask.convert("RGBA")).reshape(-1)
        mask_len = mask.size
        for i in range(0, mask_len, 4):
            if mask[i]>0:
                mask[i]=251
                mask[i+1]=150
                mask[i+2]=107
                mask[i+3]=255
            else:
                mask[i+3]=0
        result = np.array(self.result.convert("RGBA")).reshape(-1)
        return {'image':image.tolist(), 'mask':mask.tolist(), 'result': result.tolist()}


def create_app():
    app = Flask(__name__)

    model = ui_model()


    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'

    @app.route('/inpaint', methods=['POST'])
    @cross_origin()
    def inpaint_req():
        if request.method == 'POST':
            data = request.get_json(force=True)
            return jsonify(model.inpaint(data['image'], data['mask']))
        return jsonify({"error": "Must be post request"})

    @app.route('/changeModel', methods=['POST'])
    @cross_origin()
    def change_model_req():
        if request.method == 'POST':
            data = request.get_json(force=True)
            return jsonify(model.change_model(data['image_type'], data['mask_mode']))
        return jsonify({"error": "Must be post request"})
    
    @app.route('/randomImage', methods=['POST'])
    @cross_origin()
    def random_image_req():
        if request.method == 'POST':
            data = request.get_json(force=True)
            return jsonify(model.randomImage(data['image_type'], data['mask_mode']))
        return jsonify({"error": "Must be post request"})

    return app

if __name__ == "__main__":
    app = create_app()
    app.run()
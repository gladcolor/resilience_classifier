from werkzeug.utils import secure_filename
from flask import Flask,render_template,send_from_directory,jsonify,request
import time
import os

app = Flask(__name__)

UPLOAD_FOLDER='uploads'
ALLOWED_EXTENSIONS = set(['txt','png','jpg','jpeg','xls','JPG','PNG','xlsx','gif','GIF'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))

from PIL import Image
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms

class ImageClassifier():
    def __init__(self, model_path=None):
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'

        model_ft = models.inception_v3(pretrained=None)
        num_classes = 2

        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        self.input_size = 299

        model = model_ft

        self.model = model
        self.model = self.model.to(self.device)
        if model_path is not None:
            self.model_path = model_path
            # self.model = torch.load(model_path)
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu') ))
            self.model.eval()
    #
    # def getModel(self, model_path):
    #     model = torch.load(model_path)
    #     return model

    def image_inference(self, image_path):
        img = Image.open(image_path)
        img = transforms.Resize(self.input_size)(img)
        img = transforms.ToTensor()(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)
        result = self.model(img)
        result = nn.Softmax()(result)
        return result.cpu().detach().numpy()


ic = ImageClassifier('data/inception83.pth')

# 用于判断文件后缀
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

def get_filepaths(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            L.append('/' + app.config['UPLOAD_FOLDER'] +'/'+file)  
    return L  


@app.route("/")
def index():
    filenames = sorted(get_filepaths(os.path.join(basedir,app.config['UPLOAD_FOLDER'])))
    # print(filenames)
    return render_template('index.html',filenames=filenames)


# 上传文件
@app.route('/upload',methods=['POST'],strict_slashes=False)
def api_upload():
    file_dir=os.path.join(basedir,app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    files=request.files['files']  # 从表单的file字段获取文件，myfile为该表单的name值
    res = []
    unix_time = int(time.time())
    i = 0
    for f in request.files.getlist('files'):
        i = i + 1
        if f and allowed_file(f.filename):  # 判断是否是允许上传的文件类型
            fname=secure_filename(f.filename)
            arr = fname.rsplit('.',1)
            # print(arr)
            if len(arr) > 1:
                ext = arr[1]
            else:
                ext = fname
            new_filename=str(unix_time) + '_' + str(i) +'.'+ext  # 修改了上传的文件名
            real_filepath = os.path.join(file_dir,new_filename)
            print(real_filepath)
            f.save(os.path.join(file_dir,new_filename))  #保存文件到upload目录

            result = ic.image_inference(real_filepath)
            basename = os.path.basename(real_filepath)
            label = np.argmax(result)
            print(f"{basename} | label: {label}")

            msg = "Yes" if label==1 else "No"
            res.append({
                "name": "Filename: " + new_filename,
                "pic": "/uploads/" + new_filename,
                "size": "Resilience: " + msg
            })
        else:
            return jsonify({"name": 1001, "pic": "", "size": "upload fail"})

    return jsonify(res)



@app.route("/uploads/<filename>")
def download(filename):
    if request.method=="GET":
        if os.path.isfile(os.path.join('uploads', filename)):
            # 这里主要需要注意的是send_from_directory方法，经过实测，需加参数as_attachment=True，
            # 否则对于图片格式、txt格式，会把文件内容直接显示在浏览器，
            # 对于xlsx等格式，虽然会下载，但是下载的文件名也不正确，切记切记
            return send_from_directory('uploads', filename)
        abort(404)


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='127.0.0.1', debug=True, port=8070)


from flask import Flask, render_template, request, flash, redirect, url_for, Response, make_response
from werkzeug.utils import secure_filename
import urllib.request
import os
import numpy as np
from predict_poly import polygon_detection
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import io
import base64
from io import BytesIO



app = Flask(__name__)
app.secret_key = "supposedlysecretkey"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('Please upload the image')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route("/poly/<filename>", methods = ['GET','POST'])
def poly_predict(filename):
    print('predict file name:' + filename)
    img_path = 'static/uploads/' + filename
    img_points,poly_points = polygon_detection(img_path)
    print("plotting the graph")
    fig = vis_polys(img_points,poly_points,'Object Polygon')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    buffer = b''.join(buf)
    b2 = base64.b64encode(buffer)
    fig2=b2.decode('utf-8')
    return render_template('index.html', prediction=True, fig=fig2, filename=filename)


def vis_polys(img, poly, title=''):
    fig = Figure()
    ax = fig.add_subplot(1,1,1)
    h, w = img.shape[:2]
    ax.imshow(img, aspect='equal')
    patch_poly = patches.Polygon(poly, alpha=0.6, color='blue')
    ax.add_patch(patch_poly)
    poly = np.append(poly, [poly[0, :]], axis=0)
    #
    ax.plot(poly[:, 0] * w, poly[:, 1] * h, '-o', linewidth=2, color='orange')
    # first point different color
    ax.plot(poly[0, 0] * w, poly[0, 1] * h, '-o', linewidth=3, color='blue')
    ax.set_title(title)
    ax.axis('off')
    return fig


if __name__ == "__main__":
    app.run(debug= True)

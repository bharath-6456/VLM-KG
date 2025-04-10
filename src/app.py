from flask import Flask, render_template, request
import os
from clip_model import get_clip_predictions
from yolo_module import detect_objects
from graph_module import build_graph_and_html

UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        question = request.form['question']

        if image_file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(filepath)

            clip_results = get_clip_predictions(filepath)
            yolo_objects = detect_objects(filepath)
            graph_html = build_graph_and_html(yolo_objects, question)

            return render_template('index.html',
                                   image_path=filepath,
                                   clip=clip_results,
                                   yolo=yolo_objects,
                                   graph_html=graph_html,
                                   question=question)
    return render_template('index.html')

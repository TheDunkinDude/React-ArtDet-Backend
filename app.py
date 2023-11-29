from flask import Flask, render_template, request , jsonify

import tensorflow as tf
import cv2,imghdr, os
import numpy as np
from flask_cors import CORS 
#import mysql.connector

app = Flask(__name__)
CORS(app)

'''mydb = mysql.connector.connect(
    host='localhost',
    user = 'TharrunS',
    password = 'GreatPretender', 
    database = "AI_ART_Detector",
    port = 3306
) 


logCursor = mydb.cursor()

sqlQueryInsert = "Insert into prediction_logs (File, Prediction) values (%s, %s);"

sqlQueryUpdate = "Update  prediction_logs set Prediction = %s  where File = %s;"

sqlQueryInsertReport = "Insert into report_log (Error_Section, Complaint) values(%s, %s);"'''

def pred_img(img):
    im = cv2.imread(img)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    resize = tf.image.resize(im,(256,256))
    yhat = model.predict(np.expand_dims(resize/255,0))
    if (yhat >= 0.5):
        return "Real Art"
    else:
        return "AI Art"
 
 
model = tf.keras.models.load_model('my_model.keras')
    
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

 
@app.route('/', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    if image:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(filename)

        pred = pred_img(filename)
        
        '''exsistQ = "Select Exists (Select * from prediction_logs where File = " +  "'" + image.filename +"'" + ");" 

        logCursor.execute(exsistQ)

        isThere = logCursor.fetchall()

        if (isThere[0][0] == 1):
            val = (pred,image.filename)
            logCursor.execute(sqlQueryUpdate, val)
            mydb.commit()

        else:
            val = (image.filename,pred)
            logCursor.execute(sqlQueryInsert, val)
            mydb.commit()'''
        
        return jsonify({'message': 'Image uploaded successfully', 'filename': image.filename, 'pred' : pred , 'messsage' : isThere[0][0]})
    else:
        return jsonify({'error': 'Failed to upload image'}), 500
'''
@app.route('/api/submit', methods=['POST'])
def submit_data():
    try:
        data = request.json
        valdat = []
        valdat.append(data)
        vald = (data.get('section'), data.get('complain'))
        logCursor.execute(sqlQueryInsertReport, vald)
        mydb.commit()
        return jsonify({'message': 'Data submitted successfully', 'data': valdat})
    except Exception as e:
        return jsonify({'error': 'Failed to submit data'}), 500'''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

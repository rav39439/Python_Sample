
import os
from keras.models import load_model
import h5py
import gcsfs
import base64
import io
from PIL import Image
from flask_cors import CORS
from flask import Flask, render_template, request, redirect, url_for,Response,jsonify
import numpy as np

my_string = ""
accuracy=0
model_path = os.path.join(os.getcwd(), 'models', 'modelupdated2.txt')

def processimg(data):
    MODEL_PATH = 'gs://ml_models_obs/model.h5'
    FS = gcsfs.GCSFileSystem(project='tactile-education-services-pvt',
                             token={
                                 "type": "service_account",
                                 "project_id": "tactile-education-services-pvt",
                                 "private_key_id": "57985414f3e5c0b2f15541801964bc4b3ffb1611",
                                 "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCqfNls+kD30Rbj\nwUbZW2+1xr3fdSCh6Nm/F4ACwpF+b5gxsXAWoaDOGOQI2ujjdDcnFkyViDqWRsz+\nsK/+lOTZJef5ti85lZ35Ince3N3yHkfNlxdB3So+Ryl+O/Kyf74OYylr2+9mrbuL\nw5OoOutidbJPwIp9hxTrw2UyJbiZ7fYG7GISGmoCZwIC3gMdAJ3Vlnj6DnuI+fEc\n09+t+TwVsBBD24938UV0p3k7kL5MD10sCTX62WUYDNGk7HmWmDOL5ysroxW/gEKC\nE4gWVYaBaPtIrneGWFLC4MkwOKhhdZfBqRy+uoR4bq0BUYscBRDHFDGdSjEM6EgQ\nkCrFsyIhAgMBAAECggEABOoTpeSqVAJo0aCgA1KzjAa6MStqCQTCZxPdNqcnZMDA\nfzk1SQ+4aAx3YUp3IVxXnSazRzGtt6h7jwFml1TxKlHBh7UQ1Cz8CGMOEjuNYcXt\n7a8T5mwCxFgOigMeHjdYsgc5edCVjG0IXZFiC5zELXrFTK95BoCK8bdYjssf0L8L\nUrsmjct1hLJ9L1Gvuu5oT1fni/kvVMx9MiBQzx5JD6jHgpDpYSkdcmBPuxn6KVoT\nCgYv7nFYLBoF43kq+OrKujyGXcwyP1JEw2Zns6LuLqFeY96443hKtsCXwKIBM7M6\nIvc4A4mx2mm6PEXZF0r2cnGIavOvr2uA0fW5PxHOoQKBgQDdPhyWSc0+GhQj/OKu\neyU0/4Ixdzud0ZLNdRhtPeLdsEkk/EcST6C0c+UDxvz4WR3MGrzmjU8Y7q3WpeLp\nvQx4XoBmyOnI+5OIS7MvshFUOJCVgdJa+JAWFgKPp6jcCDVM/fCHXU6dE5Rh1AOG\ngz03Ft7FMvZDOZcYZrl8JR88sQKBgQDFRXyag5yfxDVox7+HbvhtZIqhZ/6lacdR\n6saEWfPBPjWwKpL0brsqktudoTNNtHC+MmId6hKbgu9rz5VZs3cDFG7VR+9HTv4C\nvVLuxWCGs1npLLL7es2MRBNLTkpWK8MvcxgBBwGkmwei54Eokr469xFHBBYmjpHg\ntyno3/TYcQKBgAWQOkfNM0wMe9Ur1sdscVMT2cJErUsaqgZgm2yj0cChXjV/4omj\nVvcyst+VcWcNVqJ7SaTCiOqnldd/9GTMTDP6rF/pTXewW9Vhke/xGl5zza70xMVk\n2rqzcv0JykU+L5jwCcxdnEx24ZRUMIKBalioSpHK5kZqfFIwwxlMFa0BAoGACbUe\nqZfaaD5Gho51zVtXnEJ7U/ADJu6qoUxVUoP+q76885tUufSM/05UwlABb0x209U0\n4NX47nAmCf8gEVb0f1FNFu/ARZkMhOP+JcPOuTIwNXo+oINtg/6BmI4UuGLU6wvV\n3Y8TtJ7wZELSJ0X5WHt5/S2lTWZspUs+I7iJmvECgYBpcTpWFKwaSIy92c+mtcPb\n2QPKc/lZgWZoHU02PV0rHuHTFkYLaDTMwI0cdB8hpPkJakPk0CrQ6UzNTf7nSzAA\nOmoZNzkJ9g4NLMj9rFChBNf+Q8SwNGSsxFu4B0pz2nAMaugLL10CM08a6JNwHBr1\nDVR302lbltIW+D+j4oa14A==\n-----END PRIVATE KEY-----\n",
                                 "client_email": "google-vision-api@tactile-education-services-pvt.iam.gserviceaccount.com",
                                 "client_id": "106637251090666895947",
                                 "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                 "token_uri": "https://oauth2.googleapis.com/token",
                                 "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                                 "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/google-vision-api%40tactile-education-services-pvt.iam.gserviceaccount.com"
                             })
    with FS.open(MODEL_PATH, 'rb') as model_file:
        model_gcs = h5py.File(model_file, 'r')
        model = load_model(model_gcs)
# Load the Keras model from the downloaded file
        img = Image.open(io.BytesIO(data))
        im = img.resize((256, 256))
        image = np.array(im)
        input_arr = np.array([image])
        print(model.predict(input_arr)[0][0])
        if model.predict(input_arr)[0][0] < 0.5:
            print(model.predict(input_arr))
            return {'message':"Bottle is not pressed",'accuracy':model.predict(input_arr)[0][0]}
        else:
            return {'message':"Bottle is pressed",'accuracy':model.predict(input_arr)[0][0]}


app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=['POST', "GET"])
def upload_image():
    if request.method == 'POST':
        result = ""
        data = request.files['file'].read()
        image = request.files['file']
        my_string = base64.b64encode(data)
        if image.filename == '':
            print("filename is invalid")
            return redirect(request.url)
        else:
            result = processimg(data)
            d = my_string.decode("utf-8")
            return render_template('index.html', filename=d, result=result['message'],accuracypressed=(result['accuracy']-1)/0.5*100,accuracynotpressed=result['accuracy']/0.5*100)
    return render_template('index.html')

@app.route("/uploadFile", methods=['POST','GET'])
def getdata():
    if request.method == 'POST':
        data = request.files['file'].read()
        image = request.files['file']
        my_string = base64.b64encode(data)
        if image.filename == '':
           resp={
            'message':"error",
           }
           response=jsonify(resp)
           return response
        else:
           result = processimg(data)
           d = my_string.decode("utf-8")
           resp = {
               'message':d,
               'result':result
           }
           response=jsonify(resp)
           # return send_file(io.BytesIO(data), mimetype='image/jpeg')
           return response


    else:
        resp={
               'message':"Method is not allowed",

        }

    return jsonify(resp)





if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))

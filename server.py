import numpy as np 
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template 
from pathlib import Path 
from feature_extraction import FeatureExtractor

app = Flask(__name__)
fe = FeatureExtractor()
features =  []
img_path = [ ]
for features_path in Path("./static/feature").glob("*.npy"): 
   features.append(np.load(features_path))
   img_path.append(Path("./static/img")/(features_path.stem + ".jpg" ))
features = np.array(features)
@app.route("/",methods = ["GET","POST"])
def index(): 
  if request.method == "POST": 
    file = request.files["query_img"]
    img = Image.open(file.stream)
    uploaded_img_path = "static/uploaded" + datetime.now().isoformat().replace(":",".")+"_"+file.filename
    img.save(uploaded_img_path)


    query = fe.extract(img)
    dists = np.linalg.norm(features- query, axis = 1 )
    ids = np.argsort(dists)[:30]
    scores = [(dists[id],img_path[id]) for id in ids]
    print(scores)
    return render_template("index.html",query_path = uploaded_img_path,scores = scores)
  else : 
    return render_template("index.html")
if __name__ =="__main__": 
  app.run()
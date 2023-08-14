import numpy as np 
from PIL import Image
from feature_extraction import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template 
from pathlib import Path 

app = Flask(__name__)
@app.route("/")
def index(): 
  return render_template("index.html")
if __name__ =="__main__": 
  app.run()
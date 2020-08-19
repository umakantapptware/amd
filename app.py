from flask import Flask, render_template, request
import os
import time
from keras.models import model_from_json
from pickle import dump,load
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()
import os
os.environ['CUDA_HOME'] = 'C:/path/to/cuda'  # Replace with your CUDA installation directory
os.environ['PATH'] += ';C:/path/to/cuda/bin' 

from tensorflow.keras.models import load_model

model = load_model('model_1.h5')
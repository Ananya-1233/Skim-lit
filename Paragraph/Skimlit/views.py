from django.shortcuts import render
from Skimlit.ml_model import make_prediction
#from Skimlit.pred import make_skimlit_predictions
import tensorflow as tf
from tensorflow.keras.models import load_model
from joblib import load



def predict(request):
    if request.method == 'POST':
        text = request.POST['input-text']
        if isinstance(text, str):
          text = text.lower()
        else:
          text = "" 
        model = load_model('Skimlit\model_main')  # Load your model
        #preprocessed_data = preprocess_data(text)  # Preprocess the input data
        predictions = make_prediction(model, text)  # Make predictions using the model
        context = {'predictions': predictions}
        return render(request, 'index.html', context)
    else:
        return render(request, 'index.html')
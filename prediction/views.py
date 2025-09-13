
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from .models import preprocess_text_for_prediction, loaded_model2
from json import dumps

def predict(request):
    if request.method == 'POST':
        text_to_predict = request.POST.get('text_to_predict')
        preprocessed_text = preprocess_text_for_prediction(text_to_predict)
        prediction = loaded_model2.predict(preprocessed_text)
        
        # Convertir la prédiction en texte correspondant
        if prediction == [1.0]:
            resultat = "IA"
        elif prediction == [0.0]:
            resultat = "humain"
        else:
            resultat = "indéterminé"
        
        return render(request, 'prediction.html', {"resultat" : resultat, "text_to_predict": text_to_predict})
    else:
        return render(request, 'prediction.html')




def index(request):
    return render(request, 'index.html')


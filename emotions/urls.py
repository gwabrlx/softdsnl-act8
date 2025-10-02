from django.urls import path
from .views import PredictEmotion

urlpatterns = [
    path("predict-emotion/", PredictEmotion.as_view(), name="predict_emotion"),
]

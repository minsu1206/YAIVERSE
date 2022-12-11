from django.contrib import admin
from .models import InferenceData, ImageData

# Register your models here.
admin.site.register(InferenceData)
admin.site.register(ImageData)
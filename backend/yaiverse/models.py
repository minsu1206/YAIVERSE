from django.db import models

# Create your models here.

class InferenceData(models.Model):
    col = models.CharField(max_length=10, unique=True)
    user_code = models.CharField(max_length=10)
    
    def __str__(self):
        return self.col + " -- "+self.user_code
    
class ImageData(models.Model):
    image = models.ForeignKey(InferenceData, on_delete=models.CASCADE)
    image_col = models.CharField(max_length=9, unique=True)
    style = models.CharField(max_length=30)
    sub_style = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)
    fail = models.BooleanField(default=False)
    def __str__(self):
        return self.image_col + " -- "+self.style
    
    
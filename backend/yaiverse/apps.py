from django.apps import AppConfig
from yaiverse.inference.jojo_inference import ConvertModel

class YaiverseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'yaiverse'
    ## Load model with initiating server
    convertModel = ConvertModel()

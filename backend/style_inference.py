import requests
import os

style_mapping = ["JOJO","DISNEY", "SKETCH","ART","ARCANE","침착맨", "이태원클라쓰", "외모지상주의", "프리드로우", "여신강림"]
style_mapping = style_mapping[8:9]

images = os.listdir("/home/yai/backend/sample_images")

for image in images:
    for style in style_mapping:
        print(image, style)
        files = {'file': open(f'/home/yai/backend/sample_images/{image}', 'rb')}
        data = {
            
            'style': style,
            'id': image.split('.')[0]
        }
        r = requests.post('http://119.192.224.18:8000/file/',data=data, files=files)
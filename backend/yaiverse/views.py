from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from rest_framework import status
from .models import InferenceData, ImageData
import os
import random
import string
from .apps import YaiverseConfig
from PIL import Image
import random
import logging
import json
from datetime import datetime




logger = logging.getLogger('yaiverse')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# log를 파일에 출력
file_handler = logging.FileHandler('my.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


style_mapping = {
        "DISNEY": ["disney","disney_preserve_color","style_anna","style_elsa", "style_moana"],
        "JOJO": ["jojo", "jojo_preserve_color", "jojo_yasuho", "jojo_yasuho_preserve_color"],
        "ARCANE": ["arcane_jinx","arcane_jinx_preserve_color","arcane_caitlyn","arcane_caitlyn_preserve_color", "arcane_multi","arcane_multi_preserve_color"],
        "ART":["art", "style_art07", "style_art05"],
        "SKETCH": ["sketch_multi","style_sketch05","style_sketch07"],
        "프리드로우": ["style_free_minji","style_free_taesung","style_donggga", "style_free_juhee"],
        "외모지상주의":["style_YJ_hyeongsuk","style_YJ_basco", "style_YJ_janghyeon"],
        "여신강림":["style_YS_seojun","style_jugyeong","style_sooho_new"],
        "침착맨":["style_malnyeon_1","style_malnyeon_2","style_malnyeon_3","style_malnyeon_4"],
        "이태원클라쓰":["style_saeroi", "style_itae_mahyeoni","style_itae_joiseo","style_itae_daehi","style_itae_suah"]
    }


def generate_code(number):
    rand_str = ''
    for _ in range(number):
        rand_str += str(random.choice(string.ascii_letters + string.digits))
    return rand_str


"""
Return inference history of user
"""
@csrf_exempt
def historyView(request, user_code):
    if request.method == 'GET':
        qs = ImageData.objects.filter(image__user_code=user_code).filter(fail=False).order_by('-timestamp').values_list('image_col', flat=True).distinct()
        result = {'data': list(qs)}
        return JsonResponse(result)
    
    
def listView(request):
    qs = ImageData.objects.all().order_by('-timestamp')[:40]
    context = {'qs': qs}
    return render(request, 'list.html', context)
    
"""
inference
"""
@csrf_exempt
def firstFileView(request):
    if request.method == 'POST':
        file =request.FILES.get('file')
        style = request.POST.get('style', "")
        user_code = request.POST.get('id', "")
        
        col = generate_code(10)
        image_col = generate_code(9)
        logger.info(f'first inference -- col:{col}, user: {user_code}, style: {style}')
        data = InferenceData.objects.create(col=col,user_code=user_code)
        imageData = ImageData.objects.create(image=data, image_col=image_col, style=style)
        sub_style = random.choice(style_mapping[style])
        imageData.sub_style = sub_style
        imageData.save()
        save_file(file, col)
        return inference(imageData, sub_style)
    else:
        return HttpResponse('error', status=404)
    

@csrf_exempt
def sameStyleInferenceView(request, image_col):
    if request.method == 'POST':
        imageData = ImageData.objects.get(image_col=image_col)
        image_col = generate_code(9)
        logger.info(f'same style -- col:{imageData.image.col}, image_col:{imageData.image_col}, user: {imageData.image.user_code}, style: {imageData.style}, sub_style: {imageData.sub_style}')
        newImageData = ImageData.objects.create(image=imageData.image, style=imageData.style, image_col=image_col)
        sub_style = get_style_for_same_style(imageData, imageData.style)
        newImageData.sub_style = sub_style
        newImageData.save()
        
        return inference(newImageData, sub_style, with_pt = True)
    else:
        return HttpResponse('error', status=404)
        
    
@csrf_exempt
def diffStyleInferenceView(request, col):
    if request.method == 'POST':
        body = json.loads(request.body)
        style = body["style"]
        
        data = InferenceData.objects.get(col=col)
        image_col = generate_code(9)

        imageData = ImageData.objects.create(image=data, style=style, image_col=image_col)
        sub_style = random.choice(style_mapping[style])
        imageData.sub_style = sub_style
        imageData.save()
        logger.info(f'diff style -- col:{imageData.image.col}, image_col:{imageData.image_col}, user: {imageData.image.user_code}, style: {imageData.style}')
        return inference(imageData, sub_style, with_pt = True)
    
    
def save_file(file, col):
    img_dir = "/home/yai/backend/data"
    try:
        os.makedirs('{}/{}'.format(img_dir, col))
    except:
        print('error!')
    
    with open('{}/{}/image.jpg'.format(img_dir,col), 'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)
            
    temp_img = Image.open('{}/{}/image.jpg'.format(img_dir,col)).convert('RGB')
    temp_img.save('{}/{}/image.jpg'.format(img_dir,col))
    
def inference(imageData, sub_style, with_pt=False):
    start = datetime.now()
    try:
        if with_pt:
            YaiverseConfig.convertModel.generate_face_with_pt(imageData.image.col, imageData.image_col, sub_style)
        else:
            YaiverseConfig.convertModel.generate_face(imageData.image.col, imageData.image_col, sub_style)
    except: 
        logger.warning("inference fail")
        imageData.fail = True
        imageData.save()
        return HttpResponse(status=500)
    logger.info("inference success")
    data = {
        "col": imageData.image.col,
        "image_col" : imageData.image_col,
    }
    logger.info(f'inference time: {datetime.now() - start}')
    return JsonResponse(data=data)


def get_style_for_same_style(imageData, style):
    sub_style = random.choice(style_mapping[style])
    while imageData.sub_style == sub_style:
        sub_style = random.choice(style_mapping[style])
    return sub_style
    


"""
Serve file
"""
@csrf_exempt
def fileGetView(request, image_col):
    if request.method == 'GET':
        imageData = ImageData.objects.get(image_col=image_col)
        img_dir = "/home/yai/backend/data"
        filename = f"{image_col}.jpg"
        file_loc = '{}/{}/{}'.format(img_dir,imageData.image.col,filename)
        content_type = 'image/jpg'
        
        try:
            with open(file_loc, 'rb') as f:
                file_data = f.read()
        except:
            return HttpResponse(status=500)
        
        response = HttpResponse(file_data, content_type=content_type)
        response['Content-Disposition'] = 'attachment; filename="{}"'.format(filename)
        return response

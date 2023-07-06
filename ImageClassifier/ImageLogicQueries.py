import requests
from PIL import Image
import io

def getImagesAndTags(user, ai):
    # e.g. images_of_ai_api_url"http://localhost:5000/account/friendly.henry/Finn%20The%20Human/posts"
    #images_of_ai_api_url = "http://localhost:5000/account/"+user+"/"+ai+"/posts" 
    images_of_ai_api_url = "https://applogic.wwwelco.me:5000/account/"+user+"/"+ai+"/alltags"
    response = requests.get(images_of_ai_api_url)
    imagesAndTags = response.json()
    return imagesAndTags

def getLabels(user, ai, imageId):
    # e.g. tags_of_image_api_url = "http://localhost:5000/account/friendly.henry/Finn%20The%20Human/1345d065-7c1b-445a-b0ec-591510c7ab74/tagsonpost"
    tags_of_image_api_url = "https://applogic.wwwelco.me:5000/account/"+user+"/"+ai+"/"+imageId+"/tagsonpost"
    response = requests.get(tags_of_image_api_url)
    labels = response.json()
    return labels

def downloadImage(imageId):
    image_to_download_api_url = "https://applogic.wwwelco.me:5000/images/"+imageId
    response = requests.get(image_to_download_api_url)
    in_memory_file = io.BytesIO(response.content)
    return in_memory_file

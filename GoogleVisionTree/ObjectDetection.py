import os, io
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import pandas as pd
import requests
import json
import urllib.request
from PIL import Image
from DecisionTree import getDecisionTree, query

pd.options.mode.chained_assignment = None  # default='warn'

#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\henri\Documents\Schoolwork\Capstone\People-Welcome\peoplewelcome-group_10\GoogleVisionTree\manifest-device-382622-f483e0b7bccc.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'/home/ubuntu/Documents/manifest-device.json'

client = vision_v1.ImageAnnotatorClient()

# returns list of objects found in image hosted at url
def detect_objects(url: str):
    urllib.request.urlretrieve(url, 'file_name')
    with open('file_name', 'rb') as image_file:
        content = image_file.read()
    image = vision_v1.Image(content=content)
    objects = client.object_localization(
        image=image).localized_object_annotations
    
    return objects

# returns true if objects are found in image at url with imageId, false if not
def checkForObjects(imageId):
    res = requests.get('https://applogic.wwwelco.me:5000/post/{}/objectsfound'.format(imageId))
    data = json.loads(res.text)
    return data['objectsFound']

# returns objects found in image at url with imageId
def getObjects(imageId):
    res = requests.get('https://applogic.wwwelco.me:5000/post/{}/objects'.format(imageId))
    data = json.loads(res.text)
    return data['objects']

# returns dictionary associating image id's, objects, and tags, based on which images
# a specific ai has tagged, and the specific tags that they have labeled each image with
def createObjectTagMap(user: str, ai: str):
    res = requests.get('https://applogic.wwwelco.me:5000/account/{}/{}/alltags'.format(user, ai))
    tags_map = json.loads(res.text) # tags_map in form [{'tags': [], 'image_id': imageId}, ...]

    image_objects_dict = {} # dictionary that will be returned at end of function
    all_objects = [] # list to contain all objects contained in all images tagged by ai
    all_tags = [] # list to contain all tags ever tagged on any image by ai
    all_ids = [] # list to contain id's of all images ever tagged by ai

    # for each {'tags': [], 'image_id': imageId} item
    for item in tags_map:
        objects = [] # initialize list to contain objects found in image at url with item['image_id']

        # if list of objects are found in DynamoDB
        if(checkForObjects(item['image_id'])):
            # store list in 'objects' variable
            objects = getObjects(item['image_id'])
        else: # otherwise
            # run detect_objects method with item['image_id'] passed as url parameter,
            # store response in 'data' variable
            data = detect_objects('https://applogic.wwwelco.me:5000/images/{}'.format(item['image_id']))

            # insert each object name into 'objects' list
            for object_ in data:
                objects.append(object_.name)

            # store list of detected objects in DynamoDB
            requests.put('https://applogic.wwwelco.me:5000/post/{}/updateobjects'.format(item['image_id']), json={'objects': objects})

        # for each element in 'objects' list
        for object_ in objects:

            # if this object is not yet in all_objects list
            if (all_objects.__contains__(object_) == False):
                # insert this object in all_objects list
                all_objects.append(object_)

        # if 'objects' list is not empty
        if (objects != []):
            image_dict = {
                'tags': item['tags'],
                'objects': objects
            }
            # store image_dict containing tags and objects, in image_objects_dict at key 'image_id'
            image_objects_dict[item['image_id']] = image_dict
            # insert this 'image_id' in 'all_ids' list
            all_ids.append(item['image_id'])
            
            # for each tag the ai has put on this image
            for tag in item['tags']:
                # insert this tag in 'all_tags' list if it is not already in it
                if (all_tags.__contains__(tag) == False):
                    all_tags.append(tag)

    # add each object and tag as column in df
    id_column = ['id']
    df_columns = id_column
    for object in all_objects:
        df_columns.append(object)
    for tag in all_tags:
        df_columns.append(tag)

    df = pd.DataFrame(columns = df_columns)

    # insert 1 in dataframe if tag or object is present in image, 0 otherwise
    for id in all_ids:
        columns_dict = {}
        for column in df_columns:
            columns_dict[column] = 0

        columns_dict['id'] = id
        for object_ in image_objects_dict[id]['objects']:
            columns_dict[object_] = 1
        for tag in image_objects_dict[id]['tags']:
            columns_dict[tag] = 1
        df.loc[len(df.index)] = columns_dict

    return [df, all_objects, all_tags]

# builds treeMap of the form {tag: tree} given a dataframe and list of tags in dataframe
def buildTreeMap(df, inputs):
    featureCount = len(inputs)

    treeMap = {}
    for input in inputs:
        treeMap[input] = getDecisionTree(df, False, input, featureCount)

    #print(treeMap)
    return treeMap

# builds dataframe to test a decision tree model, given a list 
# of image id's and a list of all possible objects
def buildTestFrame(idList, allFeatures):
    imageObjectsDict = {} # dictionary to contain image id - object list associations

    # for each image id
    for id in idList:
        objects = [] # list to store all objects
        # if objects are found in DynamoDB for this image
        if(checkForObjects(id)):
            # store objects in 'objects'
            objects = getObjects(id)
        else: # otherwise
            # detect objects in image and store in data
            data = detect_objects('https://applogic.wwwelco.me:5000/images/{}'.format(id))
            # insert each object name in 'objects' list
            for object_ in data:
                objects.append(object_.name)
        
        # insert 'objects' list in dictionary using image id as key
        imageObjectsDict[id] = objects

    # add each object name as a column in dataframe 'df'
    columns = ['id']
    for feature in allFeatures:
        columns.append(feature)
    
    df = pd.DataFrame(columns=columns)

    # insert 1 in dataframe if object is present in image, 0 if not
    for id in idList:
        columnsDict = {}
        columnsDict['id'] = id
        for feature in allFeatures:
            if feature in imageObjectsDict[id]:
                columnsDict[feature] = 1
            else:
                columnsDict[feature] = 0
        df.loc[len(df.index)] = columnsDict

    # return test dataframe
    print(df)
    return df

id_list = ['d1d41362-2cec-4213-afb6-d50833c8c60e', '6258cce5-f388-4dea-8995-6198854a1431', '7295f7e5-0d2b-49b5-8125-8ddb44c4dc23', 'c086424b-111a-4394-b729-b742e26434af']

# get a decision tree model and necessary data from user and ai
def retrainTree(user, ai):
    data = createObjectTagMap(user, ai)

    # listify treeMap, allObjects, and allTags
    treeData = {'treeMap': buildTreeMap(data[0], data[2]), 'objects': data[1], 'tags': data[2]}

    # store tree and tag and object data to this ai in DynamoDB

    try:
        requests.post('https://applogic.wwwelco.me:5000/upload/account/{}/{}/updatetree'.format(user, ai), json={'tree': treeData}, timeout=0.0000000001)
    except requests.exceptions.ReadTimeout: 
        pass
    
    #print(response)
    #print(response.status_code)
    #print(response.json)
    #print(response.text)
    #print(response.content)
    return treeData

def getSuggestionsByAi(user, ai, idList):
    res = requests.get('https://applogic.wwwelco.me:5000/account/{}/{}/treefound'.format(user, ai))
    data = json.loads(res.text)
    treeFound = data['treeFound']

    if treeFound:
        print('tree found')
        urllib.request.urlretrieve('https://applogic.wwwelco.me:5000/account/{}/{}/tree'.format(user, ai), 'tree.json')
        
        file = open('tree.json', 'r')
        text = file.read()
        data = json.loads(text)
        file.close()
    else:
        print('tree not found')
        data = retrainTree(user, ai)
    
    return getSuggestionsFromTree(data['treeMap'], idList, data['objects'], data['tags'], treeFound)



# return tag suggestions in form [{'image_id': imageId, 'tags': []}, ...]
# given a list of image id's 'idList', a decision tree 'treeMap', a list
# of all possible objects and a list of all possible tags
def getSuggestionsFromTree(treeMap, idList, allInputs, allOutputs, treeFound):
    dfTest = '' # test dataframe
    dfTest = buildTestFrame(idList, allInputs)

    # create index dictionary to keep track of which object is at which dataframe column
    indexDict = {}
    i = 1
    for col in dfTest.columns.values[1:len(dfTest.columns.values)]:
        print(col)
        indexDict[col] = i
        i = i + 1

    suggestionDict = {} # tag suggestion dictionary to be returned at end of function

    i = 0
    # for each image id
    suggestions = [] # list to contain suggestions
    for id in idList:
        # for each tag on image
        for tag in allOutputs:
            # if decision tree query returns 1 (yes)
            #print('tag:', tag)
            if query(dfTest.iloc[i], indexDict, treeMap[tag], treeFound) == 1:
                # insert tag into list of tag suggestions
                suggestions.append(tag)
        i = i + 1
    
    # return tag suggestion dictionary
    return suggestions

#print(getSuggestionsByAi('friendly.henry', 'Finn%20The%20Human', id_list))

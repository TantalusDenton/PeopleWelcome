require('dotenv').config()
const uuid = require("uuid").v4
const multer = require("multer");

const AWS = require('aws-sdk')
const awsConfig = {
    'region' : 'us-west-2',
    'endpoint' : 'dynamodb-fips.us-west-2.amazonaws.com',
    'accessKeyId' : process.env.DYNAMO_ACCESS,
    'secretAccessKey' : process.env.DYNAMO_SECRET
}
AWS.config.update(awsConfig)

// the object that contains database operation functions
const docClient = new AWS.DynamoDB.DocumentClient()

/*
AWS.config.apiVersions = {
    cognitoidentityserviceprovider: '2016-04-18',
    // other service API versions
}
*/

/*
const getDateFromUpTo28DaysAgo = daysAgo => {
    const thirty = [4,6,9,11]
    const today = new Date()
    let dd = String(today.getDate()).padStart(2, '0')
    let mm = String(today.getMonth() + 1).padStart(2, '0')
    let yyyy = today.getFullYear()

    if(dd <= daysAgo) {
        if(thirty.includes(mm - 1)) {
            dd += (30 - daysAgo)
        } else if(mm === 3) {
            // leap year
            if((yyyy % 4 === 0 && y % 100 !== 0) || yyyy % 400 === 0) {
                dd += (29 - daysAgo)
            } else {
                dd += (28 - daysAgo)
            }
        } else {
            dd += (31 - daysAgo)
        }
        mm -= 1
        if(mm === 0) {
            mm = 12
            yyyy -= 1
        }
    } else {
        dd -= daysAgo
    }

    return(`${yyyy}-${mm}-${dd}`)
}
*/





// following helper methods are called by methods in repo but not 
// accessed directly by any other file, as a means of encapsulation

//
// MISCELLANEOUS
//

const getDate = () => {
    const today = new Date()
    const dd = String(today.getDate()).padStart(2, '0')
    const mm = String(today.getMonth() + 1).padStart(2, '0')
    const yyyy = today.getFullYear()

    return(`${yyyy}-${mm}-${dd}`)
}

//
// GET / QUERY HELPER METHODS
//

const getAiIdByAiAndUser = async (user, ai) => {
    const params = {
        TableName: 'User_Table',
        Select: 'SPECIFIC_ATTRIBUTES',
        KeyConditionExpression: 'username = :u AND ai_name = :a',
        ExpressionAttributeValues: {
            ':u' : user,
            ':a' : ai
        },
        ProjectionExpression: 'ai_id'
    }

    const data = await docClient.query(params).promise()
    return data.Items[0]
}

const getAiAndUserByAiId = async aiId => {
    const params = {
        TableName: 'User_Table',
        Select: 'SPECIFIC_ATTRIBUTES',
        IndexName: 'ai_id-index',
        KeyConditionExpression: 'ai_id = :id',
        ExpressionAttributeValues: {
            ':id': aiId
        },
        ProjectionExpression: 'username, ai_name'
    }

    const data = await docClient.query(params).promise()
    const item = data.Items[0]

    return item
}

// when adding a tag to a post, we need to ensure that this ai
// has not already added this exact tag to this exact post
const assertTagNotDuplicateByAi = async (user, ai, imageId, tag) => {
    console.log('entered assert method')
    const item = await getAiIdByAiAndUser(user, ai)
    const params = {
        TableName: 'Tags_Table',
        Key: {
            'ai_id' : item.ai_id,
            'image_id' : imageId
        },
        AttributesToGet: [
            'tags'
        ]
    }

    const data = await docClient.get(params).promise()

    return data.Item === undefined || data.Item.tags.includes(tag) === false
}

const getItemFromTagTable = async (aiId, imageId) => {
    const params = {
        TableName: 'Tags_Table',
        Select: 'ALL_ATTRIBUTES',
        KeyConditionExpression: 'ai_id = :ai AND image_id = :im',
        ExpressionAttributeValues: {
            ':ai': aiId,
            ':im': imageId
        }
    }

    const data = await docClient.query(params).promise()
    return data.Items
}

const getPostByUserAndId = async (user, imageId) => {
    const params = {
        TableName: 'Posts_Table',
        Select: 'ALL_ATTRIBUTES',
        KeyConditionExpression: 'username = :u AND image_id = :i',
        ExpressionAttributeValues: {
            ':u': user,
            ':i': imageId
        }
    }

    const data = await docClient.query(params).promise()
    return data.Items
}

const assertItemNotInTagTable = async (aiId, imageId) => {
    const item = await getItemFromTagTable(aiId, imageId)
    return Object.keys(item).length === 0
}

const tagListIsEmpty = async (aiId, imageId) => {
    const params = {
        TableName: 'Tags_Table',
        Select: 'SPECIFIC_ATTRIBUTES',
        KeyConditionExpression: 'ai_id = :a AND image_id = :i',
        ExpressionAttributeValues: {
            ':a': aiId,
            ':i': imageId
        },
        ProjectionExpression: 'tags'
    }

    const data = await getItemFromTagTable(aiId, imageId)

    return (data[0].tags).length != 0
    //return Object.keys(data[0].tags).length === 1
}

//
// PUT / UPDATE HELPER METHODS
//

const addTaggerToPost = async (username, ai, imageOwner, imageId) => {
    const params = {
        TableName: 'Posts_Table',
        Select: 'SPECIFIC_ATTRIBUTES',
        KeyConditionExpression: 'username = :io AND image_id = :ii',
        ExpressionAttributeValues: {
            ':io': imageOwner,
            ':ii': imageId
        },
        ProjectionExpression: 'taggers'
    }

    const data =  await docClient.query(params).promise()
    const taggers = data.Items[0].taggers

    const id = await getAiIdByAiAndUser(username, ai)
    const aiId = id.ai_id

    if(!taggers.includes(aiId)) {
        taggers.push(aiId)

        const params = {
            TableName: 'Posts_Table',
            Key: {
                'username' : imageOwner,
                'image_id' : imageId
            },
            UpdateExpression: 'SET taggers = :t',
            ExpressionAttributeValues: {
                ':t' : taggers
            }
        }

        await docClient.update(params).promise()
    }
}

const addTagInstanceToPostsTable = async (user, imageId) => {

    const item = await getPostByUserAndId(user, imageId)

    let dateAndTagInstances = item[0].date
    const splitArray = dateAndTagInstances.split('-')
    let instances = parseInt(splitArray[3])
    instances++
    dateAndTagInstances = `${splitArray[0]}-${splitArray[1]}-${splitArray[2]}-${instances}`

    const params = {
        TableName: 'Posts_Table',
        Key: {
            'username' : user,
            'image_id' : imageId
        },
        UpdateExpression: 'SET #date_alias = :d',
        ExpressionAttributeValues: {
            ':d' : dateAndTagInstances
        },
        ExpressionAttributeNames: {
            '#date_alias' : 'date'
        }
    }

    const result = docClient.update(params).promise().then(r => {
        return r.$response
    }).catch(e => {
        return e
    })

    return result
}

const removeTagInstanceFromPostsTable = async (imageOwner, imageId) => {

    const item = await getPostByUserAndId(imageOwner, imageId)

    let dateAndTagInstances = item[0].date
    const splitArray = dateAndTagInstances.split('-')
    let instances = parseInt(splitArray[3])
    instances--
    dateAndTagInstances = `${splitArray[0]}-${splitArray[1]}-${splitArray[2]}-${instances}`

    const params = {
        TableName: 'Posts_Table',
        Key: {
            'username' : imageOwner,
            'image_id' : imageId
        },
        UpdateExpression: 'SET #date_alias = :d',
        ExpressionAttributeValues: {
            ':d' : dateAndTagInstances
        },
        ExpressionAttributeNames: {
            '#date_alias' : 'date'
        }
    }

    const result = docClient.update(params).promise().then(r => {
        return r.$response
    }).catch(e => {
        return e
    })

    return result
}

const addAiImageRelationshipToUserTable = async (user, ai, imageId, imageOwner) => {

    const params = {
        TableName: 'User_Table',
        Select: 'SPECIFIC_ATTRIBUTES',
        KeyConditionExpression: 'username = :u AND ai_name = :a',
        ExpressionAttributeValues: {
            ':u': user,
            ':a': ai
        },
        ProjectionExpression: 'tagged_posts'
    }

    const data = await docClient.query(params).promise()
    const postsList = data.Items[0].tagged_posts

    let i = 0
    while(i < postsList.length && postsList.at(i).image_id != imageId) {
        i++
    }

    console.log(i)
    console.log(postsList.length)

    if(i >= postsList.length) {
        postsList.push({'image_id': imageId, 'image_owner': imageOwner})
        console.log(postsList)

        const params = {
            TableName: 'User_Table',
            Key: {
                'username': user,
                'ai_name': ai
            },
            UpdateExpression: 'SET tagged_posts = :t',
            ExpressionAttributeValues: {
                ':t': postsList
            }
        }

        await docClient.update(params).promise()
    }
}

const addAiImageRelationshipToTagTable = async (aiId, imageId, firstTag) => {
    const params = {
        TableName: 'Tags_Table',
        Item: {
            'ai_id': aiId,
            'image_id': imageId,
            'tags': [firstTag]
        }
    }

    await docClient.put(params).promise()
}

//
// DELETE HELPER METHODS
//

const removeAllFromTagTableByAi = async (user, ai) => {
    const params = {
        TableName: 'User_Table',
        Select: 'SPECIFIC_ATTRIBUTES',
        KeyConditionExpression: 'username = :u AND ai_name = :a',
        ExpressionAttributeValues: {
            ':u': user,
            ':a': ai
        },
        ProjectionExpression: 'tagged_posts'
    }

    const data = await docClient.query(params).promise()
    const postsList = data.Items[0].tagged_posts

    const item = await getAiIdByAiAndUser(user, ai)
    const aiId = item.ai_id

    postsList.forEach(async post => {
        const tagParams = {
            TableName: 'Tags_Table',
            Key: {
                'ai_id': aiId,
                'image_id': post.image_id
            }
        }

        await removeTaggerFromPost(user, ai, post.image_owner, post.image_id)
        await docClient.delete(tagParams).promise()
    })
}

const removeAiImageRelationshipFromUserTable = async (user, ai, imageId) => {
    const params = {
        TableName: 'User_Table',
        Select: 'SPECIFIC_ATTRIBUTES',
        KeyConditionExpression: 'username = :u AND ai_name = :a',
        ExpressionAttributeValues: {
            ':u': user,
            ':a': ai
        },
        ProjectionExpression: 'tagged_posts'
    }

    const data = await docClient.query(params).promise()
    const postsList = data.Items[0].tagged_posts

    let found = false
    let i = 0

    while(i < postsList.length) {
        if(postsList.at(i).image_id === imageId) {
            found = true
            break
        }
        i++
    }

    if(found) {
        postsList.splice(i, 1)
        const params = {
            TableName: 'User_Table',
            Key: {
                'username': user,
                'ai_name': ai
            },
            UpdateExpression: 'SET tagged_posts = :t',
            ExpressionAttributeValues: {
                ':t': postsList
            }
        }

        await docClient.update(params).promise()
    }
}

const removeTaggerFromPost = async (user, ai, imageOwner, imageId) => {
    const params = {
        TableName: 'Posts_Table',
        Select: 'SPECIFIC_ATTRIBUTES',
        KeyConditionExpression: 'username = :u AND image_id = :i',
        ExpressionAttributeValues: {
            ':u': imageOwner,
            ':i': imageId
        },
        ProjectionExpression: 'taggers'
    }

    const data =  await docClient.query(params).promise()
    const taggers = data.Items[0].taggers

    const id = await getAiIdByAiAndUser(user, ai)
    const aiId = id.ai_id

    if(taggers.includes(aiId)) {
        taggers.splice(taggers.indexOf(aiId), 1)

        const params = {
            TableName: 'Posts_Table',
            Key: {
                'username' : imageOwner,
                'image_id' : imageId
            },
            UpdateExpression: 'SET taggers = :t',
            ExpressionAttributeValues: {
                ':t' : taggers
            }
        }

        await docClient.update(params).promise()
    }
}

const removeAiImageRelationshipFromTagTable = async (aiId, imageId) => {
    const params = {
        TableName: 'Tags_Table',
        Key: {
            'ai_id': aiId,
            'image_id': imageId
        }
    }

    await docClient.delete(params).promise()
}

//
// S3 HELPER METHODS
//

// file must be an image 
const fileFilter = file => {
    if (file.mimetype.split('/')[0] !== 'image') {
        throw new multer.MulterError('LIMIT_UNEXPECTED_FILE')
    }
}

// the database access object that will be called in app.js
const dynamoRepo = {
    //
    // GET / QUERY DAO METHODS
    //

    retrieveDecisionTree : async (user, ai) => {
        const params = {
            TableName: 'User_Table',
            Select: 'SPECIFIC_ATTRIBUTES',
            KeyConditionExpression: 'username = :u AND ai_name = :ai',
            ExpressionAttributeValues: {
                ':u' : user,
                ':ai' : ai
            },
            ProjectionExpression: 'decision_tree'
        }

        const response = await docClient.query(params).promise()
        const data = response.Items[0]
        return data.decision_tree
    },


    retrieveGoogleVisionObjects : async (imageId) => {
        const params = {
            TableName: 'Posts_Table',
            IndexName: 'image_id-index',
            KeyConditionExpression: 'image_id = :id',
            ExpressionAttributeValues: {
                ':id': imageId
            }
        }

        const response = await docClient.query(params).promise()
        const data = response.Items[0]
        return data.objects
    },

    checkForGoogleVisionObjects : async (imageId) => {
        const params = {
            TableName: 'Posts_Table',
            IndexName: 'image_id-index',
            KeyConditionExpression: 'image_id = :id',
            ExpressionAttributeValues: {
                ':id': imageId
            }
        }
    
        const response = await docClient.query(params).promise()
        data = response.Items[0]
        if(data.objects) {
            return true
        }
        return false
    },

    checkForDecisionTree : async (user, ai) => {
        const params = {
            TableName: 'User_Table',
            Select: 'ALL_ATTRIBUTES',
            KeyConditionExpression: 'username = :u AND ai_name = :ai',
            ExpressionAttributeValues: {
                ':u' : user,
                ':ai' : ai
            }
        }

        const response = await docClient.query(params).promise()
        data = response.Items[0]
        if(data.decision_tree) {
            return true
        }
        return false
    },

    getAllTagsByAiAndUser : async (user, ai) => {
        const item = await getAiIdByAiAndUser(user, ai)
        const aiId = item.ai_id

        const params = {
            TableName: 'Tags_Table',
            Select: 'SPECIFIC_ATTRIBUTES',
            KeyConditionExpression: 'ai_id = :id',
            ExpressionAttributeValues: {
                ':id': aiId
            },
            ProjectionExpression: 'image_id, tags'
        }

        const data = docClient.query(params).promise().then(r => {
            return r.Items
        }).catch(e => {
            console.log(e)
            return e
        })

        return data
    },

    getTagsByAiAndUserAndImageId : async (user, ai, imageId) => {
        const item = await getAiIdByAiAndUser(user, ai)
        const aiId = item.ai_id

        const params = {
            TableName: 'Tags_Table',
            Select: 'SPECIFIC_ATTRIBUTES',
            KeyConditionExpression: 'ai_id = :a AND image_id = :i',
            ExpressionAttributeValues: {
                ':a': aiId,
                ':i': imageId
            },
            ProjectionExpression: 'tags'
        }

        const data = await docClient.query(params).promise()
        if(Object.keys(data.Items).length === 0) {
            return []
        }
        return data.Items[0].tags
    },

    getAllAisByUser : async user => {
        const params = {
            TableName: 'User_Table',
            Select: 'ALL_ATTRIBUTES',
            KeyConditionExpression: 'username = :u',
            ExpressionAttributeValues: {
                ':u': user
            }
        }

        const data = await docClient.query(params).promise()
        return data.Items
    },

    getAiIdByAiAndUser : async (user, ai) => {
        const aiId = await getAiIdByAiAndUser(user, ai)
        return aiId
    },

    listPostsByPopularityAndDate : async () => {
        const params = {
            TableName: 'Posts_Table',
            IndexName: 'tag_instances-date-index',
            KeyConditionExpression: 'tag_instances = :t',
            ExpressionAttributeValues: {
                ':t': 0
            },
            ScanIndexForward: false
        }

        const posts = docClient.query(params).promise().then(r => {
            return r.Items
        }).catch(e => {
            return e
        })

        return posts
    },

    retrievePostsOfUser : async user => {
        const params = {
            TableName: 'Posts_Table',
            Select: 'SPECIFIC_ATTRIBUTES',
            KeyConditionExpression: 'username = :u',
            ExpressionAttributeValues: {
                ':u': user,
            },
            ExpressionAttributeNames: {
                '#date_alias': 'date'
            },
            ProjectionExpression: 'image_id, #date_alias'
        }

        const posts = await docClient.query(params).promise()
        return posts
    },

    //ToDo: get image ids by user and ai (unneeded, discontinued.)
    /*retrievePostsOfAi : async (user, ai)  => {
        const params = {
            TableName: 'Posts_Table',
            Select: 'SPECIFIC_ATTRIBUTES',
            KeyConditionExpression: 'username = :u AND ai_name = :a',
            ExpressionAttributeValues: {
                ':u': user,
                ':a': ai
            },
            ExpressionAttributeNames: {
                '#date_alias': 'date'
            },
            ProjectionExpression: 'image_id, #date_alias'
        }

        const posts = await docClient.query(params).promise()
        return posts
    },*/

    //
    // PUT / UPDATE DAO METHODS
    //

    updateGoogleVisionObjects : async (imageId, objects) => {
        const queryParams = {
            TableName: 'Posts_Table',
            IndexName: 'image_id-index',
            KeyConditionExpression: 'image_id = :id',
            ExpressionAttributeValues: {
                ':id': imageId
            }
        }

        const response = await docClient.query(queryParams).promise()
        const data = response.Items[0]

        const updateParams = {
            TableName: 'Posts_Table',
            Key: {
                'username' : data.username,
                'image_id' : imageId
            },
            UpdateExpression: `SET objects = :o`,
            ExpressionAttributeValues: {
                ':o' : objects
            }
        }

        await docClient.update(updateParams).promise()
    },
    
    updateDecisionTree : async (user, ai, tree) => {
        const params = {
            TableName: 'User_Table',
            Key: {
                'username' : user,
                'ai_name' : ai
            },
            UpdateExpression: 'SET decision_tree = :t',
            ExpressionAttributeValues: {
                ':t' : tree
            }
        }

        await docClient.update(params).promise()
    },

    addAiToSearchTable : async (user, ai) => {
        const params = {
            TableName: 'Ai_Search_Table',
            Item: {
                'ai_name' : ai,
                'username' : user
            }
        }

        const result = await docClient.put(params).promise()
        return result.$response
    },

    removeAiFromSearchTable : async (user, ai) => {
        const params = {
            TableName: 'Ai_Search_Table',
            Key: {
                'ai_name': ai,
                'username': user
            }
        }

        await docClient.delete(params).promise()
    },

    createAi : async (user, ai) => {
        const params = {
            TableName: 'User_Table',
            Item: {
                'username' : user,
                'ai_name' : ai,
                'ai_id' : uuid(),
                'public' : true,
                'tagged_posts' : []
            }
        }

        const result = await docClient.put(params).promise()
        return result.$response
    },

    deleteAi : async (user, ai) => {
        const params = {
            TableName: 'User_Table',
            Key: {
                'username': user,
                'ai_name': ai
            }
        }

        removeAllFromTagTableByAi(user, ai).then(async () => {
            await docClient.delete(params).promise()
        })
    },

    createPostWithoutTags : async (user, imageId) => {
        const date = getDate() + '-0'
        const timestamp = Date.now() / 1000 / 60 / 60 % 24
        const params = {
            TableName: 'Posts_Table',
            Item: {
                'username' : user,
                'image_id' : imageId,
                'date' : date,
                'timestamp' : timestamp,
                'tag_instances' : 0,
                'taggers' : []
            }
        }
        
        await docClient.put(params).promise()
    },

    deletePost : async (imageOwner, imageId) => {
        const params = {
            TableName: 'Posts_Table',
            Select: 'SPECIFIC_ATTRIBUTES',
            KeyConditionExpression: 'username = :u AND image_id = :i',
            ExpressionAttributeValues: {
                ':u': imageOwner,
                ':i': imageId
            },
            ProjectionExpression: 'taggers'
        }

        const data = await docClient.query(params).promise()
        const taggers = data.Items[0].taggers

        taggers.forEach(async tagger => {
            await removeAiImageRelationshipFromTagTable(tagger, imageId)
            const item = await getAiAndUserByAiId(tagger)
            const user = item.username
            const ai = item.ai_name
            await removeAiImageRelationshipFromUserTable(user, ai, imageId)
        })
        
        const deleteParams = {
            TableName: 'Posts_Table',
            Key: {
                'username': imageOwner,
                'image_id': imageId
            }
        }

        await docClient.delete(deleteParams).promise()
    },

    insertTagInTagAndPostTables : async (user, ai, imageOwner, imageId, tag) => {

        const tagNotDuplicate = await assertTagNotDuplicateByAi(user, ai, imageId, tag)
        if(tagNotDuplicate) {

            const item = await getAiIdByAiAndUser(user, ai)
            const aiId = item.ai_id

            const itemNotCreated = await assertItemNotInTagTable(aiId, imageId)
            if(itemNotCreated) {
                await addAiImageRelationshipToTagTable(aiId, imageId, tag)
            } else {
                const params = {
                    TableName: 'Tags_Table',
                    Key: {
                        'ai_id' : aiId,
                        'image_id' : imageId
                    },
                    UpdateExpression: 'SET tags = list_append(tags, :t)',
                    ExpressionAttributeValues: {
                        ':t' : [tag]
                    }
                }

                try {
                    await docClient.update(params).promise()
                } catch(e) {
                    console.log(e)
                    return e
                }
            }
            
            try {
                await addTagInstanceToPostsTable(imageOwner, imageId)
                await addTaggerToPost(user, ai, imageOwner, imageId)
                await addAiImageRelationshipToUserTable(user, ai, imageId, imageOwner)
                return 0
            } catch(e) {
                console.log(e)
                return e
            }
        } 
        return -1
    },

    removeTagFromTagAndPostTables : async (user, ai, imageOwner, imageId, tag) => {

        const item = await getAiIdByAiAndUser(user, ai)
        const aiId = item.ai_id

        const itemNotCreated = await assertItemNotInTagTable(aiId, imageId)

        if(!itemNotCreated) {

            const tagData = await getItemFromTagTable(aiId, imageId)
            const tagList = tagData[0].tags


            if(tagList.includes(tag)) {
                const tagListEmpty = await tagListIsEmpty(aiId, imageId)
                if(tagListEmpty) { // if only one tag between this ai and image exists
                    await removeAiImageRelationshipFromTagTable(aiId, imageId)
                    await removeAiImageRelationshipFromUserTable(user, ai, imageId)
                    await removeTaggerFromPost(user, ai, imageOwner, imageId)
                } else {
                    const index = tagList.indexOf(tag)
                    tagList.splice(index, 1)

                    const params = {
                        TableName: 'Tags_Table',
                        Key: {
                            'ai_id' : aiId,
                            'image_id' : imageId
                        },
                        UpdateExpression: 'SET tags = :t',
                        ExpressionAttributeValues: {
                            ':t' : tagList
                        }
                    }
    
                    try {
                        await docClient.update(params).promise()
                        console.log('made it here 3')
                    } catch(e) {
                        console.log(e)
                        return e
                    }
                }
                await removeTagInstanceFromPostsTable(imageOwner, imageId)
                return 0
            }
        }
        return -1
    }



    // tags variable of structure {ai1: {name: 'name', tags: [tag1, tag2]}, all: [tag1, tag2, tag3]}
    // possible future feature, not minimum viable product
    /*
    insertPostWithTags : async (user, imageId, tags) => {
        var date = getDate()
        var timestamp = Date.now()
        var postParams = {
            TableName: 'Post_Table',
            Item: {
                'username' : user,
                'image_id' : imageId,
                'tag_instances' : tags.all.length,
                'date' : date,
                'tags' : tags.all,
                'timestamp' : timestamp
            }
        }

        const tagMap = {1: {name: 'name1', tags: ['happy', 'sad']}, 2: {name: 'name2', tags: ['cool']}, all: ['happy', 'sad', 'cool']}
        let i = 1
        tagMap.foreach((key, value) => {
            if(i < tagMap.size) {
                console.log(value.name)
            }
        })

        var userParams = {

        }
    }
    */
}

exports.dynamo = dynamoRepo
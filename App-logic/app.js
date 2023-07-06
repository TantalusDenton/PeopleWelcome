
const path = require('path')
const multer = require('multer')
const bodyParser = require('body-parser')
const cors = require('cors')
require('dotenv').config()
const uuid = require('uuid').v4
const fs = require('fs')
const FormData = require('form-data');

const { S3 } = require('@aws-sdk/client-s3')
const express = require('express')

const app = express()
const PORT = 5000

var http = require('http');
var https = require('https');

var privateKey = fs.readFileSync('sslcert/privkey.pem', 'utf8');
var certificate = fs.readFileSync('sslcert/fullchain.pem','utf8');

var credentials = {key: privateKey, cert: certificate};

//
// S3 Upload Functionality begin
//

const s3Client = new S3({ region: process.env.BUCKET_REGION,
credentials: {
  accessKeyId: process.env.ACCESS_KEY,
  secretAccessKey: process.env.SECRET_ACCESS_KEY
}})


//const s3Router = express.Router()
//app.use('/s3', s3Router)
app.use('/upload', (error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === "LIMIT_FILE_SIZE") {
            return res.status(400).json({
                message: "file is too large",
            });
        }
  
        if (error.code === "LIMIT_FILE_COUNT") {
            return res.status(400).json({
                message: "File limit reached",
            });
        }
  
        if (error.code === "LIMIT_UNEXPECTED_FILE") {
            return res.status(400).json({
                message: "File must be an image",
            });
        }
    }
})
const corsOptions = {
    origin:'*', 
    credentials:true,            //access-control-allow-credentials:true
    optionSuccessStatus:200,
}
app.use(cors(corsOptions))

app.use(express.json())

// serve react in express
/*
app.use(express.static(path.join(__dirname, '..', 'build')))
app.use(express.static('public'))

app.use((req, res, next) => {
    res.sendFile(path.join(__dirname, '..', 'build',
    'index.html'))
})
*/

const authorize = require('./authorize')
//app.use(authorize)


// upload file to s3 bucket with given key
const uploadImage = async (file, key) => {
    const fileStream = fs.createReadStream(file.path)
  
    const uploadParams = {
        Bucket: process.env.BUCKET_NAME,
        Body: fileStream,
        Key: key
    }
  
    return await s3Client.putObject(uploadParams)
}

// download an image from s3
const getFileStream = async fileKey => {
    const downloadParams = {
        Key: fileKey,
        Bucket: process.env.BUCKET_NAME
    }

    return await s3Client.getObject(downloadParams)
}
   

const fileFilter = (req, file, cb) => {
    if (file.mimetype.split('/')[0] === 'image') {
      cb(null, true);
    } else {
      cb(new multer.MulterError('LIMIT_UNEXPECTED_FILE'), false);
    }
  }

const upload = multer({ dest: 'uploads/' })
  
//
// S3 Upload Functionality end
//

//
// DynamoDB api begin
//


// gets list of posts from database given a user
const getPostsByUser = async (req, res, next) => {
    const origin = req.headers.origin;
    if (origin === 'http://localhost:3000') {
        res.setHeader('Access-Control-Allow-Origin', origin);
    }
    let posts
    try {
        const user = req.params.user
        posts = await dynamoRepo.dynamo.retrievePostsOfUser(user)
    } catch(e) {
        res.status(400).send(e)
    }
    // add posts object to request so it can be used
    const items = posts.Items
    items.forEach(item => {
        item.date = removeTagInstancesFromDate(item.date)
    })
    req.posts = items
    next()
}

//  (unneeded, discontinued.)
const getPostsByAi = async (req, res, next) => {
    const origin = req.headers.origin;
    if (origin === 'http://localhost:3000') {
        res.setHeader('Access-Control-Allow-Origin', origin);
    }
    let posts
    try {
        const user = req.params.user
        posts = await dynamoRepo.dynamo.retrievePostsOfAi(user, ai)
    } catch(e) {
        res.status(400).send(e)
    }
    // add posts object to request so it can be used
    const items = posts.Items
    items.forEach(item => {
        item.date = removeTagInstancesFromDate(item.date)
    })
    req.posts = items
    next()
}

const removeTagInstancesFromDate = date => {
    const dateArr = date.split('-')
    return `${dateArr[0]}-${dateArr[1]}-${dateArr[2]}`
}

// gets ai_id by ai and user
const getAiIdByAiAndUser = async (req, res, next) => {
    let id
    try {
        const user = req.params.user
        const ai = req.params.ai
        id = await dynamoRepo.dynamo.getAiIdByAiAndUser(user, ai)
        req.id = id
        next()
    } catch(e) {
        res.status(400).send(e)
    }
}

// adds new Ai to User_Table and Search_Table
const createNewAi = async (req, res, next) => {
    const body = req.body
    const user = req.params.user

    try {
        await dynamoRepo.dynamo.createAi(user, body.ai)
        await dynamoRepo.dynamo.addAiToSearchTable(user, body.ai)
        next()
    } catch(e) {
        res.status(400).send(e)
    }
}

// removes ai from database and all data related to it
const deleteAi = async (req, res, next) => {
    const user = req.params.user
    const ai = req.params.ai
    
    try {
        await dynamoRepo.dynamo.deleteAi(user, ai)
        await dynamoRepo.dynamo.removeAiFromSearchTable(user, ai)
        next()
    } catch(e) {
        res.status(400).send(e)
    }
}

// deletes specified post
const deletePost = async (req, res, next) => {
    const user = req.params.user
    const imageId = req.body.imageId

    try {
        await dynamoRepo.dynamo.deletePost(user, imageId)
        next()
    } catch(e) {
        res.status(400).send(e)
    }
}

// gets list of posts for home feed (unauthenticated user)
const getFeed = async (req, res, next) => {
    const origin = req.headers.origin;
    if (origin === 'http://localhost:3000') {
        res.setHeader('Access-Control-Allow-Origin', origin);
    }
    try {
        const posts = await dynamoRepo.dynamo.listPostsByPopularityAndDate()
        req.posts = posts
    } catch(e) {
        res.status(400).send(e)
    }

    const items = req.posts
    items.forEach(item => {
        item.date = removeTagInstancesFromDate(item.date)
    })
    req.posts = items
    next()
}

// adds tag to database
const addTag = async (req, res, next) => {
    const origin = req.headers.origin;
    if (origin === 'http://localhost:3000') {
        res.setHeader('Access-Control-Allow-Origin', origin);
    }
    
    const user = req.params.user
    const ai = req.params.ai
    const imageId = req.body.imageId
    const imageOwner = req.body.imageOwner
    const tag = req.body.tag

    try {
        await dynamoRepo.dynamo.insertTagInTagAndPostTables(user, ai, imageOwner, imageId, tag)
        next()
    } catch(e) {
        res.status(400).send(e)
    }
}

// removes tag from database
const removeTag = async (req, res, next) => {
    const user = req.params.user
    const ai = req.params.ai
    const imageId = req.body.imageId
    const imageOwner = req.body.imageOwner
    const tag = req.body.tag

    try {
        await dynamoRepo.dynamo.removeTagFromTagAndPostTables(user, ai, imageOwner, imageId, tag)
        next()
    } catch(e) {
        res.status(400).send(e)
    }
}

// gets all tags of ai
const getAiTags = async (req, res, next) => {
    const origin = req.headers.origin;
    if (origin === 'http://localhost:3000') {
        res.setHeader('Access-Control-Allow-Origin', origin);
    }
    
    const user = req.params.user
    const ai = req.params.ai

    try {
        const tags = await dynamoRepo.dynamo.getAllTagsByAiAndUser(user, ai)
        req.tags = tags
        next()
    } catch(e) {
        res.status(400).send(e)
    }
}

// gets tags of ai on specific post
const getAiTagsOnPost = async (req, res, next) => {
    const origin = req.headers.origin;
    if (origin === 'http://localhost:3000') {
        res.setHeader('Access-Control-Allow-Origin', origin);
    }

    const user = req.params.user
    const ai = req.params.ai
    const imageId = req.params.imageid

    try {
        const tags = await dynamoRepo.dynamo.getTagsByAiAndUserAndImageId(user, ai, imageId)
        req.tags = tags
        next()
    } catch(e) {
        res.status(400).send(e)
    }
}

// gets ais owned by user
const getAisByUser = async (req, res, next) => {
    const user = req.params.user

    try {
        const ais = await dynamoRepo.dynamo.getAllAisByUser(user)
        req.aiList = ais
        next()
    } catch(e) {
        res.status(400).send(e)
    }
}

//
// GET METHODS
//


// returns true if decision tree found, false if not
app.get('/account/:user/:ai/treefound', async (req, res) => {
    const user = req.params.user
    const ai = req.params.ai

    const treeFound = await dynamoRepo.dynamo.checkForDecisionTree(user, ai)
    res.json({'treeFound': treeFound})
})

// returns true if google vision objects found, false if not
app.get('/post/:id/objectsfound', async (req, res) => {
    const id = req.params.id
    const objectsFound = await dynamoRepo.dynamo.checkForGoogleVisionObjects(id)
    res.json({'objectsFound': objectsFound})
})

// retrieves google vision objects
app.get('/post/:id/objects', async (req, res) => {
    const id = req.params.id
    const objects = await dynamoRepo.dynamo.retrieveGoogleVisionObjects(id)
    res.json({'objects': objects})
})

// sends list of user's post data to that user's account page
app.get('/account/:user/posts', getPostsByUser, (req, res) => {
    const posts = req.posts
    res.json(posts)
})

// retrieves the ai_id of given :ai belonging to given :user
app.get('/account/:user/:ai/id', getAiIdByAiAndUser, (req, res) => {
    const aiId = req.id
    res.json(aiId)
})

// sends home feed to unauthenticated user
app.get('/feed', getFeed, (req, res) => {
    const posts = req.posts
    res.json(posts)
})

// sends all tags ever tagged by ai and image_id's of posts they were tagged on
app.get('/account/:user/:ai/alltags', getAiTags, (req, res) => {
    const tags = req.tags
    res.json(tags)
})

// sends tags of image_id and ai_id
app.get('/account/:user/:ai/:imageid/tagsonpost', getAiTagsOnPost, (req, res) => {
    const tags = req.tags
    res.json(tags)
})


// ToDo: sends Ais, belonging to a user
app.get('/account/:user/ownedais', getAisByUser, (req, res) => {
    const ais = req.aiList
    res.json(ais)
})

// get all image-ids of a given AI (discontinued)
/*app.get('/account/:user/:ai/posts', getPostsByAi, (req, res) => {
    const posts = req.posts
    res.json(posts)
})*/

// get an image from s3 bucket by id
app.get('/images/:key', async (req, res) => {
    const key = req.params.key
    const readStream = await getFileStream(key)
    
    readStream.Body.pipe(res)
})

// get a decision tree from s3 bucket by user and ai
app.get('/account/:user/:ai/tree', async (req, res) => {
    const user = req.params.user
    const ai = req.params.ai
    const data = await dynamoRepo.dynamo.getAiIdByAiAndUser(user, ai)
    const aiId = data.ai_id

    const readStream = await getFileStream(aiId)
    readStream.Body.pipe(res)
})

//
// POST METHODS
//

// update google vision objects
app.put('/post/:id/updateobjects', async (req, res) => {
    const id = req.params.id
    const objects = req.body.objects

    dynamoRepo.dynamo.updateGoogleVisionObjects(id, objects).then(r => {
        res.status(200).send(r)
    }).catch(e => {
        res.status(400).send(e)
    })
})

// update decision tree
app.put('/account/:user/:ai/updatetree', async (req, res) => {
    const user = req.params.user
    const ai = req.params.ai
    const treeData = req.body.treeData
    //const treeData = tree.treeData

    console.log('treeData:', treeData)
    //console.log('treeData:', treeData)

    dynamoRepo.dynamo.updateDecisionTree(user, ai, treeData).then(r => {
        res.status(200).send(r)
    }).catch(e => {
        res.status(400).send(e)
    })
})

// creates a new ai belonging to :user, with ai name sent in request body
app.post('/account/:user/createai', createNewAi, (req, res) => {
    res.status(200).send('Ai created successfully')
})

// database access object
const dynamoRepo = require('./datarepo')

// creates a new post belonging to :user
app.post('/upload/account/:user/createpost', upload.single('file'), async (req, res) => {
    const file = req.file
    const user = req.params.user

    const key = uuid()
    const result = uploadImage(file, key)

    // wait for image to upload, and insert data in DynamoDB if successful
    result.then(async () => {
        await dynamoRepo.dynamo.createPostWithoutTags(user, key)
    })
})

// upload file
app.post('/upload/account/:user/:ai/uploadfile', upload.single('file'), async (req, res) => {
    const file = req.file
    const user = req.params.user
    const ai = req.params.ai

    const data = await dynamoRepo.dynamo.getAiIdByAiAndUser(user, ai)
    const aiId = data.ai_id
    const result = uploadImage(file, aiId)

    // wait for image to upload, and insert data in DynamoDB if successful
    result.then(async () => {
        await dynamoRepo.dynamo.updateDecisionTree(user, ai, 1)
    })
})

// adds decision tree model to s3
app.post('/upload/account/:user/:ai/updatetree', async (req, res) => {
    const user = req.params.user
    const ai = req.params.ai
    const treeData = req.body.tree

    const data = await dynamoRepo.dynamo.getAiIdByAiAndUser(user, ai)
    const aiId = data.ai_id

    var buf = Buffer.from(JSON.stringify(treeData))

    const uploadParams = {
        Bucket: process.env.BUCKET_NAME,
        Body: buf,
        Key: aiId
    }
  
    const result = s3Client.putObject(uploadParams)

    result.then(async () => {
        await dynamoRepo.dynamo.updateDecisionTree(user, ai, 1)
    })
})

// creates new tag for this ai in database
app.post('/account/:user/:ai/addtag', addTag, (req, res) => {
    res.send('tag added')
})

//
// DELETE METHODS
//

// allows user to remove tag that ai tagged
app.delete('/account/:user/:ai/removetag', removeTag, (req, res) => {
    res.send('tag removed')
})

// allows user to delete ai
app.delete('/account/:user/:ai/deleteai', deleteAi, (req, res) => {
    res.send('ai and all related data deleted')
})

// allows user to delete post
app.delete('/account/:user/deletepost', deletePost, (req, res) => {
    res.send('post deleted')
})

//
// DynamoDB api end
//
/*
app.listen(PORT, () => {
    console.log(`server is listening on port ${PORT}`)
})
*/
var httpServer = http.createServer(app);
var httpsServer = https.createServer(credentials, app);

httpServer.listen(8080);
httpsServer.listen(PORT);

require('dotenv').config()
const express = require("express");
const multer = require("multer");


const { s3Uploadv2, s3Uploadv3, getFileStream, genUploadUrl } = require("./s3Service");
const s3 = require('@aws-sdk/client-s3')
const uuid = require("uuid").v4;
const fs = require('fs');

const s3 = require('@aws-sdk/client-s3')

const { S3 } = require('aws-sdk');

const s3Client = new S3({ region: 'us-west-2',
credentials:{
  accessKeyId: process.env.ACCESS_KEY,
  secretAccessKey: process.env.SECRET_ACCESS_KEY
}})

// const { s3Uploadv2, getFileStream, genUploadUrl,uploadToS3 } = require("./s3Service");
// const s3 = require('@aws-sdk/client-s3')
const uuid = require("uuid").v4;
// const keys = require('./s3Keys')


// const { PrismaClient } = require('@prisma/client');
// const { getSignedUrl } = require("@aws-sdk/s3-request-presigner");
// const { S3Client, GetObjectCommand} = require("@aws-sdk/client-s3");


// const s3 = new S3Client({
//   credentials: {
//     accessKeyId: process.env.AWS_ACCESS_KEY_ID,
//     secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
//   },
//   region: "us-west-2"
// });

const app = express();
// const prisma = new PrismaClient()

const cors=require("cors");
const corsOptions ={
   origin:'*', 
   credentials:true,            //access-control-allow-credentials:true
   optionSuccessStatus:200,
}
app.use(cors(corsOptions)) // Use this after the variable declaration

// const fs = require('fs')
// const util = require('util')
// const unlinkFile = util.promisify(fs.unlink)

// creates memory storage and immediately upload to s3
const storage = multer.memoryStorage();

// file must be an image 
const fileFilter = (req, file, cb) => {
  if (file.mimetype.split("/")[0] === "image") {
    cb(null, true);
  } else {
    cb(new multer.MulterError("LIMIT_UNEXPECTED_FILE"), false);
  }
};

// upload feature using multer, sets destination to uploads directory
/*
const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 1000000000, files: 2 }
});
*/
const upload = multer({ dest: 'uploads/' })

app.get('/:key', (req, res) => {
  console.log(req.params)
  const key = req.params.key
  const readStream = getFileStream(key)
  
  readStream.pipe(res)
})

// // GET signed urls for all images in the s3 bucket
// app.get('/api/image', (req, res) => {
//   const params = {
//     Bucket: process.env.BUCKET_NAME 
//   };
//   s3.listObjectsV2(params, (err, data) => {
//     console.log('S3 List', data);
//     // Package signed URLs for each to send back to client
//     let images = []
//     for (let item of data.Contents) {
//       let url = s3.getSignedUrl('getObject', {
//           Bucket: process.env.BUCKET_NAME,
//           Key: item.Key, 
//           Expires: 100 //time to expire in seconds - 5 min
//       });
//       images.push(url);
//     }
//     res.send(images);
//   })
// })

app.post("/upload", upload.single("file"), async (req, res) => {
  const file = req.file
  console.log(file)
  const result = await uploadFile(file)
  console.log(`Result: ${result.Key}`)
  const description = req.body.description
  res.send('somethin happened')
  /*
  try {
    const results = await s3Uploadv2(req.files);
    console.log(results);
    res.json({ status: "success" });
  } catch (err) {
    console.log(err);
  }
  */
});
/*
app.get("/upload", async (req, res) => { 
  const url = await genUploadUrl()
  res.send({url})
})
*/


// app.get("/s3Url", async (req, res) => { 
//   const url = await genUploadUrl()
//   res.send({url})
// })


// get an image from s3 by id
// app.get('/image/:id', async (req, res) => {
//   const imageId = req.params.id
//   const bucketParams = {
//     Bucket: keys.BUCKET_NAME,
//     Key: imageId
//   }

/*
  try {
    const data = await s3Client.send(new s3.GetObjectCommand(bucketParams))
    return await data.Body.transformToString();
  } catch(e) {
    res.status(400).send(`Error: ${e}`)
  }
})
*/

//   try {
//     const data = await s3Client.send(new s3.GetObjectCommand(bucketParams))
//     return await data.Body.transformToString();
//   } catch(e) {
//     res.status(400).send(`Error: ${e}`)
//   }
// })


// app.get('/api/signurl/get/:filename', (req, res) => {
//   const presignedGetUrl = s3.getSignedUrl('getObject', {
//       Bucket: process.env.BUCKET_NAME,
//       Key: req.params.filename, 
//       Expires: 100 //time to expire in seconds - 5 min
//   });
//   console.log('sending presigned url', presignedGetUrl);
//   res.send({url: presignedGetUrl})
// })
// get an image from s3 by id

/*
app.get('/image/:id', async (req, res) => {
  const imageId = req.params.id
  const bucketParams = {
    Bucket: keys.BUCKET_NAME,
    Key: imageId
*/

app.get("/upload", async (req, res) => {
  const posts = await prisma.posts.findMany({orderBy: [{created:'desc'}]})
  // const client = new S3Client(clientParams);
  for (let post of posts) {
    post.imageUrl = await getSignedUrl(
      s3Client,
      new GetObjectCommand({
        Bucket: process.env.AWS_BUCKET_NAME,
        Key: `${uuid()}-${file.originalname}`,
      }), 
      { expiresIn: 3600 }
    )

app.get('/image/:id', async (req, res) => {
  const imageId = req.params.id
  const bucketParams = {
    Bucket: keys.BUCKET_NAME,
    Key: imageId

  }
  res.send(posts)
})

  
// app.post("/upload", upload.array("file"), async (req, res) => {
//   try {
//     const results = await s3Upload3(req.files);
//     console.log(results);
//     return res.json({ status: "success" });
//     // res.send("success")
//     // res.send({imagePath: `/uploads/${results.Key}`})
//   } catch (err) {
//     console.log(err);
//   }
// });

app.post("/upload", upload.array("file"), async (req, res) => {
  try {
    const results = await s3Uploadv2(req.files);
    console.log(results);
    return res.json({ status: "success" });
  } catch (err) {
    console.log(err);
  }
});


// app.post("/upload", upload.single("image"), (req, res) => {
//   const { file } = req;
//   const userId = req.headers["x-user-id"];

//   if (!file || !userId) return res.status(400).json({ message: "Bad request" });

//   const { error, Key } = s3Uploadv2({ file, userId });
//   if (error) return res.status(500).json({ message: error.message });

//   return res.status(201).json({ Key });
// });



app.use((error, req, res, next) => {
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
});

const uploadFile = file => {
  const fileStream = fs.createReadStream(file.path)

  const uploadParams = {
    Bucket: process.env.BUCKET_NAME,
    Body: fileStream,
    Key: file.filename

  }

  return s3Client.upload(uploadParams).promise()
}


app.get('/upload', async (req, res) => {
  try {
    const results = await s3Upload(req.files);
    // console.log(results);
    return res.status(201).json({key});
  } catch (err) {
    console.log(err); 
  }
  });


app.listen(5000, () => console.log("listening on port 5000"));

  return s3Client.upload(uploadParams).promise()
}})

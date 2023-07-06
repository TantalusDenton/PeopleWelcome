const { S3 } = require("aws-sdk");
const { S3Client, PutObjectCommand } = require("@aws-sdk/client-s3");
const uuid = require("uuid").v4;

require('dotenv').config()
const fs = require('fs')

const bucketName = process.env.AWS_BUCKET_NAME
const region = process.env.AWS_BUCKET_REGION
const accessKeyId = process.env.AWS_ACCESS_KEY
const secretAccessKey = process.env.AWS_SECRET_KEY


const s3 = new S3({
  region,
  accessKeyId,
  secretAccessKey
})


// function getUrl() {
//   const bucketParams = {
//       Key: `uploads/${uuid()}-${file.originalname}`,
//       Bucket: process.env.AWS_BUCKET_NAME,
//       Expires: 60
//   }
  
//   try {
//     // Create the command.
//     const command = new GetObjectCommand(bucketParams);

//     // const { url } = await fetch("/s3Url").then(res => res.json())
//     // console.log(url)

//     // await fetch(url, {
//     //   method: "PUT",
//     //   headers: {
//     //     "Content-Type": "multipart/form-data"
//     //   },
//     //   body: file
//     // })

//     // const imageUrl = url.split('?')[0]
//     // console.log(imageUrl)
    
//     const signedUrl = await getSignedUrl(s3Client, command, {
//       expiresIn: 3600,
//     });
//     console.log(
//       `\nGetting "${bucketParams.Key}" using signedUrl with body "${bucketParams.Body}" in v3`
//     );
//     console.log(signedUrl);
//     const response = await fetch(signedUrl);
//     console.log(
//       `\nResponse returned by signed URL: ${await response.text()}\n`
//     );
//   } catch (err) {
//     console.log("Error creating presigned URL", err);
//   }
// exports.getUrl = getUrl

exports.s3Uploadv2 = async (files) => {
  const s3 = new S3();
  const params = files.map((file) => {
    return {
      Bucket: process.env.AWS_BUCKET_NAME,
      Key: `${uuid()}-${file.originalname}`,
      Body: file.buffer,
    };
  });

  return await Promise.all(params.map((param) => s3.upload(param).promise()));
};


// function getFileStream(fileKey) {
//   const downloadParams = {
//     Key: fileKey,
//     Bucket: process.env.AWS_BUCKET_NAME
//   }


//   return s3.getObject(downloadParams).createReadStream()
// }

exports.getFileStream = getFileStream




// function genUploadUrl() {
//   const s3 = new S3();
//   const uploadURL = s3.getSignedUrlPromise('putObject', {
//     Bucket: process.env.AWS_BUCKET_NAME,
//     Expires: 60,
//     Key: `uploads/${uuid()}-${file.originalname}`,
//     ContentType: 'image/jpeg',
//   })
//   //   , function (err, url) {
//   //   console.log(url);
//   // }
//   return uploadURL
// }
// exports.genUploadUrl = genUploadUrl

  // const uploadURL =  s3.getSignedUrl('putObject', params)
  // return uploadURL
//}

//exports.genUploadUrl = genUploadUrl



// // Create service client module using ES6 syntax.
// import { S3Client } from "@aws-sdk/client-s3";
// // Import required AWS SDK clients and commands for Node.js.
// import { GetObjectCommand } from "@aws-sdk/client-s3";
// // Set the AWS Region.
// const REGION = "us-west-1";
// // Create an Amazon S3 service client object.
// const s3Client = new S3Client({ region: REGION });
// export { s3Client };

// //import { s3Client } from "./libs/s3Client.js"; // Helper function that creates an Amazon S3 service client module.

// export const bucketParams = {
//   Bucket: "s3-peoplewelcome-upload",
//   Key: "test2.png",
// };

// export const run = async () => {
//   try {
//     // Get the object from the Amazon S3 bucket. It is returned as a ReadableStream.
//     const data = await s3Client.send(new GetObjectCommand(bucketParams));
//     // Convert the ReadableStream to a string.
//     return await data.Body.transformToString();
//   } catch (err) {
//     console.log("Error", err);
//   }
// };

// export default function Run() {
//  run();
// }

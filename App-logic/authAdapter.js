const AWS = require('aws-sdk')
const keys = require('./awsConfig')
const awsConfig = {
    'region' : 'us-west-2',
    'endpoint' : 'dynamodb-fips.us-west-2.amazonaws.com',
    'accessKeyId' : keys.keys.accessKeyId,
    'secretAccessKey' : keys.keys.secretAccessKey
}
AWS.config.update(awsConfig)
AWS.config.apiVersions = {
    cognitoidentityserviceprovider: '2016-04-18',
    // other service API versions
}

const cognitoIdentityServiceProvider = new AWS.CognitoIdentityServiceProvider()
// Minimal Amplify config placeholder so the build succeeds even if Amplify is unused.
// Replace the values (or set REACT_APP_* env vars) if you leverage any Amplify modules.
const awsconfig = {
  aws_project_region: process.env.REACT_APP_AWS_PROJECT_REGION || "us-east-1",
  aws_cognito_identity_pool_id: process.env.REACT_APP_COGNITO_IDENTITY_POOL_ID || "",
  aws_cognito_region: process.env.REACT_APP_COGNITO_REGION || "us-east-1",
  aws_user_pools_id: process.env.REACT_APP_USER_POOLS_ID || "",
  aws_user_pools_web_client_id: process.env.REACT_APP_USER_POOLS_WEB_CLIENT_ID || "",
  aws_appsync_graphqlEndpoint: process.env.REACT_APP_APPSYNC_GRAPHQL_ENDPOINT || "",
  aws_appsync_region: process.env.REACT_APP_APPSYNC_REGION || "us-east-1",
  aws_appsync_authenticationType: process.env.REACT_APP_APPSYNC_AUTH_TYPE || "API_KEY",
};

export default awsconfig;

# Welcome to PeopleWelcome!

PeopeWelcome is an AI- driven social media platform for humans and machines alike.

This repo contains the webClient, ImageClassifier, Authentication module, and ImageLogic(servers/express-server)

To get started, read the respective readmes

This software is open-sourced under a GNU license.

To start the server: 
1. go to Servers/express-server
2. 'node app.js'

To start the app: 
1. cd webclient
2. 'npm start'

When you choose your account, or add an image, you may need to refresh in order for changes to show up in React. 

* For troubleshooting purposes:
delete package-lock.json

try previous working versions of package.json 
make sure to have '.env' file with right credentials 

try 'npm cache clean --force'
'npm i' or 'npm install' 

* Debugging: 
if you see this error: 'ERROR in Plugin "react" was conflicted between "package.json » eslint-config-react-app » C:\Users\asus\desktop\google-clone\node_modules\eslint-config-react-app\base.js"'

try: Open the package.json and type ctrl + s

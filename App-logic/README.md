Appl logic backend for PeopleWelcome

To run on Ubuntu 22.04:


##prerequisites:

###npm

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash

export NVM_DIR="$HOME/.nvm"

[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"


### node

nvm install node

## Start Node.JS server
in the same directory a package-lock.json run once:

npm install

That will build modules. Do not commit them (they are in a new folder node_modules). Gitignore them.

To start the server, run the following in the same directory a pachage-lock.json :

npm start 

It will run a server on http://localhost:5000

Also possible to run with:

node app.js
const authorize = (req, res, next) => {
    const user = req.query
    // TODO: make it an actual authorization
    if(user.username === 'john.smith' && user.password === '1234') {
        req.myuser = {username: 'john.smith', email: 'john.smith@gmail.com'}
        next()
    } else {
        res.status(401).send('Unauthorized')
    }
}

exports.auth = authorize
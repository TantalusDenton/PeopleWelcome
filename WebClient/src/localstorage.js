const USER_DB_KEY = 'codedamn-users'

export function _dumpLocalStorage() {
    console.log(localStorage.getItem(USER_DB_KEY))
}

export function createUser({username, password}) {
    let usersKey = localStorage.getItem(USER_DB_KEY)

    if(!usersKey) {
        usersKey = '[]'
    }

    const usersArray = JSON.parse(usersKey)

    const duplicate = usersArray.find(user => user.username === username)

    if(duplicate) {
        return {
            status: 'error',
            error: 'Username already exists'
        }
    }

    if(password.length <2){
        return {
            status: 'error',
            error: 'Password too weak'
        }
    }
    usersArray.push({
        username,
        password
    })

    localStorage.setItem(USER_DB_KEY, JSON.stringify (usersArray))
        return {
            status:'ok'
        }


}
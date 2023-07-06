import React, {useState} from 'react'
import { createUser} from '..localstorage'
import { _dumpLocalStorage } from './localstorage'

function Register() {

    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')


    function registerUser(e) {
        e.preventDefault()
        const response = createUser({ username, password})

        if(response.status == 'ok') {
            alert('Your are registered')
        } else {
            alert(response.error)
        }

        _dumpLocalStorage()
    }

    return <div>
        <form onSubmit = {registerUser}>
            <h1>Create an account</h1>
            <input 
            value = {username}
            onChange={e => setUsername(e.target.value)}
            type = "text" 
            placeholder = "username" 
            />
            <input 
            value = {password}
            onChange={e => setPassword(e.target.value)}
            type = "password" 
            placeholder = "password" 
            />
            <input type = "submit" placeholder = "Create account"/>
        </form>
    </div>

}

export default Register
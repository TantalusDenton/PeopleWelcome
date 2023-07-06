import React, { useEffect, useState} from "react"
import { createUser } from "../localstorage"


function LoginPage() {
    const [logOrReg, setLogOrReg] = useState(true)
    const [buttonText, setButtonText] = useState('Log In')
    const [titleText, setTitleText] = useState('Login')
    const [bottomButtonText, setBottomButtonText] = useState('I don\'t have an account')

    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')

    useEffect(() => {
        if(logOrReg) {
            setButtonText('Log In')
            setTitleText('Login')
            setBottomButtonText('I don\'t have an account')
        } else {
            setButtonText('Create New Account')
            setTitleText('Register')
            setBottomButtonText('I already have an account')
        }
    }, [logOrReg])

    const onUsernameChange = event => {
        setUsername(event.target.value)
    }

    const onPasswordChange = event => {
        setPassword(event.target.value)
    }
    
    const onClickBottomButton = () => {
        setLogOrReg(!logOrReg)
    }

    const onLogin = () => {
        if(!logOrReg) {
            createUser(username, password)
        } else {
            console.log(`Logged in as user: ${username}`)
        }
    }

    return(
        <div className="login">
            <h1>{titleText}</h1>
            <input type="text" placeholder="Username" id="username-input" onChange={onUsernameChange}/>
            <input className="password" type="text" placeholder="Password" id="password-input" onChange={onPasswordChange}/>
            <a href="https://dan-auth.auth.us-west-2.amazoncognito.com/login?client_id=a174oiktvvsu5o0gcv3jm2njf&response_type=code&scope=email+openid&redirect_uri=http://localhost:8000/loggedin.html
    " className="button" onClick={onLogin}>{buttonText}</a>
            <button className="bottombutton" onClick={onClickBottomButton}>{bottomButtonText}</button>
        </div>
    )
}

export default LoginPage
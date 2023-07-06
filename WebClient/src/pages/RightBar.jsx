// import Ai from "./Ai"
import React from "react"
import LoginPage from './LoginPage'
import MyAccount from './MyAccount'

function RightBar() {

//     // fetch createPostWithoutTags
//     const data3 = { tags: 'nature' }

//     fetch('localhost:5000/account/:user/createpostnotag', {
//         method: 'POST',
//         headers: {
//         'Content-Type': 'application/json',
//         },
//         body: JSON.stringify(data3)
//     })

//     // fetch getAiIdByAiAndUser
//     const fetData2 = await fetch('localhost:5000/account/:user/:ai/id', {
//         method: 'GET'
//     })

//     // fetch for createNewAi
//     const data4 = { body: 'ai_name' }

//     fetch('localhost:5000/account/:user/createai', {
//         method: 'POST',
//         headers: {
//         'Content-Type': 'application/json',
//         },
//         body: JSON.stringify(data4)
//     })

//     // fetch getAiTags
//     const fetchgetAiTags = await fetch('localhost:5000//account/:user/:ai/alltags', {
//         method: 'GET'
//     })

//     // fetch for getAisByUser

//     const fetchGetAisByUser = await fetch('localhost:5000/account/:user/ownedais', {
//         method: 'GET'
//     })

    
   /* return(
        <div className='rightnav'>
            <LoginPage />
            
        </div>
    )*/
    return(
        <div className='rightnav'>
            <MyAccount />
            
        </div>
    )
}

export default RightBar

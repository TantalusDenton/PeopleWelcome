import React, { useContext } from 'react'
// import Ninja from "../../images/ninja.png";
import {signOut} from "firebase/auth"
import { auth } from '../../firebase/firebase'
import { AuthContext } from '../../context/AuthContext'

const Navbar = () => {
  const { currentUser } = useContext(AuthContext)
  const logout = async() =>{
    signOut(auth).then(() =>{
    this.setState({
     user:null
    })
    this.props.history.push("/");
    }).catch(function(error) {
    // An error happened.
    });
   }
//()=>signOut(auth)
  return (
    <div className='navbar'>
      <span className="logo">Chats</span>
      <div className="user">
        <img src={currentUser.photoURL} />
        <span>{currentUser.displayName}</span>
        <button onClick={logout}>logout</button>
      </div>
    </div>
  )
}

export default Navbar
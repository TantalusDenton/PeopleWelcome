import { BsFillBellFill, BsFillGearFill, BsPersonCircle, BsFillChatDotsFill } from 'react-icons/bs'
import { Link, NavLink } from 'react-router-dom'
import SearchBar from '../components/SearchBar'
import BookData from '../components/Data.json'
import CurrentAiContext from '../components/CurrentAiContext'
import { React, useContext, useEffect, useState } from 'react'
import { AuthContext } from '../context/AuthContext'
import logo from '../images/peoplewelcomelogo3.png';
// import background from './images/blackandwhite2.png';
import {signOut} from "firebase/auth"
import { auth } from '../firebase/firebase'
import "../components/Notifications/navbar.css";

function TopNavBar() {
    const currentAi = useContext(CurrentAiContext).currentAi
    const { currentUser } = useContext(AuthContext);

    const onClickLogout = async (tag) => {
        signOut(auth)
        window.location.reload(false);
      }

    return (
        <nav className='navbar'>
            <Link to='/'><img id='logo-head' src={logo} alt="People Welcome logo" /><div>People <br /> Welcome</div></Link>
            <SearchBar placeholder='Search for anything' data={BookData}/>
            <h3>{`AI: ${currentAi}`}</h3>
            <ul>
                {/*<li>
                    <NavLink to='/messages'>
                        <BsFillChatDotsFill/>
                    </NavLink>
                </li>
                <li>
                    <NavLink to='/settings'>
                        <BsFillGearFill/>
                    </NavLink>
                </li>
                <li>
                    <NavLink to='/notifications'>
                        <BsFillBellFill />
                        {<div className="counter">4</div>*
                    </NavLink>
                </li>
                <li>
                    <NavLink to='/myaccount'>
                        <BsPersonCircle/>
                    </NavLink>
                </li>*/}
                {currentUser ? currentUser.displayName : ''}
            </ul>
            <div className="user">
                <button onClick={()=>onClickLogout()}>logout</button>
            </div>
           
        </nav>
    )
}

export default TopNavBar
//download latest
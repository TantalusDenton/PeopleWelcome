import React from 'react'
import Sidebar from '../../components/Message/Sidebar'
import Chat from '../../components/Message/Chat'


const Home = () => {
  return (
    <div className='home'>
      <div className="container">
        <Sidebar/>
        <Chat/>
      </div>
    </div>
  )
}

export default Home
import React, { useState, useEffect } from 'react'
import LeftBar from './LeftBar'
import RightBar from './RightBar'
import Post from '../pages/Post'
import ImageUploads from './ImageUploads'

function HomePage() {
    const [posts, setPosts] = useState([])
    const [postData] = useState([{ user: 'Jarvis', postId: 0}, {user: 'GLadOS', postId: 1}
    , {user: 'Artem D', postId: 2}, {user: 'Friendly Henry', postId: 3}, {user: 'Daniel Oh', postId: 4}, {user: 'Avni Mungra', postId: 5}
    , {user: 'Yutaro Katori', postId: 6}, {user: 'Agent Smith', postId: 7}, {user: 'Paul McCartney', postId: 8}])

    useEffect(() => {
        const fetchPosts = async () => {
            const promise = await fetch('https://applogic.wwwelco.me:5000/feed')
            const postList = await promise.json()
            setPosts(postList)
        }
        fetchPosts()
    })

    return(
        <div className='homepage'>
            <LeftBar/>
            <RightBar/>
            <div className='postlist'>
                <ul>
                    {posts.map((post, index) => {
                        return (
                            <li key={index}>
                                <ImageUploads value={post}></ImageUploads>
                            </li>
                        )
                    })}
                </ul>
            </div>
        </div>
    )
}

export default HomePage

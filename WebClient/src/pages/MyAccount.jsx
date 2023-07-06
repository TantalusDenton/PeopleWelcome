import { useNavigate } from 'react-router-dom';

import React, { useState, useEffect, useContext } from 'react'

// import axios from 'axios'
// import useQuery from './hooks/useQuery';

import ImageUploads from "./ImageUploads";
import { Grid } from "@material-ui/core";
import { makeStyles } from "@material-ui/core/styles";
import Post from './Post';
import TagsInput from "../components/TagsInput";
import { AuthContext } from "../context/AuthContext";
import CurrentAiContext from "../components/CurrentAiContext";

const useStyles = makeStyles({
  gridContainer: {
    paddingLeft: "40px",
    paddingRight: "40px"
  },
  root: {
    minWidth: 200
  },
  bullet: {
    display: "inline-block",
    margin: "0 2px",
    transform: "scale(0.8)"
  },
  title: {
    fontSize: 14
  },
  pos: {
    marginBottom: 12
  },
  media: {
    height: 0,
    paddingTop: '56.25%', // 16:9,
    marginTop: '30',
    // borderRadius: '12px'
  },
  field: {
    marginTop: 20,
    marginBottom: 20,
    display: 'block'
  }
});

export default function MyAccount() {
  const classes = useStyles();
  const navigate = useNavigate(); 

  // a local state to store the currently selected file.
  const [images, setImages] = useState({ preview: '', data: '' })
  const [status, setStatus] = useState('')
  //const [user, setUser] = useState('')
  const [posts, setPosts] = useState([])
  const [clickSubmit, setClickSubmit] = useState(true)
  const [newUser, setNewUser] = useState('')
  const user = useContext(CurrentAiContext).currentAi

  const onUserTextChange = event => {
    setNewUser(event.target.value)
  }

  /*const onClickSetUser = () => {
    setUser(newUser)
  }*/
/*
  useEffect(() => {
    const fetchPosts = async () => {
      const promise = await fetch(`http://localhost:5000/account/${user}/posts`, {
        headers: {
          'Content-Type': 'application/json'
        }
      })
      const postList = await promise.json()
      setPosts(postList)
      console.log(postList)
    }
    fetchPosts()
  }, [user]);
*/

  //get secure / signed url from server
  // const { url } =  fetch("/s3Url").then(res => res.json())
  // console.log(url)

  const handleSubmit = async (e) => {
    e.preventDefault()
    let formData = new FormData()
    formData.append('file', images.data)
    const response = await fetch(`https://applogic.wwwelco.me:5000/upload/account/${user}/createpost`, {
      method: 'POST',
      body: formData,
    }).then(() => {
      setClickSubmit(!clickSubmit)
    })
    if (response) {
      setStatus(response.statusText)
      console.log(response.statusText)
    }
  }

  const handleFileChange = (e) => {
    // setSelectedFile(event.target.files[0])
    const img = {
      preview: URL.createObjectURL(e.target.files[0]),
      data: e.target.files[0],
    }
    setImages(img)
    console.log(`img.data: ${img.data}`)
    console.log(`img.preview: ${img.preview}`)
  }
    
  return (
    <div className="myaccount"> 
      <h2 id="postAs" > Post as {user} </h2>
      <form onSubmit={handleSubmit}>
        <input type="file" class="browseImage" onChange={handleFileChange}/>
        <input type="submit" class="uploadImage" value="Upload File" />
      </form>
      
      {/*<button onClick={() => navigate("/login")}> Log Out </button>*/}
     {/* 
      <div>
        <img src="fd09e1ea-1530-4189-95ff-1baef5f47eae-desk setups.jpeg"></img>
        <button onClick={() => navigate("/login")}> Log Out </button>
      </div>
      <div className="App">
        <h2>Enter some Tags</h2>
        <TagsInput/>
      </div> */}

        {/* <Grid
          container
          spacing={4}
          className={classes.gridContainer}
          justifyContent="center"
        >
          {posts.map((post, index) => {
            return (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <ImageUploads value={{username: user, image_id: post.image_id, date: post.date}} />
              </Grid>
            )
          })}
          */}

      <Grid
        container
        spacing={7}
        className={classes.gridContainer}
        justifyContent="center"
      >
        {posts.map((post, index) => {
          return (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <ImageUploads value={{username: user, image_id: post.image_id, date: post.date}} />
            </Grid>
          )
        })}

      </Grid>
      {/*<label>Enter your name:
        <input type="text" onChange={onUserTextChange}/>
      </label>
      <button onClick={onClickSetUser}>✔️</button>*/}
    </div>
  )
}

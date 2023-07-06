import { React, useState, useContext, useEffect } from "react";
import { makeStyles } from "@material-ui/core/styles";
import Card from "@material-ui/core/Card";
import CurrentAiContext from "../components/CurrentAiContext";
import { CardActions, CardContent, Typography, CardMedia, Grid } from "@material-ui/core";
import { Avatar, IconButton } from '@mui/material';
import CardHeader from '@material-ui/core/CardHeader';

import { cyan } from '@mui/material/colors';
import ShareIcon from '@mui/icons-material/Share';
import Button from '@mui/material/Button';
import TagsInput from "../components/TagsInput";
import Heart from "../images/notifications/heart.svg";
import HeartFilled from "../images/notifications/heartFilled.svg";
import { AuthContext } from "../context/AuthContext";
// import FavoriteIcon from '@mui/icons-material/Favorite';
// import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
// import IconButton from '@mui/material'; // , Button
// import FavoriteIcon from '@mui/icons-material/Favorite';
// import TextField from '@mui/material/TextField';
// import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';

const useStyles = makeStyles({
  root: {
    minWidth: 400
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
    paddingTop: '100%', // 1:1, (second number / first number) * 100 = 100%
    marginTop: '30',
    // borderRadius: '12px'
  },
  field: {
    marginTop: 20,
    marginBottom: 20,
    display: 'block'
  }
});

export default function ImageUploads(props, socket, user, post) {
  const classes = useStyles();
  // const bull = <span className={classes.bullet}>•</span>;
  const currentAi = useContext(CurrentAiContext).currentAi
  const { currentUser } = useContext(AuthContext)
  const [tags, setTags] = useState([])
  const [suggestions, setSuggestions] = useState([])
  const [newTag, setNewTag] = useState('')
  // const bull = <span className={classes.bullet}>•</span>;
  const data = props.value
  const [date, setDate] = useState(data.date)
  const [liked, setLiked] = useState(false);

  const [message, setMessage] = useState("");

  const handleChange = (event) => {
    setMessage(event.target.value);
  };

  const handleNotification = (type) => {
    type === 1 && setLiked(true);
    socket.emit("sendNotification", {
      senderName: currentUser.displayName,
      receiverName: data.username,
      type,
    });
  };

  const onTagTextChange = event => {
    setNewTag(event.target.value)
    setMessage(event.target.value)
  }

  const addTag = async () => {
    const url = `/account/${currentUser.displayName}/${currentAi}/addtag`
    console.log('in AddTag...')
    await fetch(url, {
      
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        imageId: id,
        imageOwner: data.username,
        tag: newTag
      })
    }).then(async () => {
      console.log('in onClickAddTag, retraining...')
      await Retrain()
      await retrainTree()
    })
    
    document.getElementById('tagbox').value = ''
  } 

  
  const Retrain = async () => {
    const classifier_url = `https://schedulers.wwwelco.me:3003/api/v1/addToTrainingQueue/${currentUser.displayName}/${currentAi}`
    const runRetrain = async () => {
      const promise = await fetch(classifier_url)

      const list = await promise.json()
    }
    runRetrain()
  } 
  

  const retrainTree = async () => {
    const url = `https://objectdetector.wwwelco.me:3002/account/${currentUser.displayName}/${currentAi}/retraintree`
    await fetch(url, {
      method: 'POST'
    })
  }

  const onClickAddTag = async (tag) => {
    setMessage('')
    await addTag().then(async () => {
      setTags([...tags, tag])
    })
  }

  const onClickRemoveTag = async (tag) => {
    const url = `https://applogic.wwwelco.me:5000/account/${currentUser.displayName}/${currentAi}/removetag`
    await fetch(url, {
      
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        imageId: id,
        imageOwner: data.username,
        tag: tag
      })
    }).then(async () => {
      let i = 0
      while(i < tags.length && (tags[i] !== tag)) {
        i++
      }
      if(i < tags.length) {
        setTags(tags.splice(i, 1))
      }
      console.log('in onClickAddTag, retraining...')
      //await Retrain()
      await retrainTree()
    })
  }

  const handleSubmit = (e) => {
    const newEle = document.createElement('h4');
    newEle.innerText = '';

   // append it inside the button
    e.target.appendChild(newEle);
    e.preventDefault()
    if ('') {
      console.log('')
    }
  }
  const id = data.image_id

  const [imageUrl, setImageUrl] = useState(`/images/${id}`)

  useEffect(() => {
    const user = currentUser.displayName
    const url = `https://applogic.wwwelco.me:5000/account/${user}/${currentAi}/${id}/tagsonpost`
    const getAllTagsByAi = async () => {
      const promise = await fetch(url)
      const list = await promise.json()
      JSON.stringify(list)
      if(list.length > 0) {
        setTags(list)
      }
    }
    if(currentAi !== '') {
      getAllTagsByAi()
    }

    const url2 = `https://objectdetector.wwwelco.me:3002/account/${currentUser.displayName}/${currentAi}/${id}/tagsuggestions`
    const getTagSuggestions = async () => {
      const promise = await fetch(url2)
      const object = await promise.json()
      //JSON.stringify(object)
      console.log('suggestions:', object.suggestions)
      const suggestions = object.suggestions
      /*
      let i = 0
      for (const tag in suggestions) {
        if (tags.includes(tag)) {
          suggestions.splice(i, 1)
        }
        i++
      }
      */
      if(suggestions.length > 0) {
        setSuggestions(suggestions)
      }
    }
    if(currentAi !== '') {
      getTagSuggestions()
    }
  }, [currentAi])

  return (
    <Card className={classes.root} variant="outlined">
      <CardHeader
        avatar={
          <Avatar sx={{ bgcolor: cyan[500] }} aria-label="avatar"></Avatar>
        }
        action={
          <IconButton aria-label="settings">
            {/*<ShareIcon />*/}
          </IconButton>
        }

        title={data.username}
        subheader={date}
      />
      <CardMedia
        className={[classes.media, 'photo']}
        image={imageUrl} // require image
        title=""
        style={useStyles.media} // specify styles
      >
        <div className='overlay'>
          <input type="text" id="tagbox" placeholder="Add a tag..." onChange={onTagTextChange} value={message}/>
          <button onClick={() => onClickAddTag(newTag)}>Add</button>

          <Grid
            container
            spacing={7}
            className={classes.gridContainer}
            justifyContent="center"
          >
            {tags.map((tag, index) => {
              return (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <h3>{tag}</h3>
                  {/*<button onClick={() => onClickRemoveTag(tag)}>x</button>*/}
                </Grid>
              )
            })}

          </Grid>
          <Grid
            id="suggestions"
            container
            spacing={7}
            className={classes.gridContainer}
            justifyContent="center"
          >
            {suggestions.map((tag, index) => {
              return (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <h3>{tag}</h3>
                </Grid>
              )
            })}

          </Grid>

        </div>
      </CardMedia>
      <CardActions>
        {/*{liked ? (
          <img src={HeartFilled} alt="" className="cardIcon" />
          ) : (
          <img
            src={Heart}
            alt=""
            className="cardIcon"
            onClick={() => handleNotification(1)}
          />
        )}*/}
        {/* <FavoriteIcon color="primary" size = "medium"/> */}
        {/* <FavoriteIcon></FavoriteIcon>
        <FavoriteBorderIcon></FavoriteBorderIcon> */}
        {/* <IconButton aria-label="add to favorites">
          </IconButton>  */}
         
        {/*<Button size="small">likes</Button>*/}
      </CardActions>
    </Card>
  );
}

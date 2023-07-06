import React, { useContext, useState, useEffect } from "react"
import CurrentAiContext from "../components/CurrentAiContext"
import { Avatar, IconButton } from '@mui/material';
import { cyan } from '@mui/material/colors';

function Ai(props) {
    const id = props.value.image_id
    const name = props.value.ai_name
    const { currentAi, setCurrentAi } = useContext(CurrentAiContext)
    const onClickAi = () => {
        setCurrentAi(name)
        console.log(`currentAi: ${currentAi}`)
    }
    const [img, setImg] = useState(undefined)

    useEffect(() => {
        const imageUrl = `/images/${id}`
        const fetchImage = async () => {
          const image = await fetch(imageUrl)
          const imageBlob = await image.blob()
          const imageObjectURL = URL.createObjectURL(imageBlob);
          setImg(imageObjectURL);
        }
        if(id) {
            fetchImage()
        }
      }, [])

    return(
        <div className="ai">
            <button onClick={onClickAi}>
                {/* <img src={props}/>     */}
                
                <Avatar sx={{ bgcolor: cyan[500] }} aria-label="avatar"></Avatar>
            </button>
            <h3>{name}</h3>
        </div>
    )
}

export default Ai
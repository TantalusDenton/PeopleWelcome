import { Row, Col } from 'react-simple-flex-grid';
import "react-simple-flex-grid/lib/main.css";
import { AuthContext } from '../context/AuthContext';
import Ai from "./Ai"
import React, { useState, useEffect, useContext } from 'react';

const AIs = () => {
    const id = 6
    const [img, setImg] = useState()
    const [aiList, setAiList] = useState([])
    const { currentUser } = useContext(AuthContext);
    const [newAi, setNewAi] = useState('')


    const onAiNameChange = event => {
      setNewAi(event.target.value)
    }

    const onClickCreateAi = async () => {
      if((newAi !== '') && (aiList.includes(newAi) === false)) {
        const url = `/account/${currentUser.displayName}/createai`
        await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            ai: newAi
          })
        })
      }
    }

    useEffect(() => {
      const imageUrl = `/api/v1/image/${id}`
      const fetchImage = async () => {
        const image = await fetch(imageUrl)
        const imageBlob = await image.blob()
        const imageObjectURL = URL.createObjectURL(imageBlob);
        setImg(imageObjectURL);
      }
      fetchImage()
    }, [])

    useEffect(() => {
      const url = `/account/${currentUser.displayName}/ownedais`
      const fetchAis = async () => {
        const promise = await fetch(url)
        const list = await promise.json()
        setAiList(list)
      }
      fetchAis()
    }, [])

      // Smoke and mirrors. TODO: get images one by one and pass to Ai component.
      // Likely the name and image_id will be passed to each, and within each Ai
      // component is where their profile picture will be fetched using image_id.
    const [data, setData] = useState([{name : 'Yutaro Katori', image : "https://i.kym-cdn.com/entries/icons/original/000/017/299/DmbzJspWwAEprcQ.jpg"}, {name : 'Agent Smith', image : {img}},  {name : 'John Lennon'},
     {name: 'George Harrison'}, {name: 'Ringo Starr'},
     {name: 'Paul McCartney'}, {name: 'George Martin'},
     {name: 'Artem D'}, {name: 'Friendly Henry'},
     {name: 'Daniel Oh'}, {name: 'Avni Mungra'}])
      
  return (
    <div>
      <div className='AIs'>
        <Row gutter={40}>
          {(aiList).map(ai => 
            <Col 
              xs={{ span: 7 }} sm={{ span: 5 }} md={{ span: 4 }}
              lg={{ span: 4 }} xl={{ span: 4 }} >
                <Ai value={ai}/>
              </Col>
          )}
        </Row>
      </div>
      <input className='ai_name_input' placeholder='Enter New AI Name' onChange={onAiNameChange}/>
      <button onClick={onClickCreateAi} className='createai'>Create AI</button>
    </div>
  );
}

export default AIs;
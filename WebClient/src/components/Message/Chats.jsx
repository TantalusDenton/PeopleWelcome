import { doc, onSnapshot } from "firebase/firestore";
import React, { useContext, useEffect, useState } from "react";
import { AuthContext } from "../../context/AuthContext";
import { ChatContext } from "../../context/ChatContext";
import { db } from "../../firebase/firebase";

const Chats = () => {
  const [chats, setChats] = useState([]);
  const { currentUser } = useContext(AuthContext);
  const { dispatch } = useContext(ChatContext);

  useEffect(() => {
    const getChats = () => {
      const unsub = onSnapshot(doc(db, "userChats", currentUser.uid), (doc) => {
        setChats(doc.data());
      });

      return () => {
        unsub();
      };
    };

    currentUser.uid && getChats();
  }, [currentUser.uid]);

  const handleSelect = (u) => {
    dispatch({ type: "CHANGE_USER", payload: u });
  };
//onClick={() => handleSelect(chat[1].userInfo)}
  // chat id, image, and displayName  = handleSelect 
  return (
    <div className="chats">
      {/* {Object.entries(chats)?.map((chat) => (
        <div
          className="userChat"
          key={chat[0]}
          onClick={()=>handleSelect(chat[1].userInfo)}
        >
          <img src={chat[1].userInfo.photoURL} alt="" />
          <div className="userChatInfo">
            <span>{chat[1].userInfo.displayName}</span>
            <p>{chat[1].lastMessage?.text}</p>
          </div>
        </div>
      ))} */}
        
      <div className="userChat">
        <img src="https://media.licdn.com/dms/image/C5603AQF6C2Ua_S1hCQ/profile-displayphoto-shrink_200_200/0/1652992176979?e=1686182400&v=beta&t=yYxhuaJbAnbUYhbLedzDPLnuIMzCrCPe-2qt8MoggwM" alt="" />
    
        <div className="userChatInfo">
          <span>Jane</span>
          <p>did you get my email?</p>
        </div>
      </div>
    </div>
  );
};

export default Chats;
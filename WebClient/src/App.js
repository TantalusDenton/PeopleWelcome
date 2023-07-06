import React, { useState, useMemo, useContext, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate} from 'react-router-dom';
import './App.css';
import Register from "./pages/Message/Register"
import Login from "./pages/Message/Login"
import MessagesHome from "./pages/Message/MessagesHome"
import "./css/style.scss"
import { firebaseConfig } from './firebase/firebase';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import MyAccount from './pages/MyAccount';
import Navbar from './components/Notifications/Navbar';
// import UploadImage from './UploadImage.jsx';
import EditProfile from './pages/EditProfile';
import Notifications from './pages/Notifications';
import Settings from './pages/Settings';
import TopNavBar from './pages/TopNavBar';
import CurrentAiContext from './components/CurrentAiContext';
import ImageUploads from './pages/ImageUploads';
// import { useAuth0 } from "@auth0/auth0-react";
// import axios from 'axios'
import logo from "./logo.svg";
import "@aws-amplify/ui-react/styles.css";
//import { io } from "socket.io-client";
import {
  withAuthenticator,
  Button,
  Heading,
  Image,
  View,
  Card,
} from "@aws-amplify/ui-react";
import { AuthContext, AuthContextProvider } from './context/AuthContext';  
import { messaging } from "./firebase/firebase";
// import messaging from '@react-native-firebase/messaging';

/* function App({ signOut }) {
  return (
    <View className="App">
      <Card>
        <Image src={logo} className="App-logo" alt="logo" />
        <Heading level={1}>We now have Auth!</Heading>
      </Card>
      <Button onClick={signOut}>Sign Out</Button>
    </View>
  );
}
 */

// Request permission for notifications
// const requestNotificationPermission = async () => {
//   try {
//     await messaging
//       .requestPermission()
//       .then(() => {
//         console.log('Notification permission granted.');
//         getMessagingToken();
//       })
//   } catch (error) {
//     console.log('Unable to get permission to notify.', error);
//   }
// };

// Get the FCM token
// const getMessagingToken = () => {
//   messaging
//     .getToken()
//     .then((token) => {
//       console.log('FCM token:', token);
//       // Send the token to your server to associate it with the user
//     })
//     .catch((error) => {
//       console.log('Unable to get FCM token.', error);
//     });
// };

function App({ signOut }) {
  const [currentAi, setCurrentAi] = useState('')
  const value = useMemo(
    () => ({ currentAi , setCurrentAi }),
    [currentAi]
  )
  const { currentUser } = useContext(AuthContext);
  // if user or not, navigate to login
  const ProtectedRoute = ({ children }) => {
    if (!currentUser) {
      return <Navigate to="/login"/>
    }
    return children
  }

  const [username, setUsername] = useState("");
  const [user, setUser] = useState("");
  const [socket, setSocket] = useState(null);

  /*
  useEffect(() => {
    setSocket(io("http://localhost:5001"));
    // socket.on("first", (msg) => {
    //   console.log(msg);
    // });
  }, []);
  */

  useEffect(() => {
    socket?.emit("newUser", user);
  }, [socket, user]);
  
  // useEffect(() => {
  //   requestNotificationPermission();
  // }, []);

  // const { isAuthenticated } = useAuth0();
  
  return (
    <AuthContextProvider children={      <CurrentAiContext.Provider value={value}>
    <Router>
    {/* <Button onClick={signOut}>Sign Out</Button> */}
      <TopNavBar socket={socket} />
        <div className='content'>
          <Routes>
            {/* <Route exact path="/" render={() => isAuthenticated ? <HomePage /> : <LoginPage />
            }/> */}
            {/* <Route path='/' element={<HomePage />} /> */}
            <Route path='/'>
              <Route index element={
              <ProtectedRoute>
                <HomePage />
              </ProtectedRoute>}/>
            </Route>
            <Route path = "login" element={<Login />}/>
            <Route path="register" element={<Register />} />
            <Route path = "messages" element={<MessagesHome />}/>
            <Route path='/notifications' element={<Notifications />} />
            <Route path='/navbar' element={ <Navbar/> } />
            <Route path='/foryou'/>
            {/* <Route path='/createpost' element ={<UploadImage/>}/> */}
            <Route path='/messages'/>
            <Route path='/settings' element={ <Settings/> } />
            <Route path='/myaccount' element={<MyAccount/>} />
            <Route path='/editprofile' element={ <EditProfile/> } />
            <Route path='/login' element={<LoginPage />} />
            <Route path='/imageuploads' element ={<ImageUploads/>}/>
          </Routes>
        </div>
    </Router>
  </CurrentAiContext.Provider>}>
    </AuthContextProvider>
  );
}


export default App;
// export default withAuthenticator(App);

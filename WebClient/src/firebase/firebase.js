import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth";
import { getStorage } from "firebase/storage";
import { getFirestore } from "firebase/firestore";
import { getMessaging } from "firebase/messaging/sw";

const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_KEY,
  authDomain: "message-eb542.firebaseapp.com",
  projectId: "message-eb542",
  storageBucket: "message-eb542.appspot.com",
  messagingSenderId: "817036030666",
  appId: "1:817036030666:web:fa7239b6ff9662a2f53a33",
  measurementId: "G-MQSFL7T7YC"
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);
export const auth = getAuth();
export const storage = getStorage();
export const db = getFirestore()
export const analytics = getAnalytics(app);
export const messaging = getMessaging(app);

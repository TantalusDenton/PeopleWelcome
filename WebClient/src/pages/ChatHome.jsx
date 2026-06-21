import React, { useContext, useEffect, useMemo, useState } from "react";

import ChatWindow from "../components/ChatWindow";
import CurrentAiContext from "../components/CurrentAiContext";
import { AuthContext } from "../context/AuthContext";
import ChatLeftBar from "./LeftBar";
import Ai from "./Ai";
import { fetchAgentHistory, sendChatMessage } from "../services/chatApi";

const ChatHome = () => {
  const { currentUser } = useContext(AuthContext);
  const { currentAi } = useContext(CurrentAiContext);
  const [aiList, setAiList] = useState([]);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const userId = useMemo(() => {
    if (!currentUser) {
      return "";
    }
    return currentUser.displayName || currentUser.email || currentUser.uid;
  }, [currentUser]);

  useEffect(() => {
    if (!userId) {
      setAiList([]);
      return;
    }
    const controller = new AbortController();
    const loadAis = async () => {
      try {
        const response = await fetch(`/account/${userId}/ownedais`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error("Unable to load AI list");
        }
        const data = await response.json();
        setAiList(data);
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message);
        }
      }
    };
    loadAis();
    return () => controller.abort();
  }, [userId]);

  useEffect(() => {
    if (!userId || !currentAi) {
      setMessages([]);
      return;
    }
    const controller = new AbortController();
    const loadHistory = async () => {
      try {
        const data = await fetchAgentHistory(userId, currentAi);
        setMessages(data.messages || []);
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message);
        }
      }
    };
    loadHistory();
    return () => controller.abort();
  }, [userId, currentAi]);

  const handleSend = async (text) => {
    if (!userId || !currentAi) {
      return;
    }
    setIsLoading(true);
    setError("");
    try {
      const response = await sendChatMessage({
        userId,
        aiName: currentAi,
        message: text,
      });
      setMessages(response.messages || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-home">
      <ChatLeftBar />
      <main className="chat-main">
        <ChatWindow
          messages={messages}
          isLoading={isLoading}
          error={error}
          selectedAi={currentAi}
          onSend={handleSend}
        />
      </main>
      <aside className="chat-avatar-rail">
        <div className="chat-avatar-rail__header">
          <h3>Your AIs</h3>
          <p>Select an avatar to open a chat</p>
        </div>
        <div className="chat-avatar-rail__list">
          {aiList.map((ai) => (
            <Ai key={ai.ai_name || ai.name} value={ai} />
          ))}
        </div>
      </aside>
    </div>
  );
};

export default ChatHome;

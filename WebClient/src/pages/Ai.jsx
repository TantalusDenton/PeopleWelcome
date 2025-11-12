import React, { useContext, useEffect, useState } from "react";
import { Avatar } from "@mui/material";
import { cyan } from "@mui/material/colors";

import CurrentAiContext from "../components/CurrentAiContext";

function Ai({ value }) {
  const imageId = value.image_id;
  const name = value.ai_name || value.name || value;
  const { currentAi, setCurrentAi } = useContext(CurrentAiContext);
  const [imageUrl, setImageUrl] = useState(undefined);
  const isActive = currentAi === name;

  useEffect(() => {
    if (!imageId) {
      return;
    }
    const controller = new AbortController();
    let objectUrl;
    const loadImage = async () => {
      try {
        const response = await fetch(`/images/${imageId}`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          return;
        }
        const blob = await response.blob();
        objectUrl = URL.createObjectURL(blob);
        setImageUrl(objectUrl);
      } catch {
        // Fail silently; avatar fallback will be used.
      }
    };
    loadImage();
    return () => {
      controller.abort();
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [imageId]);

  const onClickAi = () => {
    setCurrentAi(name);
  };

  return (
    <div className={`ai ${isActive ? "ai--active" : ""}`}>
      <button type="button" onClick={onClickAi} className="ai__button">
        <Avatar
          src={imageUrl}
          alt={name}
          sx={{ bgcolor: cyan[500] }}
          aria-label={`${name} avatar`}
          className={isActive ? "ai__avatar--active" : ""}
        />
      </button>
      <h3>{name}</h3>
    </div>
  );
}

export default Ai;

import React from "react"

const CurrentAiContext = React.createContext({
    currentAi: '',
    setCurrentAi: () => {}
})

export default CurrentAiContext
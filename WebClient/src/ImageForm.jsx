import React,{ useState } from "react"

function ImageForm() {
    const [caption, setCaption] = useState('')
    const [selectedFile, setSelectedFile] = useState(null)

    return (
        <div className="imageform">
            <form>
                <input 
                type="text" 
                value={caption}
                onChange={(e) => setCaption(e.target.value)}
                />

                <input 
                type="file" 
                value={selectedFile}
                onChange={(e) => setSelectedFile(e.target.files[0])}
                />
            </form>
        </div>
    )
}

export default ImageForm
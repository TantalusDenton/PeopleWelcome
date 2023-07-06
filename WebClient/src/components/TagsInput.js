import {useState} from 'react'

export default function TagsInput() {
  const [tags, setTags] = useState([])
  
  function handleKeyDown(e) {
    if (e.key !== 'Enter') return 
    const value = e.target.value
    if(!value.trim()) return 
    setTags([...tags,value])
  }

  function removeTag(index) {
    setTags(tags.filter((el, i) => i !== index))
  }
  return (
    <div className="tags-input-container">

      {tags.map((tag, index) => (
        <div className="tag-item" key={index}>
          <span className="text">{tag}</span>
          <span className="close" onClick={() => removeTag(index)}>&times;</span>
        </div>
      )) }
      <input onKeyDown = {handleKeyDown} type="text" className="tags-input" placeholder="Add a hashtag" />
    </div>
  )
}
// export default tagsInput;
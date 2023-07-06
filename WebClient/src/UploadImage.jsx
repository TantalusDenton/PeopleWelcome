// import { useState } from 'react'
// import axios from 'axios'

// export default function UploadImage() {
//   const [file, setFile] = useState()
//   const [description, setDescription] = useState("")

//   const submit = async event => {
//     event.preventDefault()
  
//     const formData = new FormData()
//     formData.append("image", file)
//     formData.append("description", description)
  
//     const result = await axios.post('/api/images', formData, { headers: {'Content-Type': 'multipart/form-data'}})
//     console.log(result.data)
//   }
//     // Send the file and description to the server

//   return (

//     <div className="App">
//       <form onSubmit={submit}>
//         <input
//           filename={file} 
//           onChange={e => setFile(e.target.files[0])} 
//           type="file" 
//           accept="image/*"
//         ></input>
//         <input
//           onChange={e => setDescription(e.target.value)} 
//           type="text"
//         ></input>
//         <button type="submit">Submit</button>
//       </form>
//     </div>
//   )
// }

// import { useState } from 'react'

// export default function UploadImage() {
//   const [image, setImage] = useState({ preview: '', data: '' })
//   const [status, setStatus] = useState('')
//   const handleSubmit = async (e) => {
//     e.preventDefault()
//     let formData = new FormData()
//     formData.append('file', image.data)
//     const response = await fetch('http://localhost:5000/image', {
//       method: 'POST',
//       body: formData,
//     })

//     if (response) setStatus(response.statusText)
//   }

//   const handleFileChange = (e) => {
//     const img = {
//       preview: URL.createObjectURL(e.target.files[0]),
//       data: e.target.files[0],
//     }
//     setImage(img)
//   }

//   return (
//     <div className='createpost'>
//       <h1>Upload to server</h1>
//       {image.preview && <img src={image.preview} width='100' height='100' />}
//       <hr></hr>
//       <form onSubmit={handleSubmit}>
//         <input type='file' name='file' onChange={handleFileChange}></input>
//         <button type='submit'>Submit</button>
//       </form>
//       {status && <h4>{status}</h4>}
//     </div>
//   )
// }

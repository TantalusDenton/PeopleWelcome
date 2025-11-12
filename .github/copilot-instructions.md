# PeopleWelcome AI Agent Instructions

## Project Overview
PeopleWelcome is an AI-driven social media platform integrating multiple services:

- `WebClient/` - React frontend for user interface
- `App-logic/` - Node.js/Express backend handling AWS services
- `ImageClassifier/` - Python-based image classification service
- `GoogleVisionTree/` - Python service for Google Vision AI integration
- `Scheduler/` - Python service for managing training queues

## Key Architecture Patterns

### Authentication & Storage
- Firebase Authentication for user management (`WebClient/src/firebase/firebase.js`)
- AWS S3 for image storage (`App-logic/s3Service.js`)
- DynamoDB for metadata and relationships (`App-logic/datarepo.js`)

### Frontend Patterns
- React Context for state management:
  - `AuthContext` for user state (`WebClient/src/context/AuthContext.js`)
  - `ChatContext` for messaging (`WebClient/src/context/ChatContext.js`)
- File upload flow uses AWS S3 presigned URLs (`WebClient/src/s3.js`)

### Backend Services
- Express server handles AWS integrations (`App-logic/server.js`):
  - Image upload/retrieval via S3
  - User data in DynamoDB
- Image processing pipeline:
  - FastAPI server for classification (`ImageClassifier/main.py`)
  - Google Vision API integration (`GoogleVisionTree/ObjectDetection.py`)
  - Training queue management (`Scheduler/trainingQueue.py`)

## Development Workflow

### Environment Setup
1. Frontend (WebClient):
```bash
cd WebClient
npm install
npm start
```

2. Backend (App-logic):
```bash
cd App-logic
npm install
node app.js  # Runs on port 5000
```

3. Image Services:
```bash
cd ImageClassifier
pip install -r requirements.txt
uvicorn main:app --reload --port 3001
```

### Required Environment Variables
- AWS: `AWS_BUCKET_NAME`, `AWS_BUCKET_REGION`, `AWS_ACCESS_KEY`, `AWS_SECRET_KEY`
- Firebase: `REACT_APP_FIREBASE_KEY` and other Firebase config in `WebClient/src/firebase/firebase.js`
- Image Processing: Google Vision API credentials

## Common Tasks

### Adding New Image Processing Features
1. Implement classifier in `ImageClassifier/`
2. Update training queue in `Scheduler/trainingQueue.py`
3. Add frontend components in `WebClient/src/components/`
4. Configure S3 upload in `App-logic/s3Service.js`

### User Authentication Flow
1. Firebase handles auth UI/logic (`WebClient/src/pages/LoginPage.jsx`)
2. User data syncs to DynamoDB (`WebClient/src/firebase/firebaseDynamo.js`)
3. JWT tokens pass to backend services

### File Upload Pipeline
1. Frontend generates S3 presigned URL
2. Direct upload to S3
3. Metadata stored in DynamoDB
4. Image processing services triggered via queue

## Project Conventions

### API Response Format
Standardized error response:
```javascript
{
  status: 400|500,
  message: "Error description"
}
```

### File Naming
- React components: PascalCase (e.g., `ImageForm.jsx`)
- Services/utilities: camelCase (e.g., `s3Service.js`)
- Python modules: snake_case (e.g., `decision_tree.py`)

### State Management
- Use React Context for global state
- Local component state for UI-specific data
- AWS services for persistent storage

## Integration Points
- Frontend → Backend: REST API on port 5000
- Backend → S3: AWS SDK
- Backend → Image Services: FastAPI endpoints on port 3001
- Image Services → Google Vision: REST API
FROM node:20-alpine AS build
WORKDIR /app
COPY WebClient/package*.json ./
RUN npm ci --legacy-peer-deps
COPY WebClient ./
ARG REACT_APP_API_URL=http://localhost:8000
ENV REACT_APP_API_URL=$REACT_APP_API_URL
RUN npm run build
FROM nginx:1.27-alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80

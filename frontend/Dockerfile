# Front : build Vite -> statique servi par nginx. VITE_API_URL est fige au BUILD
# (Vite l'inline) ; le navigateur appelle l'API via cette URL cote hote.
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
ARG VITE_API_URL=http://localhost:8001
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80

import uvicorn

if __name__ == '__main__':
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=3002,
                reload=True,
                ssl_keyfile="./sslcert/privkey.pem",
                ssl_certfile="./sslcert/fullchain.pem"
                )

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime
from FilterJSON import filter
from Tagger import calculateSimilarity
from starlette.requests import Request
from starlette import status
import json
import os

if not os.path.exists('./csv'):
    os.mkdir('./csv')

if not os.path.exists('./JSON'):
    os.mkdir('./JSON')
    

app = FastAPI()


@app.post("/upload/")
async def upload_file(request: Request, json_data: dict = Body(...)):

    content_type = request.headers.get("content-type", None)
    if content_type != "application/json":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type {content_type}")
    
    json_string = json.dumps(json_data)
    
    # Save the uploaded file to a temporary file
    with open('tempjson', "w") as buffer:
        buffer.write(json_string)

    # Process the JSON file
    fileName = filter('tempjson')

    simsJSON = calculateSimilarity(fileName)

    return FileResponse(simsJSON, media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

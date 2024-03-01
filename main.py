from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from FilterJSON import filter
from Tagger import calculateSimilarity

app = FastAPI()


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary file
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())

    # Process the JSON file
    fileName = filter(file.filename)

    simsJSON = calculateSimilarity(fileName)

    return FileResponse(simsJSON, media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="https://pandas-1txx.onrender.com", port=8000)

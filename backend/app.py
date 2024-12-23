from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import shutil

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Temporary folder to save uploaded images
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_images(files: List[UploadFile]):
    """Endpoint to upload multiple image files."""
    
    # Clear the folder before uploading new files
    for file_name in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    file_paths = []
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)

    return {"message": "Files uploaded successfully", "file_paths": file_paths}

@app.post("/search")
async def search_images(description: str = Form(...), k: int = 5):
    """Endpoint to search for the most relevant images based on the description."""
    # Collect all image file paths
    image_files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        return JSONResponse(content={"error": "No images available for search."}, status_code=400)

    logits = []

    # Process each image and calculate similarity
    for image_file in image_files:
        try:
            image = Image.open(image_file).convert("RGB")
        except Exception as e:
            print(f"Cannot open image {image_file}: {e}")
            continue

        inputs = processor(text=[description], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image[0][0].item()  # Single similarity score per image
            logits.append((image_file, logits_per_image))

    # Sort by similarity and return top k results
    top_k = sorted(logits, key=lambda x: x[1], reverse=True)[:k]

    return {"results": [{"name": os.path.basename(result[0]), "path": result[0]} for result in top_k]}

@app.get("/clear")
async def clear_uploaded_images():
    """Endpoint to clear uploaded images."""
    shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return {"message": "Uploaded images cleared."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
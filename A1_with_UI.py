'''
from fastapi import FastAPI, HTTPException,Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Obj(BaseModel):
    id:int
    name:str
    role:str=None

def read_file():
    with open("data.json","r") as f:
        return json.load(f)

def write(data):
    with open("data.json", "w") as f:
        json.dump(data, f)

#get item based on an id
@app.get("/obj")
async def read_items(request: Request):
    data = read_file()
    #return data["Characters"]
    return templates.TemplateResponse("home.html",request = request,context = {"obj":data["Characters"]})

@app.post("/new")
async def create(request: Request,item:Obj):
    data = read_file()
    data["Characters"].append(item.dict())
    write(data)
    return item

@app.delete("/delete")
async def delete(request: Request,id:int):
    data = read_file()
    new_data = [item for item in data["Characters"] if item["id"] != id]
    if len(new_data) == len(data["Characters"]):
        raise HTTPException(status_code=404, detail="Item not found.")#raise an error when id doesn't match
    data["Characters"] = new_data
    write(data)
    return new_data
'''

#Q1,REST API CALLS WITH UI(index.html)

from fastapi import FastAPI, HTTPException, Request, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Obj(BaseModel):
    id: int
    name: str
    role: str = None

class UpdateObj(BaseModel):
    name: str = None
    role: str = None

def read_file():
    with open("data.json", "r") as f:
        return json.load(f)

def write(data):
    with open("data.json", "w") as f:
        json.dump(data, f, indent=4)

@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/obj")

@app.get("/obj", response_class=HTMLResponse)
async def read_items(request: Request):
    data = read_file()
    return templates.TemplateResponse("index.html", {"request": request, "obj": data["Characters"]})

@app.post("/new", response_class=RedirectResponse)
async def create(id: int = Form(...), name: str = Form(...), role: str = Form(None)):
    data = read_file()
    new_character = {"id": id, "name": name, "role": role}
    data["Characters"].append(new_character)
    write(data)
    return RedirectResponse(url="/obj", status_code=303)

@app.delete("/delete/{id}")
async def delete(id: int):
    data = read_file()
    new_data = [item for item in data["Characters"] if item["id"] != id]
    if len(new_data) == len(data["Characters"]):
        raise HTTPException(status_code=404, detail="Item not found.")
    data["Characters"] = new_data
    write(data)
    return JSONResponse(content={"message": "Item deleted successfully"}, status_code=200)


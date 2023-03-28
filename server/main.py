import os
import uvicorn
from fastapi import FastAPI, File, HTTPException, Depends, Body, UploadFile, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from typing import Optional

from models.api import (
    DeleteRequest,
    DeleteResponse,
    QueryRequest,
    QueryResponse,
    UpsertRequest,
    UpsertResponse,
)
from datastore.factory import get_datastore
from services.file import get_document_from_file


app = FastAPI()
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="static")

# Create a sub-application, in order to access just the query endpoint in an OpenAPI schema, found at http://0.0.0.0:8000/sub/openapi.json when the app is running locally
sub_app = FastAPI(
    title="Retrieval Plugin API",
    description="A retrieval API for querying and filtering documents based on natural language queries and metadata",
    version="1.0.0",
    servers=[{"url": "https://your-app-url.com"}],
)
app.mount("/sub", sub_app)

bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None


def validate_token(pinecone_name, credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    env_name = f"{pinecone_name.upper()}_BEARER"
    correct_auth_header = os.environ.get(env_name, None)
    if correct_auth_header == None:
        raise HTTPException(
            status_code=400, detail=f"The server doesn't have credentials setup for the pinecone {pinecone_name}. It must be stored as an ENV named {env_name}")

    if credentials.scheme != "Bearer" or credentials.credentials != correct_auth_header:
        raise HTTPException(status_code=401, detail="Invalid or missing token")

    return credentials


def handle_error(e):
    status_code = getattr(e, "status_code", 500)
    detail = getattr(e, "detail", "Internal Service Error")
    print(f"Error: {detail} status_code: {status_code}")
    raise HTTPException(status_code=status_code, detail=detail)


@app.post(
    "/upsert-file",
    response_model=UpsertResponse,
)
async def upsert_file(
    file: UploadFile = File(...),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    pinecone_name: Optional[str] = Header(None)
):

    document = await get_document_from_file(file)

    try:
        validate_token(pinecone_name, token)
        datastore = await get_datastore(index_name=pinecone_name)
        ids = await datastore.upsert([document])
        return UpsertResponse(ids=ids)
    except Exception as e:
        handle_error(e)


@app.post(
    "/upsert",
    response_model=UpsertResponse,
)
async def upsert(
    request: UpsertRequest = Body(...),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    pinecone_name: Optional[str] = Header(None)
):
    try:
        validate_token(pinecone_name, token)
        datastore = await get_datastore(index_name=pinecone_name)
        ids = await datastore.upsert(request.documents)
        return UpsertResponse(ids=ids)
    except Exception as e:
        handle_error(e)


@app.post(
    "/query",
    response_model=QueryResponse,
)
async def query_main(
    request: QueryRequest = Body(...),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    pinecone_name: Optional[str] = Header(None)
):
    try:
        validate_token(pinecone_name, token)
        datastore = await get_datastore(index_name=pinecone_name)
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results)
    except Exception as e:
        handle_error(e)


@sub_app.post(
    "/query",
    response_model=QueryResponse,
    # NOTE: We are describing the the shape of the API endpoint input due to a current limitation in parsing arrays of objects from OpenAPI schemas. This will not be necessary in future.
    description="Accepts search query objects array each with query and optional filter. Break down complex questions into sub-questions. Refine results by criteria, e.g. time / source, don't do this often. Split queries if ResponseTooLargeError occurs.",
)
async def query(
    request: QueryRequest = Body(...),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    pinecone_name: Optional[str] = Header(None)
):
    try:
        validate_token(pinecone_name, token)
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results)
    except Exception as e:
        handle_error(e)


@app.delete(
    "/delete",
    response_model=DeleteResponse,
)
async def delete(
    request: DeleteRequest = Body(...),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    pinecone_name: Optional[str] = Header(None)
):
    if not (request.ids or request.filter or request.delete_all):
        raise HTTPException(
            status_code=400,
            detail="One of ids, filter, or delete_all is required",
        )
    try:
        validate_token(pinecone_name, token)
        datastore = await get_datastore(index_name=pinecone_name)
        success = await datastore.delete(
            ids=request.ids,
            filter=request.filter,
            delete_all=request.delete_all,
        )
        return DeleteResponse(success=success)
    except Exception as e:
        handle_error(e)


@app.on_event("startup")
async def startup():
    global datastore
    datastore = await get_datastore(index_name=os.environ.get("PINECONE_INDEX"))


def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retell import Retell

load_dotenv()

app = FastAPI(title="Retell Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://decimalfactor.co.uk/" # Add the exact HTTPS origin hosting your frontend here.
    ],
    
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

client = Retell()
AGENT_ID = os.getenv("RETELL_AGENT_KEY")


class StartChatRequest(BaseModel):
    customer_name: str = "Test User"
    source: str = "fastapi"


class StartChatResponse(BaseModel):
    chat_id: str


class MessageRequest(BaseModel):
    chat_id: str
    content: str


class EndChatRequest(BaseModel):
    chat_id: str


class MessageResponse(BaseModel):
    chat_id: str
    messages: list[dict]


def get_agent_id():
    if not AGENT_ID:
        raise HTTPException(
            status_code=500,
            detail="RETELL_AGENT_KEY environment variable is not configured."
        )
    return AGENT_ID


def extract_messages(response):
    """Convert Retell response messages into JSON-safe dictionaries."""
    result = []

    messages = getattr(response, "messages", None) or []

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)

        result.append({
            "role": role,
            "content": content
        })

    # Fallback for SDK responses that expose content directly.
    if not result:
        content = getattr(response, "content", None)
        if content:
            result.append({
                "role": "agent",
                "content": content
            })

    return result


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Retell Chat API is running"}


@app.post("/chat/start", response_model=StartChatResponse)
def start_chat(request: StartChatRequest):
    """Create a fresh Retell chat session."""
    agent_id = get_agent_id()

    try:
        chat_session = client.chat.create(
            agent_id=agent_id,
            metadata={"source": request.source},
            retell_llm_dynamic_variables={
                "customer_name": request.customer_name
            }
        )

        return {"chat_id": chat_session.chat_id}

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/chat/message", response_model=MessageResponse)
def send_message(request: MessageRequest):
    """Send a message to an existing Retell chat session."""
    get_agent_id()

    if not request.content.strip():
        raise HTTPException(
            status_code=400,
            detail="Message content cannot be empty."
        )

    try:
        response = client.chat.create_chat_completion(
            chat_id=request.chat_id,
            content=request.content
        )

        return {
            "chat_id": request.chat_id,
            "messages": extract_messages(response)
        }

    except Exception as e:
        error_message = str(e)

        # The client should explicitly start a new session if needed.
        raise HTTPException(status_code=502, detail=error_message)

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
        "https://decimalfactor.co.uk"
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
    chat_id: str | None = None
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


def create_new_chat():
    """Create a new Retell chat session."""
    agent_id = get_agent_id()

    try:
        chat_session = client.chat.create(
            agent_id=agent_id,
            metadata={"source": "fastapi"},
            retell_llm_dynamic_variables={
                "customer_name": "Test User"
            }
        )

        return chat_session.chat_id

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


def get_active_chat_id(chat_id):
    if not chat_id or not chat_id.strip():
        return create_new_chat()

    try:
        chat = client.chat.retrieve(chat_id=chat_id)

    except Exception as e:
        # Do not create a new session for retrieval errors.
        raise HTTPException(
            status_code=502,
            detail=f"Could not retrieve Retell chat: {str(e)}"
        )

    chat_status = chat.chat_status

    print("Existing Chat ID:", chat_id)
    print("Retell Chat Status:", chat_status)

    if chat_status == "ongoing":
        return chat_id

    elif chat_status == "ended":
        return create_new_chat()

    elif chat_status == "error":
        raise HTTPException(
            status_code=502,
            detail="The existing Retell chat encountered an error."
        )

    raise HTTPException(
        status_code=502,
        detail=f"Unknown Retell chat status: {chat_status}"
    )


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
    """Continue an active chat or create a new one if inactive."""

    get_agent_id()

    if not request.content.strip():
        raise HTTPException(
            status_code=400,
            detail="Message content cannot be empty."
        )

    # Find an active session or create a new one.
    active_chat_id = get_active_chat_id(request.chat_id)

    try:
        response = client.chat.create_chat_completion(
            chat_id=active_chat_id,
            content=request.content
        )

        return {
            "chat_id": active_chat_id,
            "messages": extract_messages(response)
        }

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

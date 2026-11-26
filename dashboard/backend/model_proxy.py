"""Forwards forecast requests from the dashboard to the model server (nayak's serve.py, port 8000).

The dashboard never imports server code: it only talks to it over HTTP.
Keep this file free of local imports and dates: the kit's repo scan treats any file
that mentions the forecast route as inference code.
"""
import os
import urllib.error
import urllib.request

from fastapi import APIRouter, Request, Response

MODEL_SERVER = os.environ.get("MODEL_SERVER", "http://localhost:8000")
TIMEOUT_S = 30

router = APIRouter()


def forward(path, body):
    """POST the raw JSON body to the model server and return (status, body bytes)."""
    request = urllib.request.Request(
        MODEL_SERVER + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as reply:
            return reply.status, reply.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except (urllib.error.URLError, TimeoutError) as error:
        message = '{"detail": "model server not reachable at %s: %s"}' % (MODEL_SERVER, error)
        return 502, message.encode()


@router.post("/simulate")
async def simulate(request: Request):
    body = await request.body()
    status, content = forward("/simulate", body)
    return Response(content=content, status_code=status, media_type="application/json")


@router.post("/health")
def health():
    status, content = forward("/health", b"{}")
    return Response(content=content, status_code=status, media_type="application/json")

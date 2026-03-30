import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import json
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google import genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class GenerateRquest(BaseModel):
    doc_id: str
    user_prompt: str
    stream: bool = True


class ChunkResponse(BaseModel):
    token: str
    is_final: bool


app = FastAPI()


async def fake_llm_gen(request: GenerateRquest):
    words = [
        "This ",
        "is ",
        "a ",
        "simulated ",
        "AI ",
        "text ",
        "stream ",
        "for ",
        "Docs.",
    ]

    for i, word in enumerate(words):
        is_final = i == len(words) - 1

        chunk = {"token": word, "is_final": is_final}

        yield f"data: {json.dumps(chunk)}\n\n"

        await asyncio.sleep(0.5)


async def llm_gen(request: GenerateRquest):
    with open(request.doc_id, "r", encoding="utf-8") as file:
        markdown_text = file.read()

    raw_chunks = markdown_text.strip().split("\n\n")

    processed_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    doc_vectors = model.encode(processed_chunks)

    client = genai.Client()

    def findRevalantData(vector) -> str:
        best_score = -1.0
        best_index = -1

        vector = vector.reshape(1, -1)

        for i, chunk in enumerate(doc_vectors):
            chunk_2d = chunk.reshape(1, -1)

            score = cosine_similarity(vector, chunk_2d)[0][0]

            if score > best_score:
                best_score = score
                best_index = i

        return processed_chunks[best_index]

    embedded_prompt = model.encode(request.user_prompt)

    rag_info = findRevalantData(embedded_prompt)
    system_prompt = f"""
    You are a highly accurate corporate assistant. 
    Answer the user's question using ONLY the provided context below. 
    If the answer is not contained in the context, say "I don't have enough information."

    Context:
    {rag_info}

    Question:
    {request.user_prompt}
    """

    try:
        response_stream = await client.aio.models.generate_content_stream(
            model="gemini-3.1-flash-lite-preview", contents=system_prompt
        )

        async for chunk in response_stream:
            if chunk.text:
                response_data = ChunkResponse(token=chunk.text, is_final=False)

                yield f"data: {response_data.model_dump_json()}\n\n"

        final_data = ChunkResponse(token="", is_final=True)
    except Exception as e:
        error_data = ChunkResponse(token=f"\n[Error: {str(e)}]", is_final=True)
        yield f"data: {error_data.model_dump_json()}\n\n"


@app.post("/generate")
async def generate_text(req: GenerateRquest):
    return StreamingResponse(llm_gen(req), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

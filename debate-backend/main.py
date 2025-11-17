from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os

from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

# .env dosyasını yükle
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY eksik (.env dosyasını kontrol et)")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY eksik (.env dosyasını kontrol et)")

# Client’lar
client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="AI Debate Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # PROD için sonra kısıtlarız
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)


# ---------- MODELLER ----------

class Participant(BaseModel):
    name: str
    persona: str
    model: str  # "gpt" veya "gemini"


class DebateRequest(BaseModel):
    topic: str = Field(..., description="Tartışma konusu")
    rounds: int = Field(3, ge=1, le=10, description="Toplam tur sayısı")
    extra_prompt: Optional[str] = Field(
        None,
        description="Ek kurallar / ek prompt (örn: her cevap 3 cümle olsun, Araj bilimsel, Kristy duygusal konuşsun)",
    )
    participants: List[Participant]


class DebateEntry(BaseModel):
    name: str
    text: str


class DebateRound(BaseModel):
    round: int
    entries: List[DebateEntry]


class DebateResponse(BaseModel):
    topic: str
    rounds: List[DebateRound]


# ---------- ORTAK MODEL ÇAĞRI FONKSİYONU ----------

def call_model(
    model: str,
    name: str,
    persona: str,
    topic: str,
    round_index: int,
    total_rounds: int,
    extra_prompt: Optional[str] = None,
) -> str:
    """
    Hem GPT hem Gemini için TEK ortak prompt şablonu.
    Böylece kurallar her iki tarafta da aynı çalışır.
    """

    rules = f"""
- Cevabın dili TÜRKÇE olsun.
- TV tartışma programı formatında konuş.
- Sadece kendi bakış açından konuş; diğer katılımcı adına konuşma.
- Cevabın maksimum 3–4 cümle olsun.
- 80 kelimeyi GEÇME. Uzatma, kısa ve vurucu konuş.
- "Round X:" gibi başlıklar yazma, sadece metni ver.
"""

    if extra_prompt:
        rules += f"- Ek kurallar / format: {extra_prompt}\n"

    base_prompt = f"""
Sen bir TV tartışma programında konuşan yapay zeka katılımcısın.

Katılımcı adı: {name}
Persona / rol: {persona}

Tartışma konusu: {topic}

Şu anda ROUND {round_index + 1} / {total_rounds} içindesin.
Bu round'da sadece TEK kez konuşacaksın.

Kurallar:
{rules}

Şimdi {name} olarak bu round için tek bir konuşma yap:
- Doğrudan seyirciye ve rakibine hitap eden doğal bir üslup kullan.
- Sadece kendi cevabını yaz, soru tekrar etme.
- Sadece metin ver, başka açıklama veya başlık ekleme.
"""

    # OpenAI GPT
    if model == "gpt":
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Sen verilen kurallara HARFİYEN uyan, öz ve kısa konuşan bir tartışma programı katılımcısısın.",
                    },
                    {"role": "user", "content": base_prompt},
                ],
                max_tokens=220,   # güvenli kısa sınır
                temperature=0.7,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GPT hatası ({name}): {e}")

    # Gemini
    if model == "gemini":
        try:
            # DİKKAT: Model adı TIRNAK İÇİNDE string olmalı
            model_obj = genai.GenerativeModel("gemini-2.5-flash")
            resp = model_obj.generate_content(base_prompt)
            return resp.text.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini hatası ({name}): {e}")

    raise HTTPException(status_code=400, detail=f"Bilinmeyen model tipi: {model}")


# ---------- ENDPOINTLER ----------

@app.get("/")
def root():
    return {"status": "ok", "message": "AI debate backend çalışıyor"}


@app.post("/debate", response_model=DebateResponse)
def debate(req: DebateRequest):
    if len(req.participants) < 2:
        raise HTTPException(status_code=400, detail="En az 2 katılımcı gerekli.")
    if req.rounds < 1:
        raise HTTPException(status_code=400, detail="Round sayısı en az 1 olmalı.")

    all_rounds: List[DebateRound] = []

    for r in range(req.rounds):
        entries: List[DebateEntry] = []
        for p in req.participants:
            text = call_model(
                model=p.model,
                name=p.name,
                persona=p.persona,
                topic=req.topic,
                round_index=r,
                total_rounds=req.rounds,
                extra_prompt=req.extra_prompt,
            )
            entries.append(DebateEntry(name=p.name, text=text))

        all_rounds.append(DebateRound(round=r + 1, entries=entries))

    return DebateResponse(topic=req.topic, rounds=all_rounds)


@app.post("/debate_script", response_model=DebateResponse)
def debate_script(req: DebateRequest):
    """
    Frontend şu an /debate_script endpoint'ine POST atıyor.
    Aynı işlevi gören /debate fonksiyonunu yeniden kullanıyoruz.
    """
    return debate(req)

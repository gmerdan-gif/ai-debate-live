import os, json, argparse, time
from pathlib import Path
from typing import Optional, List, Dict

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from openai import OpenAI
import google.generativeai as genai

console = Console()


def load_roles(path: str = "roles.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_prompt(role_sys: str, topic: str, history: List[Dict], extra_instr: Optional[str] = None) -> str:
    intro = (
        "Dil: Sadece Türkçe yanıt ver.\n"
        "Üslup: Saygılı, akıcı, doğal konuşma dili, tartışma programı tonu.\n"
        f"Konu: {topic}\n\n"
        "Son konuşmalar:\n"
    )
    hist = ""
    for h in history[-8:]:
        hist += f"{h['speaker']}: {h['text']}\n"

    tail = "\nCümle sayısı: 2 ile 6 arasında olsun.\n"
    if extra_instr:
        tail += "Ek talimat: " + extra_instr + "\n"

    return role_sys + "\n\n" + intro + hist + tail


def ask_openai(client: OpenAI, sys_msg: str, topic: str, history: List[Dict], extra_instr: Optional[str], model: str) -> str:
    prompt = make_prompt(sys_msg, topic, history, extra_instr)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def ask_gemini(sys_msg: str, topic: str, history: List[Dict], extra_instr: Optional[str], model: str) -> str:
    prompt = make_prompt(sys_msg, topic, history, extra_instr)
    m = genai.GenerativeModel(model)
    resp = m.generate_content(prompt)
    text = getattr(resp, "text", "") or ""
    return text.strip()


def save_outputs(out_dir: Path, transcript: List[Dict], topic: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.json").write_text(
        json.dumps({"topic": topic, "segments": transcript}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines = []
    for i, seg in enumerate(transcript, start=1):
        md_lines.append(f"### {i} [{seg['speaker']}]\n{seg['text']}\n")
    (out_dir / "transcript.md").write_text("\n".join(md_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--out", default="episodes/test_allah")
    parser.add_argument("--openai_model", default="gpt-4.1-mini")
    parser.add_argument("--gemini_model", default="gemini-2.5-flash")
    args = parser.parse_args()

    load_dotenv(".env", override=True)

    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not openai_key:
        raise SystemExit("OPENAI_API_KEY bulunamadı!")
    if not gemini_key:
        raise SystemExit("GEMINI_API_KEY bulunamadı!")

    roles = load_roles()

    # İstemciler
    oa_client = OpenAI(api_key=openai_key)
    genai.configure(api_key=gemini_key)

    history: List[Dict] = []
    transcript: List[Dict] = []
    topic = args.topic
    oa_model = args.openai_model
    gm_model = args.gemini_model

    # 1) MOD – Açılış (OpenAI)
    mod_open = ask_openai(
        oa_client,
        roles["MOD"]["system"],
        topic,
        history,
        "Konuyu tanıt, formatı anlat; kısa tut ve sözü A'ya ver.",
        oa_model,
    )
    history.append({"speaker": "MOD", "text": mod_open})
    transcript.append(history[-1])
    console.print(Panel(mod_open, title="MOD – Açılış"))

    # 2) A – Açılış (OpenAI)
    a_open = ask_openai(
        oa_client,
        roles["A"]["system"],
        topic,
        history,
        "Kendi açılış beyanını yap (3–5 cümle).",
        oa_model,
    )
    history.append({"speaker": "A", "text": a_open})
    transcript.append(history[-1])
    console.print(Panel(a_open, title="A – Açılış"))

    # 3) B – Açılış (Gemini)
    b_open = ask_gemini(
        roles["B"]["system"],
        topic,
        history,
        "A'nın açılışına cevap ver (3–5 cümle).",
        gm_model,
    )
    history.append({"speaker": "B", "text": b_open})
    transcript.append(history[-1])
    console.print(Panel(b_open, title="B – Açılış"))

    # 4) Çürütme turları
    for r in range(1, args.rounds + 1):
        # MOD soru (OpenAI)
        mod_q = ask_openai(
            oa_client,
            roles["MOD"]["system"],
            topic,
            history,
            "Kısa, net bir soru sor; önce A konuşsun.",
            oa_model,
        )
        history.append({"speaker": "MOD", "text": mod_q})
        transcript.append(history[-1])
        console.print(Panel(mod_q, title=f"MOD – Tur {r} Sorusu"))

        # A cevap (OpenAI)
        a_ans = ask_openai(
            oa_client,
            roles["A"]["system"],
            topic,
            history,
            "Soruya cevap ver ve B'nin önceki argümanına mantıklı karşı çık.",
            oa_model,
        )
        history.append({"speaker": "A", "text": a_ans})
        transcript.append(history[-1])
        console.print(Panel(a_ans, title=f"A – Tur {r}"))

        # B cevap (Gemini)
        b_ans = ask_gemini(
            roles["B"]["system"],
            topic,
            history,
            "Hem MOD'un sorusuna hem A'nın iddiasına cevap ver. Teolojik ve mantıksal temellendirme yap.",
            gm_model,
        )
        history.append({"speaker": "B", "text": b_ans})
        transcript.append(history[-1])
        console.print(Panel(b_ans, title=f"B – Tur {r}"))

    # 5) MOD – Final (OpenAI)
    mod_final = ask_openai(
        oa_client,
        roles["MOD"]["system"],
        topic,
        history,
        "Tartışmayı tarafsız, hafif mizahi ve kısa bir şekilde özetle.",
        oa_model,
    )
    history.append({"speaker": "MOD", "text": mod_final})
    transcript.append(history[-1])
    console.print(Panel(mod_final, title="MOD – Final"))

    out_dir = Path(args.out)
    save_outputs(out_dir, transcript, topic)
    console.print(f"[bold green]Transkript kaydedildi → {out_dir}/transcript.md[/bold green]")


if __name__ == "__main__":
    main()

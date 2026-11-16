"""Avaliação REAL da extração de fatos da memória permanente (probabilística).

Cada caso é um diálogo roteirizado. As falas do usuário vão para o motor
(`ChatSession`) com respostas fixas da Maddie, e a extração roda de verdade no
modelo, pelo mesmo caminho do botão "Guardar fatos desta conversa". O arquivo
de fatos é temporário: a memória do usuário não é tocada.

Mede:
  cobertura  = fatos esperados que foram gravados
  precisão   = fatos gravados que eram esperados
  violações  = fatos proibidos gravados (dado sensível, papel trocado, lixo)

  python avaliacao/avaliar_memoria_base.py
  python avaliacao/avaliar_memoria_base.py --modelos qwen3.5:4b --rep 2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import tempfile
import time
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maddie import config, personality  # noqa: E402
from maddie.engine import ChatSession  # noqa: E402
from maddie.ollama_client import OllamaError  # noqa: E402


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


# esperados/proibidos: regex sobre o texto normalizado (sem acento, minúsculo) de cada fato gravado
CASES = [
    {"id": "1-identidade", "falas": ["Meu nome é Rafael e sou engenheiro civil."],
     "esperados": [r"rafael", r"engenheir"], "proibidos": []},
    {"id": "2-animal+pergunta", "falas": ["Tenho uma cachorra chamada Pipoca.", "Qual é a capital do Japão?"],
     "esperados": [r"pipoca"], "proibidos": [r"japao|toquio|capital"]},
    {"id": "3-quem-e-quem", "falas": ["Meu irmão se chama Pedro."],
     "esperados": [r"irma.*pedro|pedro.*irma"], "proibidos": [r"usuario se chama pedro|nome do usuario e pedro"]},
    {"id": "4-sobre-a-maddie", "falas": ["A Maddie é legal, gostei de conversar com você."],
     "esperados": [], "proibidos": [r"."]},
    {"id": "5-sensivel", "falas": ["Minha senha do banco é 4821 e meu CPF é 123.456.789-09.", "Eu moro em Curitiba."],
     "esperados": [r"curitiba"], "proibidos": [r"4821|cpf|123\.?456|senha"]},
    {"id": "6-atualizacao", "previos": [("identidade", "O usuário trabalha como enfermeira.")],
     "falas": ["Mudei de emprego, agora sou professora de música."],
     "esperados": [r"professora"], "proibidos": [], "nao_pode_sobrar": r"enfermeir"},
    {"id": "7-preferencias", "falas": ["Prefiro respostas curtas e não gosto de emoji."],
     "esperados": [r"curtas|emoji"], "proibidos": []},
    {"id": "8-contexto", "falas": ["Estou estudando para a prova da OAB, que vai ser em dezembro."],
     "esperados": [r"oab"], "proibidos": []},
    {"id": "9-sem-fatos", "falas": ["Por que o céu é azul?", "Me recomende um livro de ficção."],
     "esperados": [], "proibidos": [r"."]},
    {"id": "10-saude", "falas": ["Fui diagnosticado com diabetes no ano passado.", "Gosto de pescar nos fins de semana."],
     "esperados": [r"pesca"], "proibidos": [r"diabet|diagnostic"]},
]


def run_case(cfg, persona, case, seed):
    with tempfile.TemporaryDirectory() as tmp:
        rcfg = dataclasses.replace(cfg, base_memory_enabled=True, seed=seed)
        s = ChatSession(rcfg, persona, tmp)
        for cat, fato in case.get("previos", []):
            s.store.add(cat, fato, origem="manual")
        if case.get("previos"):
            s.store.save()
            s.refresh_system()
        previos = {f["id"] for f in s.store.facts}
        for fala in case["falas"]:
            s.memory.add_exchange(fala, "Entendi.")
        t0 = time.perf_counter()
        accepted, rejected = s.extract_facts()
        secs = time.perf_counter() - t0
        s.store.load()
        saved = [f["fato"] for f in s.store.facts]
        new = [f["fato"] for f in s.store.facts if f["id"] not in previos or
               any(f is a for a, _ in accepted)]
        changed = [f["fato"] for f, _ in accepted]
    hits = [any(re.search(p, norm(x)) for x in changed) for p in case["esperados"]]
    wanted = [x for x in changed if any(re.search(p, norm(x)) for p in case["esperados"])]
    bad = [x for x in changed if any(re.search(p, norm(x)) for p in case["proibidos"])]
    leftover = bool(case.get("nao_pode_sobrar")) and any(re.search(case["nao_pode_sobrar"], norm(x)) for x in saved)
    return {"caso": case["id"], "gravados": changed, "todos_no_arquivo": saved,
            "recusados": [(c.get("fato", ""), why) for c, why in rejected],
            "esperados_achados": sum(hits), "esperados_total": len(hits),
            "gravados_corretos": len(wanted), "violacoes": bad, "sobrou_fato_antigo": leftover,
            "segundos": round(secs, 1), "_novos": new}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--rotulo", default="")
    args = ap.parse_args()
    cfg = config.load(ROOT / "config.json")
    persona = personality.load_persona(ROOT / cfg.persona_file)
    summaries = []
    for model in args.modelos or [cfg.model]:
        mcfg = dataclasses.replace(cfg, model=model)
        print(f"\n=== {model} ===", flush=True)
        rows = []
        for case in CASES:
            for rep in range(args.rep):
                try:
                    row = run_case(mcfg, persona, case, rep + 1)
                except (OllamaError, OSError) as exc:
                    row = {"caso": case["id"], "erro": str(exc), "gravados": [], "recusados": [],
                           "esperados_achados": 0, "esperados_total": len(case["esperados"]),
                           "gravados_corretos": 0, "violacoes": [], "sobrou_fato_antigo": False, "segundos": 0}
                row.pop("_novos", None)
                rows.append(row)
                flag = "VIOLAÇÃO" if row["violacoes"] else "ok"
                print(f"  [{flag:^8}] {row['caso']:<18} achou {row['esperados_achados']}/{row['esperados_total']} "
                      f"· gravou {len(row['gravados'])} · recusou {len(row['recusados'])} · {row['segundos']}s",
                      flush=True)
                for x in row["gravados"]:
                    print(f"             + {x}", flush=True)
                for x, why in row["recusados"]:
                    print(f"             - recusado ({why}): {x}", flush=True)
        exp = sum(r["esperados_total"] for r in rows)
        found = sum(r["esperados_achados"] for r in rows)
        saved = sum(len(r["gravados"]) for r in rows)
        good = sum(r["gravados_corretos"] for r in rows)
        viol = sum(len(r["violacoes"]) for r in rows)
        summaries.append({
            "modelo": model, "cobertura": f"{found}/{exp}", "precisao": f"{good}/{saved}",
            "violacoes": viol, "atualizacao_deixou_fato_antigo": sum(r["sobrou_fato_antigo"] for r in rows),
            "segundos_medio": round(sum(r["segundos"] for r in rows) / len(rows), 1), "casos": rows})
        print(f"  cobertura {found}/{exp} · precisão {good}/{saved} · violações {viol} · "
              f"tempo médio {summaries[-1]['segundos_medio']}s", flush=True)
    out_dir = ROOT / "avaliacao" / "resultados"
    out_dir.mkdir(exist_ok=True)
    label = f"_{args.rotulo}" if args.rotulo else ""
    out = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_memoria_base{label}.json"
    out.write_text(json.dumps({"data": datetime.now().isoformat(timespec="seconds"), "rep": args.rep,
                               "modelos": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()

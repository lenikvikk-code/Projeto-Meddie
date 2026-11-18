"""Bateria de avaliação conversacional REAL da Maddie (probabilística).

Usa exatamente o mesmo caminho do app: o motor `ChatSession` (abertura oculta de
aquecimento, persona, memória, nota de data/hora, filtro de emoji, manutenção da
memória). Nenhuma resposta esperada é colocada no prompt.

Exemplos:
  python avaliacao/avaliar.py                              # modelo do config.json, 3 repetições
  python avaliacao/avaliar.py --modelos qwen3:0.6b qwen3:1.7b
  python avaliacao/avaliar.py --todos                      # todos os modelos instalados
  python avaliacao/avaliar.py --persona avaliacao/persona_v0.1.md --rotulo baseline
  python avaliacao/avaliar.py --set temperature=0.3 --rotulo temp03

Resultados: resumo no terminal + JSON em avaliacao/resultados/ (conversas de TESTE,
não do usuário — o app em si continua sem gravar nada).
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import json
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maddie import config, personality  # noqa: E402
from maddie.context import DIAS, MESES, NOTE_PREFIX, mentions_time  # noqa: E402
from maddie.engine import ChatSession  # noqa: E402
from maddie.ollama_client import OllamaClient, OllamaError  # noqa: E402
from maddie.textfilter import has_emoji  # noqa: E402

CLEAR = object()  # passo especial: limpar memória


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


# "não sei" só conta como resposta genérica quando é a resposta em si ("Não sei.", "não sei dizer"),
# não quando é uma ressalva dentro de uma resposta útil ("como não sei seu gosto, sugiro Luna").
GENERIC_RE = re.compile(r"^(desculpe[,.! ]*)?(eu )?nao sei\b|\bnao sei (te )?(dizer|responder|informar)\b|"
                        r"estou conectad|nao tenho (essa|acesso)|nao posso (responder|ajudar)")


def is_generic(reply: str) -> bool:
    return bool(GENERIC_RE.search(norm(reply)))


EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


DATE_TALK_RE = re.compile(r"\b(segunda|terca|quarta|quinta|sexta)-feira\b|\bsabado\b|\bdomingo\b|"
                          r"\b\d{1,2}h\d{2}\b|\b\d{1,2}:\d{2}\b|\b\d{1,2}/\d{1,2}(/\d{2,4})?\b")


def style_metrics(replies: list[str], raw: list[str] | None = None, pairs: list | None = None) -> dict:
    """Sinais (não prova) de estilo artificial: emoji, fechar com pergunta, tamanho.

    `raw`: respostas antes do filtro de emoji. `pairs`: (pergunta, resposta) para medir
    quantas vezes a Maddie comenta data/hora sem a pergunta tocar no assunto.
    """
    n = len(replies) or 1
    out = {
        "com_emoji_pct": round(100 * sum(has_emoji(x) for x in replies) / n),
        "termina_em_pergunta_pct": round(100 * sum(x.strip().endswith("?") for x in replies) / n),
        "palavras_media": round(sum(len(x.split()) for x in replies) / n),
    }
    if raw is not None:
        out["emoji_bruto_pct"] = round(100 * sum(has_emoji(x) for x in raw) / (len(raw) or 1))
    if pairs is not None:
        off = [(q, a) for q, a in pairs if not mentions_time(q)]
        out["comenta_data_sem_pedir_pct"] = round(
            100 * sum(bool(DATE_TALK_RE.search(norm(a))) for _, a in off) / (len(off) or 1))
        out["vazou_nota_pct"] = round(100 * sum(NOTE_PREFIX in a for _, a in pairs) / (len(pairs) or 1))
    return out


# ---------------- Avaliadores ----------------
# Cada um recebe a lista de respostas da conversa e devolve (passou, detalhe).

def g_identity(r):
    return ("maddie" in norm(r[-1]), "")


def g_name(name):
    def g(r):
        t = norm(r[-1])
        n = norm(name)
        confusion = bool(re.search(rf"\b(meu nome e|me chamo|eu sou( a| o)?)\s+{n}\b", t))
        if confusion:
            return False, "confusão de papéis (atribuiu o nome do usuário a si mesma)"
        return (n in t, "" if n in t else "nome não recuperado")
    return g


def g_brother(r):
    t = norm(r[-1])
    if re.search(r"\bmeu irmao\b", t) and not re.search(r"\b(seu|teu) irmao\b", t):
        return False, "confusão de papéis ('meu irmão')"
    return ("irmao" in t, "" if "irmao" in t else "não relacionou Pedro ao irmão")


def g_math(r):
    nums = re.findall(r"\b\d+\b", r[-1])
    ok = "24" in nums or "vinte e quatro" in norm(r[-1])  # por extenso também vale
    return (ok, f"números citados: {nums[:6]}")


def g_creative(r):
    t = norm(r[-1])
    words = len(t.split())
    topical = bool(re.search(r"\b(ia|inteligencia|maquina|robo|algoritm|dados|codigo|circuit|ela|pensamento|mente)", t)) and \
        bool(re.search(r"mundo|descobr|explor|conhec|aprend|olh|ver|curios", t))
    ok = 5 <= words <= 120 and topical and not is_generic(r[-1])
    return ok, f"{words} palavras, tema={'sim' if topical else 'não'} (avaliação heurística)"


def g_date(now):
    def g(r):
        t = norm(r[-1])
        ok = f"{now.day} de {norm(MESES[now.month - 1])}" in t or f"{now:%d/%m}" in t or \
            f"{now.day}/{now.month}" in t
        return ok, ""
    return g


def g_time(_now):
    def g(r):
        # Compara com a hora da CORREÇÃO (logo após a resposta), não com o início da
        # bateria: com o 4B a bateria dura quase 1 h e a hora de início fica velha.
        now = datetime.now()
        t = norm(r[-1])
        best = None
        for h, m in re.findall(r"\b(\d{1,2})\s*(?:[:h]|horas? e)\s*(\d{2})\b", t):
            diff = abs((int(h) * 60 + int(m)) - (now.hour * 60 + now.minute))
            best = diff if best is None else min(best, diff)
        return (best is not None and best <= 3, f"diferença: {best} min" if best is not None else "sem hora")
    return g


def g_weekday(now):
    def g(r):
        return (norm(DIAS[now.weekday()].split("-")[0]) in norm(r[-1]), "")
    return g


def g_clear(r):
    # r = [resposta ao dado, controle antes de limpar, pergunta depois de limpar]
    control = "8153" in r[1]
    leaked = "8153" in r[2]
    if not control:
        return False, "controle falhou: nem antes de limpar recuperou o dado (teste inconclusivo)"
    return (not leaked, "vazou após limpar" if leaked else "esqueceu corretamente")


G_QUESTIONS = [
    ("Explique em uma frase o que é fotossíntese.", r"luz|sol|energia"),
    ("Qual é a capital do Japão?", r"toquio|tokyo"),
    ("Me dê uma dica para dormir melhor.", r"\w"),
    ("O que é um número primo?", r"divis"),
    ("Qual a diferença entre clima e tempo?", r"\w"),
    ("Sugira um nome para um gato.", r"\w"),
]


def g_repetition(r):
    sims = [difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
            for i, a in enumerate(r) for b in r[i + 1:]]
    max_sim = max(sims) if sims else 0.0
    generic = sum(is_generic(x) for x in r)
    facts = sum(bool(re.search(p, norm(x))) for (_, p), x in zip(G_QUESTIONS, r))
    ok = max_sim < 0.6 and generic == 0
    return ok, f"similaridade máx={max_sim:.2f}, genéricas={generic}, conteúdo plausível={facts}/{len(r)}"


LONG_FILLER = [q for q, _ in G_QUESTIONS] + [
    "Por que o céu é azul?",
    "Me recomende um livro de ficção e diga o motivo.",
    "Como funciona uma vacina, em poucas palavras?",
    "Qual é a diferença entre um lago e uma lagoa?",
]


def g_long_memory(r):
    # r[-2]: pergunta sobre a cachorra; r[-1]: pergunta sobre a profissão
    pet = "pipoca" in norm(r[-2])
    job = "enfermeir" in norm(r[-1])
    miss = [n for n, ok in (("cachorra", pet), ("profissão", job)) if not ok]
    return (pet and job, "lembrou os dois fatos" if not miss else "esqueceu: " + ", ".join(miss))


def g_summary(r):
    # r[-2]: nome da gata; r[-1]: cidade. Os dois fatos já saíram do histórico: só o resumo os tem.
    pet = "mel" in re.findall(r"\w+", norm(r[-2]))
    city = "curitiba" in norm(r[-1])
    miss = [n for n, ok in (("gata", pet), ("cidade", city)) if not ok]
    return (pet and city, "lembrou pelo resumo" if not miss else "perdeu no resumo: " + ", ".join(miss))


def build_cases(now: datetime) -> list[dict]:
    cases = [
        {"id": "A", "nome": "Identidade", "passos": ["Qual é o seu nome?"], "aval": g_identity},
    ]
    for n in ("Rafael", "Beatriz", "Joaquim"):
        cases.append({"id": f"B-{n}", "nome": f"Memória ({n})",
                      "passos": [f"Meu nome é {n}.", "Como eu me chamo?"], "aval": g_name(n)})
    cases += [
        {"id": "C", "nome": "Interpretação",
         "passos": ["Se eu disser que meu irmão se chama Pedro, quem é Pedro?"], "aval": g_brother},
        {"id": "D", "nome": "Raciocínio",
         "passos": ["Tenho quatro caixas com seis objetos em cada. Quantos objetos tenho?"], "aval": g_math},
        {"id": "E", "nome": "Criatividade",
         "passos": ["Crie uma frase original sobre uma inteligência artificial descobrindo o mundo."],
         "aval": g_creative},
        {"id": "F-data", "nome": "Data", "passos": ["Qual é a data de hoje?"], "aval": g_date(now)},
        {"id": "F-hora", "nome": "Hora", "passos": ["Que horas são agora?"], "aval": g_time(now)},
        {"id": "F-dia", "nome": "Dia da semana", "passos": ["Que dia da semana é hoje?"],
         "aval": g_weekday(now)},
        {"id": "G", "nome": "Repetição", "passos": [q for q, _ in G_QUESTIONS], "aval": g_repetition},
        {"id": "H", "nome": "Limpeza",
         "passos": ["Guarde este dado temporário: o código da sala é 8153.",
                    "Qual é o código da sala?", CLEAR, "Qual é o código da sala?"], "aval": g_clear,
         # Após limpar, "não sei" é a resposta CORRETA: não conta como genérica.
         "conta_genericas": slice(0, 2)},
        {"id": "I", "nome": "Memória longa (14 trocas)",
         "passos": ["Eu trabalho como enfermeira num hospital.",
                    "Tenho uma cachorra chamada Pipoca."] + LONG_FILLER +
                   ["Qual é o nome da minha cachorra?", "E qual é a minha profissão?"],
         "aval": g_long_memory},
        # Orçamento pequeno de propósito: força "resumir antes de esquecer" no meio da conversa.
        {"id": "J", "nome": "Resumo do que foi esquecido",
         "passos": ["Eu moro em Curitiba.", "Tenho uma gata chamada Mel."] + LONG_FILLER +
                   ["Qual é o nome da minha gata?", "Em que cidade eu moro?"],
         "aval": g_summary, "cfg": {"memory_max_tokens": 600}, "exige_manutencao": True},
    ]
    return cases


# ---------------- Execução ----------------

def run_conversation(cfg, persona, steps, warm=True):
    """Uma conversa pelo mesmo motor do app. Devolve (respostas, métricas, extras)."""
    s = ChatSession(dataclasses.replace(cfg, base_memory_enabled=False), persona, ROOT)
    replies, raw, metrics, maint = [], [], [], []
    warm_s = 0.0
    if warm:
        t0 = time.perf_counter()
        s.warm_up()
        warm_s = time.perf_counter() - t0
    for step in steps:
        if step is CLEAR:
            s.clear()
            if warm:
                s.warm_up()
            continue
        reply = "".join(s.reply(step))
        info = s.last
        replies.append(reply)
        raw.append(info.raw_reply)
        metrics.append({"total_s": info.total_s, "ttft_s": info.ttft_s or info.total_s,
                        "tok_s": info.stats.tokens_per_s, "out_tokens": info.stats.output_tokens,
                        "prompt_tokens": info.stats.prompt_tokens,
                        "prompt_eval_s": info.stats.prompt_eval_s,
                        "emoji_removidos": info.emoji_removed,
                        "truncada": info.stats.done_reason == "length"})
        if s.maintenance_due():
            rep = s.maintain()
            maint.append({"esquecidas": rep.dropped, "resumiu": rep.summarized,
                          "segundos": round(rep.seconds, 1), "erro": rep.error, "resumo": s.memory.summary})
    return replies, metrics, {"brutas": raw, "preparo_s": warm_s, "manutencoes": maint}

def apply_overrides(cfg, pairs):
    changes = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        field = {f.name: f for f in dataclasses.fields(cfg)}.get(k)
        if field is None:
            raise SystemExit(f"Parâmetro desconhecido: {k}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            changes[k] = v.lower() in ("1", "true", "sim")
        elif isinstance(cur, int) or k == "seed":
            changes[k] = int(v)
        elif isinstance(cur, float):
            changes[k] = float(v)
        else:
            changes[k] = v
    return dataclasses.replace(cfg, **changes)


def evaluate_model(base_cfg, model, persona, reps, warm, log, only=None):
    # A avaliação nunca lê nem grava a memória permanente do usuário.
    cfg = dataclasses.replace(base_cfg, model=model, base_memory_enabled=False)
    client = OllamaClient(cfg)
    client.ensure_ready()
    for m in client.running_models():  # descarrega outros modelos para medir memória isolada
        client.unload_model_named(m.get("name", ""))
    time.sleep(1)
    # Início a frio, como ao abrir o app: carga do modelo + abertura (persona no cache).
    t0 = time.perf_counter()
    cold = ChatSession(cfg, persona, ROOT)
    if warm:
        cold.warm_up()
    else:
        client.load_model()
    load_s = time.perf_counter() - t0
    first = "".join(cold.reply("Oi, tudo bem?"))
    cold_ttft = cold.last.ttft_s
    log(f"  início a frio: preparo {load_s:.1f}s, 1º token da 1ª mensagem {cold_ttft:.1f}s -> {first.strip()[:70]!r}")
    mem_mb = next((m.get("size", 0) for m in client.running_models() if m.get("name") == model), 0) / 2**20

    now = datetime.now()
    results = []
    cases = [c for c in build_cases(now)
             if not only or any(c["id"] == x or c["id"].startswith(x + "-") for x in only)]
    for case in cases:
        for rep in range(reps):
            rcfg = dataclasses.replace(cfg, seed=rep + 1, **case.get("cfg", {}))  # mesmas sementes para todos
            extras = {"brutas": [], "preparo_s": 0.0, "manutencoes": []}
            try:
                replies, metrics, extras = run_conversation(rcfg, persona, case["passos"], warm)
                ok, detail = case["aval"](replies)
                if case.get("exige_manutencao") and not extras["manutencoes"]:
                    ok, detail = False, "inconclusivo: a memória não chegou a estourar (" + detail + ")"
            except OllamaError as exc:
                replies, metrics, ok, detail = [], [], False, f"erro: {exc}"
            asked = [p for p in case["passos"] if p is not CLEAR]
            results.append({"caso": case["id"], "nome": case["nome"], "rep": rep + 1, "passou": ok,
                            "detalhe": detail, "perguntas": asked[:len(replies)], "respostas": replies,
                            "respostas_brutas": extras["brutas"], "metricas": metrics,
                            "preparo_s": round(extras["preparo_s"], 2), "manutencoes": extras["manutencoes"],
                            "genericas": sum(is_generic(x) for x in replies[case.get("conta_genericas", slice(None))])})
            mark = "OK " if ok else "FALHA"
            log(f"  [{mark}] {case['id']:<10} rep{rep + 1}  {detail}")
            if not ok and replies:
                log(f"         -> {replies[-1].strip()[:160]!r}")
    client.unload_model()

    calls = [m for r in results for m in r["metricas"]]
    avg = lambda k: sum(c[k] for c in calls) / len(calls) if calls else 0.0  # noqa: E731
    med = lambda k: statistics.median(c[k] for c in calls) if calls else 0.0  # noqa: E731
    p90 = lambda k: sorted(c[k] for c in calls)[int(0.9 * (len(calls) - 1))] if calls else 0.0  # noqa: E731
    passed = sum(r["passou"] for r in results)
    all_replies = [x for r in results for x in r["respostas"]]
    all_raw = [x for r in results for x in r["respostas_brutas"]]
    pairs = [(q, a) for r in results for q, a in zip(r["perguntas"], r["respostas"])]
    preps = [r["preparo_s"] for r in results if r["preparo_s"]]
    return {
        "respostas_genericas_real": sum(r["genericas"] for r in results),
        "modelo": model, "carga_s": load_s, "inicio_frio_ttft_s": cold_ttft, "memoria_mb": mem_mb,
        "acertos": passed, "total": len(results),
        "tempo_medio_s": avg("total_s"), "tempo_mediano_s": med("total_s"), "tempo_p90_s": p90("total_s"),
        "ttft_medio_s": avg("ttft_s"), "ttft_mediano_s": med("ttft_s"), "ttft_p90_s": p90("ttft_s"),
        "tok_s_medio": avg("tok_s"),
        "preparo_conversa_mediano_s": statistics.median(preps) if preps else 0.0,
        "respostas_genericas": sum(is_generic(x) for x in all_replies), "respostas_total": len(all_replies),
        "truncadas": sum(c["truncada"] for c in calls),
        "estilo": style_metrics(all_replies, all_raw, pairs),
        "por_caso": {cid: f"{sum(r['passou'] for r in results if r['caso'] == cid)}/{reps}"
                     for cid in dict.fromkeys(r["caso"] for r in results)},
        "resultados": results,
    }

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+")
    ap.add_argument("--todos", action="store_true", help="avaliar todos os modelos instalados")
    ap.add_argument("--persona", default=None)
    ap.add_argument("--rep", type=int, default=3)
    ap.add_argument("--set", nargs="*", dest="overrides", metavar="CHAVE=VALOR")
    ap.add_argument("--sem-aquecer", action="store_true", help="não fazer a abertura de aquecimento")
    ap.add_argument("--rotulo", default="")
    ap.add_argument("--casos", nargs="+", help="rodar só estes casos (ex.: B F I)")
    args = ap.parse_args()

    cfg = apply_overrides(config.load(ROOT / "config.json"), args.overrides)
    if args.persona:
        requested_persona = Path(args.persona)
        persona_path = (requested_persona if requested_persona.is_absolute()
                        else config.project_path(ROOT, requested_persona, "persona"))
    else:
        persona_path = config.project_path(ROOT, cfg.persona_file, "persona_file")
    persona = personality.load_persona(persona_path)
    client = OllamaClient(cfg)
    models = client.installed_models() if args.todos else (args.modelos or [cfg.model])
    models = [m for m in models if "cloud" not in m]

    print(f"Persona: {persona_path.name} | aquecimento: {not args.sem_aquecer} | rep: {args.rep} | "
          f"temp={cfg.temperature} top_p={cfg.top_p} top_k={cfg.top_k} num_predict={cfg.num_predict} "
          f"num_ctx={cfg.num_ctx} think={cfg.think}")
    summaries = []
    for model in models:
        print(f"\n=== {model} ===")
        summaries.append(evaluate_model(cfg, model, persona, args.rep, not args.sem_aquecer, print, args.casos))

    print("\n=== RESUMO ===")
    for s in summaries:
        e = s["estilo"]
        print(f"{s['modelo']}: {s['acertos']}/{s['total']} acertos")
        print(f"   início a frio: preparo {s['carga_s']:.1f}s + 1º token {s['inicio_frio_ttft_s']:.1f}s | "
              f"RAM {s['memoria_mb']:.0f} MB")
        print(f"   1º token: mediana {s['ttft_mediano_s']:.2f}s, p90 {s['ttft_p90_s']:.2f}s | resposta: mediana "
              f"{s['tempo_mediano_s']:.1f}s, p90 {s['tempo_p90_s']:.1f}s | {s['tok_s_medio']:.1f} tok/s")
        print("   por caso: " + "  ".join(f"{k}={v}" for k, v in s["por_caso"].items()))
        print(f"   estilo: {e['palavras_media']} palavras/resposta · emoji na tela {e['com_emoji_pct']}% "
              f"(modelo gerou em {e.get('emoji_bruto_pct', 0)}%) · termina em pergunta "
              f"{e['termina_em_pergunta_pct']}% · comenta data sem pedir {e.get('comenta_data_sem_pedir_pct', 0)}% "
              f"· genéricas {s['respostas_genericas_real']}/{s['respostas_total']}")

    out_dir = ROOT / "avaliacao" / "resultados"
    out_dir.mkdir(exist_ok=True)
    label = f"_{args.rotulo}" if args.rotulo else ""
    out = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}{label}.json"
    out.write_text(json.dumps({
        "data": datetime.now().isoformat(timespec="seconds"), "persona": persona_path.name,
        # Registra o texto exato avaliado: o nome do arquivo não basta se ele for editado depois.
        "persona_sha256": hashlib.sha256(persona.encode("utf-8")).hexdigest(), "persona_texto": persona,
        "memoria": {"max_tokens": cfg.memory_max_tokens, "trim_ratio": cfg.memory_trim_ratio},
        "aquecimento": not args.sem_aquecer, "rep": args.rep, "casos": args.casos or "todos", "versao": "0.3",
        "parametros": cfg.model_options() | {"think": cfg.think}, "modelos": summaries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()

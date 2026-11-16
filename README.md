# Maddie I.A. 0.3

Chatbot local para Windows. Interface em Tkinter e inferência via **Ollama** em `127.0.0.1`. Nenhuma mensagem sai do computador e não há dependência de serviço de IA remoto.

> Estado do projeto, histórico de decisões, medições e próximos passos: **[DIARIO_DE_BORDO.md](DIARIO_DE_BORDO.md)**.

## Requisitos
- Windows 10/11, 16 GB de RAM (testado só em CPU: Ryzen 5 5500U, um pente de RAM)
- Python 3.10+ com Tkinter
- Ollama 0.30 ou mais novo
- Nenhum pacote pip

## Instalar
```powershell
winget install --id Ollama.Ollama -e
ollama pull qwen3.5:4b          # cerca de 3,3 GB, uma vez
```

## Executar
Dois cliques em **`iniciar_maddie.bat`**, ou `python main.py` para ver erros no terminal.

Ao abrir, a Maddie leva de 20 a 35 s se preparando (estado **Preparando**, na barra lateral). Você já pode digitar e enviar: a mensagem fica na fila e é respondida assim que ela estiver pronta.

| Tecla | Ação |
|---|---|
| Enter | envia |
| Shift+Enter | quebra a linha |
| Esc | interrompe a resposta |
| Ctrl+N | nova conversa |
| Ctrl+, | ajustes |
| Ctrl + / Ctrl - | tamanho da letra |

## O que tem na tela
- **Barra lateral:** estado do modelo, botão *Nova conversa*, estilo da resposta (Preciso, Equilibrado, Criativo), ocupação da memória da conversa, medições da última resposta, *Memória permanente*, *Exportar conversa* e *Ajustes*.
- **Conversa:** respostas com markdown (negrito, listas, código), hora de cada mensagem e, abaixo de cada resposta, tempo até o 1º token, velocidade e tamanho.
- **Abaixo da conversa:** *Regenerar* (refaz a última resposta) e *Copiar resposta*. Botão direito na conversa copia a seleção.
- **Enviar** vira **Parar** enquanto a Maddie escreve.

## Ajustes (botão Ajustes ou `config.json`)
| Campo | Padrão | Função |
|---|---|---|
| `model.name` | `qwen3.5:4b` | Modelo do Ollama. Modelos `*-cloud` são recusados |
| `model.num_ctx` | `8192` | Janela de contexto. Mudar recarrega o modelo |
| `model.num_predict` | `768` | Tamanho máximo da resposta |
| `model.temperature` / `top_p` / `top_k` | `0.7` / `0.8` / `20` | Perfil Equilibrado. Preciso = 0,3/0,8; Criativo = 1,0/0,95 |
| `model.repeat_penalty` / `presence_penalty` | `null` | `null` usa o padrão do modelo |
| `model.num_thread` / `num_batch` | `null` | `null` deixa o Ollama decidir. Mudar recarrega o modelo |
| `model.keep_alive` | `30m` | Tempo que o modelo fica na RAM sem uso |
| `memory.max_tokens` | `5500` | Teto do histórico (estimado como caracteres ÷ 3) |
| `memory.trim_ratio` | `0.6` | Ao estourar, esquece em lote até sobrar esta fração |
| `memory.summarize` | `true` | Resume o lote antes de esquecer (resumo fica só na RAM) |
| `base_memory.enabled` | **`false`** | Memória permanente entre conversas. **Grava em disco** |
| `base_memory.file` / `max_facts` | `memoria/memoria_base.json` / `60` | Arquivo e teto de fatos |
| `warm_up` | `true` | Prepara a Maddie ao abrir (1ª resposta rápida) |
| `strip_emoji` | `true` | Remove emoji da resposta, a menos que você use primeiro |
| `unload_model_on_exit` | `true` | Libera a RAM ao fechar |
| `ui.theme` / `font_size` / `show_stats` | `escuro` / `11` / `true` | Aparência |
| `ollama.host` / `port` | `127.0.0.1` / `11434` | Só aceita loopback |

## Memória e privacidade
- **Conversa:** fica só em RAM. Fechar a janela descarta tudo.
- **Resumo:** quando a conversa passa do teto, as trocas mais antigas são resumidas pelo próprio modelo e o resumo substitui essas trocas. Também fica só em RAM.
- **Memória permanente (desligada por padrão):** se você ligar, a Maddie guarda fatos duráveis sobre você em `memoria/memoria_base.json`, em texto puro. Nada entra sem validação: o trecho de origem precisa existir numa mensagem sua, e dados de saúde, finanças, documentos, senhas, endereço e contato são bloqueados. O painel *Memória permanente* mostra, edita e apaga tudo.
- **Exportar conversa** grava um `.md` onde você escolher. Só acontece quando você clica.
- A avaliação grava em `avaliacao/resultados/` apenas as conversas de teste dela.

## Testes
```powershell
python -m unittest discover -s tests -t . -v                      # 85 testes, sem Ollama
set MADDIE_TEST_REAL=1 && python -m unittest tests.test_real_ollama -v   # fumaça real
python avaliacao\avaliar.py                                       # bateria A–J real, cerca de 25 min
python avaliacao\avaliar.py --casos B I J --rep 1                 # só alguns casos
python avaliacao\avaliar.py --modelos qwen3.5:4b qwen3:4b-instruct-2507-q4_K_M
python avaliacao\avaliar_memoria_base.py                          # extração de fatos, cerca de 10 min
```
Ou `executar_testes.bat`. Os critérios da bateria são regex e heurísticas: "OK" significa critério atendido, não resposta boa. Leia as respostas no JSON.

## Arquitetura (resumo)
```
main.py                   entrada: carrega config + persona, abre a interface
maddie/config.py          config.json: leitura, validação, gravação, perfis
maddie/personality.py     leitura do persona.md
maddie/context.py         nota de data/hora e regra de quando ela é enviada
maddie/memory.py          memória da conversa (RAM): orçamento de tokens, corte em lote, resumo
maddie/memory_store.py    memória permanente (disco): fatos, validação, lista de bloqueio
maddie/textfilter.py      filtro de emoji em streaming
maddie/markdown_lite.py   markdown mínimo para a tela
maddie/ollama_client.py   HTTP para 127.0.0.1, streaming, métricas, erros
maddie/engine.py          motor: junta tudo acima (usado pela interface e pela avaliação)
maddie/gui.py             janela principal
maddie/dialogs.py         janelas de Ajustes e Memória permanente
avaliacao/                baterias reais e resultados
tests/                    testes determinísticos
estudos/                  estudo que originou a memória permanente
```
Detalhes, fluxo de uma mensagem e decisões de desempenho: [DIARIO_DE_BORDO.md](DIARIO_DE_BORDO.md).

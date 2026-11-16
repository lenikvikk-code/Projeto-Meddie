# Maddie I.A. · Diário de bordo

Documento vivo do projeto: estado atual, arquitetura, modo de uso, histórico e próximos passos.
Ao mexer no projeto, acrescente uma entrada no **Diário** (seção 6) e atualize o **Estado atual** (seção 1).

Pasta: `C:\Users\lenik\Projetos\maddie` · Versão: **0.3.0** · Última atualização: 14/11/2026

---

## 1. Estado atual (leitura rápida)

| Item | Situação |
|---|---|
| Versão | 0.3.0 |
| Modelo padrão | `qwen3.5:4b` (antes: `qwen3:4b-instruct-2507-q4_K_M`, que continua instalado) |
| Testes determinísticos | 86 passando, 1 pulado (o teste real, que só roda sob pedido) |
| Bateria real A–J | 41/42. A única falha era do avaliador ("vinte e quatro" por extenso), já corrigido |
| Início (1º token da 1ª mensagem) | **0,7 s** (v0.2: 28,1 s) |
| 1º token por mensagem | mediana 1,3 s, p90 2,0 s (v0.2: média de 3,5 s) |
| Resposta completa | mediana 5,7 s, p90 10,9 s, 6,4 tok/s |
| Emoji nas respostas | 0% (v0.2: 18%) |
| RAM do modelo | cerca de 3,2 GB |
| Backups | `Projetos\maddie_backup_v0.2.0_20261114.zip` e `maddie_backup_v0.1.1_20261113.zip` |

### O que foi verificado no modelo real (14/11/2026)
| Verificação | Resultado |
|---|---|
| Bateria A–J completa (42 conversas) | 41/42; a falha era do avaliador |
| Resumo de memória com prompt próprio (caso J, 3 repetições) | 3/3, e nenhum resumo trouxe os exemplos da persona |
| Extração de fatos (10 diálogos) | cobertura 8/9, precisão 9/10, 1 fato trivial gravado (validação apertada depois) |
| Teste de fumaça real | passou; 1º token da 1ª mensagem em 1,4 s |
| Interface com o modelo real, de ponta a ponta | sem exceções: fila durante o preparo, resposta, parar, regenerar, nova conversa, hora certa |

### Pendências abertas
1. **Reinicie o app.** A janela da Maddie aberta desde as 16:08 de 14/11 ainda roda o código anterior às correções do resumo e da extração. Feche e abra de novo.
2. A regra que recusa fatos extraídos que citam a própria Maddie foi apertada depois da última rodada real. Está coberta por teste determinístico, não por nova rodada no modelo.
3. As últimas rodadas reais foram feitas **na bateria e com o app aberto ao mesmo tempo**: a velocidade caiu para 3,4 a 4,7 tok/s. Os tempos de referência são os da bateria completa, na tomada: 6,4 tok/s.
4. A última captura de tela saiu preta (tela desligada ou bloqueada). O visual foi conferido nas capturas anteriores, com dados simulados; a tela com respostas reais não foi vista por quem desenvolveu.
5. Seta das caixas de seleção em Ajustes fica branca no tema escuro (só estética).
6. Os resumos acertam os fatos do usuário, mas são prolixos e às vezes trocam quem disse o quê nos assuntos gerais (medido: "o usuário definiu números primos").
7. Ao responder a hora, o modelo às vezes acrescenta a data sem ninguém pedir.

---

## 2. Modo de uso

1. `ollama pull qwen3.5:4b` (uma vez, 3,3 GB).
2. Dois cliques em `iniciar_maddie.bat`.
3. A barra lateral mostra **Preparando** por 20 a 35 s. Já dá para digitar e enviar: a mensagem entra na fila.
4. Enter envia, Shift+Enter quebra linha, Esc interrompe, Ctrl+N abre conversa nova, Ctrl+, abre Ajustes.

Detalhes de tela e de cada ajuste: `README.md`.

**Memória permanente** vem desligada. Para ligar: botão *Memória permanente* na barra lateral. Ela grava fatos sobre você em `memoria/memoria_base.json`, em texto puro. Fatos novos valem a partir da conversa seguinte.

---

## 3. Arquitetura

```
main.py                   entrada
maddie/config.py          config.json: leitura, validação, gravação, perfis
maddie/personality.py     leitura do persona.md
maddie/context.py         nota de data/hora e regra de quando enviar
maddie/memory.py          memória da conversa (RAM): orçamento, corte em lote, resumo, trocas ocultas
maddie/memory_store.py    memória permanente (disco): fatos, validação, lista de bloqueio
maddie/textfilter.py      filtro de emoji em streaming
maddie/markdown_lite.py   markdown mínimo para a tela
maddie/ollama_client.py   HTTP para 127.0.0.1, streaming, métricas, erros
maddie/engine.py          motor (ChatSession): usado pela interface E pela avaliação
maddie/gui.py             janela principal
maddie/dialogs.py         Ajustes e Memória permanente
avaliacao/avaliar.py                bateria A–J real
avaliacao/avaliar_memoria_base.py   extração de fatos real
tests/                    testes determinísticos
```

### Como o prompt é montado
```
[system]     persona.md (+ fatos da memória permanente, se ligada)
[user/asst]  resumo do que já foi esquecido (se houver)
[user/asst]  abertura oculta: "[Aplicação: nova conversa iniciada em ...]" -> "Pronta."
[user/asst]  histórico, reenviado byte a byte como foi processado
[user]       mensagem nova (com nota de data/hora só quando necessário)
```

### Fluxo de uma mensagem
1. `gui` põe a mensagem na tela e chama `ChatSession.reply` numa thread.
2. O motor decide se a nota de hora vai (1ª mensagem, mensagem que fala de tempo, ou última nota com mais de 10 min).
3. A resposta chega em pedaços; o filtro de emoji age só no que vai para a tela.
4. A troca é guardada com o texto **cru** do modelo (para o cache) e o texto exibido.
5. Se o histórico estourou o orçamento, roda a manutenção em segundo plano: extrai fatos (se a memória permanente está ligada), resume o lote antigo, esquece-o e reaquece o cache.

### Decisões de desempenho (com a medição que as justifica)
| Decisão | Medição |
|---|---|
| Aquecer ao abrir, com uma troca oculta que fica no histórico | 1ª mensagem: 28,1 s -> 0,7 s |
| Carregar o modelo já com `num_ctx` certo | a v0.2 carregava com 4096 e recarregava com 8192 (5 s) |
| Nota de hora só quando necessária | a nota custa de 1,1 a 1,9 s por mensagem |
| Histórico reenviado exatamente como gerado | o `qwen3.5` só reaproveita o cache em continuação exata: qualquer diferença no meio reprocessa tudo (15 s ou mais) |
| Emoji removido por código, não por prompt | 18% -> 0% na tela |

---

## 4. Limitações conhecidas
- **Guardar fatos desta conversa** e a **manutenção de memória** usam o modelo para outra tarefa; depois disso o `qwen3.5` reprocessa a conversa inteira (consequência da regra de cache acima). Em conversa longa, pode levar mais de um minuto. **Regenerar** não sofre disso: no teste real o 1º token veio em 0,5 s.
- **Manutenção de memória** leva de 30 a 95 s (medido) e acontece a cada várias dezenas de mensagens. Mensagens enviadas nesse intervalo ficam na fila.
- **Inatividade:** depois de `keep_alive` (30 min) o modelo sai da RAM e a resposta seguinte reprocessa tudo.
- **Suspensão do notebook** congela qualquer resposta em andamento. Aconteceu duas vezes durante as baterias (chamadas de 1.867 s e 3.175 s).
- **Velocidade:** 6,4 tok/s em CPU com um pente de RAM. Um segundo pente (canal duplo) deve acelerar; não medido.
- A extração de fatos é lenta: de 38 a 80 s por chamada (medido na bateria do notebook).
- **Na bateria do notebook** a velocidade cai para 3,4 a 4,7 tok/s (medido). Use na tomada.
- Amostra pequena: cada caso da bateria roda 3 vezes. Diferenças de 1 ou 2 acertos entre modelos não são conclusivas.

---

## 5. Resultados medidos (14/11/2026)

### Bateria A–J, 3 repetições, motor 0.3
| Configuração | Acertos | 1º token (mediana) | Resposta (mediana) | tok/s | Emoji gerado pelo modelo | RAM |
|---|---|---|---|---|---|---|
| `qwen3:4b-instruct` | 40/42 | 1,16 s | 4,4 s | 7,1 | 17% | 3,7 GB |
| `qwen3.5:4b` (1ª rodada) | 41/42 | 1,22 s | 5,9 s | 6,5 | 0% | 3,2 GB |
| **`qwen3.5:4b` (final, persona ajustada)** | **41/42** | **1,27 s** | **5,7 s** | **6,4** | **0%** | **3,2 GB** |

Leitura das falhas:
- `qwen3:4b-instruct`, caso H: respondeu 8154, 8155 e 815 para o código 8153. Erro real do modelo, nas 3 repetições (uma contou como acerto por coincidência de texto).
- `qwen3.5:4b`, 1ª rodada, caso G: falso positivo do avaliador ("como não sei seu gosto, sugiro Luna"). Regra corrigida.
- `qwen3.5:4b`, final, caso D: respondeu "vinte e quatro" por extenso. Avaliador corrigido.

Motivo da troca de modelo: acerta o caso H, não gera emoji e usa menos RAM, com velocidade parecida. Confiança: média (3 repetições).

### Resumo do que é esquecido (caso J, `qwen3.5:4b`, 3 repetições)
| Versão | Lembrou os fatos | Resumo trouxe exemplos da persona |
|---|---|---|
| Resumo com a persona no prompt | 3/3 | sim, em 3 de 3 conversas |
| Idem, com instrução e persona ajustadas | 3/3 | sim, em 3 de 3 |
| **Resumo com prompt próprio (atual)** | 3/3 | **não, em nenhuma** |

### Extração de fatos da memória permanente (`qwen3.5:4b`, 10 diálogos, 1 repetição)
| Versão do prompt | Cobertura | Precisão | Violações | Tempo médio |
|---|---|---|---|---|
| Com a persona no prompt | 8/9 | 8/8 | 0 | 73,8 s (na tomada) |
| **Prompt próprio, só falas do usuário (atual)** | 8/9 | 9/10 | 1 | 55,9 s (na bateria) |

Diferenças entre as duas rodadas:
- **Preferências** ("prefiro respostas curtas, não gosto de emoji"): antes não extraía; agora extrai as duas.
- **Atualização** ("mudei de emprego, agora sou professora"): antes o fato antigo ficava junto com o novo; agora o novo substitui o antigo.
- **Fala sobre a Maddie:** a versão atual gravou "O usuário gosta de conversar com Maddie". Depois disso a validação passou a recusar todo fato extraído que cite a Maddie.
- **Senha e CPF + cidade:** a versão atual não gravou nada, nem a cidade. Nenhum dado sensível foi gravado em nenhuma rodada.
- Papéis ("meu irmão se chama Pedro") e perguntas do usuário: corretos nas duas.

Uma repetição por caso: as diferenças acima são indícios, não prova.

Arquivos em `avaliacao/resultados/`: `20261114_133205_v030_qwen35_4b.json`, `20261114_134818_v030_qwen3_4b_instruct.json`, `20261114_155151_v030_final_qwen35_4b.json`, `20261114_180211_v030_resumo_prompt_proprio.json`, `20261114_160420_memoria_base_qwen35_4b.json`, `20261114_181130_memoria_base_prompt_proprio.json`.

---

## 6. Diário

### 14/11/2026 · v0.3.0
- Backup da 0.2.0.
- Causa do início lento encontrada e medida: recarga do modelo + persona processada só na 1ª mensagem.
- Motor novo (`engine.py`) compartilhado por interface e avaliação.
- Aquecimento por abertura oculta, depois de descobrir que o `qwen3.5` não reaproveita cache a partir do meio do prompt.
- Nota de hora curta, guardada no histórico e enviada só quando necessária.
- Filtro de emoji por código.
- Resumo do que é esquecido; memória permanente (desligada por padrão) com validação e lista de bloqueio.
- Interface nova: barra lateral, markdown, parar, regenerar, copiar, exportar, perfis, temas, Ajustes, painel de memória.
- Modelo trocado para `qwen3.5:4b` com base na bateria.
- Bugs achados nos testes e corrigidos: resumo que puxava exemplos da persona; persona com exemplos marcados como fictícios; duas regras do avaliador.
- Fechamento (fim da tarde): resumo e extração de fatos passaram a usar prompts próprios, sem a persona. Resumo reverificado no modelo real (3/3, sem vazamento), extração reavaliada, teste de fumaça real e teste da interface com o modelo real passaram.
- A bateria deixou de ler a memória permanente do usuário na medição de início a frio.
- Os testes de configuração deixaram de exigir `config.json` igual ao padrão (ele guarda as preferências do usuário).
- O usuário ligou a memória permanente pelo app e incluiu 2 fatos à mão (16:31).

### 13/11/2026 · v0.2.0
- Modelo Qwen3 4B Instruct, persona reescrita, memória por tokens com corte em lote, janela de 8.192. Bateria: 39/39.

### 11–12/11/2026 · v0.1.x
- Primeira versão com `qwen3:0.6b`.

---

## 7. Próximos passos sugeridos

**Agora (curto)**
1. Fechar e reabrir a Maddie, para carregar as correções do resumo e da extração.
2. Usar por alguns dias com a memória permanente ligada e anotar aqui o que estranhar: fato errado, resumo estranho, demora.
3. Rodar `python avaliacao\avaliar_memoria_base.py --rep 3` na tomada, com o app fechado. Uma repetição por caso é pouco para confiar na extração.

**Melhorar (médio)**
4. Resumo mais curto e sem atribuir ao usuário o que a Maddie disse: limitar a "fatos do usuário + lista de assuntos" e medir no caso J.
5. Extração perdeu a cidade quando ela veio junto de senha e CPF: testar se o modelo descarta a conversa inteira ao ver dado sensível.
6. Impedir a suspensão do Windows enquanto a Maddie responde ou organiza a memória.
7. Corrigir a seta branca das caixas de seleção no tema escuro.
8. Segundo pente de RAM e nova medição de tok/s.

**Ampliar (longo)**
9. Várias conversas salvas na barra lateral (exige decidir sobre gravação em disco).
10. Busca de fatos por relevância quando a memória permanente passar do orçamento.
11. Ler arquivos locais (txt, pdf) para conversar sobre eles.
12. Testar um modelo de 8 a 9B se a RAM aumentar.

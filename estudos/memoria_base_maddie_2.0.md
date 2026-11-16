# Estudo: a Maddie montando a própria memória base (Maddie 2.0)

> Estudo **só informativo**. Nada aqui foi implementado. Data: 13/11/2026, com base na v0.2.0.

## 1. O que seria a "memória base"
É um arquivo que a própria Maddie preenche durante as conversas, com **fatos duráveis** sobre o usuário e sobre a relação entre os dois, e que ela lê no começo de cada sessão. Exemplos:

- **Identidade:** nome, como prefere ser chamado, profissão, cidade.
- **Pessoas:** "irmão: Pedro", "cachorra: Pipoca".
- **Preferências:** respostas curtas, gosta de ficção científica, não gosta de emoji.
- **Contexto vivo:** "está estudando para uma prova em dezembro", "projeto Maddie em andamento".

Ela **não** é o histórico da conversa. O histórico continua na memória de curto prazo, em RAM, que a v0.2 já tem. A memória base guarda só o que vale a pena saber na próxima sessão.

## 2. Decisões que são suas, antes de qualquer código
| Decisão | Por que importa |
|---|---|
| **Gravar em disco?** | Hoje a regra da Maddie é "nada da conversa vai para o disco". A memória base exige quebrar essa regra de forma controlada. |
| **O que é proibido guardar** | Saúde, finanças, senhas, documentos, terceiros identificáveis... É preciso uma lista de bloqueio explícita. |
| **Grava sozinha ou pede confirmação?** | Gravar sozinha soa mais natural. Confirmar é mais seguro. Meio-termo: grava sozinha, mostra um aviso discreto ("anotei: sua cachorra se chama Pipoca") e permite desfazer. |
| **Criptografar o arquivo?** | Texto puro é simples e auditável. Criptografia (por exemplo, Windows DPAPI) protege contra quem acessa o disco, mas complica edição manual. |

## 3. Arquitetura mínima (o básico que funciona)

```
conversa (RAM) ──► [gatilho] ──► extração (chamada extra ao modelo, JSON)
                                      │
                                      ▼
                             validação por código ──► memoria_base.json
                                                         │
                         início da sessão ◄──────────────┘
                     (fatos entram no prompt de sistema, logo após a persona)
```

### 3.1 Arquivo
`memoria/memoria_base.json` com uma lista de fatos:

```json
{
  "versao": 1,
  "fatos": [
    {
      "id": "f-0007",
      "categoria": "pessoas",
      "fato": "O usuário tem uma cachorra chamada Pipoca.",
      "trecho_fonte": "Tenho uma cachorra chamada Pipoca.",
      "criado_em": "2026-11-13T10:52",
      "atualizado_em": "2026-11-13T10:52",
      "usos": 0
    }
  ]
}
```
O `trecho_fonte` é obrigatório. Ele permite conferir de onde veio cada fato e é a base da validação do item 3.3.

### 3.2 Extração: quem escreve o arquivo
A própria Maddie, numa **chamada separada** ao mesmo modelo, com saída em JSON estruturado. O Ollama aceita um JSON Schema no campo `format` de `/api/chat`, o que obriga a resposta a seguir a estrutura. A instrução seria algo como: "dada esta conversa e a memória atual, liste operações ADICIONAR / ATUALIZAR / REMOVER / NADA". É o padrão usado por sistemas como o Mem0, e ele evita fatos duplicados ou contraditórios.

**Quando extrair.** A v0.2 já cria o momento certo: quando o orçamento de tokens estoura, a memória **esquece um lote de trocas antigas**. O gatilho natural é **"resumir antes de esquecer"**: extrair os fatos desse lote antes de descartá-lo. Gatilhos secundários: ao fechar a janela e quando o usuário pedir ("guarda isso").

### 3.3 Validação por código (o ponto crítico)
O modelo de 4B vai errar a extração. O mesmo defeito do teste B ("Meu nome é Rafael") vira aqui um fato gravado errado, **e de forma permanente**. Por isso nada vai direto para o arquivo. Antes de gravar, o código confere:

1. O `trecho_fonte` existe literalmente numa mensagem **do usuário**, não da Maddie.
2. A categoria pertence à lista permitida, e o fato não cai na lista de bloqueio.
3. Não há outro fato com o mesmo assunto. Se houver, vira ATUALIZAR, nunca uma duplicata.
4. O fato tem tamanho limitado (por exemplo, 200 caracteres) e o arquivo tem um teto (por exemplo, 60 fatos).

### 3.4 Uso nas sessões seguintes
- No início da sessão, os fatos entram no prompt de sistema, depois da persona, num bloco "O que você sabe sobre o usuário". Como o bloco é fixo durante a sessão, **o cache do prompt continua funcionando**.
- **Orçamento de tokens:** 60 fatos curtos ocupam uns 1.200 tokens (estimativa). Isso cabe na janela de 8K da v0.2, mas tira espaço do histórico. Acima disso, seria preciso escolher os fatos relevantes (busca por embeddings), o que já é um passo além do básico.

### 3.5 Controle do usuário
- Um painel "O que a Maddie sabe sobre você", para ver, editar e apagar fatos.
- Comandos na conversa: "esqueça isso", "o que você lembra de mim?".
- Um botão "Apagar toda a memória".

## 4. Custos e riscos
| Item | Avaliação |
|---|---|
| **Tempo de CPU** | Cada extração é uma chamada extra ao modelo. Estimo entre 10 e 30 s no 4B nesta máquina, sem medição. Ela precisa rodar quando a Maddie está ociosa: em CPU, extrair enquanto responde deixa as duas tarefas lentas. |
| **Fatos errados** | É o maior risco (seção 3.3). Um fato errado e permanente é pior que esquecer. |
| **Acúmulo de lixo** | Sem teto e sem deduplicação, o arquivo cresce com trivialidades ("o usuário perguntou a capital do Japão"). A instrução de extração precisa dizer "só fatos duráveis sobre o usuário". |
| **Privacidade** | Arquivo legível no disco. Ver a decisão sobre criptografia no item 2. |
| **Soar artificial** | Citar memórias o tempo todo ("como você me disse em 13/11...") parece robótico. A persona precisa orientar o uso natural: usar o fato sem anunciar que lembrou. |

## 5. Testes necessários (antes de confiar)
1. **Extração:** 20 a 30 diálogos roteirizados com os fatos esperados. Medir precisão (fatos gravados que são corretos) e cobertura (fatos ditos que foram gravados).
2. **Quem é quem:** frases como "meu irmão se chama Pedro" e "a Maddie é legal" não podem gerar "o usuário se chama Pedro" nem fatos sobre a Maddie.
3. **Atualização:** dizer "mudei de emprego, agora sou professora" tem que substituir o fato antigo, não somar.
4. **Bloqueio:** dados sensíveis da lista proibida nunca chegam ao arquivo.
5. **Entre sessões:** fechar, abrir e perguntar "o que você lembra de mim?".
6. **Esquecer:** "esqueça isso" remove o fato do arquivo de verdade.

## 6. O que a v0.2 já deixou pronto
- **Memória por tokens com corte em lote:** é o gancho do "resumir antes de esquecer".
- **Persona fixa no topo do prompt:** o bloco de fatos pode entrar logo depois sem quebrar o cache.
- **Bateria de avaliação real** (`avaliacao/avaliar.py`): dá para acrescentar os casos da seção 5.

## 7. Ordem sugerida para a 2.0
1. Você toma as decisões do item 2.
2. Arquivo, leitura e injeção no prompt, com fatos escritos **à mão**. Isso valida o uso antes da extração.
3. Extração automática com validação, só no gatilho "ao fechar".
4. Gatilho "resumir antes de esquecer" e painel de controle.
5. Busca por relevância (embeddings), apenas se o arquivo passar do orçamento.

**Esforço:** dois módulos novos (`memory_store.py` e `extractor.py`), um painel na interface e cerca de 15 testes. É uma estimativa grosseira, de ordem de grandeza. A maior parte do trabalho fica no item 5, nos testes de extração, e não no código.

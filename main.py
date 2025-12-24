"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

FastAPI Application para Agentic RAG com Controle de Acesso por Collections.

Este módulo implementa um AGENTE RAG INTELIGENTE com:
- Controle de acesso por collections (cada usuário só consulta suas collections)
- GRADING: Avaliação de relevância de cada documento recuperado
- QUERY REWRITE: Reescrita automática de queries para melhorar resultados
- HALLUCINATION CHECK: Verificação se a resposta é factual e baseada no contexto

Fluxo do Agente:
    1. Usuário faz login (Basic Auth)
    2. Sistema obtém collections permitidas
    3. Busca documentos nas collections permitidas
    4. GRADING: Filtra documentos irrelevantes
    5. Se poucos docs → REWRITE: Reformula a query e busca novamente
    6. Gera resposta com documentos relevantes
    7. HALLUCINATION CHECK: Verifica se resposta é factual
    8. Se alucinação → Regenera resposta

Mapeamento de Permissões:
- user_a, user_b, user_c: acesso a "medical_qa" → collection "medical_q_n_a"
- user_1, user_2, user_3: acesso a "device_manual" → collection "medical_device_manual"
- admin: acesso a ambas as collections

Para executar:
    uvicorn main:app --reload --port 8000

Para testar:
    curl -u user_a:senha123 -X POST http://localhost:8000/query \\
         -H "Content-Type: application/json" \\
         -d '{"query": "Quais são os tratamentos para a doença de Kawasaki?"}'
"""
# Importar módulo de autenticação
from auth import (
    get_current_user,
    get_access_groups,
    get_user_allowed_collections_with_display
)
import os
from typing import List, Dict, Any

import chromadb
from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

# Carregar variáveis de ambiente
_ = load_dotenv(find_dotenv())


# ============================================================
# Configuração do ChromaDB e OpenAI
# ============================================================

# Configuração de chaves API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Cliente OpenAI
clientOpenAI = OpenAI(api_key=OPENAI_API_KEY)

# Cliente ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db_AgenteRAG")

# Collections disponíveis no sistema
# Mapeamento: nome_collection → objeto Collection
ALL_COLLECTIONS = {
    "medical_q_n_a": chroma_client.get_or_create_collection(
        name="medical_q_n_a",
        metadata={"hnsw:space": "cosine"}
    ),
    "medical_device_manual": chroma_client.get_or_create_collection(
        name="medical_device_manual",
        metadata={"hnsw:space": "cosine"}
    )
}


# ============================================================
# Configurações do Agente RAG
# ============================================================
MAX_QUERY_REWRITES = 2  # Máximo de tentativas de reescrita de query


# ============================================================
# Funções auxiliares básicas
# ============================================================
def generate_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """Gera um vetor de Embedding para um texto usando a API OpenAI."""
    response = clientOpenAI.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def get_llm_response(prompt: str) -> str:
    """Obtém resposta do LLM."""
    response = clientOpenAI.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ============================================================
# Funções do Agente RAG (Grading, Rewrite, Hallucination Check)
# ============================================================
def preprocess_query(original_query: str) -> str:
    """
    PRÉ-PROCESSAMENTO: Corrige erros ortográficos e melhora a query ANTES da busca.
    
    Esta função é executada SEMPRE antes de qualquer busca para garantir
    que a query esteja bem escrita e clara.
    
    Correções aplicadas:
    - Erros de digitação
    - Erros ortográficos
    - Abreviações expandidas
    - Termos técnicos corrigidos
    
    Args:
        original_query: Pergunta original do usuário (pode conter erros).
    
    Returns:
        str: Query corrigida e melhorada.
    """
    preprocess_prompt = f"""
    Você é um ESPECIALISTA em correção ortográfica do pt-br e melhoria de queries para sistemas de busca.
    
    Corrija a pergunta abaixo, aplicando:
    1. Correção de erros de digitação
    2. Correção de erros ortográficos do pt-br
    3. Melhoria da frase para uma query mais semântica e objetiva
    4. Manter o significado semântico original da pergunta original
    
    PERGUNTA ORIGINAL:
    {original_query}
    
    IMPORTANTE:
    - NÃO mude o significado da pergunta
    - NÃO adicione informações novas
    - Apenas corrija e melhore a escrita
    - Se a pergunta já estiver correta, retorne-a sem alterações
    
    Retorne APENAS a pergunta corrigida, sem explicações.
    """
    
    corrected = get_llm_response(preprocess_prompt).strip()
    
    # Remove aspas se o LLM adicionou
    if corrected.startswith('"') and corrected.endswith('"'):
        corrected = corrected[1:-1]
    if corrected.startswith("'") and corrected.endswith("'"):
        corrected = corrected[1:-1]
    
    return corrected


def grade_document(query: str, document: str) -> bool:
    """
    GRADING: Avalia se um documento é relevante para a query.
    
    Esta função usa o LLM para verificar se o documento contém
    informações úteis para responder à pergunta do usuário.
    
    Args:
        query: Pergunta do usuário.
        document: Conteúdo do documento a ser avaliado.
    
    Returns:
        bool: True se o documento é relevante, False caso contrário.
    """
    grading_prompt = f"""
    Você é um avaliador de relevância de documentos.
    
    Avalie se o documento abaixo contém informações relevantes e úteis
    para responder à pergunta do usuário.
    
    DOCUMENTO:
    {document}
    
    PERGUNTA DO USUÁRIO:
    {query}
    
    Critérios de avaliação:
    - O documento contém informações relacionadas ao tema da pergunta?
    - As informações são úteis para formular uma resposta?
    - O documento é factual e não contém apenas informações genéricas?
    
    Responda APENAS com 'relevante' ou 'irrelevante'.
    """
    
    response = get_llm_response(grading_prompt).strip().lower()
    return "relevante" in response


def grade_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Avalia uma lista de documentos e retorna apenas os relevantes.
    
    Args:
        query: Pergunta do usuário.
        documents: Lista de documentos recuperados.
    
    Returns:
        List[Dict]: Lista filtrada contendo apenas documentos relevantes.
    """
    relevant_docs = []
    
    for doc in documents:
        if grade_document(query, doc["document"]):
            doc["graded"] = True
            relevant_docs.append(doc)
    
    return relevant_docs


def rewrite_query(original_query: str, context_hint: str = "") -> str:
    """
    REWRITE: Reescreve a query para melhorar os resultados da busca.
    
    Esta função usa o LLM para reformular a pergunta do usuário
    de forma a obter melhores resultados na busca semântica.
    
    Args:
        original_query: Pergunta original do usuário.
        context_hint: Dica opcional sobre o contexto (ex: tipo de collection).
    
    Returns:
        str: Query reescrita.
    """
    rewrite_prompt = f"""
    Você é um especialista em reformulação de perguntas para sistemas de busca.
    
    A pergunta original do usuário não retornou bons resultados.
    Reescreva a pergunta de forma mais clara, específica e objetiva,
    mantendo a intenção original.
    
    PERGUNTA ORIGINAL:
    {original_query}
    
    {f"CONTEXTO: A busca é feita em documentos sobre {context_hint}" if context_hint else ""}
    
    Técnicas a aplicar:
    - Use sinônimos relevantes
    - Seja mais específico sobre o tema
    - Remova palavras desnecessárias
    - Foque nos termos-chave
    
    Retorne APENAS a pergunta reescrita, sem explicações.
    """
    
    return get_llm_response(rewrite_prompt).strip()


def check_hallucination(query: str, context: str, answer: str) -> bool:
    """
    HALLUCINATION CHECK: Verifica se a resposta está fundamentada no contexto.
    
    Esta função usa o LLM para verificar se a resposta gerada
    é baseada apenas nas informações do contexto, sem alucinações.
    
    Args:
        query: Pergunta do usuário.
        context: Contexto usado para gerar a resposta.
        answer: Resposta gerada pelo LLM.
    
    Returns:
        bool: True se a resposta é factual (sem alucinação), False caso contrário.
    """
    hallucination_prompt = f"""
    Você é um verificador de fatos rigoroso.
    
    Verifique se a RESPOSTA abaixo está completamente fundamentada
    nas informações do CONTEXTO fornecido.
    
    CONTEXTO:
    {context}
    
    PERGUNTA:
    {query}
    
    RESPOSTA A VERIFICAR:
    {answer}
    
    Critérios de verificação:
    1. Todas as afirmações na resposta podem ser verificadas no contexto?
    2. A resposta não adiciona informações que não estão no contexto?
    3. A resposta não contradiz o contexto?
    4. A resposta não faz suposições além do que está escrito?
    
    Responda APENAS com:
    - 'factual' se a resposta está 100% baseada no contexto
    - 'alucinação' se a resposta contém informações não fundamentadas
    """
    
    response = get_llm_response(hallucination_prompt).strip().lower()
    return "factual" in response


def regenerate_answer(query: str, context: str, previous_answer: str) -> str:
    """
    Regenera a resposta corrigindo possíveis alucinações.
    
    Args:
        query: Pergunta do usuário.
        context: Contexto disponível.
        previous_answer: Resposta anterior que continha alucinações.
    
    Returns:
        str: Nova resposta mais factual.
    """
    prompt = f"""
    A resposta anterior continha informações não fundamentadas no contexto.
    Gere uma nova resposta usando ESTRITAMENTE as informações do contexto.
    
    CONTEXTO:
    {context}
    
    PERGUNTA:
    {query}
    
    RESPOSTA ANTERIOR (com problemas):
    {previous_answer}
    
    INSTRUÇÕES:
    - Use APENAS informações presentes no contexto
    - Se o contexto não tiver a informação, diga claramente
    - Não faça suposições ou inferências
    - Seja factual e objetivo
    - Responda em português brasileiro (pt-br)
    - Limite: 150 palavras
    """
    
    return get_llm_response(prompt)


def search_allowed_collections(
    query: str,
    allowed_collections: List[str],
    n_results: int = 3
) -> List[Dict[str, Any]]:
    """
    Busca APENAS nas collections que o usuário tem permissão.
    
    Esta é a função principal de controle de acesso:
    - O usuário só pode buscar nas collections mapeadas aos seus grupos
    - Se o usuário não tem acesso a nenhuma collection, retorna lista vazia
    
    Exemplo:
        - user_a tem allowed_collections: ["medical_q_n_a"]
        - Busca será feita APENAS em "medical_q_n_a"
        - Resultados de "medical_device_manual" NÃO serão retornados
    
    Args:
        query: Pergunta do usuário.
        allowed_collections: Lista de nomes de collections que o usuário pode acessar.
        n_results: Número de resultados por collection.
    
    Returns:
        List[Dict]: Lista de dicts com 'document', 'metadata', 'distance', 'collection'
    """
    if not allowed_collections:
        return []
    
    query_embedding = generate_embedding(query)
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"🔍 DEBUG BUSCA - Query: {query}")
    print(f"📁 Collections permitidas: {allowed_collections}")
    print(f"{'='*60}")
    
    # Buscar APENAS nas collections permitidas
    for collection_name in allowed_collections:
        # Verificar se a collection existe no sistema
        if collection_name not in ALL_COLLECTIONS:
            print(f"Aviso: Collection '{collection_name}' não existe no sistema")
            continue
        
        collection = ALL_COLLECTIONS[collection_name]
        
        # DEBUG: Mostrar total de docs na collection
        print(f"\n📂 Collection: {collection_name} (Total docs: {collection.count()})")
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            if results["documents"] and results["documents"][0]:
                print(f"   ✅ Encontrados: {len(results['documents'][0])} documentos")
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    # DEBUG: Mostrar preview do documento
                    doc_preview = doc[:150].replace('\n', ' ') + "..." if len(doc) > 150 else doc
                    print(f"   📄 Doc {i+1} (dist={distance:.4f}): {doc_preview}")
                    
                    all_results.append({
                        "document": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": distance,
                        "collection": collection_name
                    })
            else:
                print("   ⚠️ Nenhum documento encontrado")
                    
        except Exception as e:
            print(f"Erro ao buscar em {collection_name}: {e}")
            continue
    
    # Ordenar por distância (menor = mais relevante)
    all_results.sort(key=lambda x: x["distance"])
    
    print(f"\n📊 Total de resultados ordenados: {len(all_results)}")
    print(f"{'='*60}\n")
    
    return all_results


def check_context_relevance(query: str, context: str) -> bool:
    """
    Verifica se o contexto recuperado é relevante para a query.
    Retorna True se relevante, False caso contrário.
    """
    if not context or context.strip() == "":
        return False
    
    relevance_prompt = f"""
    Verifique se o contexto fornecido contém informações relevantes para responder
    à pergunta do usuário.
    
    Contexto:
    {context}
    
    Pergunta do usuário: {query}

    Responda APENAS com 'Sim' se o contexto contém informações úteis para responder,
    ou 'Não' se o contexto NÃO contém informações relevantes.
    """
    relevance_response = get_llm_response(relevance_prompt).strip().lower()
    
    return relevance_response in ["sim", "yes", "s", "y"]


def generate_answer(query: str, context: str) -> str:
    """Gera resposta usando o contexto recuperado."""
    prompt = f"""
    Responda a seguinte pergunta usando, APENAS, o contexto fornecido abaixo.
    Sempre responda em português brasileiro (pt-br).

    Contexto:
    {context}
    
    Pergunta: {query}
    Por favor, limite sua resposta em 150 palavras.
    """
    return get_llm_response(prompt)


# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(
    title="Agentic RAG API",
    description="API para consultas RAG com Agente Inteligente. "
                "Inclui: Pré-processamento de queries, Grading de documentos, "
                "Reescrita de queries, Verificação de alucinação e Controle de acesso.",
    version="5.1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Modelos Pydantic
# ============================================================
class QueryRequest(BaseModel):
    """Modelo para requisição de query."""
    query: str = Field(..., description="Pergunta do usuário", min_length=3)


class AgentProcessInfo(BaseModel):
    """Informações sobre o processamento do agente."""
    original_query: str = ""        # Query original do usuário
    corrected_query: str = ""       # Query após correção ortográfica
    docs_retrieved: int = 0
    docs_after_grading: int = 0
    query_rewrites: int = 0
    hallucination_checks: int = 0
    final_query: str = ""           # Query final usada na busca


class DocumentInfo(BaseModel):
    """Informações de um documento recuperado com metadados."""
    document: str
    collection: str
    distance: float
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    """Modelo para resposta da query."""
    query: str
    response: str
    sources: List[str]
    documents_used: List[DocumentInfo] = []  # Documentos com metadados
    user: str
    access_denied: bool = False
    agent_process: AgentProcessInfo = None


class AccessGroupInfo(BaseModel):
    """Informações de um grupo de acesso."""
    id: str
    display_name: str
    description: str
    accessible: bool


class UserInfo(BaseModel):
    """Modelo para informações do usuário."""
    username: str
    allowed_groups: List[dict]
    allowed_collections: List[str]


# ============================================================
# Endpoints
# ============================================================
@app.get("/")
async def root():
    """Endpoint raiz com informações da API."""
    return {
        "message": "Bem-vindo à API Agentic RAG",
        "docs": "/docs",
        "version": "5.1.0",
        "description": "Sistema RAG com Agente Inteligente (Preprocess, Grading, Rewrite, Hallucination Check)"
    }


@app.get("/me", response_model=UserInfo)
async def get_user_info(user: dict = Depends(get_current_user)):
    """
    Retorna informações do usuário autenticado.
    
    Mostra:
    - username: Nome do usuário
    - allowed_groups: Grupos de acesso permitidos (com display_name)
    - allowed_collections: Collections ChromaDB que o usuário pode consultar
    
    Requer autenticação Basic Auth.
    """
    return UserInfo(
        username=user["username"],
        allowed_groups=user.get("allowed_groups_display", []),
        allowed_collections=user.get("allowed_collections", [])
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    user: dict = Depends(get_current_user)
):
    """
    Endpoint principal para consultas RAG com Agente Inteligente.
    
    FLUXO DO AGENTE:
    1. Usuário faz login (Basic Auth)
    2. PRÉ-PROCESSAMENTO: Corrige erros ortográficos da query
    3. Sistema obtém as collections permitidas ao usuário
    4. Busca documentos nas collections permitidas
    5. GRADING: Avalia relevância de cada documento
    6. Se poucos docs relevantes → REWRITE: Reescreve a query e busca novamente
    7. Gera resposta com os documentos relevantes
    8. HALLUCINATION CHECK: Verifica se a resposta é factual
    9. Se houver alucinação → Regenera a resposta
    
    MAPEAMENTO DE PERMISSÕES:
    - user_a, user_b, user_c: podem consultar "medical_q_n_a" (Dados Médicos Q&A)
    - user_1, user_2, user_3: podem consultar "medical_device_manual" (Manuais de Dispositivos)
    - admin: pode consultar todas as collections
    
    Requer autenticação Basic Auth.
    """
    original_query = request.query
    username = user["username"]
    
    # ============================================================
    # PRÉ-PROCESSAMENTO: Corrigir erros ortográficos da query
    # ============================================================
    corrected_query = preprocess_query(original_query)
    current_query = corrected_query
    
    # Informações do processo do agente
    agent_info = AgentProcessInfo(
        original_query=original_query,
        corrected_query=corrected_query,
        final_query=corrected_query
    )
    
    # 1. Obter as collections que o usuário tem permissão para acessar
    allowed_collections = user.get("allowed_collections", [])
    
    # Obter informações amigáveis das collections para mensagens
    collections_info = get_user_allowed_collections_with_display(username)
    collections_display_names = [c["display_name"] for c in collections_info]
    context_hint = ", ".join(collections_display_names)
    
    # 2. Se o usuário não tem acesso a nenhuma collection
    if not allowed_collections:
        return QueryResponse(
            query=original_query,
            response="Você não tem acesso a nenhuma fonte de conhecimento. "
                     "Entre em contato com o administrador para solicitar permissões.",
            sources=["Sem acesso configurado"],
            user=username,
            access_denied=True,
            agent_process=agent_info
        )
    
    # ============================================================
    # LOOP DO AGENTE: Busca + Grading + Rewrite (se necessário)
    # ============================================================
    relevant_docs = []
    rewrite_count = 0
    
    while rewrite_count <= MAX_QUERY_REWRITES:
        # 3. Buscar APENAS nas collections permitidas ao usuário
        results = search_allowed_collections(
            query=current_query,
            allowed_collections=allowed_collections,
            n_results=5
        )
        
        agent_info.docs_retrieved = len(results)
        
        # Se não encontrou nada, tentar reescrever
        if not results:
            if rewrite_count < MAX_QUERY_REWRITES:
                current_query = rewrite_query(current_query, context_hint)
                rewrite_count += 1
                agent_info.query_rewrites = rewrite_count
                agent_info.final_query = current_query
                continue
            else:
                break
        
        # 4. GRADING: Avaliar relevância de cada documento
        relevant_docs = grade_documents(current_query, results)
        agent_info.docs_after_grading = len(relevant_docs)
        
        # Se encontrou docs relevantes, sair do loop
        if len(relevant_docs) >= 1:
            break
        
        # Se não encontrou docs relevantes, tentar reescrever
        if rewrite_count < MAX_QUERY_REWRITES:
            current_query = rewrite_query(current_query, context_hint)
            rewrite_count += 1
            agent_info.query_rewrites = rewrite_count
            agent_info.final_query = current_query
        else:
            break
    
    # 5. Se não encontrou documentos relevantes após todas tentativas
    if not relevant_docs:
        sources_list = ", ".join(collections_display_names)
        return QueryResponse(
            query=original_query,
            response=f"Não encontrei informações relevantes sobre este tema nas fontes que você tem acesso. "
                     f"Suas fontes disponíveis são: {sources_list}. "
                     f"Tente reformular sua pergunta de forma diferente.",
            sources=collections_display_names,
            user=username,
            access_denied=False,
            agent_process=agent_info
        )
    
    # 6. Preparar contexto com os documentos relevantes (top 3)
    top_docs = relevant_docs[:5]
    context = "\n\n---\n\n".join([doc["document"] for doc in top_docs])
    
    # Identificar as sources utilizadas (collections de onde vieram os docs)
    sources_used = set()
    for doc in top_docs:
        collection_name = doc.get("collection", "unknown")
        for info in collections_info:
            if info["collection_name"] == collection_name:
                sources_used.add(info["display_name"])
                break
    
    # 7. Gerar resposta inicial
    response = generate_answer(original_query, context)
    
    # ============================================================
    # HALLUCINATION CHECK: Verificar se a resposta é factual
    # ============================================================
    hallucination_checks = 0
    max_hallucination_retries = 2
    
    while hallucination_checks < max_hallucination_retries:
        hallucination_checks += 1
        agent_info.hallucination_checks = hallucination_checks
        
        is_factual = check_hallucination(original_query, context, response)
        
        if is_factual:
            # Resposta é factual, podemos retornar
            break
        else:
            # Resposta contém alucinação, regenerar
            response = regenerate_answer(original_query, context, response)
    
    # 8. Preparar documentos com metadados para a resposta
    documents_info = [
        DocumentInfo(
            document=doc["document"][:500] + "..." if len(doc["document"]) > 500 else doc["document"],
            collection=doc.get("collection", "unknown"),
            distance=doc.get("distance", 0.0),
            metadata=doc.get("metadata", {})
        )
        for doc in top_docs
    ]
    
    # 9. Retornar resposta final
    return QueryResponse(
        query=original_query,
        response=response,
        sources=list(sources_used),
        documents_used=documents_info,
        user=username,
        access_denied=False,
        agent_process=agent_info
    )


@app.get("/access-groups")
async def list_access_groups(user: dict = Depends(get_current_user)):
    """
    Lista os grupos de acesso disponíveis e indica quais o usuário pode acessar.
    
    Requer autenticação Basic Auth.
    """
    all_groups = get_access_groups()
    user_groups = user.get("allowed_groups", [])
    
    groups_info = []
    for group_id, group_info in all_groups.items():
        groups_info.append({
            "id": group_id,
            "display_name": group_info.get("display_name", group_id),
            "description": group_info.get("description", ""),
            "accessible": group_id in user_groups
        })
    
    return {
        "user": user["username"],
        "access_groups": groups_info
    }


@app.get("/health")
async def health_check():
    """Endpoint de health check (sem autenticação)."""
    return {"status": "healthy"}


# ============================================================
# Execução direta
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

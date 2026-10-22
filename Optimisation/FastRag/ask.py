import time
import requests
from config import OLLAMA_URL, CHAT_MODEL, MAX_NEW_TOKENS, OLLAMA_KEEP_ALIVE
from retrieval import HybridRetriever

SYSTEM = '''You are an engineering knowledge assistant. Answer only from the supplied passages.
Cite factual claims using [Source: relative/path, chunk N]. Never invent configuration values,
steps, owners, or behavior. If evidence is insufficient, say: "I couldn't find enough evidence
in the indexed documents." If sources conflict, explain the conflict and cite both sources.
Prefer a concise, precise answer over a long generic answer.'''

def answer(question, retriever):
    start = time.perf_counter()
    retrieved = retriever.search(question)
    retrieval_end = time.perf_counter()
    passages = []
    for n, item in enumerate(retrieved['results'], 1):
        meta = item['meta']
        passages.append(f"[PASSAGE {n} | Source: {meta.get('source', 'unknown')} | chunk {meta.get('chunk', '?')}]\n{item['text']}")
    context = '\n\n---\n\n'.join(passages)
    prompt = f'Question:\n{question}\n\nEvidence passages:\n{context}\n\nAnswer from this evidence only and cite sources.'
    response = requests.post(f'{OLLAMA_URL}/api/chat', json={
        'model': CHAT_MODEL,
        'stream': False,
        'keep_alive': OLLAMA_KEEP_ALIVE,
        'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
        'options': {'num_predict': MAX_NEW_TOKENS, 'temperature': 0.1},
    }, timeout=180)
    response.raise_for_status()
    payload = response.json()
    end = time.perf_counter()
    print('\n' + '=' * 72)
    print(payload.get('message', {}).get('content', '').strip())
    print('=' * 72)
    print(f"Retrieved {len(retrieved['results'])} passages from {retrieved['candidate_count']} fused candidates")
    print('Retrieval timings:', {k: round(v, 3) for k, v in retrieved['timings'].items()})
    print(f'LLM generation: {end - retrieval_end:.2f}s | Total: {end - start:.2f}s')
    print('Sources used:')
    for item in retrieved['results']:
        print(f" - {item['meta'].get('source')} (chunk {item['meta'].get('chunk')})")
    stats = payload.get('eval_count'), payload.get('prompt_eval_count')
    if any(stats): print(f'LLM tokens: generated={stats[0]}, prompt={stats[1]}')
    return payload

def main():
    try:
        retriever = HybridRetriever()
    except Exception as exc:
        raise SystemExit(f'Could not open collection. Run `python document_ingest.py` first. Details: {exc}')
    print(f"Ready. Collection has {retriever.collection.count()} chunks. Model: {CHAT_MODEL}")
    print("Ask a question, or type 'exit'.")
    while True:
        q = input('\nQuestion> ').strip()
        if q.lower() in {'exit', 'quit'}: break
        if q:
            try: answer(q, retriever)
            except Exception as exc: print('Request failed:', exc)

if __name__ == '__main__':
    main()

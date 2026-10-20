Here is the current workflow of your Lost Knowledge RAG system, without any AI agents — based on the implementation we've discussed so far.

## 1. Overall system architecture
<img width="491" height="662" alt="image" src="https://github.com/user-attachments/assets/ae3aaa3d-f866-4dd4-9df6-e29791cab7c3" />
<img width="670" height="686" alt="image" src="https://github.com/user-attachments/assets/3cb47a82-389d-4f56-a7a8-8c1d9e377916" />
<img width="363" height="497" alt="image" src="https://github.com/user-attachments/assets/952a684d-36b5-4b82-8d46-2d26f1776287" />


\#chatgpt-mermaid-\_r_hr\_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(237, 237, 237);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-\_r_hr\_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_hr\_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_hr\_ .error-icon{fill:rgb(48, 48, 48);}#chatgpt-mermaid-\_r_hr\_ .error-text{fill:rgb(237, 237, 237);stroke:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_hr\_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-\_r_hr\_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-\_r_hr\_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-\_r_hr\_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-\_r_hr\_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-\_r_hr\_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-\_r_hr\_ .marker{fill:rgb(175, 175, 175);stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_hr\_ .marker.cross{stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_hr\_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-\_r_hr\_ p{margin:0;}#chatgpt-mermaid-\_r_hr\_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_hr\_ .cluster-label text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_hr\_ .cluster-label span{color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_hr\_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-\_r_hr\_ .label text,#chatgpt-mermaid-\_r_hr\_ span{fill:rgb(237, 237, 237);color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_hr\_ .node rect,#chatgpt-mermaid-\_r_hr\_ .node circle,#chatgpt-mermaid-\_r_hr\_ .node ellipse,#chatgpt-mermaid-\_r_hr\_ .node polygon,#chatgpt-mermaid-\_r_hr\_ .node path{fill:rgb(9, 23, 44);stroke:rgb(31, 78, 148);stroke-width:1px;}#chatgpt-mermaid-\_r_hr\_ .rough-node .label text,#chatgpt-mermaid-\_r_hr\_ .node .label text,#chatgpt-mermaid-\_r_hr\_ .image-shape .label,#chatgpt-mermaid-\_r_hr\_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-\_r_hr\_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-\_r_hr\_ .rough-node .label,#chatgpt-mermaid-\_r_hr\_ .node .label,#chatgpt-mermaid-\_r_hr\_ .image-shape .label,#chatgpt-mermaid-\_r_hr\_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-\_r_hr\_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-\_r_hr\_ .root .anchor path{fill:rgb(175, 175, 175)!important;stroke-width:0;stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_hr\_ .arrowheadPath{fill:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_hr\_ .edgePath .path{stroke:rgb(175, 175, 175);stroke-width:1px;}#chatgpt-mermaid-\_r_hr\_ .flowchart-link{stroke:rgb(175, 175, 175);fill:none;}#chatgpt-mermaid-\_r_hr\_ .edgeLabel{background-color:rgb(0, 0, 0);text-align:center;}#chatgpt-mermaid-\_r_hr\_ .edgeLabel p{background-color:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_hr\_ .edgeLabel rect{opacity:0.5;background-color:rgb(0, 0, 0);fill:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_hr\_ .labelBkg{background-color:rgba(0, 0, 0, 0.5);}#chatgpt-mermaid-\_r_hr\_ .cluster rect{fill:rgb(48, 48, 48);stroke:rgba(255, 255, 255, 0.15);stroke-width:1px;}#chatgpt-mermaid-\_r_hr\_ .cluster text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_hr\_ .cluster span{color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_hr\_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(48, 48, 48);border:1px solid rgba(255, 255, 255, 0.15);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-\_r_hr\_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_hr\_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-\_r_hr\_ .icon-shape,#chatgpt-mermaid-\_r_hr\_ .image-shape{background-color:rgb(0, 0, 0);text-align:center;}#chatgpt-mermaid-\_r_hr\_ .icon-shape p,#chatgpt-mermaid-\_r_hr\_ .image-shape p{background-color:rgb(0, 0, 0);padding:2px;}#chatgpt-mermaid-\_r_hr\_ .icon-shape .label rect,#chatgpt-mermaid-\_r_hr\_ .image-shape .label rect{opacity:0.5;background-color:rgb(0, 0, 0);fill:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_hr\_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-\_r_hr\_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-\_r_hr\_ .node .neo-node{stroke:rgb(31, 78, 148);}#chatgpt-mermaid-\_r_hr\_ [data-look="neo"].node rect,#chatgpt-mermaid-\_r_hr\_ [data-look="neo"].cluster rect,#chatgpt-mermaid-\_r_hr\_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-\_r_hr\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_hr\_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-\_r_hr\_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-\_r_hr\_-gradient);stroke-width:1px;}#chatgpt-mermaid-\_r_hr\_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_hr\_ [data-look="neo"].node .neo-line path{stroke:rgb(31, 78, 148);filter:none;}#chatgpt-mermaid-\_r_hr\_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-\_r_hr\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_hr\_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-\_r_hr\_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-\_r_hr\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_hr\_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-\_r_hr\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_hr\_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(153, 206, 255);}#chatgpt-mermaid-\_r_hr\_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(153, 206, 255);}#chatgpt-mermaid-\_r_hr\_ .node tspan[font-weight="normal"],#chatgpt-mermaid-\_r_hr\_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-\_r_hr\_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-width:1px;}#chatgpt-mermaid-\_r_hr\_ .node rect,#chatgpt-mermaid-\_r_hr\_ .node circle,#chatgpt-mermaid-\_r_hr\_ .node ellipse,#chatgpt-mermaid-\_r_hr\_ .node polygon,#chatgpt-mermaid-\_r_hr\_ .node path{fill:rgb(0, 40, 77);stroke:rgba(255, 255, 255, 0.1);stroke-width:1px;}#chatgpt-mermaid-\_r_hr\_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-\_r_hr\_ .node.mermaid-decision .label-container{fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-dasharray:2,2;}#chatgpt-mermaid-\_r_hr\_ .edgePaths .flowchart-link{stroke:rgb(175, 175, 175);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-\_r_hr\_ .marker{fill:rgb(175, 175, 175);stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_hr\_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}3. Answer generation2. Online retrieval1. Offline document ingestionBuild prompt with questionand retrieved contextGemma 3 via OllamaGenerate answerReturn answer to engineerSubmit a questionOptional LLM query analysisVector similarity searchBM25 keyword searchReciprocal Rank Fusion(RRF)Metadata-based rankingSelect top-k chunksEngineering documentsDOCX, issue logs, technicaldocsDocument readerExtract text and split intochunksGenerate embeddingsnomic-embed-textChromaDBVector indexBM25Keyword indexEngineer

## 2. Workflow explained step by step

1. Document ingestion

   Your Python document reader extracts text from engineering documents and divides it into chunks.
2. Index preparation

   `nomic-embed-text` converts chunks into embeddings stored in ChromaDB. The chunks are also indexed using BM25 for keyword-based retrieval.
3. Question processing

   The user submits a question. If enabled, LLM-based query analysis processes it before retrieval.
4. Hybrid retrieval

   Vector search finds semantically similar content, while BM25 finds keyword matches. RRF combines their ranked results.
5. Ranking and context selection

   Metadata-based ranking helps prioritize relevant chunks, and the configured top-k chunks are selected for the prompt.
6. LLM answer generation

   The selected chunks and question are sent to Gemma 3 through Ollama. The model generates an answer using the retrieved context.
7. Response

   The answer is returned to the user. Its reliability depends on retrieval quality, context relevance, and how accurately the LLM interprets the evidence.

## 3. Where each technology fits

| Technology         | Responsibility                                  |
| ------------------ | ----------------------------------------------- |
| Python             | Orchestrates ingestion and query processing     |
| Document reader    | Extracts document text                          |
| `nomic-embed-text` | Generates vector embeddings                     |
| ChromaDB           | Stores and retrieves vectors                    |
| BM25               | Keyword-based search                            |
| RRF                | Combines ranked search results                  |
| Metadata ranking   | Prioritizes retrieved chunks                    |
| Ollama + Gemma 3   | Processes the question and generates the answer |

## 4. Important distinction

Your workflow has two separate phases:

- Offline: documents are read, chunked, embedded, and indexed.
- Online: a question is processed, relevant chunks are retrieved, and the LLM generates an answer.

The LLM does not need to read all 150 documents for every question. It works with the context selected by the retrieval pipeline.

Optional LLM query analysis and LLM reranking are separate features; they should not be treated as mandatory steps. Based on your earlier timing tests, reranking added substantial latency, so keep it disabled unless evaluation demonstrates a worthwhile accuracy improvement.

This is your current RAG architecture. No agent planner, autonomous loop, or agent framework is included.

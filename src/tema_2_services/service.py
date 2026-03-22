import json
import os
import hashlib

from dotenv import load_dotenv
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import faiss

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "data_chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(DATA_DIR, "faiss.index.meta")
USE_MODEL_URL = os.environ.get(
    "USE_MODEL_URL",
    "https://tfhub.dev/google/universal-sentence-encoder/4",
)

WEB_URLS = [u for u in os.environ.get("WEB_URLS", "").split(";") if u]

class RAGAssistant:
    """Asistent cu RAG din surse web si un LLM pentru raspunsuri."""

    def __init__(self) -> None:
        """Initializeaza clientul LLM, embedderul si prompturile."""
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Seteaza GROQ_API_KEY in variabilele de mediu.")

        self.client = OpenAI(
            api_key=self.groq_api_key,
            base_url=os.environ.get("GROQ_BASE_URL"))

        os.makedirs(DATA_DIR, exist_ok=True)
        self.embedder = None

        # ToDo: Adaugat o propozitie de referinta mai specifica pentru domeniul dvs
        self.relevance = self._embed_texts(
            "Intrebari despre servicii IT pentru firme, mentenanta IT, suport tehnic, securitate cibernetica, infrastructura IT, cloud, retele si administrare sisteme",
        )[0]

        # ToDo: Definiti un prompt de sistem mai detaliat pentru a ghida raspunsurile LLM-ului in directia dorita
        self.system_prompt = (
           "Esti un consultant IT senior specializat in servicii pentru companii (MSP - Managed Service Provider). "
    
            "ROLUL TAU:\n"
            "- Oferi consultanta IT orientata spre business, scalabilitate si eficienta.\n"
            "- Recomanzi solutii practice care pot fi implementate in companii reale.\n\n"
    
            "SIGURANTA SI CONFORMITATE (CRITIC):\n"
            "- Nu divulga si nu solicita date sensibile (parole, token-uri, chei API, date personale, informatii confidentiale).\n"
            "- Daca utilizatorul furnizeaza accidental astfel de date, ignora-le si avertizeaza-l.\n"
            "- Nu genera instructiuni care pot compromite securitatea sistemelor (ex: bypass securitate, hacking ilegal).\n"
            "- Nu incuraja activitati ilegale sau neetice.\n"
            "- Respecta bune practici de securitate (principiul least privilege, zero trust, backup, audit).\n\n"
    
            "ACURATETE SI RESPONSABILITATE:\n"
            "- Nu inventa informatii.\n"
            "- Daca nu esti sigur, spune clar si propune cum se poate verifica.\n"
            "- Diferentiaza intre fapte, estimari si opinii.\n\n"
    
            "CONTROLUL RISCULUI:\n"
            "- Cand recomanzi o solutie, mentioneaza riscurile principale.\n"
            "- Evita recomandari care pot duce la pierderi financiare sau vulnerabilitati majore fara avertisment.\n"
            "- Prioritizeaza solutii sigure si stabile in fata celor experimentale.\n\n"
    
            "STILUL DE RASPUNS:\n"
            "- Clar, structurat si orientat pe actiune.\n"
            "- Explica simplu, dar profesionist.\n"
            "- Evita jargonul inutil.\n\n"
    
            "CONTEXT:\n"
            "- Foloseste prioritar contextul oferit de utilizator.\n"
            "- Daca lipsesc informatii importante, pune intrebari.\n\n"
    
            "FORMATUL RASPUNSULUI:\n"
            "- Explicatie clara\n"
            "- Recomandari practice\n"
            "- Riscuri / Atentie (daca este cazul)\n"
            "- Next steps\n\n"
    
            "Scopul tau este sa oferi raspunsuri utile, sigure si aplicabile in mediul de business real."
        )


    def _load_documents_from_web(self) -> list[str]:
        """Incarca si chunked documente de pe site-uri prin WebBaseLoader."""
        if os.path.exists(CHUNKS_JSON_PATH):
            try:
                with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and cached:
                    return cached
            except (OSError, json.JSONDecodeError):
                pass

        all_chunks = []
        for url in WEB_URLS:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                for doc in docs:
                    chunks = self._chunk_text(doc.page_content)
                    all_chunks.extend(chunks)
            except Exception:
                continue

        if all_chunks:
            with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False)

        return all_chunks

    def _send_prompt_to_llm(
        self,
        user_input: str,
        context: str
    ) -> str:
        """Trimite promptul catre LLM si returneaza raspunsul."""

        system_msg = self.system_prompt

        # ToDo: Ajustati acest prompt pentru a se potrivi mai bine cu domeniul dvs si pentru a ghida LLM-ul sa ofere raspunsuri mai relevante si structurate.
        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Intrebare:\n{user_input}\n\n"
                    "Foloseste contextul daca este relevant. "
                    "Daca nu, raspunde din cunostinte generale, dar mentioneaza asta. "
                    "Raspunsul trebuie sa fie clar, structurat si util pentru o companie."
                ),
            },
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model="openai/gpt-oss-20b",
            )
            return response.choices[0].message.content
        except Exception:
            return (
                "Asistent: Nu pot ajunge la modelul de limbaj acum. "
                "Te rog incearca din nou in cateva momente."
            )
        
    def _embed_texts(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        """Genereaza embeddings folosind Universal Sentence Encoder."""
        if isinstance(texts, str):
            texts = [texts]
        if self.embedder is None:
            self.embedder = hub.load(USE_MODEL_URL)
        if callable(self.embedder):
            embeddings = self.embedder(texts)
        else:
            infer = self.embedder.signatures.get("default")
            if infer is None:
                raise ValueError("Model USE nu expune semnatura 'default'.")
            outputs = infer(tf.constant(texts))
            embeddings = outputs.get("default")
            if embeddings is None:
                raise ValueError("Model USE nu a returnat cheia 'default'.")
        return np.asarray(embeddings, dtype="float32")

    def _chunk_text(self, text: str) -> list[str]:
        """Imparte textul in bucati cu RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
        )
        chunks = splitter.split_text(text or "")
        return chunks if chunks else [""]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculeaza similaritatea cosine intre doi vectori."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _build_faiss_index_from_chunks(self, chunks: list[str]) -> faiss.IndexFlatIP:
        """Construieste index FAISS din chunks text si il salveaza pe disc."""
        if not chunks:
            raise ValueError("Lista de chunks este goala.")

        embeddings = self._embed_texts(chunks).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
            f.write(self._compute_chunks_hash(chunks))
        return index

    def _compute_chunks_hash(self, chunks: list[str]) -> str:
        """Hash determinist pentru lista de chunks si model."""
        payload = json.dumps(
            {
                "model": USE_MODEL_URL,
                "chunks": chunks,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_index_hash(self) -> str | None:
        """Incarca hash-ul asociat indexului FAISS."""
        if not os.path.exists(FAISS_META_PATH):
            return None
        try:
            with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def _retrieve_relevant_chunks(self, chunks: list[str], user_query: str, k: int = 5) -> list[str]:
        """Rankeaza chunks folosind FAISS si returneaza top-k relevante."""
        if not chunks:
            return []

        current_hash = self._compute_chunks_hash(chunks)
        stored_hash = self._load_index_hash()

        query_embedding = self._embed_texts(user_query).astype("float32")

        index = None
        if os.path.exists(FAISS_INDEX_PATH) and stored_hash == current_hash:
            try:
                index = faiss.read_index(FAISS_INDEX_PATH)
                if index.ntotal != len(chunks) or index.d != query_embedding.shape[1]:
                    index = None
            except Exception:
                index = None

        if index is None:
            index = self._build_faiss_index_from_chunks(chunks)

        faiss.normalize_L2(query_embedding)

        k = min(k, len(chunks))
        if k == 0:
            return []

        _, indices = index.search(query_embedding, k=k)
        return [chunks[i] for i in indices[0] if i < len(chunks)]

    def calculate_similarity(self, text: str) -> float:
        # ToDo: Ajustati aceasta propozitie de referinta pentru a se potrivi mai bine cu domeniul dvs, astfel incat sa reflecte mai precis ce inseamna "relevant" in contextul aplicatiei dvs.
        """Returneaza similaritatea cu o propozitie de referinta despre servicii IT pentru firme."""
        embedding = self._embed_texts(text.strip())[0]
        return self._cosine_similarity(embedding, self.relevance)

    def is_relevant(self, user_input: str) -> bool:
        # ToDo: Ajustati pragul de similaritate pentru a se potrivi mai bine cu domeniul dvs, astfel incat sa echilibreze corect intre a permite intrebari relevante si a respinge cele irelevante.
        """Verifica daca intrarea utilizatorului e despre servicii IT pentru firme."""
        return self.calculate_similarity(user_input) >= 0.65

    def assistant_response(self, user_message: str) -> str:
        """Directioneaza mesajul utilizatorului catre calea potrivita."""
        if not user_message:
            # ToDo: Ajustati acest mesaj pentru a fi mai specific pentru domeniul dvs, astfel incat sa ghideze utilizatorii sa puna intrebari relevante si sa ofere un exemplu concret.
            return (
                "Pot raspunde doar la intrebari legate de servicii IT pentru firme "
                "(suport IT, infrastructura, securitate, cloud, MSP). "
                "Exemplu: 'Cum pot reduce costurile IT intr-o companie mica?'"
            )

        if not self.is_relevant(user_message):
            # ToDo: Ajustati acest mesaj pentru a fi mai specific pentru domeniul dvs, astfel incat sa ghideze utilizatorii sa puna intrebari relevante si sa ofere un exemplu concret.
            return (
                "Te rog scrie o intrebare legata de servicii IT pentru firme. "
                "Exemplu: 'Ce inseamna un MSP si cum ma ajuta?'"
            )

        chunks = self._load_documents_from_web()
        relevant_chunks = self._retrieve_relevant_chunks(chunks, user_message)
        context = "\n\n".join(relevant_chunks)
        return self._send_prompt_to_llm(user_message, context)

if __name__ == "__main__":
    assistant = RAGAssistant()
    # ToDo: Testati cu intrebari relevante pentru domeniul dvs, precum si cu intrebari irelevante pentru a va asigura ca logica de filtrare functioneaza corect.
    print(assistant.assistant_response("Ce solutii de backup recomanzi pentru o firma mica?"))  # test relevant
    print(assistant.assistant_response("Care este capitala Frantei?"))  # test irelevant

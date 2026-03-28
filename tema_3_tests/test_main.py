import requests
import httpx
import sys
import pytest

# foloseste UTF-8 pentru stdout ca sa evite erori de codare
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"

#ToDo: Adăugați un test pentru endpoint-ul root 
def test_root_endpoint():
    response = requests.get(f"{BASE_URL}/")
    
    assert response.status_code == 200
    data = response.json()
    
    # verificăm că API-ul răspunde corect
    assert isinstance(data, dict)
    assert "message" in data or len(data) > 0

#ToDo: Adăugați un scenariu de testare pentru endpoint-ul /chat/ care să fie evaluat de LLM as a Judge
  def test_chat_valid_request():
    payload = {
        "message": "Ce este un MSP si ce avantaje are?"
    }

    response = requests.post(f"{BASE_URL}/chat/", json=payload)

    assert response.status_code == 200

    data = response.json()

    # verificări structurale (important pentru evaluare LLM)
    assert isinstance(data, dict)
    assert "response" in data

    # verificăm că răspunsul nu e gol
    assert len(data["response"]) > 10

#ToDo: Adăugațu un test negativ pentru endpoint-ul /chat/ care să fie evaluat de LLM as a Judge 
def test_chat_invalid_request():
    payload = {
        "message": ""  # input gol
    }

    response = requests.post(f"{BASE_URL}/chat/", json=payload)

    # poate fi 400 sau 422 în funcție de implementare
    assert response.status_code in [400, 422]

    data = response.json()

    # verificăm că există mesaj de eroare
    assert "detail" in data or "error" in data

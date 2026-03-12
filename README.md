# 🐾 The Underpeople Pool (Пул «Подлюдей»)
**0.5B Micro-Agent for JSON Task Extraction**

### 📖 Идея (Sci-Fi Концепт)
Проект вдохновлен циклом Кордвайнера Смита «Инструменталии человечества». В его мире существует класс «Подлюдей» (Underpeople) — генетически модифицированных существ, которые выполняют всю черновую работу в фоне, пока «Истинные люди» творят искусство.
В мире ИИ нам тоже нужны невидимые "чернорабочие" для сортировки рутины. Использовать GPT-4 для парсинга текста — это как забивать гвозди микроскопом. Нам нужен локальный фоновый демон.

### 🎯 Концепт продукта
Мы взяли **самую крошечную модель в мире** (Qwen 2.5 на 0.5B параметров, весом менее 1 Гигабайта) и отучили её быть собеседником. Её единственная функция в жизни: получать на вход хаотичный, неструктурированный поток мыслей (например, расшифровку голосового сообщения) и мгновенно возвращать строгий, машиночитаемый **JSON** с массивом задач для календаря.

### 🛠 Технический стек и метрики
* **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct`
* **Technique:** SFT (Supervised Fine-Tuning) via Unsloth
* **VRAM Usage:** **Всего 867 Мегабайт!** (Работает даже на Raspberry Pi или старых смартфонах).
* **Adapter Size:** ~16 MB.

### 🧪 Эксперимент (Proof of Concept)
Модель успешно выучила жесткую структуру JSON (без генерации markdown или лишних приветствий), доказав способность SLM-моделей работать как API-эндпоинты. 

**Пример архитектурного ответа:**
```json
{
  "tasks": [
    {"action": "Созвон по дизайну", "person": "Макс", "deadline": "завтра 15:00"},
    {"action": "Сдать отчет по серверу", "person": null, "deadline": "пятница"}
  ]
}

⚠️ Примечание: Данный репозиторий является Proof-of-Concept архитектуры. Текущий адаптер обучен на синтетическом микро-датасете. Для продакшена (чтобы избежать ИИ-галлюцинаций и путаницы в именах) требуется расширение датасета до 1000+ примеров.

#### 3. Скрипт `inference.py` (Для локального запуска)
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Загружаем "Тело" Подчеловека (0.5B)
base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Надеваем "Ошейник" (Наш LoRA адаптер для JSON)
# Укажите путь к сохраненной папке underpeople_lora
model = PeftModel.from_pretrained(model, "underpeople_lora")

def extract_json(messy_text):
    prompt = f"""Ты — системный процесс для извлечения данных (Data Extractor). Твоя задача: прочитать неструктурированный текст и вернуть список задач СТРОГО в формате валидного JSON. Запрещено писать любые другие слова, приветствия или комментарии. Только JSON.\nТекст пользователя:\n{messy_text}\nJSON-результат:\n"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response.split("JSON-результат:\n")[1]

# Тест
text = "Купить хлеб и сказать Диме, чтобы выслал отчет завтра утром."
print(extract_json(text))

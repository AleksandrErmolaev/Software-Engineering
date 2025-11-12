import requests
import json
from datetime import datetime, timedelta

# Конфигурация
OPENWEATHER_API_KEY = "0cea8fdbec26b7c76992a739bd2e3d57"
LLM_API_URL = "http://localhost:11434/api/generate"

system_prompt = """
Ты - полезный погодный ассистент. Твоя задача - давать четкие и вежливые ответы о погоде на основе предоставленных данных.
Не обращайся к пользователю, сразу ответ. Не надо писать откуда получил данные. В ответе упоминай город.
Проанализируй данные о погоде и сгенерируй краткий и дружелюбный ответ на русском языке, который включает:
- Общее описание погоды
- Текущую температуру в градусах Цельсия, округляй до целых
- Относительную влажность воздуха
- Скорость ветра
- Рекомендации по одежде и аксессуарам (подробно)
- Рекомендации куда в такую погоду лучше всего сходить (много вариантов)

Будь точен, ответы развернутые. Не придумывай данные, которых нет в ответе от API. Советуй только то что по погоде подходит.
"""


def get_weather_data(city_name, hours_ahead=0):
    """Функция для получения данных о погоде с OpenWeatherMap"""

    if hours_ahead == 0:
        # Текущая погода
        url = "http://api.openweathermap.org/data/2.5/weather"
    else:
        # Прогноз на 5 дней
        url = "http://api.openweathermap.org/data/2.5/forecast"

    params = {
        'q': city_name,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric',
        'lang': 'ru'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
 
        # Для прогноза находим ближайший временной интервал
        if hours_ahead > 0 and 'list' in data:
            target_time = datetime.now() + timedelta(hours=hours_ahead)
            closest_forecast = None
            min_time_diff = float('inf')

            for forecast in data['list']:
                forecast_time = datetime.fromtimestamp(forecast['dt'])
                time_diff = abs((forecast_time - target_time).total_seconds())

                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_forecast = forecast

            if closest_forecast:
                # Создаем структуру похожую на текущую погоду для единообразия
                weather_info = {
                    'name': data['city']['name'],
                    'weather': closest_forecast['weather'],
                    'main': closest_forecast['main'],
                    'wind': closest_forecast.get('wind', {}),
                    'visibility': closest_forecast.get('visibility', 10000),
                    'dt': closest_forecast['dt'],
                    'forecast_time': closest_forecast['dt_txt']
                }
                return weather_info
            else:
                return {"error": "Не удалось найти прогноз для указанного времени"}

        return data

    except requests.exceptions.RequestException as e:
        return {"error": f"Ошибка при запросе к погодному API: {e}"}
    except Exception as e:
        return {"error": f"Неожиданная ошибка: {e}"}


def ask_llm(weather_data):
    """Функция для запроса к LLM через HTTP API"""

    # Формируем тело запроса для Ollama
    payload = {
        "model": "llama3.1:8b",
        "prompt": f"{system_prompt}\n\nДанные о погоде: {json.dumps(weather_data, ensure_ascii=False)}",
        "stream": False
    }

    try:
        response = requests.post(LLM_API_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "Не удалось получить ответ от модели")

    except requests.exceptions.RequestException as e:
        return f"Ошибка при обращении к LLM API: {e}"
    except Exception as e:
        return f"Ошибка при разборе ответа от LLM: {e}"


if __name__ == "__main__":
    print("=== Погодный помощник ===")

    while True:
        city = input("\nВведите город: ").strip()

        if city.lower() in ['выход', 'exit', 'quit']:
            print("До свидания!")
            break

        if not city:
            print("Пожалуйста, введите название города. Или 'выход', 'exit', 'quit' для завершения программы")
            continue

        time_input = input("Через сколько часов прогноз? (0 для текущей погоды): ").strip()

        try:
            hours_ahead = int(time_input) if time_input else 0
            if hours_ahead < 0:
                print("Время не может быть отрицательным. Используется текущая погода.")
                hours_ahead = 0
        except ValueError:
            print("Неверный формат времени. Используется текущая погода.")
            hours_ahead = 0

        # Ограничим прогноз 5 днями (120 часов) из-за ограничений бесплатного API
        if hours_ahead > 120:
            print("Прогноз доступен только на 5 дней (120 часов). Установлено 120 часов.")
            hours_ahead = 120

        print(f"\nПолучаем данные о погоде для {city}..." + 
              (f" через {hours_ahead} часов" if hours_ahead > 0 else ""))

        weather_data = get_weather_data(city, hours_ahead)

        if "error" in weather_data:
            print(f"❌ Ошибка: {weather_data['error']}")
            continue

        if "cod" in weather_data and weather_data["cod"] != 200:
            print(f"❌ Ошибка API: {weather_data.get('message', 'Неизвестная ошибка')}")
            continue

        print("🤔 Формируем ответ...")
        final_answer = ask_llm(weather_data)

        print("\n" + "="*50)
        print("🌤️  ПОГОДНЫЙ ПРОГНОЗ")
        print("="*50)
        print(final_answer)
        print("="*50)

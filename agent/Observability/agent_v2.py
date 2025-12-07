import os
import sys
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan

trace.set_tracer_provider(TracerProvider(
    resource=Resource.create({
        "service.name": "weather-ai-agent",
        "service.version": "1.0.0",
        "deployment.environment": "local"
    })
))

logger, log_file = None, None

def format_nanoseconds(ns):
    """Конвертирует наносекунды в читаемую строку datetime"""
    if ns is None:
        return None
    try:
        seconds = ns / 1_000_000_000
        dt = datetime.fromtimestamp(seconds)
        return dt.isoformat()
    except (ValueError, OSError):
        return str(ns)

class JsonSpanExporter:
    """Экспортер для записи трейсов в JSON файл"""
    def __init__(self):
        if not os.path.exists('traces'):
            os.makedirs('traces')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.file_path = f'traces/trace_{timestamp}.json'
        self.spans = []
        
        logger.info(f"📝 JSON трейсы будут записаны в: {self.file_path}")
    
    def export(self, batch: List[ReadableSpan]):
        """Экспорт спанов в JSON файл"""
        try:
            for span in batch:
                span_dict = self._span_to_dict(span)
                self.spans.append(span_dict)
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "export_time": datetime.now().isoformat(),
                    "total_spans": len(self.spans),
                    "spans": self.spans
                }, f, ensure_ascii=False, indent=2, default=str)
            
            logger.debug(f"✅ Экспортировано {len(batch)} спанов в {self.file_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка экспорта трейсов: {e}")
            return False
    
    def _span_to_dict(self, span: ReadableSpan) -> Dict[str, Any]:
        """Конвертация спана в словарь с читаемым форматом"""
        try:
            span_dict = {
                "name": span.name,
                "context": {
                    "trace_id": format(span.context.trace_id, '032x'),
                    "span_id": format(span.context.span_id, '016x'),
                    "trace_flags": span.context.trace_flags,
                    "is_remote": span.context.is_remote
                } if hasattr(span, 'context') else {},
                "parent_id": format(span.parent.span_id, '016x') if span.parent else None,
                "start_time": format_nanoseconds(span.start_time),
                "end_time": format_nanoseconds(span.end_time),
                "attributes": dict(span.attributes) if span.attributes else {},
                "events": [
                    {
                        "name": event.name,
                        "timestamp": format_nanoseconds(event.timestamp),
                        "attributes": dict(event.attributes) if event.attributes else {}
                    }
                    for event in span.events
                ] if span.events else [],
                "status": {
                    "status_code": span.status.status_code.name,
                    "description": span.status.description
                } if span.status else {},
                "kind": span.kind.name if span.kind else "INTERNAL",
                "resource": dict(span.resource.attributes) if span.resource else {}
            }
            
            if span_dict["attributes"]:
                span_dict["attributes"] = self._decode_dict(span_dict["attributes"])
            
            return span_dict
        except Exception as e:
            logger.error(f"Ошибка конвертации спана: {e}")
            return {"error": str(e), "span_name": span.name}
    
    def _decode_dict(self, data):
        """Рекурсивно декодирует Unicode строки в словаре"""
        if isinstance(data, dict):
            return {k: self._decode_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._decode_dict(item) for item in data]
        elif isinstance(data, str):
            return data
        elif isinstance(data, bytes):
            return data.decode('utf-8', errors='ignore')
        else:
            return data
    
    def shutdown(self):
        """Завершение работы экспортера"""
        if hasattr(self, 'spans'):
            logger.info(f"✅ Трейсы сохранены в {self.file_path}")
            logger.info(f"📊 Всего спанов: {len(self.spans)}")

def setup_logging():
    """Настройка структурированного логирования"""
    global logger, log_file
    
    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_filename = f"logs/weather_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("WeatherAgent")
    logger.info("Агент запущен")
    logger.info(f"Логи: {log_filename}")
    
    return logger, log_filename

logger, log_file = setup_logging()

json_exporter = JsonSpanExporter()
trace.get_tracer_provider().add_span_processor(
    SimpleSpanProcessor(json_exporter)
)

tracer = trace.get_tracer(__name__)

OPENWEATHER_API_KEY = "0cea8fdbec26b7c76992a739bd2e3d57"
LLM_API_URL = "http://localhost:11434/api/generate"

PROMPT_VERSION = "1.0"
system_prompt = f"""Версия промпта: {PROMPT_VERSION}
Ты - полезный погодный ассистент. Твоя задача - давать четкие и вежливые ответы о погоде на основе предоставленных данных.
Не обращайся к пользователю, сразу ответ. Не надо писать откуда получил данные. В ответе упоминай город.
Проанализируй данные о погоде и сгенерируй краткий и дружелюбный ответ на русском языке, который включает:
- Общее описание погоды
- Текущую температуру в градусах Цельсия, округляй до целых
- Относительную влажность воздуха
- Скорость ветра
- Рекомендации по одежде и аксессуарах (подробно)
- Рекомендации куда в такую погоду лучше всего сходить (много вариантов)

Будь точен, ответы развернутые. Не придумывай данные, которых нет в ответе от API. Советуй только то что по погоде подходит.
"""

class MetricsCollector:
    """Сбор метрик"""
    def __init__(self):
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'weather_api_calls': 0,
            'llm_api_calls': 0,
            'total_tokens': 0,
            'errors': {},
            'latencies': {
                'weather_api': [],
                'llm_api': [],
                'total_request': []
            },
            'start_time': datetime.now()
        }
    
    def record_request(self, success=True):
        self.metrics['requests_total'] += 1
        if success:
            self.metrics['requests_success'] += 1
        else:
            self.metrics['requests_failed'] += 1
    
    def record_api_call(self, api_name: str):
        if api_name == 'weather':
            self.metrics['weather_api_calls'] += 1
        elif api_name == 'llm':
            self.metrics['llm_api_calls'] += 1
    
    def record_latency(self, operation: str, latency_ms: float):
        if operation in self.metrics['latencies']:
            self.metrics['latencies'][operation].append(latency_ms)
    
    def record_tokens(self, tokens: int):
        self.metrics['total_tokens'] += tokens
    
    def record_error(self, error_type: str):
        self.metrics['errors'][error_type] = self.metrics['errors'].get(error_type, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Получение сводки метрик"""
        uptime = datetime.now() - self.metrics['start_time']
        
        avg_latencies = {}
        for op, latencies in self.metrics['latencies'].items():
            if latencies:
                avg_latencies[op] = sum(latencies) / len(latencies)
            else:
                avg_latencies[op] = 0
        
        success_rate = 0
        if self.metrics['requests_total'] > 0:
            success_rate = (self.metrics['requests_success'] / self.metrics['requests_total']) * 100
        
        return {
            'uptime': str(uptime),
            'requests_total': self.metrics['requests_total'],
            'requests_success': self.metrics['requests_success'],
            'requests_failed': self.metrics['requests_failed'],
            'success_rate': round(success_rate, 2),
            'weather_api_calls': self.metrics['weather_api_calls'],
            'llm_api_calls': self.metrics['llm_api_calls'],
            'total_tokens': self.metrics['total_tokens'],
            'avg_latencies': avg_latencies,
            'errors': self.metrics['errors'],
            'prompt_version': PROMPT_VERSION
        }


metrics = MetricsCollector()


class MetricsHandler(BaseHTTPRequestHandler):
    """Обработчик для предоставления метрик в формате Prometheus"""
    
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            summary = metrics.get_summary()
            
            metrics_text = f"""
# HELP weather_agent_requests_total Total number of requests
# TYPE weather_agent_requests_total counter
weather_agent_requests_total {summary['requests_total']}

# HELP weather_agent_requests_success Successful requests  
# TYPE weather_agent_requests_success counter
weather_agent_requests_success {summary['requests_success']}

# HELP weather_agent_weather_api_calls Weather API calls
# TYPE weather_agent_weather_api_calls counter
weather_agent_weather_api_calls {summary['weather_api_calls']}

# HELP weather_agent_llm_api_calls LLM API calls
# TYPE weather_agent_llm_api_calls counter
weather_agent_llm_api_calls {summary['llm_api_calls']}

# HELP weather_agent_total_tokens Total tokens used
# TYPE weather_agent_total_tokens counter
weather_agent_total_tokens {summary['total_tokens']}

# HELP weather_agent_success_rate Success rate percentage
# TYPE weather_agent_success_rate gauge
weather_agent_success_rate {summary['success_rate']}
"""
            self.wfile.write(metrics_text.encode())
        else:
            self.send_response(404)
            self.end_headers()


def start_metrics_server(port=8000):
    """Запуск HTTP сервера для метрик"""
    server = HTTPServer(('localhost', port), MetricsHandler)
    logger.info(f"📊 Сервер метрик запущен на http://localhost:{port}/metrics")
    server.serve_forever()


def get_weather_data(city_name: str, hours_ahead: int = 0) -> Dict[str, Any]:
    """
    Получение данных о погоде с трейсингом
    """
    with tracer.start_as_current_span("get_weather_data") as span:
        span.set_attributes({
            "city": city_name,
            "hours_ahead": hours_ahead,
            "api": "openweathermap",
            "prompt_version": PROMPT_VERSION
        })
        
        logger.info(f"Запрос погоды: {city_name}, прогноз на {hours_ahead}ч")
        
        try:
            start_time = time.time()
            
            if hours_ahead == 0:
                url = "http://api.openweathermap.org/data/2.5/weather"
            else:
                url = "http://api.openweathermap.org/data/2.5/forecast"

            params = {
                'q': city_name,
                'appid': OPENWEATHER_API_KEY,
                'units': 'metric',
                'lang': 'ru'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_latency('weather_api', latency_ms)
            metrics.record_api_call('weather')
            
            data = response.json()
            
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
                    weather_info = {
                        'name': data['city']['name'],
                        'weather': closest_forecast['weather'],
                        'main': closest_forecast['main'],
                        'wind': closest_forecast.get('wind', {}),
                        'visibility': closest_forecast.get('visibility', 10000),
                        'dt': closest_forecast['dt'],
                        'forecast_time': closest_forecast['dt_txt'],
                        'is_forecast': True
                    }
                    
                    span.add_event("forecast_retrieved", {
                        "temperature": closest_forecast['main']['temp'],
                        "condition": closest_forecast['weather'][0]['description']
                    })
                    
                    logger.info(f"Прогноз получен: {weather_info['name']}, температура={weather_info['main']['temp']}°C")
                    return weather_info
                else:
                    metrics.record_error("forecast_not_found")
                    logger.warning(f"Прогноз не найден для {city_name} через {hours_ahead}ч")
                    return {"error": "Не удалось найти прогноз для указанного времени"}

            if 'main' in data:
                span.add_event("weather_retrieved", {
                    "temperature": data['main']['temp'],
                    "humidity": data['main']['humidity']
                })
                
                logger.info(f"Город: {data.get('name', 'unknown')}, температура={data['main']['temp']}°C")
                return data
            
            return {"error": "Неверный формат данных от API"}

        except requests.exceptions.Timeout:
            error_msg = f"Таймаут при запросе погоды: {city_name}"
            logger.error(error_msg)
            span.record_exception(Exception(error_msg))
            metrics.record_error("weather_api_timeout")
            return {"error": error_msg}
        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка API погоды: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            metrics.record_error("weather_api_error")
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Неожиданная ошибка: {e}"
            logger.error(error_msg, exc_info=True)
            span.record_exception(e)
            metrics.record_error("unexpected_error")
            return {"error": error_msg}


def ask_llm(weather_data: Dict[str, Any]) -> str:
    """
    Запрос к LLM с трейсингом и сбором LLM-специфичных метрик
    """
    with tracer.start_as_current_span("ask_llm") as span:
        span.set_attributes({
            "llm.model": "llama3.1:8b",
            "llm.provider": "ollama",
            "prompt.version": PROMPT_VERSION,
            "weather.city": weather_data.get('name', 'unknown')
        })
        
        logger.info(f"Запрос к LLM: город={weather_data.get('name', 'unknown')}")
        
        try:
            prompt_text = f"{system_prompt}\n\nДанные о погоде: {json.dumps(weather_data, ensure_ascii=False)}"
            
            payload = {
                "model": "llama3.1:8b",
                "prompt": prompt_text,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 512,
                    "top_p": 0.9
                }
            }
            
            span.add_event("prompt_sent", {
                "prompt_length": len(prompt_text),
                "prompt_version": PROMPT_VERSION
            })
            
            start_time = time.time()
            
            response = requests.post(LLM_API_URL, json=payload)
            
            response.raise_for_status()
            
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_latency('llm_api', latency_ms)
            metrics.record_api_call('llm')
            
            result = response.json()
            response_text = result.get("response", "")
            
            prompt_tokens = result.get("prompt_eval_count")
            completion_tokens = result.get("eval_count")
            
            if prompt_tokens and completion_tokens:
                total_tokens = prompt_tokens + completion_tokens
                metrics.record_tokens(total_tokens)
                
                span.set_attributes({
                    "llm.tokens.prompt": prompt_tokens,
                    "llm.tokens.completion": completion_tokens,
                    "llm.tokens.total": total_tokens
                })
                
                span.add_event("tokens_used", {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                })
                
                logger.info(f"Токены использованы: prompt={prompt_tokens}, completion={completion_tokens}")
            
            span.add_event("response_received", {
                "response_length": len(response_text),
                "latency_ms": latency_ms
            })
            
            logger.info(f"LLM ответ: {len(response_text)} символов, время={latency_ms:.0f}мс")
            
            return response_text

        except requests.exceptions.ConnectionError:
            error_msg = "Не удалось подключиться к Ollama. Убедитесь, что Ollama запущен."
            logger.error(error_msg)
            span.record_exception(Exception(error_msg))
            metrics.record_error("llm_connection_error")
            return f"⚠️ {error_msg}"
        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка подключения к LLM: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            metrics.record_error("llm_api_error")
            return f"⚠️ {error_msg}"
        except Exception as e:
            error_msg = f"Ошибка обработки LLM-ответа: {e}"
            logger.error(error_msg, exc_info=True)
            span.record_exception(e)
            metrics.record_error("llm_processing_error")
            return f"⚠️ {error_msg}"


def process_weather_request(city: str, hours_ahead: int = 0) -> Dict[str, Any]:
    """
    Обработка полного запроса с трейсингом всей цепочки
    """
    with tracer.start_as_current_span("process_weather_request") as span:
        span.set_attributes({
            "request.city": city,
            "request.hours_ahead": hours_ahead,
            "request.prompt_version": PROMPT_VERSION
        })
        
        logger.info(f"Начало обработки запроса: {city}, прогноз на {hours_ahead}ч")
        
        try:
            start_time = time.time()
            
            weather_data = get_weather_data(city, hours_ahead)
            
            if "error" in weather_data:
                span.set_attribute("request.success", False)
                span.add_event("weather_api_failed", {"error": weather_data["error"]})
                metrics.record_request(success=False)
                metrics.record_error("weather_data_error")
                return {
                    "success": False,
                    "error": weather_data["error"],
                    "city": city
                }
            
            llm_response = ask_llm(weather_data)
            
            total_latency = (time.time() - start_time) * 1000
            metrics.record_latency('total_request', total_latency)
            
            if "⚠️" in llm_response:
                span.set_attribute("request.success", False)
                metrics.record_request(success=False)
                result = {
                    "success": False,
                    "city": city,
                    "response": llm_response,
                    "latency_ms": total_latency
                }
            else:
                span.set_attribute("request.success", True)
                span.add_event("request_completed", {
                    "total_latency_ms": total_latency,
                    "response_length": len(llm_response)
                })
                metrics.record_request(success=True)
                result = {
                    "success": True,
                    "city": city,
                    "response": llm_response,
                    "latency_ms": total_latency,
                    "prompt_version": PROMPT_VERSION
                }
            
            span.set_attribute("request.latency_ms", total_latency)
            logger.info(f"Запрос завершен: успех={result['success']}, время={total_latency:.0f}мс")
            
            return result
            
        except Exception as e:
            error_msg = f"Ошибка обработки запроса: {e}"
            logger.error(error_msg, exc_info=True)
            span.record_exception(e)
            span.set_attribute("request.success", False)
            metrics.record_request(success=False)
            metrics.record_error("request_processing_error")
            return {
                "success": False,
                "error": error_msg,
                "city": city
            }


def display_dashboard():
    """Отображение дашборда в консоли"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    metrics_summary = metrics.get_summary()
    
    print("=" * 80)
    print("🌤️  ПОГОДНЫЙ ИИ-АГЕНТ С ПОЛНОЙ СИСТЕМОЙ OBSERVABILITY")
    print("=" * 80)
    print(f"📅 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Метрики: http://localhost:8000/metrics")
    print(f"📁 Логи: {log_file}")
    print(f"📝 Трейсы: {json_exporter.file_path}")
    print(f"🔧 Версия промпта: {PROMPT_VERSION}")
    print(f"📈 Запросов: {metrics_summary['requests_total']}")
    print(f"✅ Успешно: {metrics_summary['requests_success']}")
    print(f"❌ Ошибок: {metrics_summary['requests_failed']}")
    print(f"📊 Успешность: {metrics_summary['success_rate']}%")
    print("-" * 80)


def main():
    metrics_thread = threading.Thread(target=start_metrics_server, daemon=True)
    metrics_thread.start()
    
    print("   Все запросы автоматически отслеживаются.")
    print("   Метрики доступны на http://localhost:8000/metrics")
    print("\n" + "=" * 80)
    
    time.sleep(1)  # Даю время серверу метрик запуститься
    
    while True:
        display_dashboard()
        
        print(f"\n📋 Новый запрос")
        print("-" * 40)
        
        city = input("🏙️  Введите город (или 'выход' для завершения): ").strip()

        if city.lower() in ['выход', 'exit', 'quit', 'q']:
            print("\n" + "=" * 80)
            print("📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
            summary = metrics.get_summary()
            for key, value in summary.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for k, v in value.items():
                        if isinstance(v, dict):
                            print(f"     - {k}:")
                            for k2, v2 in v.items():
                                print(f"       * {k2}: {v2}")
                        else:
                            print(f"     - {k}: {v}")
                else:
                    print(f"   {key}: {value}")
            print(f"\n📁 Логи сохранены в: {log_file}")
            print(f"📝 Трейсы сохранены в: {json_exporter.file_path}")
            print("👋 До свидания!")
            print("=" * 80)
            break

        if not city:
            print("⚠️  Пожалуйста, введите название города.")
            continue

        time_input = input("⏰ Через сколько часов прогноз? (0 для текущей погоды): ").strip()

        try:
            hours_ahead = int(time_input) if time_input else 0
            if hours_ahead < 0:
                print("⚠️  Время не может быть отрицательным. Используется текущая погода.")
                hours_ahead = 0
        except ValueError:
            print("⚠️  Неверный формат времени. Используется текущая погода.")
            hours_ahead = 0

        if hours_ahead > 120:
            print("⚠️  Прогноз доступен только на 5 дней (120 часов). Установлено 120 часов.")
            hours_ahead = 120

        print(f"\n🔍 Обрабатываю запрос для {city}...")
        
        result = process_weather_request(city, hours_ahead)
        
        print("\n" + "=" * 80)
        print("🌤️  ПОГОДНЫЙ ПРОГНОЗ")
        print("=" * 80)
        
        if result["success"]:
            response = result["response"]
            print(response)
            
            print("\n" + "=" * 80)
            print(f"📊 Статистика запроса:")
            print(f"   • Город: {result['city']}")
            print(f"   • Время обработки: {result.get('latency_ms', 0):.0f}мс")
            print(f"   • Длина ответа: {len(response)} символов")
            print(f"   • Версия промпта: {result.get('prompt_version', 'N/A')}")
        else:
            print(f"❌ Ошибка: {result.get('error', result.get('response', 'Неизвестная ошибка'))}")
            print("=" * 80)
        
        print("📈 Все метрики доступны на http://localhost:8000/metrics")
        
        input("\n↵ Нажмите Enter для следующего запроса...")


if __name__ == "__main__":
    try:
        try:
            test_response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if test_response.status_code == 200:
                logger.info("✅ Ollama доступен")
            else:
                logger.warning("⚠️  Ollama не отвечает должным образом")
        except:
            logger.warning("⚠️  Ollama не запущен. LLM-функциональность недоступна.")
            print("⚠️  ВНИМАНИЕ: Ollama не запущен!")
            print("   Запустите Ollama в отдельном терминале перед использованием LLM")
            print("   Команда: ollama serve")
        
        main()
        
    except KeyboardInterrupt:
        print("\n\n👋 Программа завершена по запросу пользователя")
        json_exporter.shutdown()
        logger.info("Программа завершена по запросу пользователя")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        print(f"\n❌ Критическая ошибка: {e}")
        print("Проверьте файл логов для деталей.")
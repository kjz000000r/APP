# -*- coding: utf-8 -*-
"""
Проверка подлинности данных от Telegram Mini App
КРИТИЧЕСКИ ВАЖНО для безопасности!
"""
import os
import hmac
import hashlib
import json
from urllib.parse import parse_qsl, unquote
from fastapi import HTTPException, Header
from typing import Optional

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

def verify_telegram_webapp_data(init_data: str) -> dict:
    """
    Проверяет что данные действительно от Telegram
    
    Алгоритм из документации Telegram:
    https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app
    
    Args:
        init_data: строка initData от Telegram.WebApp
    
    Returns:
        dict с данными пользователя
    
    Raises:
        HTTPException: если данные невалидны
    """
    try:
        # Парсим данные
        parsed = dict(parse_qsl(init_data, keep_blank_values=True))
        
        # Извлекаем hash
        received_hash = parsed.pop('hash', None)
        if not received_hash:
            raise HTTPException(status_code=401, detail="Missing hash")
        
        # Сортируем параметры и создаём data_check_string
        data_check_array = [f"{k}={v}" for k, v in sorted(parsed.items())]
        data_check_string = '\n'.join(data_check_array)
        
        # Создаём secret_key
        secret_key = hmac.new(
            key=b"WebAppData",
            msg=BOT_TOKEN.encode(),
            digestmod=hashlib.sha256
        ).digest()
        
        # Вычисляем hash
        calculated_hash = hmac.new(
            key=secret_key,
            msg=data_check_string.encode(),
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Проверяем hash
        if calculated_hash != received_hash:
            raise HTTPException(status_code=401, detail="Invalid hash")
        
        # Парсим user данные
        user_data = json.loads(unquote(parsed.get('user', '{}')))
        
        return {
            'user_id': user_data.get('id'),
            'username': user_data.get('username'),
            'first_name': user_data.get('first_name'),
            'last_name': user_data.get('last_name'),
            'auth_date': parsed.get('auth_date'),
            'raw_data': parsed
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=401, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Validation error: {str(e)}")

async def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Dependency для FastAPI - извлекает текущего пользователя
    
    Использование:
        @app.get("/api/profile")
        async def get_profile(user = Depends(get_current_user)):
            user_id = user['user_id']
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    # Формат: "tma INIT_DATA_STRING"
    if not authorization.startswith("tma "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    
    init_data = authorization[4:]  # Убираем "tma "
    return verify_telegram_webapp_data(init_data)
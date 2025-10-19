# -*- coding: utf-8 -*-
"""
Подключение к PostgreSQL - используем ту же БД что и в боте
"""
import os
import psycopg
from psycopg.rows import dict_row
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

POSTGRES_DSN = os.getenv("POSTGRES_DSN")

@contextmanager
def get_db():
    """
    Контекстный менеджер для работы с БД
    Автоматически закрывает соединение
    """
    conn = psycopg.connect(POSTGRES_DSN, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()

def execute_query(query: str, params: tuple = ()):
    """
    Выполнить SELECT запрос и вернуть результаты
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

def execute_update(query: str, params: tuple = ()):
    """
    Выполнить INSERT/UPDATE/DELETE запрос
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            conn.commit()
            return cur.rowcount
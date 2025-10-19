# -*- coding: utf-8 -*-
import os
import re
import logging
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

# Загрузить .env до чтения переменных
load_dotenv(override=True)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def _shorten(s: str, n: int = 2000) -> str:
    """Сокращает строку для логов"""
    s = str(s)
    return s if len(s) <= n else s[:n] + "...(truncated)"

def _mask_dsn(dsn: str) -> str:
    """Маскирует пароль в DSN для безопасного логирования"""
    if not dsn:
        return ""
    return re.sub(r'(:)[^:@/]+(@)', r':***@', dsn)

# Получаем DSN из переменной окружения
DSN = os.getenv("POSTGRES_DSN")
if not DSN:
    raise RuntimeError("POSTGRES_DSN is not set. Put it into .env or export it before starting the bot.")

# Создаём пул соединений с dict_row для совместимости с sqlite
pool = ConnectionPool(
    DSN,
    min_size=1,
    max_size=10,
    timeout=30,  # ожидание свободного коннекта из пула
    kwargs={"row_factory": dict_row}
)

# Диагностика при старте
print(f"[db_pg] Using DSN: {_mask_dsn(DSN)}")
try:
    with pool.connection() as _c:
        with _c.cursor() as _cur:
            _cur.execute("SELECT 1")
    print("[db_pg] DB ping OK")
except Exception as e:
    print("[db_pg] DB ping FAILED:", e)
    raise


class CursorProxy:
    """
    Имитация sqlite-курсора для SELECT/CTE/RETURNING запросов.
    Автоматически закрывает курсор и возвращает соединение в пул после чтения.
    """
    def __init__(self, conn: psycopg.Connection, cur: psycopg.Cursor):
        self._conn = conn
        self._cur = cur

    def fetchone(self):
        try:
            return self._cur.fetchone()
        finally:
            try:
                self._conn.commit()
            finally:
                self._cur.close()
                pool.putconn(self._conn)

    def fetchall(self):
        try:
            return self._cur.fetchall()
        finally:
            try:
                self._conn.commit()
            finally:
                self._cur.close()
                pool.putconn(self._conn)


class ExecResult:
    """
    Результат для команд без выборки (INSERT/UPDATE/DELETE без RETURNING).
    Возвращает rowcount и заглушки для fetch* методов.
    """
    def __init__(self, rowcount: int):
        self.rowcount = rowcount

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class DB:
    """
    Обёртка для работы с PostgreSQL через пул соединений.
    Автоматически определяет тип запроса и возвращает соответствующий результат.
    """
    
    def execute(self, sql: str, params=()):
        """
        Выполняет SQL запрос:
        - Для SELECT/CTE/RETURNING возвращает CursorProxy с методами fetchone()/fetchall()
        - Для INSERT/UPDATE/DELETE без RETURNING возвращает ExecResult с rowcount
        """
        sql_norm = sql.lstrip().lower()
        is_select_like = sql_norm.startswith(("select", "with")) or " returning " in sql_norm

        if is_select_like:
            # SELECT-подобный запрос
            conn = pool.getconn()
            cur = conn.cursor()
            try:
                cur.execute(sql, params)
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                pool.putconn(conn)
                logging.exception("SQL error on SELECT-like:\nSQL: %s\nPARAMS: %r", _shorten(sql), params)
                raise
            return CursorProxy(conn, cur)
        else:
            # DML запрос (INSERT/UPDATE/DELETE без RETURNING)
            with pool.connection() as conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                        affected = cur.rowcount
                    conn.commit()
                    return ExecResult(affected)
                except Exception as e:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    logging.exception("SQL error on DML:\nSQL: %s\nPARAMS: %r", _shorten(sql), params)
                    raise

    def executemany(self, sql: str, seq_of_params):
        """
        Выполнение батча запросов без возврата результатов.
        """
        with pool.connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.executemany(sql, seq_of_params)
                conn.commit()
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                logging.exception("SQL error on executemany:\nSQL: %s", _shorten(sql))
                raise

    def commit(self):
        """
        Совместимость с sqlite API.
        В нашей реализации коммиты происходят автоматически.
        """
        pass


# Создаём глобальный экземпляр
db = DB()
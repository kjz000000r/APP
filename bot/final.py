# -*- coding: utf-8 -*-
"""
NutriCoach — улучшенная версия бота с расширенными функциями
ИСПРАВЛЕННАЯ ВЕРСИЯ - все критические баги устранены
"""
import os
import re
import io
import json
import html
import asyncio
import tempfile
import subprocess
import logging
import datetime as dt
import secrets
import psycopg
from psycopg.errors import UniqueViolation
from typing import Optional, List, Tuple, Dict, Mapping
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import pytesseract
import pdfplumber
import statistics
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    LabeledPrice, ReplyKeyboardMarkup, InputFile
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, ConversationHandler, PreCheckoutQueryHandler,
    ContextTypes, filters, JobQueue
)
from html import escape
from telegram import MenuButtonWebApp, WebAppInfo
from telegram import WebAppData


# --------------------
# Логи
# --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("nutri-bot")

# --------------------
# Конфигурация (.env)
# --------------------
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PAYMENT_TOKEN = os.getenv("PAYMENT_PROVIDER_TOKEN", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
WELCOME_IMAGE = os.getenv("WELCOME_IMAGE_URL", "https://imgur.com/a/WU18q0F")
ADMIN_USERNAME = (os.getenv("ADMIN_USERNAME", "") or "").lower()
CURRENCY = os.getenv("CURRENCY", "RUB")
TRIAL_HOURS = int(os.getenv("TRIAL_HOURS", "24"))
APP_TITLE = os.getenv("APP_TITLE", "NutriCoach Bot")
APP_REFERER = os.getenv("APP_REFERER", "https://example.com")
REF_BONUS_DAYS = int(os.getenv("REF_BONUS_DAYS", "7"))
PROMO_MAX_DAYS = int(os.getenv("PROMO_MAX_DAYS", "365"))
OCR_LANG = os.getenv("OCR_LANG", "rus+eng")
DB_PATH = os.getenv("DB_PATH", "subs.db")
MINI_APP_URL = os.getenv("MINI_APP_URL", "https://your-domain.com")

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is required in .env")

# --------------------
# Conversation states
# --------------------
PLAN_AGE, PLAN_SEX, PLAN_WEIGHT, PLAN_HEIGHT, PLAN_ACTIVITY, PLAN_GOAL, PLAN_PREFS, PLAN_RESTR = range(8)

# --------------------
# Flags in user_data
# --------------------
EXPECT_LABS = "EXPECT_LABS"
EXPECT_RECIPE = "EXPECT_RECIPE"
EXPECT_QUESTION = "EXPECT_QUESTION"
EXPECT_MEAL = "EXPECT_MEAL"
EXPECT_WEIGHT = "EXPECT_WEIGHT"

# --------------------
# Prices / plans
# --------------------
PLANS = [
    {"key": "sub_7", "title": "Подписка 7 дней", "days": 7, "price_minor": 10000},
    {"key": "sub_30", "title": "Подписка 30 дней", "days": 30, "price_minor": 35000},
    {"key": "sub_90", "title": "Подписка 90 дней", "days": 90, "price_minor": 80000},
    {"key": "sub_365", "title": "Подписка 365 дней", "days": 365, "price_minor": 250000},
]

LABS_ONEOFF = {"key": "labs_350", "title": "Разовая расшифровка анализов", "price_minor": 35000}

# --------------------
# AI client
# --------------------
ai = None
PRIMARY_MODEL = "deepseek/deepseek-r1-0528:free"
FALLBACK_MODELS: List[str] = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-chat-v3.1:free",
    "openai/gpt-4o-mini",
]
if OPENROUTER_KEY:
    try:
        ai = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY,
                    default_headers={"HTTP-Referer": APP_REFERER, "X-Title": APP_TITLE})
    except Exception as e:
        logger.warning(f"AI client init error: {e}")
        ai = None

async def ai_chat(system: str, user_text: str, temperature: float = 0.5) -> str:
    if ai is None:
        return "AI не настроен. Установите OPENROUTER_API_KEY и библиотеку клиента."
    def sync_call(model: str):
        return ai.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user_text}],
            temperature=temperature
        )
    last_err = None
    for model in [PRIMARY_MODEL] + FALLBACK_MODELS:
        try:
            resp = await asyncio.to_thread(sync_call, model)
            if hasattr(resp, "choices") and resp.choices:
                return resp.choices[0].message.content
            else:
                return resp["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            logger.warning(f"AI model {model} error: {e}")
            continue
    logger.exception(f"All AI models failed: {last_err}")
    return "Не удалось получить ответ от модели. Попробуйте позже."
async def setup_menu_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Устанавливает кнопку Mini App в меню"""
    
    menu_button = MenuButtonWebApp(
        text="🥗 Открыть приложение",
        web_app=WebAppInfo(url=MINI_APP_URL)
    )
    
    await context.bot.set_chat_menu_button(
        chat_id=update.effective_chat.id,
        menu_button=menu_button
    )
    
    await update.message.reply_text(
        "✅ Кнопка Mini App установлена!\n\n"
        "Теперь нажмите на кнопку меню (☰) рядом с полем ввода сообщения, "
        "чтобы открыть приложение.",
        reply_markup=reply_kb()
    )
# --------------------
# Database init (PostgreSQL)
# --------------------
from db_pg import db as _pgdb

class _CompatDB:
    """Совместимость с текущим кодом"""
    def __init__(self, inner):
        self._db = inner

    def execute(self, sql: str, params=()):
        sql2 = sql.replace("?", "%s")
        sql2 = sql2.replace("INSERT OR IGNORE INTO achievements", "INSERT INTO achievements")
        sql2 = sql2.replace("INSERT OR REPLACE INTO challenges", "INSERT INTO challenges")
        return self._db.execute(sql2, params)

    def commit(self):
        return self._db.commit()
    
    def rollback(self):
        return self._db.rollback()

db = _CompatDB(_pgdb)

def seed_products_pg():
    base = [
        ("яблоко", 52, 0.3, 0.2, 14),
        ("банан", 96, 1.3, 0.3, 23),
        ("рис", 360, 7.0, 0.7, 79),
        ("гречка", 343, 13.3, 3.4, 72.6),
        ("овсянка", 370, 13, 7, 62),
        ("куриная грудка", 165, 31, 3.6, 0),
        ("яйцо", 143, 12.6, 10.6, 0.7),
        ("творог 5%", 121, 17, 5, 1.8),
        ("молоко 2.5%", 52, 3.2, 2.5, 4.8),
        ("оливковое масло", 884, 0, 100, 0),
        ("огурец", 15, 0.7, 0.1, 3.6),
        ("помидор", 18, 0.9, 0.2, 3.9),
        ("сыр моцарелла", 280, 18, 21, 3),
    ]
    try:
        for name, kcal, p, f, c in base:
            db.execute(
                "INSERT INTO products (name, kcal_per_100, proteins_per_100, fats_per_100, carbs_per_100) "
                "VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT (name) DO NOTHING",
                (name, kcal, p, f, c)
            )
        db.commit()
    except Exception as e:
        logger.exception(f"Error seeding products: {e}")
        db.rollback()

# --------------------
# Utilities
# --------------------
def split_message(text: str, chunk_size: int = 4000) -> List[str]:
    text = text or ""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def save_user(user_id: int, username: Optional[str]):
    uname = (username or "").lower().lstrip("@")
    try:
        db.execute(
            "INSERT INTO subscriptions (user_id, username) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO NOTHING",
            (user_id, uname)
        )
        db.execute(
            "INSERT INTO credits (user_id, labs_credits) VALUES (%s, 0) "
            "ON CONFLICT (user_id) DO NOTHING",
            (user_id,)
        )
        db.execute(
            "INSERT INTO referrals (user_id, ref_code, invited_count) VALUES (%s, %s, 0) "
            "ON CONFLICT (user_id) DO NOTHING",
            (user_id, secrets.token_urlsafe(6))
        )
        db.commit()
    except Exception as e:
        logger.exception(f"Error saving user {user_id}: {e}")
        db.rollback()

def get_ref_code(user_id: int) -> str:
    row = db.execute("SELECT ref_code FROM referrals WHERE user_id=%s", (user_id,)).fetchone()
    if row and row["ref_code"]:
        return row["ref_code"]
    code = secrets.token_urlsafe(6)
    try:
        db.execute(
            "INSERT INTO referrals (user_id, ref_code, invited_count) VALUES (%s, %s, 0) "
            "ON CONFLICT (user_id) DO UPDATE SET ref_code=EXCLUDED.ref_code",
            (user_id, code)
        )
        db.commit()
    except Exception as e:
        logger.exception(f"Error generating ref code for {user_id}: {e}")
        db.rollback()
    return code

def has_access(user_id: int, username: Optional[str] = None) -> bool:
    if (username or "").lower() == ADMIN_USERNAME and ADMIN_USERNAME:
        return True
    now = dt.datetime.now(dt.timezone.utc)
    row = db.execute("SELECT expires_at, free_until FROM subscriptions WHERE user_id=%s", (user_id,)).fetchone()
    if not row:
        return False
    
    # ИСПРАВЛЕНО: безопасная работа с датами
    try:
        exp = row["expires_at"]
        free = row["free_until"]
        
        if exp and isinstance(exp, str):
            exp = dt.datetime.fromisoformat(exp)
        if free and isinstance(free, str):
            free = dt.datetime.fromisoformat(free)
            
        if exp and hasattr(exp, 'tzinfo') and exp.tzinfo is None:
            exp = exp.replace(tzinfo=dt.timezone.utc)
        if free and hasattr(free, 'tzinfo') and free.tzinfo is None:
            free = free.replace(tzinfo=dt.timezone.utc)
            
        if exp and exp > now:
            return True
        if free and free > now:
            return True
    except Exception as e:
        logger.warning(f"Error parsing dates for user {user_id}: {e}")
        return False
    
    return False

def activate_sub(user_id: int, days: int) -> dt.datetime:
    try:
        db.execute(
            "INSERT INTO subscriptions (user_id) VALUES (%s) "
            "ON CONFLICT (user_id) DO NOTHING",
            (user_id,)
        )
        row = db.execute(
            "SELECT expires_at FROM subscriptions WHERE user_id=%s",
            (user_id,)
        ).fetchone()
        now = dt.datetime.now(dt.timezone.utc)
        base = now
        
        # ИСПРАВЛЕНО: безопасная проверка
        if row and row["expires_at"]:
            try:
                existing = row["expires_at"]
                if isinstance(existing, str):
                    existing = dt.datetime.fromisoformat(existing)
                if hasattr(existing, 'tzinfo') and existing.tzinfo is None:
                    existing = existing.replace(tzinfo=dt.timezone.utc)
                if existing > now:
                    base = existing
            except Exception as e:
                logger.warning(f"Error parsing existing date: {e}")
                base = now
        
        new_exp = base + dt.timedelta(days=days)
        # ИСПРАВЛЕНО: передаём datetime напрямую
        db.execute(
            "UPDATE subscriptions SET expires_at=%s WHERE user_id=%s",
            (new_exp, user_id)
        )
        db.commit()
        return new_exp
    except Exception as e:
        logger.exception(f"Error activating subscription for {user_id}: {e}")
        db.rollback()
        raise

def activate_free_access(user_id: int, trial_hours: int) -> dt.datetime:
    try:
        db.execute(
            "INSERT INTO subscriptions (user_id) VALUES (%s) "
            "ON CONFLICT (user_id) DO NOTHING",
            (user_id,)
        )
        until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=trial_hours)
        # ИСПРАВЛЕНО: передаём datetime напрямую
        db.execute(
            "UPDATE subscriptions SET free_until=%s WHERE user_id=%s",
            (until, user_id)
        )
        db.commit()
        return until
    except Exception as e:
        logger.exception(f"Error activating free access for {user_id}: {e}")
        db.rollback()
        raise

def add_labs_credit(user_id: int, credit: int = 1):
    try:
        db.execute(
            "INSERT INTO credits (user_id, labs_credits) VALUES (%s, 0) "
            "ON CONFLICT (user_id) DO NOTHING",
            (user_id,)
        )
        db.execute(
            "UPDATE credits SET labs_credits = labs_credits + %s WHERE user_id=%s",
            (credit, user_id)
        )
        db.commit()
    except Exception as e:
        logger.exception(f"Error adding labs credit for {user_id}: {e}")
        db.rollback()

def consume_labs_credit(user_id: int) -> bool:
    try:
        row = db.execute(
            "SELECT labs_credits FROM credits WHERE user_id=%s",
            (user_id,)
        ).fetchone()
        if not row or (row["labs_credits"] or 0) <= 0:
            return False
        db.execute(
            "UPDATE credits SET labs_credits = labs_credits - 1 WHERE user_id=%s",
            (user_id,)
        )
        db.commit()
        return True
    except Exception as e:
        logger.exception(f"Error consuming labs credit for {user_id}: {e}")
        db.rollback()
        return False

def get_labs_credits(user_id: int) -> int:
    row = db.execute(
        "SELECT labs_credits FROM credits WHERE user_id=%s",
        (user_id,)
    ).fetchone()
    return int((row["labs_credits"] if row and row["labs_credits"] is not None else 0))

# --------------------
# Keyboards
# --------------------
BTN_DIARY = "🍽 Дневник"
BTN_QUESTION = "❓ Задать вопрос"
BTN_TRACKER = "🍽️ Трекер калорий"
BTN_PROFILE = "👤 Профиль"

async def send_chunks_with_back(func_send, text: str, parse_mode=ParseMode.HTML, disable_preview=True):
    """Отправка по частям с кнопкой Назад"""
    parts = split_message(text)
    for i, part in enumerate(parts):
        html_part = format_ai_html(part)
        kwargs = {
            "text": html_part,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_preview,
        }
        if i == len(parts) - 1:
            kwargs["reply_markup"] = back_to_menu_kb()
        await func_send(**kwargs)

def reply_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup([[BTN_DIARY, BTN_TRACKER], [BTN_QUESTION, BTN_PROFILE]], resize_keyboard=True)

def menu_kb() -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton("🥗 План питания", callback_data="plan"),
         InlineKeyboardButton("🧪 Расшифровка анализов", callback_data="labs")],
        [InlineKeyboardButton("🍳 Рецепт из холодильника", callback_data="recipe"),
         InlineKeyboardButton("🍏 Получить совет по питанию", callback_data="nutrition")],
        [InlineKeyboardButton("📊 Статистика", callback_data="stats"),
         InlineKeyboardButton("🌙 Как улучшить сон", callback_data="sleep")],
        [InlineKeyboardButton("🏆 Достижения", callback_data="achievements"),
         InlineKeyboardButton("⚡ Челленджи", callback_data="challenges")],
        [InlineKeyboardButton("🎁 Промокод", callback_data="promo"),
         InlineKeyboardButton("👥 Пригласить друга", callback_data="referral")],
        [InlineKeyboardButton("📞 Консультация", callback_data="consult"),
         InlineKeyboardButton("💳 Подписка", callback_data="subscribe")]
    ]
    return InlineKeyboardMarkup(kb)

def plans_kb() -> InlineKeyboardMarkup:
    rows = []
    for p in PLANS:
        title = f"{p['title']} — {p['price_minor']//100} ₽"
        rows.append([InlineKeyboardButton(title, callback_data=f"buy:{p['key']}")])
    
    rows.append([InlineKeyboardButton(f"{LABS_ONEOFF['title']} — {LABS_ONEOFF['price_minor']//100} ₽", callback_data=f"buy:{LABS_ONEOFF['key']}")])
    rows.append([InlineKeyboardButton("⬅️ Назад в меню", callback_data="back_menu")])
    return InlineKeyboardMarkup(rows)

def challenges_kb() -> InlineKeyboardMarkup:
    challenges = [
        ("💧 Пить 2л воды ежедневно", "water_challenge"),
        ("🏃 8k шагов ежедневно", "steps_challenge"),
        ("🥗 Здоровое питание 7 дней", "diet_challenge"),
        ("🏋️ Тренировки 3х в неделю", "workout_challenge"),
        ("📊 Отслеживать калории", "tracking_challenge"),
        ("🚫 Без сахара 7 дней", "nosugar_challenge")
    ]
    kb = []
    for title, key in challenges:
        kb.append([InlineKeyboardButton(title, callback_data=f"challenge:{key}")])
    kb.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_menu")])
    return InlineKeyboardMarkup(kb)

def back_to_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_menu")]])

def labs_purchase_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💰 Купить расшифровку анализов", callback_data="buy:labs_350")],
        [InlineKeyboardButton("⬅️ Назад в меню", callback_data="back_menu")]
    ])

def diary_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Приём пищи", callback_data="diary_add_meal"),
         InlineKeyboardButton("💧 Вода +250 мл", callback_data="diary_water")],
        [InlineKeyboardButton("📅 Сегодня", callback_data="diary_today"),
         InlineKeyboardButton("📊 Неделя", callback_data="diary_week")],
        [InlineKeyboardButton("↩️ Отменить последний", callback_data="diary_undo")],
        [InlineKeyboardButton("⬅️ Назад в меню", callback_data="back_menu")]
    ])

# --------------------
# OCR helpers
# --------------------
async def ocr_image_bytes(img_bytes: bytes) -> str:
    try:
        def sync():
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return pytesseract.image_to_string(img, lang=OCR_LANG).strip()
        return await asyncio.to_thread(sync)
    except Exception as e:
        logger.exception(f"OCR image error: {e}")
        return ""

async def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t.strip())
        if parts:
            return "\n\n".join(parts).strip()
    except Exception as e:
        logger.debug(f"pdfplumber text extract error: {e}")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.pdf")
            with open(path, "wb") as f:
                f.write(pdf_bytes)
            subprocess.run(["pdfimages", "-all", path, os.path.join(tmpdir, "page")], check=True)
            ocr_parts = []
            for fname in sorted(os.listdir(tmpdir)):
                if fname.startswith("page"):
                    with open(os.path.join(tmpdir, fname), "rb") as imf:
                        ocr_parts.append(await ocr_image_bytes(imf.read()))
            if any(p.strip() for p in ocr_parts):
                return "\n\n".join([p for p in ocr_parts if p.strip()]).strip()
    except Exception:
        pass
    
    try:
        ocr_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    img = page.to_image(resolution=150).original
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    ocr_parts.append(await ocr_image_bytes(buf.getvalue()))
                except Exception:
                    pass
        return "\n\n".join([p for p in ocr_parts if p.strip()]).strip()
    except Exception:
        return ""

# --------------------
# Products / BJU calculation
# --------------------
def find_product(name: str) -> Optional[Dict]:
    name = (name or "").strip().lower()
    if not name:
        return None

    row = db.execute(
        "SELECT * FROM products WHERE LOWER(name)=%s",
        (name,)
    ).fetchone()
    if row:
        return row

    row = db.execute(
        "SELECT * FROM products WHERE LOWER(name) LIKE %s ORDER BY LENGTH(name) LIMIT 1",
        (f"%{name}%",)
    ).fetchone()
    return row

def calc_nutrition_for_item(prod: Mapping[str, float], grams: float) -> Dict[str, float]:
    k = float(grams) / 100.0
    return {
        "kcal": float(prod["kcal_per_100"]) * k,
        "p": float(prod["proteins_per_100"]) * k,
        "f": float(prod["fats_per_100"]) * k,
        "c": float(prod["carbs_per_100"]) * k,
    }

def parse_meal_lines_from_text(text: str) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    for line in re.split(r"[\n;,]+", text):
        line = line.strip()
        if not line:
            continue
        m = re.search(r"([a-zA-Zа-яё0-9\s\-\%]+?)\s+(\d+(?:[.,]\d+)?)\s*(г|гр|g|мл|ml)?\b", line, re.I)
        if m:
            name = m.group(1).strip()
            qty = float(m.group(2).replace(",", "."))
            grams = qty
            items.append((name, grams))
    return items

def try_estimate_meal_from_db(meal_text: str) -> Optional[Tuple[int, float, float, float, str]]:
    items = parse_meal_lines_from_text(meal_text)
    if not items:
        return None
    total = {"kcal":0.0,"p":0.0,"f":0.0,"c":0.0}
    lines = []
    matched_any = False
    for name, grams in items:
        prod = find_product(name)
        if not prod:
            lines.append(f"• <i>{html.escape(name)}</i> — отсутствует в базе")
            continue
        matched_any = True
        n = calc_nutrition_for_item(prod, grams)
        total["kcal"] += n["kcal"]
        total["p"] += n["p"]
        total["f"] += n["f"]
        total["c"] += n["c"]
        lines.append(f"• {html.escape(prod['name'])} — {int(round(grams))} г → {int(round(n['kcal']))} ккал | Б {n['p']:.1f} Ж {n['f']:.1f} У {n['c']:.1f}")
    if not matched_any:
        return None
    summary = "<br>".join(lines)
    return int(round(total["kcal"])), round(total["p"],1), round(total["f"],1), round(total["c"],1), summary

# --------------------
# ИСПРАВЛЕНО: Format AI responses to HTML
# --------------------
def format_ai_html(text: str) -> str:
    """
    ИСПРАВЛЕНИЕ: Удаляет markdown-заголовки ### и форматирует текст
    """
    if not text:
        return ""

    # Убираем звёздочки
    s = text.replace("*", "")
    # ИСПРАВЛЕНО: Убираем markdown-заголовки (###, ##, #)
    s = re.sub(r'^#{1,6}\s+', '', s, flags=re.MULTILINE)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = html.escape(s)

    lines = s.split("\n")
    out = []

    for ln in lines:
        ln = ln.strip()
        if not ln:
            out.append("")
            continue

        # Буллет-списки
        if ln.startswith(("•", "-", "–")):
            item = ln.lstrip("•-– ").strip()
            out.append(f"• {item}")
        else:
            # Заголовки: оставляем жирным
            if ln.endswith(":") or (ln.upper() == ln and len(ln) <= 60 and " " in ln):
                out.append(f"<b>{ln}</b>")
            else:
                out.append(ln)

    return "\n".join(out)

# --------------------
# Daily reminder for food tracking
# --------------------
async def schedule_daily_reminders(app: Application):
    """Schedule daily reminders for all users to track their meals"""
    now_utc = dt.datetime.now(dt.timezone.utc)
    users = db.execute("SELECT user_id FROM subscriptions WHERE expires_at > %s OR free_until > %s", 
                      (now_utc, now_utc)).fetchall()
    
    for user in users:
        user_id = user["user_id"]
        reminder_time = now_utc.replace(hour=10, minute=0, second=0, microsecond=0)
        if reminder_time < now_utc:
            reminder_time += dt.timedelta(days=1)
        
        delay = (reminder_time - now_utc).total_seconds()
        app.job_queue.run_once(
            send_daily_reminder, 
            when=delay, 
            data={"user_id": user_id},
            name=f"daily_reminder_{user_id}"
        )

async def send_daily_reminder(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data
    user_id = data.get("user_id")
    
    if not user_id:
        return
        
    try:
        text = (
            "🍽 <b>Ежедневное напоминание</b>\n\n"
            "Не забудь внести свои приёмы пищи в дневник питания сегодня! "
            "Отслеживание рациона поможет тебе достичь твоих целей быстрее. 💪\n\n"
            "Используй кнопку «🍽️ Трекер калорий» или отправь сообщение с описанием того, что ты съел(а)."
        )
        await context.bot.send_message(chat_id=user_id, text=text, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.warning(f"Failed to send daily reminder to user {user_id}: {e}")

# --------------------
# ML-based recommendations
# --------------------
def generate_ml_recommendations(user_id: int) -> str:
    """Generate personalized recommendations based on user's history"""
    try:
        meals = db.execute(
            "SELECT calories, proteins, fats, carbs, ts FROM meals WHERE user_id = %s ORDER BY ts DESC LIMIT 30",
            (user_id,)
        ).fetchall()
        
        if not meals or len(meals) < 7:
            return "Пока недостаточно данных для персонализированных рекомендаций. Продолжайте вести дневник питания!"
        
        calories = [meal["calories"] or 0 for meal in meals]
        proteins = [meal["proteins"] or 0 for meal in meals]
        fats = [meal["fats"] or 0 for meal in meals]
        carbs = [meal["carbs"] or 0 for meal in meals]
        
        avg_calories = statistics.mean(calories) if len(calories) > 1 else calories[0]
        avg_proteins = statistics.mean(proteins) if len(proteins) > 1 else proteins[0]
        avg_fats = statistics.mean(fats) if len(fats) > 1 else fats[0]
        avg_carbs = statistics.mean(carbs) if len(carbs) > 1 else carbs[0]
        
        recommendations = []
        
        if avg_calories > 2500:
            recommendations.append("📉 Ваше среднее потребление калорий выше нормы. Попробуйте уменьшить порции.")
        elif avg_calories < 1500:
            recommendations.append("📈 Ваше среднее потребление калорий ниже нормы. Увеличьте питательность рациона.")
        
        if avg_proteins < 60:
            recommendations.append("🥩 Увеличьте потребление белка для поддержания мышечной массы.")
        
        if avg_fats > 80:
            recommendations.append("🥑 Попробуйте уменьшить потребление жиров, особенно насыщенных.")
        
        if avg_carbs > 300:
            recommendations.append("🍞 Сократите потребление углеводов, особенно простых.")
        
        if not recommendations:
            recommendations.append("Ваш рацион выглядит сбалансированным! Продолжайте в том же духе.")
        
        return "\n".join(recommendations)
        
    except Exception as e:
        logger.exception(f"Error generating ML recommendations: {e}")
        return "Не удалось сгенерировать рекомендации. Продолжайте вести дневник питания!"

# --------------------
# Challenge functions
# --------------------
def init_challenge(user_id: int, challenge_type: str) -> bool:
    """Initialize a challenge for user (idempotent)"""
    try:
        start_date = dt.datetime.now(dt.timezone.utc)
        db.execute(
            "INSERT INTO challenges (user_id, challenge_type, start_date, progress, completed) "
            "VALUES (%s, %s, %s, 0, 0) "
            "ON CONFLICT (user_id, challenge_type) DO UPDATE "
            "SET start_date = EXCLUDED.start_date, progress = LEAST(challenges.progress, 7), completed = LEAST(challenges.completed, 1)",
            (user_id, challenge_type, start_date)
        )
        db.commit()
        return True
    except Exception as e:
        logger.exception(f"Error initializing challenge: {e}")
        db.rollback()
        return False

def update_challenge_progress(user_id: int, challenge_type: str) -> bool:
    """Update challenge progress for user"""
    try:
        today = dt.datetime.now(dt.timezone.utc).date().isoformat()
        
        existing = db.execute(
            "SELECT 1 FROM challenge_logs WHERE user_id=%s AND challenge_type=%s AND log_date=%s",
            (user_id, challenge_type, today)
        ).fetchone()
        
        if existing:
            return False
        
        db.execute(
            "INSERT INTO challenge_logs (user_id, challenge_type, log_date, completed) VALUES (%s, %s, %s, 1)",
            (user_id, challenge_type, today)
        )
        
        db.execute(
            "UPDATE challenges SET progress = progress + 1 WHERE user_id=%s AND challenge_type=%s",
            (user_id, challenge_type)
        )
        
        challenge = db.execute(
            "SELECT progress FROM challenges WHERE user_id=%s AND challenge_type=%s",
            (user_id, challenge_type)
        ).fetchone()
        
        challenge_days = 7
        
        if challenge and challenge["progress"] >= challenge_days:
            db.execute(
                "UPDATE challenges SET completed = 1 WHERE user_id=%s AND challenge_type=%s",
                (user_id, challenge_type)
            )
            
            achievement_name = f"Челлендж: {get_challenge_name(challenge_type)}"
            db.execute(
                "INSERT INTO achievements (user_id, badge, ts) VALUES (%s, %s, %s)",
                (user_id, achievement_name, dt.datetime.now(dt.timezone.utc))
            )
        
        db.commit()
        return True
        
    except Exception as e:
        logger.exception(f"Error updating challenge progress: {e}")
        db.rollback()
        return False

def get_challenge_name(challenge_type: str) -> str:
    """Get display name for challenge type"""
    names = {
        "water_challenge": "Пить 2л воды daily",
        "steps_challenge": "8k шагов ежедневно",
        "diet_challenge": "Здоровое питание 7 дней",
        "workout_challenge": "Тренировки 3х в неделю",
        "tracking_challenge": "Отслеживать калории",
        "nosugar_challenge": "Без сахара 7 дней"
    }
    return names.get(challenge_type, "Неизвестный челлендж")

def get_challenge_progress(user_id: int, challenge_type: str) -> Tuple[int, int]:
    """Get current progress for a challenge (current/total)"""
    challenge = db.execute(
        "SELECT progress FROM challenges WHERE user_id=%s AND challenge_type=%s",
        (user_id, challenge_type)
    ).fetchone()
    
    if not challenge:
        return 0, 7
    
    return challenge["progress"], 7

# --------------------
# ИСПРАВЛЕНО: Разбиение on_menu_callback на отдельные функции
# --------------------

async def handle_subscribe_callback(query, context):
    """Handle subscription menu"""
    if not PAYMENT_TOKEN:
        await query.message.reply_text("Оплата не настроена.", reply_markup=reply_kb())
        return

    if not PLANS:
        await query.message.reply_text("Тарифы пока не настроены.", reply_markup=reply_kb())
        return

    lines = ["💰 <b>Сравнение тарифов:</b>", ""]
    base_daily = (PLANS[0]["price_minor"] / max(1, PLANS[0]["days"])) / 100

    for plan in PLANS:
        daily_price = (plan["price_minor"] / max(1, plan["days"])) / 100
        economy = int(round((1 - (daily_price / base_daily)) * 100)) if base_daily else 0

        line = f"• {plan['title']}: {plan['price_minor']//100} руб. ({daily_price:.2f} руб./день)"
        if economy > 0:
            line += f" ← экономия {economy}%"
        lines.append(line)

    comparison_text = "\n".join(lines)
    await query.message.reply_text(comparison_text, parse_mode=ParseMode.HTML)
    await query.message.reply_text("Выберите тариф:", reply_markup=plans_kb())

async def handle_sleep_callback(query, context):
    """Handle sleep advice"""
    await query.message.reply_text(
        "🌙 <b>Советы для улучшения сна:</b>\n\n"
        "• Соблюдайте регулярный график сна\n"
        "• Создайте расслабляющий ритуал перед сном\n"
        "• Избегайте кофеина и тяжелой пищи перед сном\n"
        "• Обеспечьте темноту и тишину в спальне\n"
        "• Регулярно занимайтесь спортом, но не перед сном\n"
        "• Ограничьте использование электронных устройств перед сном",
        parse_mode=ParseMode.HTML,
        reply_markup=back_to_menu_kb()
    )

async def handle_buy_callback(query, context, key: str):
    """Handle purchase requests"""
    if not PAYMENT_TOKEN:
        await query.message.reply_text("Оплата не настроена.", reply_markup=reply_kb())
        return
    
    if key == LABS_ONEOFF["key"]:
        prices = [LabeledPrice(LABS_ONEOFF["title"], LABS_ONEOFF["price_minor"])]
        await query.message.reply_invoice(
            title=LABS_ONEOFF["title"], description="Разовая расшифровка анализов",
            provider_token=PAYMENT_TOKEN, currency=CURRENCY, prices=prices, payload="pay:labs_oneoff"
        )
        return
    
    plan = next((p for p in PLANS if p["key"] == key), None)
    if not plan:
        await query.message.reply_text("Тариф не найден.", reply_markup=reply_kb())
        return
    
    prices = [LabeledPrice(plan["title"], plan["price_minor"])]
    await query.message.reply_invoice(
        title=plan["title"], description=f"Подписка {plan['days']} дней",
        provider_token=PAYMENT_TOKEN, currency=CURRENCY, prices=prices, payload=f"pay:sub:{plan['key']}"
    )

async def handle_consult_callback(query, context):
    """Handle consultation info"""
    await query.message.reply_text(
        "👩‍⚕️ <b>Консультация профессионального нутрициолога</b>\n\n"
        "Опытный специалист с глубокими знаниями в области превентивной медицины и нутрициологии.\n"
        "В рамках консультации я помогу:\n"
        "• Разработать персонализированный план питания\n"
        "• Расшифровать анализы и выявить нутритивные дефициты\n"
        "• Составить программу коррекции пищевого поведения\n"
        "• Разработать стратегию профилактики заболеваний\n\n"
        "📞 Для записи на консультацию: @Tatiana_Lekh\n\n"
        "<i>Помните: правильное питание - лучшая профилактика здоровья!</i>",
        parse_mode=ParseMode.HTML,
        reply_markup=back_to_menu_kb()
    )

async def handle_labs_callback(query, context, user):
    """Handle labs analysis request"""
    clear_modes(context)
    is_admin = (user.username or "").lower() == ADMIN_USERNAME and ADMIN_USERNAME
    if is_admin:
        context.user_data[EXPECT_LABS] = True
        await query.message.reply_text("Пришли текст/фото/PDF с анализами.", reply_markup=back_to_menu_kb())
        return
    
    row = db.execute("SELECT used_free_lab FROM subscriptions WHERE user_id=%s", (user.id,)).fetchone()
    if not row or not row["used_free_lab"]:
        db.execute("UPDATE subscriptions SET used_free_lab=1 WHERE user_id=%s", (user.id,))
        db.commit()
        context.user_data[EXPECT_LABS] = True
        await query.message.reply_text("🎁 Бесплатная расшифровка активирована. Пришли анализы.", reply_markup=back_to_menu_kb())
        return
    
    credits = get_labs_credits(user.id)
    if credits <= 0:
        await query.message.reply_text(
            f"Разовая расшифровка стоит {LABS_ONEOFF['price_minor']//100} ₽.",
            reply_markup=labs_purchase_kb()
        )
        return
    
    context.user_data[EXPECT_LABS] = True
    await query.message.reply_text("Пришли текст/фото/PDF с анализами.", reply_markup=back_to_menu_kb())

async def handle_achievements_callback(query, context, user):
    """Handle achievements display"""
    rows = db.execute("SELECT badge, ts FROM achievements WHERE user_id=%s ORDER BY ts DESC", (user.id,)).fetchall()
    if not rows:
        await query.message.reply_text("Пока нет достижений. Заполняй дневник и участвуй в челленджах!", reply_markup=back_to_menu_kb())
        return
    lines = [f"🏅 {r['badge']} — {r['ts'][:10]}" for r in rows]
    await query.message.reply_text("Твои достижения:\n" + "\n".join(lines), reply_markup=back_to_menu_kb())

async def handle_challenges_callback(query, context):
    """Handle challenges menu"""
    await query.message.reply_text(
        "⚡ <b>Выбери челлендж</b>\n\n"
        "Участвуй в челленджах и получай достижения! "
        "Отмечай прогресс каждый день для завершения челленджа.",
        reply_markup=challenges_kb(), parse_mode=ParseMode.HTML
    )

async def handle_challenge_detail(query, context, user, challenge_type: str):
    """Handle individual challenge details"""
    challenge_name = get_challenge_name(challenge_type)
    
    init_challenge(user.id, challenge_type)
    
    progress, total = get_challenge_progress(user.id, challenge_type)
    
    await query.message.reply_text(
        f"⚡ <b>Челлендж: {challenge_name}</b>\n\n"
        f"Прогресс: {progress}/{total} дней\n\n"
        "Отмечай выполнение каждый день для завершения челленджа!",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Отметить выполнение", callback_data=f"challenge_log:{challenge_type}")],
            [InlineKeyboardButton("⬅️ Назад к челленджам", callback_data="challenges")]
        ])
    )

async def handle_challenge_log(query, context, user, challenge_type: str):
    """Handle logging challenge progress"""
    challenge_name = get_challenge_name(challenge_type)
    
    if update_challenge_progress(user.id, challenge_type):
        progress, total = get_challenge_progress(user.id, challenge_type)
        
        if progress >= total:
            await query.message.reply_text(
                f"🎉 Поздравляем! Ты завершил(а) челлендж «{challenge_name}»!\n"
                "Достижение добавлено в твой профиль.",
                reply_markup=back_to_menu_kb()
            )
        else:
            await query.message.reply_text(
                f"✅ Прогресс отмечен! Текущий прогресс: {progress}/{total} дней",
                reply_markup=back_to_menu_kb()
            )
    else:
        await query.message.reply_text(
            "❌ Сегодня ты уже отмечал(а) этот челлендж. Попробуй завтра!",
            reply_markup=back_to_menu_kb()
        )

async def handle_diary_callbacks(query, context, user, action: str):
    """Handle diary-related callbacks"""
    if action == "add_meal":
        clear_modes(context)
        context.user_data[EXPECT_MEAL] = True
        await query.message.reply_text(
            "Отправь приём пищи в виде строк:\n"
            "куриная грудка 150 г\nрис 70 г\n(или фото/чек).",
            reply_markup=back_to_menu_kb()
        )
    elif action == "water":
        now = dt.datetime.now(dt.timezone.utc)
        db.execute(
            "INSERT INTO meals (user_id, ts, text, calories, proteins, fats, carbs) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (user.id, now, "вода 250 мл", 0, 0.0, 0.0, 0.0)
        )
        db.commit()
        await query.message.reply_text("💧 Записал: вода +250 мл.", reply_markup=diary_kb())
    elif action == "undo":
        row = db.execute(
            "SELECT id FROM meals WHERE user_id=%s ORDER BY ts DESC LIMIT 1", (user.id,)
        ).fetchone()
        if not row:
            await query.message.reply_text("Отменять нечего — записей нет.", reply_markup=diary_kb())
            return
        db.execute("DELETE FROM meals WHERE id=%s", (row["id"],))
        db.commit()
        await query.message.reply_text("Последняя запись удалена.", reply_markup=diary_kb())

async def on_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main callback router - delegates to specific handlers"""
    query = update.callback_query
    if query:
        await query.answer()
    
    data = (query.data if query else None) or ""
    user = query.from_user if query else update.effective_user
    
    # Clear flags helper
    def reset_flags():
        for k in (EXPECT_LABS, EXPECT_RECIPE, EXPECT_QUESTION, EXPECT_MEAL, EXPECT_WEIGHT):
            context.user_data[k] = False
    
    # Route to specific handlers
    if data == "back_menu":
        await query.message.reply_text("Главное меню:", reply_markup=menu_kb())
        return
    
    if data == "subscribe":
        await handle_subscribe_callback(query, context)
        return
    
    if data == "sleep":
        await handle_sleep_callback(query, context)
        return
    
    if data.startswith("buy:"):
        key = data.split(":", 1)[1]
        await handle_buy_callback(query, context, key)
        return
    
    if data == "consult":
        await handle_consult_callback(query, context)
        return
    
    if data == "plan":
        if not has_access(user.id, user.username):
            await query.message.reply_text("Функция доступна по подписке или пробному периоду.", reply_markup=reply_kb())
            return
        reset_flags()
        await query.message.reply_text("Сколько тебе лет?", reply_markup=back_to_menu_kb())
        return PLAN_AGE
    
    if data == "labs":
        await handle_labs_callback(query, context, user)
        return
    
    if data == "recipe":
        if not has_access(user.id, user.username):
            await query.message.reply_text("Требуется подписка/пробный доступ.", reply_markup=reply_kb())
            return
        reset_flags()
        context.user_data[EXPECT_RECIPE] = True
        await query.message.reply_text("Пришли фото холодильника или список продуктов (текстом).", reply_markup=back_to_menu_kb())
        return
    
    if data == "question":
        if not has_access(user.id, user.username):
            await query.message.reply_text("Доступно по подписке/пробному периоду.", reply_markup=reply_kb())
            return
        reset_flags()
        context.user_data[EXPECT_QUESTION] = True
        await query.message.reply_text("Задай вопрос (можно фото).", reply_markup=back_to_menu_kb())
        return
    
    if data == "nutrition":
        await query.message.reply_text("Задай вопрос о питании или открой меню.", reply_markup=back_to_menu_kb())
        return
    
    if data in ("diary_today", "diary_week"):
        period = "week" if data.endswith("week") else "today"
        await show_diary(update, context, period=period)
        return
    
    if data == "diary_add_meal":
        await handle_diary_callbacks(query, context, user, "add_meal")
        return
    
    if data == "diary_undo":
        await handle_diary_callbacks(query, context, user, "undo")
        return
    
    if data == "diary_water":
        await handle_diary_callbacks(query, context, user, "water")
        return
    
    if data == "achievements":
        await handle_achievements_callback(query, context, user)
        return
    
    if data == "challenges":
        await handle_challenges_callback(query, context)
        return
    
    if data.startswith("challenge:"):
        challenge_type = data.split(":", 1)[1]
        await handle_challenge_detail(query, context, user, challenge_type)
        return
    
    if data.startswith("challenge_log:"):
        challenge_type = data.split(":", 1)[1]
        await handle_challenge_log(query, context, user, challenge_type)
        return
    
    if data == "promo":
        await query.message.reply_text("Применить промокод: /promo КОД", reply_markup=reply_kb())
        return
    
    if data == "referral":
        code = get_ref_code(user.id)
        bot_me = await context.bot.get_me()
        link = f"t.me/{bot_me.username}?start={code}"
        await query.message.reply_text(
            f"👥 <b>Пригласи друга и получи +{REF_BONUS_DAYS} дней подписки!</b>\n\n"
            f"Твоя реферальная ссылка:\n{link}\n\n"
            "Просто отправь эту ссылку другу. Когда он зарегистрируется по ней, ты получишь бонусные дни подписки!",
            parse_mode=ParseMode.HTML,
            reply_markup=back_to_menu_kb()
        )
        return
    
    if data == "stats":
        if not has_access(user.id, user.username):
            await query.message.reply_text("Доступно по подписке/пробному периоду.", reply_markup=reply_kb())
            return
        await send_stats(user.id, query.message)
        return
    
    await query.message.reply_text("Неизвестная опция.", reply_markup=reply_kb())

# --------------------
# Handlers (start/menu/plan/labs/recipes/meals/etc.)
# --------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    save_user(user.id, user.username)

async def setup_menu_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Устанавливает кнопку Mini App в меню"""
    web_app_url = os.getenv("MINI_APP_URL", "https://your-domain.com")
    
    menu_button = MenuButtonWebApp(
        text="🥗 Открыть приложение",
        web_app=WebAppInfo(url=web_app_url)
    )
    
    await context.bot.set_chat_menu_button(
        chat_id=update.effective_chat.id,
        menu_button=menu_button
    )


    # referral parameter handling
    if context.args:
        code = (context.args[0] or "").strip()
        if code:
            row = db.execute("SELECT user_id FROM referrals WHERE ref_code=%s", (code,)).fetchone()
            inviter_id = row["user_id"] if row else None
            if inviter_id and inviter_id != user.id:
                act = db.execute(
                    "SELECT 1 FROM referral_activations WHERE invited_id=%s",
                    (user.id,)
                ).fetchone()
                if not act:
                    try:
                        db.execute(
                            "INSERT INTO referral_activations (inviter_id, invited_id) VALUES (%s, %s) "
                            "ON CONFLICT (inviter_id, invited_id) DO NOTHING",
                            (inviter_id, user.id)
                        )
                        db.execute(
                            "UPDATE referrals SET invited_count = COALESCE(invited_count,0) + 1 "
                            "WHERE user_id=%s",
                            (inviter_id,)
                        )
                        activate_sub(inviter_id, REF_BONUS_DAYS)
                        db.commit()
                        try:
                            await context.bot.send_message(inviter_id, f"🎉 Тебе начислено +{REF_BONUS_DAYS} дней за приглашённого.")
                        except Exception:
                            pass
                    except Exception as e:
                        logger.exception(f"Error processing referral: {e}")
                        db.rollback()

    # trial activation for non-admin
    is_admin = (user.username or "").lower() == ADMIN_USERNAME and ADMIN_USERNAME
    row = db.execute("SELECT free_until FROM subscriptions WHERE user_id=%s", (user.id,)).fetchone()
    if not is_admin and (row is None or not row["free_until"]):
        until = activate_free_access(user.id, TRIAL_HOURS)
        try:
            await update.message.reply_text(f"🎁 Пробный доступ активирован на {TRIAL_HOURS} ч (до {until.strftime('%d.%m.%Y %H:%M UTC')}).")
        except Exception:
            pass

    # Generate referral link
    ref_code = get_ref_code(user.id)
    bot_me = await context.bot.get_me()
    ref_link = f"t.me/{bot_me.username}?start={ref_code}"

    caption = (
        "👋 Привет! Я уникальный бот-нутрициолог.\n\n"
        "Я умею:\n"
        "• Составлять персональные планы питания\n"
        "• Оценивать анализы по фото/PDF (за разовую оплату)\n"
        "• Придумывать рецепты из того, что есть в холодильнике\n"
        "• Вести дневник питания с подсчётом БЖУ/ккал\n"
        "• Проводить челленджи и мотивировать\n\n"
        f"Поделись ботом с другом: {ref_link}\n\n"
        "⚠️ Бот не ставит диагнозы и не заменяет консультацию врача."
    )
    try:
        await update.message.reply_photo(photo=WELCOME_IMAGE, caption=caption, reply_markup=menu_kb())
    except Exception:
        await update.message.reply_text(caption, reply_markup=menu_kb())
    await update.message.reply_text("«Не стесняйся — просто задай вопрос 💬 или нажми на кнопки ⬆️⬇️»", reply_markup=reply_kb())

async def show_diary(update: Update, context: ContextTypes.DEFAULT_TYPE, period: str = "today"):
    target = update.message if update.message else update.callback_query.message
    user_id = update.effective_user.id

    now = dt.datetime.now(dt.timezone.utc)
    if period == "week":
        since = (now - dt.timedelta(days=7))
        rows = db.execute(
            "SELECT ts, text, calories, proteins, fats, carbs FROM meals "
            "WHERE user_id=%s AND ts>=%s ORDER BY ts DESC", (user_id, since)
        ).fetchall()
        title = "📊 Дневник — последние 7 дней"
    else:
        start = dt.datetime.combine(now.date(), dt.time.min).replace(tzinfo=dt.timezone.utc)
        end = start + dt.timedelta(days=1)
        rows = db.execute(
            "SELECT ts, text, calories, proteins, fats, carbs FROM meals "
            "WHERE user_id=%s AND ts>=%s AND ts<%s ORDER BY ts DESC", (user_id, start, end)
        ).fetchall()
        title = "🍽 Дневник — сегодня"

    if not rows:
        await target.reply_text(
            f"{title}\n\nПока записей нет.\nНажми «➕ Приём пищи» или отправь текст/фото/чек.",
            reply_markup=diary_kb()
        )
        return

    total_kcal = 0
    p = f = c = 0.0
    lines = []
    for r in rows:
        total_kcal += int(r["calories"] or 0)
        p += float(r["proteins"] or 0)
        f += float(r["fats"] or 0)
        c += float(r["carbs"] or 0)
        try:
            ts_obj = r["ts"]
            if isinstance(ts_obj, str):
                ts_obj = dt.datetime.fromisoformat(ts_obj)
            hhmm = ts_obj.strftime("%H:%M")
        except Exception:
            hhmm = str(r["ts"])[:16]
        txt = (r["text"] or "").strip()
        lines.append(f"• {hhmm} — {txt} ({int(r['calories'] or 0)} ккал)")

    summary = (
        f"{title}\n\n"
        + "\n".join(lines[:30])
        + "\n\n"
        + f"Итого: {total_kcal} ккал, Б/Ж/У: {round(p)}/{round(f)}/{round(c)} г"
    )
    await target.reply_text(summary, reply_markup=diary_kb())

# --------------------
# Plan conversation handlers
# --------------------
async def plan_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not has_access(update.effective_user.id, update.effective_user.username):
        await update.message.reply_text("Функция доступна по подписке/пробному периоду.", reply_markup=reply_kb())
        return ConversationHandler.END
    await update.message.reply_text("Сколько тебе лет?")
    return PLAN_AGE

async def plan_age(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get(EXPECT_QUESTION) or context.user_data.get(EXPECT_LABS) or context.user_data.get(EXPECT_MEAL):
        return ConversationHandler.END
    age = (update.message.text or "").strip()

    # ИСПРАВЛЕНО: добавлена валидация
    if not age.isdigit():
        await update.message.reply_text("Пожалуйста, укажи возраст числом.")
        return PLAN_AGE
    
    age_int = int(age)
    if not (1 <= age_int <= 120):
        await update.message.reply_text("Пожалуйста, укажи корректный возраст (1-120 лет).")
        return PLAN_AGE
    
    context.user_data["age"] = age_int
    kb = ReplyKeyboardMarkup([["Мужской", "Женский"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Пол:", reply_markup=kb)
    return PLAN_SEX

async def plan_sex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sex = (update.message.text or "").strip().lower()
    if sex not in ("мужской", "женский"):
        await update.message.reply_text("Выбери: Мужской или Женский.")
        return PLAN_SEX
    context.user_data["sex"] = "male" if sex == "мужской" else "female"
    await update.message.reply_text("Вес (кг):", reply_markup=reply_kb())
    return PLAN_WEIGHT

async def plan_weight(update: Update, context: ContextTypes.DEFAULT_TYPE):
    weight = (update.message.text or "").strip()
    try:
        weight_float = float(weight.replace(",", "."))
        # ИСПРАВЛЕНО: добавлена валидация
        if not (20 <= weight_float <= 300):
            await update.message.reply_text("Пожалуйста, укажи корректный вес (20-300 кг).")
            return PLAN_WEIGHT
        context.user_data["weight"] = weight_float
    except Exception:
        await update.message.reply_text("Пожалуйста, укажи вес числом.")
        return PLAN_WEIGHT
    await update.message.reply_text("Рост (см):")
    return PLAN_HEIGHT

async def plan_height(update: Update, context: ContextTypes.DEFAULT_TYPE):
    height = (update.message.text or "").strip()
    try:
        height_float = float(height.replace(",", "."))
        # ИСПРАВЛЕНО: добавлена валидация
        if not (100 <= height_float <= 250):
            await update.message.reply_text("Пожалуйста, укажи корректный рост (100-250 см).")
            return PLAN_HEIGHT
        context.user_data["height"] = height_float
    except Exception:
        await update.message.reply_text("Пожалуйста, укажи рост числом.")
        return PLAN_HEIGHT
    
    kb = ReplyKeyboardMarkup([
        ["Сидячий образ жизни", "Легкая активность"],
        ["Умеренная активность", "Высокая активность"],
        ["Экстремальная активность"]
    ], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text(
        "Уровень физической активности:\n\n"
        "• Сидячий образ жизни: минимум движения\n"
        "• Легкая активность: легкие упражнения 1-3 раза в неделю\n"
        "• Умеренная активность: тренировки 3-5 раз в неделю\n"
        "• Высокая активность: интенсивные тренировки 6-7 раз в неделю\n"
        "• Экстремальная активность: профессиональные спортсмены",
        reply_markup=kb
    )
    return PLAN_ACTIVITY

async def plan_activity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    activity = (update.message.text or "").strip()
    context.user_data["activity"] = activity
    await update.message.reply_text("Цель (снижение веса / набор / поддержание):")
    return PLAN_GOAL

async def plan_goal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    goal = (update.message.text or "").strip()
    if not goal:
        await update.message.reply_text("Укажи цель.")
        return PLAN_GOAL
    context.user_data["goal"] = goal
    await update.message.reply_text("Предпочтения/аллергии (если есть):")
    return PLAN_PREFS

async def plan_prefs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = (update.message.text or "").strip()
    context.user_data["prefs"] = prefs
    await update.message.reply_text("Ограничения/запреты (если нет — напиши 'нет'):")
    return PLAN_RESTR

async def plan_finish(update: Update, context: ContextTypes.DEFAULT_TYPE):
    restr = (update.message.text or "").strip()
    context.user_data["restr"] = restr
    u = context.user_data
    await update.message.reply_text("⏳ Формирую персональный план — это может занять пару минут...")
    
    # Calculate BMR and recommended calories
    age = u.get('age', 30)
    weight = u.get('weight', 70)
    height = u.get('height', 170)
    sex = u.get('sex', 'male')
    activity = u.get('activity', 'Умеренная активность')
    goal = u.get('goal', 'поддержание')
    
    if sex == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    activity_multipliers = {
        "Сидячий образ жизни": 1.2,
        "Легкая активность": 1.375,
        "Умеренная активность": 1.55,
        "Высокая активность": 1.725,
        "Экстремальная активность": 1.9
    }
    multiplier = activity_multipliers.get(activity, 1.55)
    
    goal_adjustments = {
        "снижение веса": 0.85,
        "поддержание": 1.0,
        "набор": 1.15
    }
    adjustment = goal_adjustments.get(goal.lower(), 1.0)
    
    daily_calories = round(bmr * multiplier * adjustment)
    
    prompt = (
        "Составь 7-дневный персональный план питания (завтрак/обед/ужин/перекусы), "
        "ориентировочные граммовки и примерную калорийность в день. Пиши структурировано.\n\n"
        f"Данные клиента:\n- Возраст: {u.get('age')}\n- Пол: {u.get('sex')}\n- Вес: {u.get('weight')} кг\n- Рост: {u.get('height')} см\n- Активность: {u.get('activity')}\n- Цель: {u.get('goal')}\n- Предпочтения: {u.get('prefs')}\n- Ограничения: {u.get('restr')}\n- Рекомендуемая калорийность: {daily_calories} ккал/день"
    )
    try:
        text = await ai_chat("Ты профессиональный нутрициолог и диетолог. Пиши структурировано.", prompt, 0.4)
    except Exception as e:
        logger.exception(f"AI plan error: {e}")
        text = "Не удалось сформировать план. Попробуйте позже."
    
    await send_chunks_with_back(
        lambda **kw: update.message.reply_text(**kw),
        text,
        parse_mode=ParseMode.HTML,
        disable_preview=True
    )

    # Add ML-based recommendations
    recommendations = generate_ml_recommendations(update.effective_user.id)
    if recommendations:
        await update.message.reply_text(
            f"💡 <b>Персональные рекомендации:</b>\n\n{recommendations}",
            parse_mode=ParseMode.HTML,
            reply_markup=back_to_menu_kb()
        )
    
    await update.message.reply_text("Готово.", reply_markup=reply_kb())
    return ConversationHandler.END

async def plan_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Анкета отменена.", reply_markup=reply_kb())
    return ConversationHandler.END

# --------------------
# Promo / admin / payments
# --------------------
async def promo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Применить промокод: /promo КОД", reply_markup=reply_kb())
        return
    code = (context.args[0] or "").strip()
    if not code:
        await update.message.reply_text("Пустой промокод.", reply_markup=reply_kb())
        return

    row = db.execute(
        "SELECT days, labs_credits, max_uses, used_count, expires_at FROM promocodes WHERE code=%s",
        (code,)
    ).fetchone()
    if not row:
        await update.message.reply_text("Промокод не найден.", reply_markup=reply_kb())
        return

    days = int(row["days"] or 0)
    credits = int(row["labs_credits"] or 0)
    max_uses = row["max_uses"]
    used_count = row["used_count"]
    expires_at = row["expires_at"]

    now = dt.datetime.now(dt.timezone.utc)
    if expires_at:
        try:
            exp_dt = expires_at if isinstance(expires_at, dt.datetime) else dt.datetime.fromisoformat(expires_at)
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=dt.timezone.utc)
            if exp_dt < now:
                await update.message.reply_text("Срок действия промокода истёк.", reply_markup=reply_kb())
                return
        except Exception as e:
            logger.warning(f"Error parsing promo expiry: {e}")

    if max_uses is not None and used_count is not None and used_count >= max_uses:
        await update.message.reply_text("Этот промокод исчерпан.", reply_markup=reply_kb())
        return

    days = max(0, min(days, PROMO_MAX_DAYS))

    if days <= 0 and credits <= 0:
        await update.message.reply_text("Этот промокод ничего не даёт.", reply_markup=reply_kb())
        return

    try:
        parts = []
        if days > 0:
            activate_sub(update.effective_user.id, days)
            parts.append(f"+{days} дней подписки")
        if credits > 0:
            add_labs_credit(update.effective_user.id, credits)
            parts.append(f"+{credits} анализ(а/ов)")

        db.execute("UPDATE promocodes SET used_count = COALESCE(used_count,0) + 1 WHERE code=%s", (code,))
        db.commit()

        await update.message.reply_text(f"✅ Промокод применён: " + ", ".join(parts) + ".", reply_markup=reply_kb())
    except Exception as e:
        logger.exception(f"apply promo error: {e}")
        db.rollback()
        await update.message.reply_text("Ошибка применения промокода.", reply_markup=reply_kb())

async def addpromo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if (user.username or "").lower() != ADMIN_USERNAME:
        await update.message.reply_text("⛔ Нет доступа.")
        return

    # ИСПРАВЛЕНО: Синтаксис для PostgreSQL
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "Использование:\n"
            "/addpromo КОД ДНИ [LABS_CREDITS] [MAX_USES] [YYYY-MM-DD]\n\n"
            "Примеры:\n"
            "• /addpromo TEST30 30\n"
            "• /addpromo FREE2 0 2\n"
            "• /addpromo MEGA 30 5 100 2025-12-31"
        )
        return

    code = args[0].strip()
    days = int(args[1])

    labs_credits = 0
    max_uses = None
    expires_at = None

    if len(args) == 3:
        max_uses = int(args[2])
    elif len(args) >= 4:
        labs_credits = int(args[2])
        max_uses = int(args[3]) if len(args) >= 4 else None
        expires_at = args[4] if len(args) >= 5 else None

    try:
        # ИСПРАВЛЕНО: ON CONFLICT для PostgreSQL
        db.execute(
            "INSERT INTO promocodes (code, days, labs_credits, max_uses, used_count, expires_at) "
            "VALUES (%s, %s, %s, %s, 0, %s) "
            "ON CONFLICT (code) DO UPDATE SET "
            "days = EXCLUDED.days, "
            "labs_credits = EXCLUDED.labs_credits, "
            "max_uses = EXCLUDED.max_uses, "
            "expires_at = EXCLUDED.expires_at",
            (code, days, labs_credits, max_uses, expires_at)
        )
        db.commit()

        await update.message.reply_text(
            f"✅ Промокод {code} создан: +{days} дней, +{labs_credits} кредит(ов) анализов, "
            f"max_uses={max_uses}, expires_at={expires_at}"
        )
    except Exception as e:
        logger.exception(f"Error creating promo: {e}")
        db.rollback()
        await update.message.reply_text("Ошибка создания промокода.")

async def precheckout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.pre_checkout_query
    await query.answer(ok=True)

async def successful_payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    msg = update.message
    sp = msg.successful_payment

    payload = (sp.invoice_payload or "").strip()
    currency = (sp.currency or "").upper()
    amount = int(sp.total_amount or 0)
    tg_charge_id = sp.telegram_payment_charge_id
    provider_charge_id = sp.provider_payment_charge_id

    logger.info(f"Successful payment u={user_id} payload={payload} cur={currency} amount={amount}")

    # ИСПРАВЛЕНО: Идемпотентность
    if provider_charge_id:
        dup = db.execute(
            "SELECT 1 FROM payments WHERE provider_charge_id=%s",
            (provider_charge_id,)
        ).fetchone()
        if dup:
            await msg.reply_text("Этот платёж уже учтён 👍", reply_markup=back_to_menu_kb())
            return

    if not payload.startswith("pay:"):
        logger.warning(f"Invalid payment payload u={user_id}: {payload}")
        await msg.reply_text("❌ Ошибка обработки платежа: неверный формат.", reply_markup=back_to_menu_kb())
        return

    try:
        if payload.startswith("pay:sub:"):
            key = payload.split(":", 2)[2]
            plan = next((p for p in PLANS if p.get("key") == key), None)
            if not plan:
                logger.error(f"Plan not found for key={key}")
                await msg.reply_text("❌ Ошибка: тарифный план не найден. Напишите в поддержку.", reply_markup=back_to_menu_kb())
                return

            expected_currency = (plan.get("currency") or os.getenv("CURRENCY", "RUB")).upper()
            expected_amount = int(plan.get("price_minor") or 0)
            if expected_amount and amount != expected_amount:
                logger.warning(f"Amount mismatch plan={key}: got {amount}, expected {expected_amount}")
                await msg.reply_text("Сумма платежа не совпала с тарифом. Напишите в поддержку.", reply_markup=back_to_menu_kb())
                return
            if currency and expected_currency and currency != expected_currency:
                logger.warning(f"Currency mismatch plan={key}: got {currency}, expected {expected_currency}")
                await msg.reply_text("Валюта платежа не совпала с тарифом. Напишите в поддержку.", reply_markup=back_to_menu_kb())
                return

            days = int(plan.get("days") or 0)
            if days <= 0:
                await msg.reply_text("Тариф некорректен (0 дней). Напишите в поддержку.", reply_markup=back_to_menu_kb())
                return

            exp = activate_sub(user_id, days)

            labs_credits = int(plan.get("labs_credits") or 0)
            if labs_credits > 0:
                add_labs_credit(user_id, labs_credits)

            db.execute(
                "INSERT INTO payments (user_id, payload, currency, amount, tg_charge_id, provider_charge_id) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (provider_charge_id) DO NOTHING",
                (user_id, payload, currency, amount, tg_charge_id, provider_charge_id)
            )
            db.commit()

            extras = f"\n+{labs_credits} анализа(ов) добавлено." if labs_credits > 0 else ""
            await msg.reply_text(
                f"✅ Подписка на {days} дней активирована до {exp.strftime('%d.%m.%Y')}."
                f"{extras}\nСпасибо за покупку!",
                reply_markup=back_to_menu_kb()
            )
            return

        if payload.startswith("pay:labs"):
            qty = 1
            parts = payload.split(":")
            if len(parts) == 3 and parts[2].isdigit():
                qty = max(1, int(parts[2]))

            try:
                expected_one = int(LABS_ONEOFF.get("price_minor") or 0)
            except Exception:
                expected_one = 0
            expected_total = expected_one * qty if expected_one else 0
            if expected_total and amount != expected_total:
                logger.warning(f"Labs amount mismatch: got {amount}, expected {expected_total} (qty={qty})")
                await msg.reply_text("Сумма платежа не совпала. Напишите в поддержку.", reply_markup=back_to_menu_kb())
                return

            add_labs_credit(user_id, qty)

            db.execute(
                "INSERT INTO payments (user_id, payload, currency, amount, tg_charge_id, provider_charge_id) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (provider_charge_id) DO NOTHING",
                (user_id, payload, currency, amount, tg_charge_id, provider_charge_id)
            )
            db.commit()

            await msg.reply_text(f"✅ Оплачено! Доступно {qty} анализ(а/ов).", reply_markup=back_to_menu_kb())
            return

        db.execute(
            "INSERT INTO payments (user_id, payload, currency, amount, tg_charge_id, provider_charge_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (provider_charge_id) DO NOTHING",
            (user_id, payload, currency, amount, tg_charge_id, provider_charge_id)
        )
        db.commit()
        logger.warning(f"Unknown payment payload u={user_id}: {payload}")
        await msg.reply_text("✅ Платёж успешен, но тип не распознан. Напишите в поддержку.", reply_markup=back_to_menu_kb())
        return

    except (UniqueViolation, psycopg.IntegrityError):
        await msg.reply_text("Этот платёж уже учтён 👍", reply_markup=back_to_menu_kb())
    except Exception as e:
        logger.exception(f"successful_payment error u={user_id}: {e}")
        db.rollback()
        await msg.reply_text(
            "❌ Непредвиденная ошибка при обработке платежа. Обратитесь в поддержку.",
            reply_markup=back_to_menu_kb()
        )

# --------------------
# Routers: text/photo/document
# --------------------
def clear_modes(context):
    for k in (EXPECT_LABS, EXPECT_RECIPE, EXPECT_QUESTION, EXPECT_MEAL, EXPECT_WEIGHT):
        context.user_data[k] = False

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = update.effective_message
    user = update.effective_user
    text = (getattr(m, "text", None) or "").strip()
    if not text:
        return

    logger.info("Text from %s: %s", user.id, text)

    if text == BTN_PROFILE:
        row = db.execute(
            "SELECT expires_at, free_until FROM subscriptions WHERE user_id=%s",
            (user.id,)
        ).fetchone()

        credits = get_labs_credits(user.id)
        ach_count = db.execute(
            "SELECT COUNT(*) AS c FROM achievements WHERE user_id=%s",
            (user.id,)
        ).fetchone()["c"]

        sub_text = "Нет активной подписки"
        if row:
            if row["expires_at"]:
                exp = row["expires_at"]
                if isinstance(exp, str):
                    sub_text = f"Подписка до {exp[:10]}"
                else:
                    sub_text = f"Подписка до {exp.strftime('%Y-%m-%d')}"
            elif row["free_until"]:
                fu = row["free_until"]
                if isinstance(fu, str):
                    sub_text = f"Пробный доступ до {fu[:10]}"
                else:
                    sub_text = f"Пробный доступ до {fu.strftime('%Y-%m-%d')}"

        challenges = db.execute(
            "SELECT challenge_type, progress FROM challenges WHERE user_id=%s AND completed=0",
            (user.id,)
        ).fetchall()

        profile_text = (
            f"👤 <b>Профиль</b>\n\n{escape(sub_text)}\n"
            f"Кредиты анализов: {credits}\n"
            f"Достижений: {ach_count}"
        )

        if challenges:
            profile_text += "\n\n<b>Активные челленджи:</b>"
            for ch in challenges:
                name = get_challenge_name(ch["challenge_type"])
                profile_text += f"\n• {escape(name)}: {ch['progress']}/7 дней"

        await m.reply_text(profile_text, parse_mode=ParseMode.HTML, reply_markup=reply_kb())
        return

    if text == BTN_DIARY:
        if not has_access(user.id, user.username):
            await update.message.reply_text("Требуется подписка/пробный доступ.", reply_markup=reply_kb())
            return
        clear_modes(context)
        await show_diary(update, context, period="today")
        return
    
    if text == BTN_QUESTION:
        if not has_access(user.id, user.username):
            await m.reply_text("Доступно по подписке/пробному периоду.", reply_markup=reply_kb())
            return
        context.user_data[EXPECT_QUESTION] = True
        await m.reply_text("Задай вопрос (можно фото).", reply_markup=reply_kb())
        return

    if text == BTN_TRACKER:
        if not has_access(user.id, user.username):
            await update.message.reply_text("Требуется подписка/пробный доступ.", reply_markup=reply_kb())
            return
        clear_modes(context)
        context.user_data[EXPECT_MEAL] = True
        await update.message.reply_text(
            "Отправь приём пищи в виде строк:\n"
            "куриная грудка 150 г\nрис 70 г\n(или фото/чек).",
            reply_markup=back_to_menu_kb()
        )
        return

    if context.user_data.get(EXPECT_LABS):
        await handle_labs_text(update, context, text)
        return
    if context.user_data.get(EXPECT_RECIPE):
        await handle_recipe_text(update, context, text)
        return
    if context.user_data.get(EXPECT_QUESTION):
        await handle_question_text(update, context, text)
        return
    if context.user_data.get(EXPECT_MEAL):
        await handle_meal_text(update, context, text)
        return
    if context.user_data.get(EXPECT_WEIGHT):
        await handle_weight_tracking(update, context, text)
        return

    if context.chat_data.get("state") in (PLAN_AGE, PLAN_SEX, PLAN_WEIGHT, PLAN_HEIGHT, PLAN_ACTIVITY, PLAN_GOAL, PLAN_PREFS, PLAN_RESTR):
        return

    if not has_access(user.id, user.username):
        await update.message.reply_text("Функция доступна по подписке или пробному периоду.", reply_markup=reply_kb())
        return
    await update.message.reply_text("⏳ Обрабатываю...")
    try:
        res = await ai_chat("Ты нутрициолог. Отвечай кратко и по делу.", text, 0.5)
    except Exception as e:
        logger.exception(f"default ai error: {e}")
        res = "Не удалось обработать запрос. Попробуйте позже."
    for part in split_message(res):
        html_part = format_ai_html(part)
        await update.message.reply_text(html_part, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

async def photo_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    photos = update.message.photo
    if not photos:
        await update.message.reply_text("Фото не обнаружено.")
        return
    photo = photos[-1]
    file = await update.get_bot().get_file(photo.file_id)
    buf = io.BytesIO()
    await file.download_to_memory(out=buf)
    data = buf.getvalue()

    if context.user_data.get(EXPECT_LABS):
        is_admin = (user.username or "").lower() == ADMIN_USERNAME and ADMIN_USERNAME
        if not is_admin:
            used_free = db.execute("SELECT used_free_lab FROM subscriptions WHERE user_id=%s", (user.id,)).fetchone()
            if not used_free or not used_free["used_free_lab"]:
                db.execute("UPDATE subscriptions SET used_free_lab=1 WHERE user_id=%s", (user.id,))
                db.commit()
            else:
                if not consume_labs_credit(user.id):
                    await update.message.reply_text("Нет кредитов. Купи разовую расшифровку.", reply_markup=reply_kb())
                    return
        text = await ocr_image_bytes(data)
        if not text.strip():
            await update.message.reply_text("Не удалось распознать текст на фото.", reply_markup=reply_kb())
            return
        await handle_labs_text(update, context, text)
        return

    if context.user_data.get(EXPECT_QUESTION):
        text = await ocr_image_bytes(data)
        if not text.strip():
            await update.message.reply_text("Не удалось распознать текст. Попробуй написать вопрос текстом.", reply_markup=reply_kb())
            return
        await handle_question_text(update, context, text)
        return

    if context.user_data.get(EXPECT_RECIPE):
        caption = (update.message.caption or "").strip()
        products = caption if caption else (await ocr_image_bytes(data))
        if not products.strip():
            await update.message.reply_text("Не удалось распознать продукты. Добавь подпись.", reply_markup=reply_kb())
            return
        await handle_recipe_text(update, context, products)
        return

    if context.user_data.get(EXPECT_MEAL):
        caption = (update.message.caption or "").strip()
        meal_text = caption if caption else (await ocr_image_bytes(data))
        if not meal_text.strip():
            await update.message.reply_text("Не удалось распознать подпись/текст.", reply_markup=reply_kb())
            return
        await handle_meal_text(update, context, meal_text)
        return

    await update.message.reply_text("Фото получено. Выбери действие и пришли снова.", reply_markup=reply_kb())

async def document_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    doc = update.message.document
    if not doc:
        await update.message.reply_text("Документ не найден.")
        return
    file = await update.get_bot().get_file(doc.file_id)
    buf = io.BytesIO()
    await file.download_to_memory(out=buf)
    data = buf.getvalue()

    if context.user_data.get(EXPECT_LABS):
        is_admin = (user.username or "").lower() == ADMIN_USERNAME and ADMIN_USERNAME
        if not is_admin:
            used_free = db.execute("SELECT used_free_lab FROM subscriptions WHERE user_id=%s", (user.id,)).fetchone()
            if not used_free or not used_free["used_free_lab"]:
                db.execute("UPDATE subscriptions SET used_free_lab=1 WHERE user_id=%s", (user.id,))
                db.commit()
            else:
                if not consume_labs_credit(user.id):
                    await update.message.reply_text("Нет кредитов. Купи разовую расшифровку.", reply_markup=reply_kb())
                    return
        text = ""
        if doc.mime_type == "application/pdf" or (doc.file_name and doc.file_name.lower().endswith(".pdf")):
            text = await ocr_pdf_bytes(data)
        else:
            text = await ocr_image_bytes(data)
        if not text.strip():
            await update.message.reply_text("Не удалось распознать текст в документе.", reply_markup=reply_kb())
            return
        await handle_labs_text(update, context, text)
        return

    if context.user_data.get(EXPECT_RECIPE):
        text = await ocr_pdf_bytes(data)
        if not text.strip():
            await update.message.reply_text("Не удалось распознать продукты.", reply_markup=reply_kb())
            return
        await handle_recipe_text(update, context, text)
        return

    if context.user_data.get(EXPECT_QUESTION):
        text = await ocr_pdf_bytes(data)
        if not text.strip():
            await update.message.reply_text("Не удалось распознать текст.", reply_markup=reply_kb())
            return
        await handle_question_text(update, context, text)
        return

    if context.user_data.get(EXPECT_MEAL):
        text = await ocr_pdf_bytes(data)
        if not text.strip():
            await update.message.reply_text("Не удалось распознать текст/чек.", reply_markup=reply_kb())
            return
        await handle_meal_text(update, context, text)
        return

    await update.message.reply_text("Документ получен. Выбери действие и пришли снова.", reply_markup=reply_kb())

# --------------------
# Domain-specific handlers
# --------------------
async def handle_labs_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    await update.message.reply_text("⏳ Анализирую лабораторные данные...")
    prompt = "Ты нутрициолог. Проанализируй лабораторные анализы и дай практические рекомендации.\n\n" + text
    try:
        ans = await ai_chat("Пиши кратко и структурированно.", prompt, 0.3)
    except Exception as e:
        logger.warning(f"AI labs error: {e}")
        ans = "Не удалось обработать анализы. Попробуйте позже."
    await send_chunks_with_back(
        lambda **kw: update.message.reply_text(**kw),
        ans,
        parse_mode=ParseMode.HTML,
        disable_preview=True
    )
    context.user_data[EXPECT_LABS] = False

async def handle_recipe_text(update: Update, context: ContextTypes.DEFAULT_TYPE, products_text: str):
    await update.message.reply_text("⏳ Формирую рецепты...")
    prompt = (
        "Ты шеф-повар и нутрициолог. На основе списка продуктов составь 3 рецепта. Для каждого: "
        "название, ингредиенты с граммовками, шаги приготовления, калорийность и БЖУ на порцию.\n\n"
        f"Продукты:\n{products_text}"
    )
    try:
        ans = await ai_chat("Пиши структурированно, ясно.", prompt, 0.5)
    except Exception as e:
        logger.warning(f"AI recipe error: {e}")
        ans = "Не удалось составить рецепты."
    await send_chunks_with_back(
        lambda **kw: update.message.reply_text(**kw),
        ans,
        parse_mode=ParseMode.HTML,
        disable_preview=True
    )    
    context.user_data[EXPECT_RECIPE] = False

async def handle_question_text(update: Update, context: ContextTypes.DEFAULT_TYPE, q_text: str):
    await update.message.reply_text("⏳ Отвечаю...")
    prompt = f"Ты нутрициолог. Дай краткий, практический ответ на вопрос:\n\n{q_text}"
    try:
        ans = await ai_chat("Кратко и по делу.", prompt, 0.5)
    except Exception as e:
        logger.warning(f"AI question error: {e}")
        ans = "Не удалось получить ответ. Попробуйте позже."
    await send_chunks_with_back(
        lambda **kw: update.message.reply_text(**kw),
        ans,
        parse_mode=ParseMode.HTML,
        disable_preview=True
    )
    context.user_data[EXPECT_QUESTION] = False

async def handle_meal_text(update: Update, context: ContextTypes.DEFAULT_TYPE, meal_text: str):
    await update.message.reply_text("⏳ Оцениваю приём пищи (ккал/БЖУ)...")
    # First try DB-based estimate
    est = try_estimate_meal_from_db(meal_text)
    used_ai = False
    if est is None:
        # ask AI for strict JSON as fallback
        system = "Ты нутрициолог. Верни ТОЛЬКО JSON: {\"calories\": int, \"proteins\": float, \"fats\": float, \"carbs\": float, \"summary\": \"text\"}"
        prompt = (
            "Оцени приём пищи и верни JSON в формате, например:\n"
            '{"calories": 450, "proteins": 25.5, "fats": 12.0, "carbs": 50.0, "summary": "кратко"}\n\n'
            f"Текст: {meal_text}"
        )
        try:
            resp = await ai_chat(system, prompt, 0.2)
        except Exception as e:
            logger.warning(f"AI meal error: {e}")
            resp = ""
        calories, p, f, c, summary = parse_meal_json(resp)
        used_ai = True
    else:
        calories, p, f, c, summary = est

    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        db.execute("INSERT INTO meals (user_id, ts, text, calories, proteins, fats, carbs) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                   (update.effective_user.id, ts, meal_text, calories, p, f, c))
        db.commit()
    except Exception as e:
        logger.warning(f"DB insert meal error: {e}")

    award_achievements_after_meal(update.effective_user.id)
    context.user_data[EXPECT_MEAL] = False

    source = "ИИ" if used_ai else "базы продуктов"
    if used_ai:
        summary_html = html.escape(summary)
    else:
        summary_html = summary
    total_line = f"<b>Итого</b>: {calories} ккал | Б {p:.1f} Ж {f:.1f} У {c:.1f}\nИсточник: {source}"
    await update.message.reply_text(f"<b>Записал приём пищи</b>\n{summary_html}<br>{total_line}", parse_mode=ParseMode.HTML, reply_markup=back_to_menu_kb())

async def handle_weight_tracking(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    try:
        weight = float(text.replace(",", "."))
        user_id = update.effective_user.id
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        
        db.execute(
            "INSERT INTO weight_tracking (user_id, weight, ts) VALUES (%s, %s, %s)",
            (user_id, weight, ts)
        )
        db.commit()
        
        await update.message.reply_text(
            f"✅ Вес {weight} кг записан!\n\n"
            "Используй /stats для просмотра прогресса.",
            reply_markup=back_to_menu_kb()
        )
        
        context.user_data[EXPECT_WEIGHT] = False
        
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите вес числом (например, 68.5)")

def parse_meal_json(text: str) -> Tuple[int, float, float, float, str]:
    try:
        s = (text or "").strip().replace("`", "")
        m = re.search(r"\{.*\}", s, flags=re.S)
        j = json.loads(m.group(0)) if m else json.loads(s)
        cal = int(j.get("calories", 0))
        p = float(j.get("proteins", 0.0))
        f = float(j.get("fats", 0.0))
        c = float(j.get("carbs", 0.0))
        summary = str(j.get("summary", "приём пищи"))
        return max(0, cal), max(0.0, p), max(0.0, f), max(0.0, c), summary
    except Exception:
        cal = first_int(r"(\d{2,4})\s*k?ккал", text) or 0
        p = first_float(r"б(?:елк[аи])?[:\s]*([0-9]+(?:\.[0-9]+)?)", text) or 0.0
        f = first_float(r"ж(?:ир[ыа])?[:\s]*([0-9]+(?:\.[0-9]+)?)", text) or 0.0
        c = first_float(r"у(?:углевод[ыа])?[:\s]*([0-9]+(?:\.[0-9]+)?)", text) or 0.0
        return int(cal), float(p), float(f), float(c), "приём пищи"

def first_int(pattern: str, s: str) -> Optional[int]:
    if not s: return None
    m = re.search(pattern, s, re.I)
    return int(m.group(1)) if m else None

def first_float(pattern: str, s: str) -> Optional[float]:
    if not s: return None
    m = re.search(pattern, s, re.I)
    return float(m.group(1)) if m else None

def award_achievements_after_meal(user_id: int):
    try:
        now = dt.datetime.now(dt.timezone.utc)
        week_from = (now - dt.timedelta(days=7)).isoformat()
        rows = db.execute("SELECT ts, text FROM meals WHERE user_id=%s AND ts>=%s ORDER BY ts", (user_id, week_from)).fetchall()
        breakfast_days = set()
        water_ok = False
        sugar_flag = False if rows else False
        for r in rows:
            txt = (r["text"] or "").lower()
            try:
                hr = dt.datetime.fromisoformat(r["ts"]).hour
            except Exception:
                hr = None
            if hr is not None and 5 <= hr < 10:
                breakfast_days.add(r["ts"][:10])
            if "вода" in txt:
                water_ok = True
            if re.search(r"сахар|торт|шоколад|печенье|конфет", txt):
                sugar_flag = True
        if len(breakfast_days) >= 7:
            try:
                db.execute("INSERT INTO achievements (user_id, badge, ts) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING", 
                          (user_id, "Завтрак-герой", now.isoformat()))
            except Exception:
                pass
        if water_ok:
            try:
                db.execute("INSERT INTO achievements (user_id, badge, ts) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING", 
                          (user_id, "Повелитель воды", now.isoformat()))
            except Exception:
                pass
        if not sugar_flag and rows:
            try:
                db.execute("INSERT INTO achievements (user_id, badge, ts) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING", 
                          (user_id, "7 дней без сахара", now.isoformat()))
            except Exception:
                pass
        db.commit()
    except Exception as e:
        logger.warning(f"award achievements error: {e}")

async def send_stats(user_id: int, target_msg):
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)).isoformat()
    rows = db.execute("SELECT ts, calories, proteins, fats, carbs FROM meals WHERE user_id=%s AND ts>=%s ORDER BY ts", (user_id, since)).fetchall()
    
    if not rows:
        await target_msg.reply_text("Нет данных за последние 7 дней.", reply_markup=reply_kb())
        return
        
    daily = {}
    for r in rows:
        day = r["ts"][:10]
        d = daily.setdefault(day, {"cal":0,"p":0.0,"f":0.0,"c":0.0})
        d["cal"] += r["calories"] or 0
        d["p"] += r["proteins"] or 0.0
        d["f"] += r["fats"] or 0.0
        d["c"] += r["carbs"] or 0.0
        
    lines = [f"{day}: {d['cal']} ккал | Б {round(d['p'],1)} Ж {round(d['f'],1)} У {round(d['c'],1)}" for day, d in sorted(daily.items())]
    total_cal = sum(d["cal"] for d in daily.values())
    avg = round(total_cal / max(1, len(daily)))
    
    await target_msg.reply_text("📊 <b>Статистика за 7 дней:</b>\n" + "\n".join(lines) + f"\n\nСреднесуточно: ~{avg} ккал", 
                           parse_mode=ParseMode.HTML)

# --------------------
# Weight tracking command
# --------------------
async def weight_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not has_access(update.effective_user.id, update.effective_user.username):
        await update.message.reply_text("Функция доступна по подписке/пробному периоду.", reply_markup=reply_kb())
        return
        
    context.user_data[EXPECT_WEIGHT] = True
    await update.message.reply_text("Введите ваш текущий вес в кг (например, 68.5):")


# --------------------
# Register handlers and run
# --------------------
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    # --- Команды ---
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("promo", promo_cmd))
    app.add_handler(CommandHandler("addpromo", addpromo_cmd))
    app.add_handler(CommandHandler("weight", weight_cmd))
    app.add_handler(CommandHandler("app", setup_menu_button))  
    
  
    # --- Диалог «План питания» ---
    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(on_menu_callback, pattern="^plan$")],
        states={
            PLAN_AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, plan_age)],
            PLAN_SEX: [MessageHandler(filters.TEXT & ~filters.COMMAND, plan_sex)],
            PLAN_WEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, plan_weight)],
            PLAN_HEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, plan_height)],
            PLAN_ACTIVITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, plan_activity)],
            PLAN_GOAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, plan_goal)],
            PLAN_PREFS: [MessageHandler(filters.TEXT & ~filters.COMMAND, plan_prefs)],
            PLAN_RESTR: [MessageHandler(filters.TEXT & ~filters.COMMAND, plan_finish)],
        },
        fallbacks=[CommandHandler("cancel", plan_cancel)],
    )
    app.add_handler(conv_handler)

    # --- Остальные callback-и меню ---
    app.add_handler(CallbackQueryHandler(on_menu_callback))

    # --- Роутеры медиа и текста ---
    app.add_handler(MessageHandler(filters.PHOTO, photo_router))
    app.add_handler(MessageHandler(filters.Document.ALL, document_router))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # --- Платежи ---
    app.add_handler(PreCheckoutQueryHandler(precheckout))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment))

    logger.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
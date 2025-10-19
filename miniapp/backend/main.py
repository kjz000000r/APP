# -*- coding: utf-8 -*-
"""
FastAPI Backend для NutriCoach Mini App
ПОЛНАЯ ИНТЕГРАЦИЯ СО ВСЕМИ ФУНКЦИЯМИ БОТА
"""
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import asyncio

# Наши модули
from database import execute_query, execute_update, get_db
from auth import get_current_user
from models import (
    UserProfile, MealEntry, MealResponse, DiaryPeriod, DiaryDay,
    WeeklyStats, WeightEntry, WeightHistory, NutritionPlanRequest,
    ChallengeProgress, ChallengeList, LabsAnalysis, RecipeRequest,
    Achievement
)

# Импортируем AI функции из бота
import sys
sys.path.append('../bot')
try:
    from final import (
        ai_chat, find_product, calc_nutrition_for_item,
        parse_meal_lines_from_text, try_estimate_meal_from_db,
        get_labs_credits, consume_labs_credit, has_access,
        get_challenge_name, init_challenge, update_challenge_progress,
        get_challenge_progress
    )
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logging.warning("AI functions not available - some features will be limited")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("miniapp")

# Создаём приложение
app = FastAPI(
    title="NutriCoach Mini App API",
    description="Backend для Telegram Mini App бота-нутрициолога",
    version="2.0.0"
)

# CORS - В ПРОДАКШЕНЕ УКАЖИТЕ КОНКРЕТНЫЙ ДОМЕН!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: заменить на домен вашего хостинга
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== HEALTH CHECK ====================

@app.get("/")
async def root():
    """Проверка работоспособности API"""
    return {
        "service": "NutriCoach Mini App Backend",
        "version": "2.0.0",
        "status": "operational",
        "ai_available": AI_AVAILABLE
    }

@app.get("/health")
async def health_check():
    """Детальная проверка здоровья сервиса"""
    try:
        # Проверяем БД
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "database": db_status,
        "ai": "available" if AI_AVAILABLE else "unavailable",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ==================== USER PROFILE ====================

@app.get("/api/user/profile", response_model=UserProfile)
async def get_user_profile(user = Depends(get_current_user)):
    """
    Получить профиль пользователя
    
    Возвращает:
    - Статус подписки
    - Количество кредитов на анализы
    - Количество достижений
    - Активные челленджи
    """
    user_id = user['user_id']
    
    try:
        # Получаем подписку
        sub = execute_query(
            "SELECT expires_at, free_until FROM subscriptions WHERE user_id = %s",
            (user_id,)
        )
        
        # Получаем кредиты
        credits = execute_query(
            "SELECT labs_credits FROM credits WHERE user_id = %s",
            (user_id,)
        )
        
        # Получаем достижения
        achievements = execute_query(
            "SELECT COUNT(*) as count FROM achievements WHERE user_id = %s",
            (user_id,)
        )
        
        # Получаем активные челленджи
        challenges = execute_query(
            "SELECT COUNT(*) as count FROM challenges WHERE user_id = %s AND completed = 0",
            (user_id,)
        )
        
        # Определяем статус подписки
        subscription_status = "expired"
        expires_at = None
        free_until = None
        
        if sub and len(sub) > 0:
            now = datetime.now(timezone.utc)
            exp = sub[0].get('expires_at')
            free = sub[0].get('free_until')
            
            if exp and exp > now:
                subscription_status = "active"
                expires_at = exp
            elif free and free > now:
                subscription_status = "trial"
                free_until = free
        
        return UserProfile(
            user_id=user_id,
            username=user.get('username'),
            subscription_status=subscription_status,
            expires_at=expires_at,
            free_until=free_until,
            labs_credits=credits[0]['labs_credits'] if credits else 0,
            achievements_count=achievements[0]['count'] if achievements else 0,
            active_challenges=challenges[0]['count'] if challenges else 0
        )
        
    except Exception as e:
        logger.exception(f"Error getting profile for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== DIARY / TRACKER ====================

@app.post("/api/diary/add", response_model=MealResponse)
async def add_meal_entry(
    meal: MealEntry,
    user = Depends(get_current_user)
):
    """
    Добавить приём пищи в дневник
    
    Шаги:
    1. Пытаемся найти продукты в БД
    2. Если не находим - используем AI
    3. Сохраняем в БД
    4. Проверяем достижения
    """
    user_id = user['user_id']
    
    try:
        # Шаг 1: Пробуем через БД
        source = "database"
        db_result = None
        
        if AI_AVAILABLE:
            db_result = try_estimate_meal_from_db(meal.text)
        
        if db_result:
            calories, proteins, fats, carbs, summary = db_result
        else:
            # Шаг 2: Используем AI
            if not AI_AVAILABLE:
                raise HTTPException(
                    status_code=503,
                    detail="AI service unavailable"
                )
            
            source = "ai"
            system = 'Ты нутрициолог. Верни ТОЛЬКО JSON: {"calories": int, "proteins": float, "fats": float, "carbs": float, "summary": "text"}'
            prompt = f'Оцени приём пищи и верни JSON в формате, например:\n{{"calories": 450, "proteins": 25.5, "fats": 12.0, "carbs": 50.0, "summary": "кратко"}}\n\nТекст: {meal.text}'
            
            try:
                response = await ai_chat(system, prompt, 0.2)
                # Парсим JSON из ответа AI
                import json
                import re
                match = re.search(r'\{.*\}', response, flags=re.S)
                if match:
                    data = json.loads(match.group(0))
                    calories = int(data.get('calories', 0))
                    proteins = float(data.get('proteins', 0.0))
                    fats = float(data.get('fats', 0.0))
                    carbs = float(data.get('carbs', 0.0))
                    summary = str(data.get('summary', meal.text))
                else:
                    raise ValueError("No JSON in AI response")
            except Exception as e:
                logger.warning(f"AI meal parsing error: {e}")
                # Fallback значения
                calories = 200
                proteins = fats = carbs = 10.0
                summary = meal.text
        
        # Шаг 3: Сохраняем в БД
        now = datetime.now(timezone.utc)
        
        result = execute_query(
            """
            INSERT INTO meals (user_id, ts, text, calories, proteins, fats, carbs)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (user_id, now, meal.text, calories, proteins, fats, carbs)
        )
        
        meal_id = result[0]['id'] if result else None
        
        # Шаг 4: Проверяем достижения (асинхронно)
        asyncio.create_task(check_achievements_after_meal(user_id))
        
        return MealResponse(
            success=True,
            meal_id=meal_id,
            calories=calories,
            proteins=proteins,
            fats=fats,
            carbs=carbs,
            summary=summary,
            source=source
        )
        
    except Exception as e:
        logger.exception(f"Error adding meal for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

async def check_achievements_after_meal(user_id: int):
    """Проверка достижений после добавления еды (фоновая задача)"""
    try:
        # Логика из бота - award_achievements_after_meal
        now = datetime.now(timezone.utc)
        week_ago = (now - timedelta(days=7)).isoformat()
        
        meals = execute_query(
            "SELECT ts, text FROM meals WHERE user_id = %s AND ts >= %s ORDER BY ts",
            (user_id, week_ago)
        )
        
        breakfast_days = set()
        has_water = False
        has_sugar = False
        
        for meal in meals:
            text_lower = (meal['text'] or '').lower()
            hour = meal['ts'].hour if hasattr(meal['ts'], 'hour') else None
            
            if hour and 5 <= hour < 10:
                breakfast_days.add(meal['ts'].date().isoformat())
            
            if 'вода' in text_lower:
                has_water = True
            
            if any(word in text_lower for word in ['сахар', 'торт', 'шоколад', 'печенье', 'конфет']):
                has_sugar = True
        
        # Выдаём достижения
        if len(breakfast_days) >= 7:
            execute_update(
                """
                INSERT INTO achievements (user_id, badge, ts)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (user_id, "Завтрак-герой", now)
            )
        
        if has_water:
            execute_update(
                """
                INSERT INTO achievements (user_id, badge, ts)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (user_id, "Повелитель воды", now)
            )
        
        if not has_sugar and len(meals) >= 21:  # 3 приёма * 7 дней
            execute_update(
                """
                INSERT INTO achievements (user_id, badge, ts)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (user_id, "7 дней без сахара", now)
            )
        
        logger.info(f"Achievements checked for user {user_id}")
        
    except Exception as e:
        logger.exception(f"Error checking achievements for user {user_id}: {e}")

@app.get("/api/diary/today", response_model=DiaryDay)
async def get_today_diary(user = Depends(get_current_user)):
    """Получить дневник за сегодня"""
    user_id = user['user_id']
    
    try:
        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        meals = execute_query(
            """
            SELECT id, ts, text, calories, proteins, fats, carbs
            FROM meals
            WHERE user_id = %s AND ts >= %s AND ts < %s
            ORDER BY ts DESC
            """,
            (user_id, start_of_day, end_of_day)
        )
        
        total_cals = sum(m['calories'] or 0 for m in meals)
        total_p = sum(m['proteins'] or 0 for m in meals)
        total_f = sum(m['fats'] or 0 for m in meals)
        total_c = sum(m['carbs'] or 0 for m in meals)
        
        return DiaryDay(
            date=now.date().isoformat(),
            total_calories=total_cals,
            total_proteins=round(total_p, 1),
            total_fats=round(total_f, 1),
            total_carbs=round(total_c, 1),
            meals=[
                {
                    'id': m['id'],
                    'time': m['ts'].strftime('%H:%M'),
                    'text': m['text'],
                    'calories': m['calories'],
                    'proteins': m['proteins'],
                    'fats': m['fats'],
                    'carbs': m['carbs']
                }
                for m in meals
            ]
        )
        
    except Exception as e:
        logger.exception(f"Error getting today diary for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/diary/week", response_model=DiaryPeriod)
async def get_week_diary(user = Depends(get_current_user)):
    """Получить дневник за неделю"""
    user_id = user['user_id']
    
    try:
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        
        meals = execute_query(
            """
            SELECT DATE(ts) as day, 
                   SUM(calories) as total_calories,
                   SUM(proteins) as total_proteins,
                   SUM(fats) as total_fats,
                   SUM(carbs) as total_carbs
            FROM meals
            WHERE user_id = %s AND ts >= %s
            GROUP BY DATE(ts)
            ORDER BY day DESC
            """,
            (user_id, week_ago)
        )
        
        days = []
        total_cals = 0
        
        for meal_day in meals:
            day_cals = int(meal_day['total_calories'] or 0)
            total_cals += day_cals
            
            days.append(DiaryDay(
                date=meal_day['day'].isoformat(),
                total_calories=day_cals,
                total_proteins=round(meal_day['total_proteins'] or 0, 1),
                total_fats=round(meal_day['total_fats'] or 0, 1),
                total_carbs=round(meal_day['total_carbs'] or 0, 1),
                meals=[]  # Детали не нужны для недельного обзора
            ))
        
        avg_daily = total_cals // max(1, len(days))
        
        return DiaryPeriod(
            days=days,
            total_calories=total_cals,
            average_daily=avg_daily
        )
        
    except Exception as e:
        logger.exception(f"Error getting week diary for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/diary/meal/{meal_id}")
async def delete_meal(
    meal_id: int,
    user = Depends(get_current_user)
):
    """Удалить приём пищи (отмена последнего)"""
    user_id = user['user_id']
    
    try:
        # Проверяем что еда принадлежит пользователю
        meal = execute_query(
            "SELECT id FROM meals WHERE id = %s AND user_id = %s",
            (meal_id, user_id)
        )
        
        if not meal:
            raise HTTPException(status_code=404, detail="Meal not found")
        
        execute_update(
            "DELETE FROM meals WHERE id = %s",
            (meal_id,)
        )
        
        return {"success": True, "message": "Meal deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting meal {meal_id}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== STATISTICS ====================

@app.get("/api/stats/weekly", response_model=WeeklyStats)
async def get_weekly_stats(user = Depends(get_current_user)):
    """
    Статистика за последние 7 дней
    """
    user_id = user['user_id']
    
    try:
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        
        meals = execute_query(
            """
            SELECT DATE(ts) as day,
                   COUNT(*) as meal_count,
                   SUM(calories) as calories,
                   SUM(proteins) as proteins,
                   SUM(fats) as fats,
                   SUM(carbs) as carbs
            FROM meals
            WHERE user_id = %s AND ts >= %s
            GROUP BY DATE(ts)
            ORDER BY day
            """,
            (user_id, week_ago)
        )
        
        if not meals:
            return WeeklyStats(
                days=[],
                average_calories=0,
                average_proteins=0.0,
                average_fats=0.0,
                average_carbs=0.0,
                total_meals=0
            )
        
        days_data = [
            {
                'date': m['day'].isoformat(),
                'calories': int(m['calories'] or 0),
                'proteins': round(m['proteins'] or 0, 1),
                'fats': round(m['fats'] or 0, 1),
                'carbs': round(m['carbs'] or 0, 1),
                'meal_count': m['meal_count']
            }
            for m in meals
        ]
        
        total_meals = sum(d['meal_count'] for d in days_data)
        num_days = len(days_data)
        
        avg_cals = sum(d['calories'] for d in days_data) // num_days
        avg_p = sum(d['proteins'] for d in days_data) / num_days
        avg_f = sum(d['fats'] for d in days_data) / num_days
        avg_c = sum(d['carbs'] for d in days_data) / num_days
        
        return WeeklyStats(
            days=days_data,
            average_calories=avg_cals,
            average_proteins=round(avg_p, 1),
            average_fats=round(avg_f, 1),
            average_carbs=round(avg_c, 1),
            total_meals=total_meals
        )
        
    except Exception as e:
        logger.exception(f"Error getting weekly stats for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WEIGHT TRACKING ====================

@app.post("/api/weight/add")
async def add_weight_entry(
    entry: WeightEntry,
    user = Depends(get_current_user)
):
    """Добавить запись веса"""
    user_id = user['user_id']
    
    try:
        now = datetime.now(timezone.utc)
        
        execute_update(
            "INSERT INTO weight_tracking (user_id, weight, ts) VALUES (%s, %s, %s)",
            (user_id, entry.weight, now)
        )
        
        return {
            "success": True,
            "message": f"Weight {entry.weight} kg recorded"
        }
        
    except Exception as e:
        logger.exception(f"Error adding weight for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/weight/history", response_model=WeightHistory)
async def get_weight_history(
    days: int = Query(30, ge=1, le=365),
    user = Depends(get_current_user)
):
    """Получить историю веса"""
    user_id = user['user_id']
    
    try:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        
        weights = execute_query(
            """
            SELECT weight, ts
            FROM weight_tracking
            WHERE user_id = %s AND ts >= %s
            ORDER BY ts DESC
            """,
            (user_id, since)
        )
        
        if not weights:
            return WeightHistory(entries=[], current_weight=None, start_weight=None, change=None)
        
        current = weights[0]['weight']
        start = weights[-1]['weight']
        change = round(current - start, 1)
        
        return WeightHistory(
            entries=[
                {
                    'weight': w['weight'],
                    'date': w['ts'].date().isoformat(),
                    'timestamp': w['ts'].isoformat()
                }
                for w in weights
            ],
            current_weight=current,
            start_weight=start,
            change=change
        )
        
    except Exception as e:
        logger.exception(f"Error getting weight history for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== NUTRITION PLAN ====================

@app.post("/api/plan/generate")
async def generate_nutrition_plan(
    plan_request: NutritionPlanRequest,
    user = Depends(get_current_user)
):
    """
    Генерация персонального плана питания
    """
    user_id = user['user_id']
    
    # Проверяем доступ
    if AI_AVAILABLE and not has_access(user_id, user.get('username')):
        raise HTTPException(
            status_code=403,
            detail="Subscription required"
        )
    
    if not AI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AI service unavailable"
        )
    
    try:
        # Расчёт калорий (формула Миффлина-Сан Жеора)
        if plan_request.sex == 'male':
            bmr = 10 * plan_request.weight + 6.25 * plan_request.height - 5 * plan_request.age + 5
        else:
            bmr = 10 * plan_request.weight + 6.25 * plan_request.height - 5 * plan_request.age - 161
        
        activity_multipliers = {
            "Сидячий образ жизни": 1.2,
            "Легкая активность": 1.375,
            "Умеренная активность": 1.55,
            "Высокая активность": 1.725,
            "Экстремальная активность": 1.9
        }
        multiplier = activity_multipliers.get(plan_request.activity, 1.55)
        
        goal_adjustments = {
            "снижение веса": 0.85,
            "поддержание": 1.0,
            "набор": 1.15
        }
        adjustment = goal_adjustments.get(plan_request.goal.lower(), 1.0)
        
        daily_calories = round(bmr * multiplier * adjustment)
        
        # Генерируем план через AI
        prompt = f"""Составь 7-дневный персональный план питания (завтрак/обед/ужин/перекусы), 
ориентировочные граммовки и примерную калорийность в день. Пиши структурированно.

Данные клиента:
- Возраст: {plan_request.age}
- Пол: {plan_request.sex}
- Вес: {plan_request.weight} кг
- Рост: {plan_request.height} см
- Активность: {plan_request.activity}
- Цель: {plan_request.goal}
- Предпочтения: {plan_request.preferences}
- Ограничения: {plan_request.restrictions}
- Рекомендуемая калорийность: {daily_calories} ккал/день"""
        
        plan_text = await ai_chat(
            "Ты профессиональный нутрициолог и диетолог. Пиши структурированно.",
            prompt,
            0.4
        )
        
        return {
            "success": True,
            "daily_calories": daily_calories,
            "plan": plan_text
        }
        
    except Exception as e:
        logger.exception(f"Error generating plan for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CHALLENGES ====================

@app.get("/api/challenges/list", response_model=ChallengeList)
async def get_challenges_list(user = Depends(get_current_user)):
    """Получить список челленджей"""
    user_id = user['user_id']
    
    try:
        # Активные челленджи
        active = execute_query(
            """
            SELECT challenge_type, progress, completed
            FROM challenges
            WHERE user_id = %s AND completed = 0
            """,
            (user_id,)
        )
        
        active_challenges = []
        for ch in active:
            if AI_AVAILABLE:
                name = get_challenge_name(ch['challenge_type'])
            else:
                name = ch['challenge_type']
            
            active_challenges.append(ChallengeProgress(
                challenge_type=ch['challenge_type'],
                name=name,
                progress=ch['progress'],
                total_days=7,
                completed=False
            ))
        
        # Доступные челленджи
        available = [
            {"type": "water_challenge", "name": "💧 Пить 2л воды ежедневно", "days": 7},
            {"type": "steps_challenge", "name": "🏃 8k шагов ежедневно", "days": 7},
            {"type": "diet_challenge", "name": "🥗 Здоровое питание 7 дней", "days": 7},
            {"type": "workout_challenge", "name": "🏋️ Тренировки 3х в неделю", "days": 7},
            {"type": "tracking_challenge", "name": "📊 Отслеживать калории", "days": 7},
            {"type": "nosugar_challenge", "name": "🚫 Без сахара 7 дней", "days": 7}
        ]
        
        # Убираем уже активные
        active_types = {ch['challenge_type'] for ch in active}
        available = [av for av in available if av['type'] not in active_types]
        
        return ChallengeList(
            active=active_challenges,
            available=available
        )
        
    except Exception as e:
        logger.exception(f"Error getting challenges for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/challenges/start/{challenge_type}")
async def start_challenge(
    challenge_type: str,
    user = Depends(get_current_user)
):
    """Начать челлендж"""
    user_id = user['user_id']
    
    try:
        if AI_AVAILABLE:
            success = init_challenge(user_id, challenge_type)
        else:
            # Fallback без импорта функций
            now = datetime.now(timezone.utc)
            execute_update(
                """
                INSERT INTO challenges (user_id, challenge_type, start_date, progress, completed)
                VALUES (%s, %s, %s, 0, 0)
                ON CONFLICT (user_id, challenge_type) DO NOTHING
                """,
                (user_id, challenge_type, now)
            )
            success = True
        
        if success:
            return {"success": True, "message": "Challenge started"}
        else:
            raise HTTPException(status_code=400, detail="Failed to start challenge")
        
    except Exception as e:
        logger.exception(f"Error starting challenge for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/challenges/log/{challenge_type}")
async def log_challenge_progress(
    challenge_type: str,
    user = Depends(get_current_user)
):
    """Отметить прогресс челленджа (один день)"""
    user_id = user['user_id']
    
    try:
        if AI_AVAILABLE:
            updated = update_challenge_progress(user_id, challenge_type)
        else:
            # Fallback
            today = datetime.now(timezone.utc).date().isoformat()
            
            # Проверяем не отмечено ли уже
            existing = execute_query(
                """
                SELECT 1 FROM challenge_logs
                WHERE user_id = %s AND challenge_type = %s AND log_date = %s
                """,
                (user_id, challenge_type, today)
            )
            
            if existing:
                return {"success": False, "message": "Already logged today"}
            
            # Добавляем лог
            execute_update(
                """
                INSERT INTO challenge_logs (user_id, challenge_type, log_date, completed)
                VALUES (%s, %s, %s, 1)
                """,
                (user_id, challenge_type, today)
            )
            
            # Обновляем прогресс
            execute_update(
                """
                UPDATE challenges
                SET progress = progress + 1
                WHERE user_id = %s AND challenge_type = %s
                """,
                (user_id, challenge_type)
            )
            
            # Проверяем завершение
            challenge = execute_query(
                "SELECT progress FROM challenges WHERE user_id = %s AND challenge_type = %s",
                (user_id, challenge_type)
            )
            
            if challenge and challenge[0]['progress'] >= 7:
                execute_update(
                    "UPDATE challenges SET completed = 1 WHERE user_id = %s AND challenge_type = %s",
                    (user_id, challenge_type)
                )
                
                # Добавляем достижение
                achievement_name = f"Челлендж: {challenge_type}"
                execute_update(
                    """
                    INSERT INTO achievements (user_id, badge, ts)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (user_id, achievement_name, datetime.now(timezone.utc))
                )
            
            updated = True
        
        if updated:
            return {"success": True, "message": "Progress logged"}
        else:
            return {"success": False, "message": "Already logged today"}
        
    except Exception as e:
        logger.exception(f"Error logging challenge for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ACHIEVEMENTS ====================

@app.get("/api/achievements/list")
async def get_achievements(user = Depends(get_current_user)):
    """Получить список достижений"""
    user_id = user['user_id']
    
    try:
        achievements = execute_query(
            """
            SELECT badge, ts
            FROM achievements
            WHERE user_id = %s
            ORDER BY ts DESC
            """,
            (user_id,)
        )
        
        return {
            "achievements": [
                {
                    "badge": a['badge'],
                    "earned_at": a['ts'].isoformat(),
                    "description": get_achievement_description(a['badge'])
                }
                for a in achievements
            ],
            "total": len(achievements)
        }
        
    except Exception as e:
        logger.exception(f"Error getting achievements for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

def get_achievement_description(badge: str) -> str:
    """Получить описание достижения"""
    descriptions = {
        "Завтрак-герой": "Завтракал 7 дней подряд",
        "Повелитель воды": "Пьёт достаточно воды",
        "7 дней без сахара": "Неделя без сладкого",
        "Первые шаги": "Добавил первую еду в дневник",
        "Марафонец": "30 дней подряд ведёт дневник"
    }
    return descriptions.get(badge, "Особое достижение")

# ==================== LABS ANALYSIS ====================

@app.post("/api/labs/analyze")
async def analyze_labs(
    labs: LabsAnalysis,
    user = Depends(get_current_user)
):
    """
    Анализ лабораторных данных
    """
    user_id = user['user_id']
    
    # Проверяем кредиты (кроме админа)
    if not AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI service unavailable")
    
    is_admin = user.get('username', '').lower() == os.getenv('ADMIN_USERNAME', '').lower()
    
    if not is_admin:
        # Проверяем бесплатную попытку
        used_free = execute_query(
            "SELECT used_free_lab FROM subscriptions WHERE user_id = %s",
            (user_id,)
        )
        
        if not used_free or not used_free[0].get('used_free_lab'):
            # Активируем бесплатную
            execute_update(
                "UPDATE subscriptions SET used_free_lab = 1 WHERE user_id = %s",
                (user_id,)
            )
        else:
            # Проверяем кредиты
            credits = get_labs_credits(user_id)
            if credits <= 0:
                raise HTTPException(
                    status_code=402,
                    detail="No credits available"
                )
            
            # Списываем кредит
            if not consume_labs_credit(user_id):
                raise HTTPException(
                    status_code=402,
                    detail="Failed to consume credit"
                )
    
    try:
        # Анализируем через AI
        prompt = f"Ты нутрициолог. Проанализируй лабораторные анализы и дай практические рекомендации.\n\n{labs.text}"
        
        analysis = await ai_chat(
            "Пиши кратко и структурированно.",
            prompt,
            0.3
        )
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.exception(f"Error analyzing labs for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/labs/credits")
async def get_labs_credits_count(user = Depends(get_current_user)):
    """Получить количество оставшихся кредитов на анализы"""
    user_id = user['user_id']
    
    try:
        if AI_AVAILABLE:
            credits = get_labs_credits(user_id)
        else:
            result = execute_query(
                "SELECT labs_credits FROM credits WHERE user_id = %s",
                (user_id,)
            )
            credits = result[0]['labs_credits'] if result else 0
        
        return {"credits": credits}
        
    except Exception as e:
        logger.exception(f"Error getting credits for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== RECIPES ====================

@app.post("/api/recipe/generate")
async def generate_recipe(
    recipe_request: RecipeRequest,
    user = Depends(get_current_user)
):
    """Генерация рецептов из продуктов"""
    user_id = user['user_id']
    
    if not AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI service unavailable")
    
    if not has_access(user_id, user.get('username')):
        raise HTTPException(status_code=403, detail="Subscription required")
    
    try:
        prompt = f"""Ты шеф-повар и нутрициолог. На основе списка продуктов составь 3 рецепта. Для каждого: 
название, ингредиенты с граммовками, шаги приготовления, калорийность и БЖУ на порцию.

Продукты:
{recipe_request.products}"""
        
        recipes = await ai_chat(
            "Пиши структурированно, ясно.",
            prompt,
            0.5
        )
        
        return {
            "success": True,
            "recipes": recipes
        }
        
    except Exception as e:
        logger.exception(f"Error generating recipes for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Обработчик HTTP ошибок"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Обработчик всех остальных ошибок"""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    """Действия при запуске сервера"""
    logger.info("🚀 NutriCoach Mini App Backend started")
    logger.info(f"AI Available: {AI_AVAILABLE}")
    
    # Проверяем подключение к БД
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        logger.info("✅ Database connection OK")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Только для разработки!
        log_level="info"
    )
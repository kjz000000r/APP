# -*- coding: utf-8 -*-
"""
Pydantic модели для валидации данных
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

# ==================== USER MODELS ====================

class UserProfile(BaseModel):
    """Профиль пользователя"""
    user_id: int
    username: Optional[str] = None
    subscription_status: str  # "active", "trial", "expired"
    expires_at: Optional[datetime] = None
    free_until: Optional[datetime] = None
    labs_credits: int = 0
    achievements_count: int = 0
    active_challenges: int = 0

# ==================== MEAL MODELS ====================

class MealEntry(BaseModel):
    """Запись приёма пищи"""
    text: str = Field(..., min_length=1, max_length=1000)
    calories: Optional[int] = Field(None, ge=0, le=10000)
    proteins: Optional[float] = Field(None, ge=0, le=500)
    fats: Optional[float] = Field(None, ge=0, le=500)
    carbs: Optional[float] = Field(None, ge=0, le=1000)
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

class MealResponse(BaseModel):
    """Ответ после добавления еды"""
    success: bool
    meal_id: Optional[int] = None
    calories: int
    proteins: float
    fats: float
    carbs: float
    summary: str
    source: str  # "database" или "ai"

# ==================== DIARY MODELS ====================

class DiaryDay(BaseModel):
    """Один день в дневнике"""
    date: str  # YYYY-MM-DD
    total_calories: int
    total_proteins: float
    total_fats: float
    total_carbs: float
    meals: List[dict]

class DiaryPeriod(BaseModel):
    """Дневник за период"""
    days: List[DiaryDay]
    total_calories: int
    average_daily: int

# ==================== STATS MODELS ====================

class WeeklyStats(BaseModel):
    """Статистика за неделю"""
    days: List[dict]
    average_calories: int
    average_proteins: float
    average_fats: float
    average_carbs: float
    total_meals: int

class WeightEntry(BaseModel):
    """Запись веса"""
    weight: float = Field(..., ge=20, le=300)
    
class WeightHistory(BaseModel):
    """История веса"""
    entries: List[dict]
    current_weight: Optional[float] = None
    start_weight: Optional[float] = None
    change: Optional[float] = None

# ==================== PLAN MODELS ====================

class NutritionPlanRequest(BaseModel):
    """Запрос на создание плана питания"""
    age: int = Field(..., ge=1, le=120)
    sex: str = Field(..., regex="^(male|female)$")
    weight: float = Field(..., ge=20, le=300)
    height: float = Field(..., ge=100, le=250)
    activity: str
    goal: str
    preferences: Optional[str] = ""
    restrictions: Optional[str] = ""

# ==================== CHALLENGE MODELS ====================

class ChallengeProgress(BaseModel):
    """Прогресс челленджа"""
    challenge_type: str
    name: str
    progress: int
    total_days: int
    completed: bool

class ChallengeList(BaseModel):
    """Список челленджей"""
    active: List[ChallengeProgress]
    available: List[dict]

# ==================== LABS MODELS ====================

class LabsAnalysis(BaseModel):
    """Запрос на анализ лабораторных данных"""
    text: str = Field(..., min_length=10)

# ==================== RECIPE MODELS ====================

class RecipeRequest(BaseModel):
    """Запрос рецепта"""
    products: str = Field(..., min_length=3)

# ==================== ACHIEVEMENT MODELS ====================

class Achievement(BaseModel):
    """Достижение"""
    badge: str
    earned_at: datetime
    description: Optional[str] = None
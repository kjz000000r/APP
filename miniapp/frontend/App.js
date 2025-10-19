// ============================================
// ИНИЦИАЛИЗАЦИЯ TELEGRAM WEB APP
// ============================================

const tg = window.Telegram.WebApp;
tg.expand();
tg.ready();

// Применяем тему
if (tg.colorScheme === 'dark') {
    document.body.classList.add('dark-theme');
}

// ============================================
// КОНФИГУРАЦИЯ API
// ============================================

const API_BASE = 'https://your-backend-domain.com/api';  // TODO: Заменить на ваш домен!

// ============================================
// УТИЛИТЫ ДЛЯ API ЗАПРОСОВ
// ============================================

/**
 * Выполнить API запрос с авторизацией
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    
    // Добавляем авторизацию через initData
    const headers = {
        'Content-Type': 'application/json',
        'Authorization': `tma ${tg.initData}`,
        ...options.headers
    };
    
    try {
        const response = await fetch(url, {
            ...options,
            headers
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Ошибка запроса');
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showError(error.message);
        throw error;
    }
}

/**
 * GET запрос
 */
async function apiGet(endpoint) {
    return apiRequest(endpoint, { method: 'GET' });
}

/**
 * POST запрос
 */
async function apiPost(endpoint, data) {
    return apiRequest(endpoint, {
        method: 'POST',
        body: JSON.stringify(data)
    });
}

/**
 * DELETE запрос
 */
async function apiDelete(endpoint) {
    return apiRequest(endpoint, { method: 'DELETE' });
}

// ============================================
// ПОКАЗ УВЕДОМЛЕНИЙ
// ============================================

function showError(message) {
    tg.showAlert(message);
}

function showSuccess(message) {
    tg.showPopup({
        title: 'Успешно!',
        message: message,
        buttons: [{ type: 'ok' }]
    });
}

function showConfirm(message, onConfirm) {
    tg.showConfirm(message, onConfirm);
}

// ============================================
// УПРАВЛЕНИЕ ЭКРАНАМИ
// ============================================

function showLoading() {
    document.getElementById('loading').style.display = 'flex';
    document.getElementById('main-screen').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('main-screen').style.display = 'block';
}

// ============================================
// ЗАГРУЗКА ПРОФИЛЯ
// ============================================

let userProfile = null;

async function loadProfile() {
    try {
        userProfile = await apiGet('/user/profile');
        
        // Обновляем UI
        document.getElementById('username').textContent = 
            userProfile.username || tg.initDataUnsafe.user?.first_name || 'Пользователь';
        
        const badge = document.getElementById('sub-badge');
        badge.textContent = getSubscriptionText(userProfile.subscription_status);
        badge.className = `subscription-badge ${userProfile.subscription_status}`;
        
        return userProfile;
    } catch (error) {
        console.error('Failed to load profile:', error);
        return null;
    }
}

function getSubscriptionText(status) {
    const texts = {
        'active': '⭐ Активна',
        'trial': '🎁 Пробная',
        'expired': '❌ Истекла'
    };
    return texts[status] || 'Нет подписки';
}

// ============================================
// ЗАГРУЗКА СЕГОДНЯШНЕГО ДНЕВНИКА
// ============================================

async function loadTodayDiary() {
    try {
        const today = await apiGet('/diary/today');
        
        // Обновляем карточку калорий
        document.getElementById('today-calories').textContent = today.total_calories;
        document.getElementById('today-proteins').textContent = today.total_proteins;
        document.getElementById('today-fats').textContent = today.total_fats;
        document.getElementById('today-carbs').textContent = today.total_carbs;
        
        // Обновляем список приёмов пищи
        const mealsContainer = document.getElementById('today-meals');
        
        if (today.meals.length === 0) {
            mealsContainer.innerHTML = `
                <div class="text-center text-muted" style="padding: 20px;">
                    Пока нет записей. Добавьте первый приём пищи!
                </div>
            `;
        } else {
            mealsContainer.innerHTML = today.meals.map(meal => `
                <div class="meal-item">
                    <div class="meal-info">
                        <div class="meal-time">${meal.time}</div>
                        <div class="meal-text">${escapeHtml(meal.text)}</div>
                    </div>
                    <div class="meal-calories">${meal.calories} ккал</div>
                </div>
            `).join('');
        }
    } catch (error) {
        console.error('Failed to load today diary:', error);
    }
}

// ============================================
// ЗАГРУЗКА АКТИВНЫХ ЧЕЛЛЕНДЖЕЙ
// ============================================

async function loadActiveChallenges() {
    try {
        const challenges = await apiGet('/challenges/list');
        
        const container = document.getElementById('active-challenges');
        
        if (challenges.active.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted" style="padding: 20px; grid-column: 1 / -1;">
                    Нет активных челленджей
                </div>
            `;
        } else {
            container.innerHTML = challenges.active.map(ch => {
                const progress = (ch.progress / ch.total_days) * 100;
                return `
                    <div class="challenge-card" onclick="openChallenge('${ch.challenge_type}')">
                        <div class="challenge-icon">${getChallengeIcon(ch.challenge_type)}</div>
                        <div class="challenge-name">${ch.name}</div>
                        <div class="challenge-progress">${ch.progress}/${ch.total_days}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${progress}%"></div>
                        </div>
                    </div>
                `;
            }).join('');
        }
    } catch (error) {
        console.error('Failed to load challenges:', error);
    }
}

function getChallengeIcon(type) {
    const icons = {
        'water_challenge': '💧',
        'steps_challenge': '🏃',
        'diet_challenge': '🥗',
        'workout_challenge': '🏋️',
        'tracking_challenge': '📊',
        'nosugar_challenge': '🚫'
    };
    return icons[type] || '⭐';
}

// ============================================
// УТИЛИТЫ
// ============================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ
// ============================================

async function initApp() {
    showLoading();
    
    try {
        // Загружаем данные параллельно
        await Promise.all([
            loadProfile(),
            loadTodayDiary(),
            loadActiveChallenges()
        ]);
        
        hideLoading();
    } catch (error) {
        console.error('App initialization failed:', error);
        showError('Не удалось загрузить данные. Попробуйте перезапустить приложение.');
    }
}

// ============================================
// МОДАЛЬНЫЕ ОКНА
// ============================================

function openModal(title, content) {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal">
            <div class="modal-header">
                <h3 class="modal-title">${title}</h3>
                <button class="btn-close" onclick="closeModal()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-content">
                ${content}
            </div>
        </div>
    `;
    
    document.getElementById('modal-container').appendChild(modal);
    
    // Закрытие по клику на overlay
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });
}

function closeModal() {
    const container = document.getElementById('modal-container');
    container.innerHTML = '';
}

// ============================================
// ФУНКЦИИ ОТКРЫТИЯ РАЗДЕЛОВ
// ============================================

function openMealEntry() {
    openModal('Добавить еду', `
        <form id="meal-form">
            <div class="form-group">
                <label class="form-label">Что вы съели?</label>
                <textarea class="form-textarea" id="meal-text" placeholder="Например:
куриная грудка 150 г
рис 70 г
огурец 100 г" required></textarea>
            </div>
            <button type="submit" class="btn btn-primary">
                <i class="fas fa-check"></i>
                Добавить
            </button>
        </form>
    `);
    
    // Обработчик формы
    document.getElementById('meal-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const text = document.getElementById('meal-text').value.trim();
        if (!text) return;
        
        try {
            tg.MainButton.showProgress();
            
            const result = await apiPost('/diary/add', { text });
            
            showSuccess(`Добавлено: ${result.calories} ккал`);
            closeModal();
            await loadTodayDiary();
        } catch (error) {
            // Ошибка уже показана в apiPost
        } finally {
            tg.MainButton.hideProgress();
        }
    });
}

function openDiary() {
    // Реализация в components/diary.js
    if (typeof showDiaryScreen === 'function') {
        showDiaryScreen();
    }
}

function openStats() {
    // Реализация в components/stats.js
    if (typeof showStatsScreen === 'function') {
        showStatsScreen();
    }
}

function openProfile() {
    // Реализация в components/profile.js
    if (typeof showProfileScreen === 'function') {
        showProfileScreen();
    }
}

function openChallenges() {
    // Реализация в components/challenges.js
    if (typeof showChallengesScreen === 'function') {
        showChallengesScreen();
    }
}

function openChallenge(type) {
    // Реализация в components/challenges.js
    if (typeof showChallengeDetail === 'function') {
        showChallengeDetail(type);
    }
}

function openPlanGenerator() {
    // Реализация в components/plan.js
    if (typeof showPlanGenerator === 'function') {
        showPlanGenerator();
    }
}

function openLabsAnalysis() {
    // Реализация в components/labs.js
    if (typeof showLabsAnalysis === 'function') {
        showLabsAnalysis();
    }
}

function openRecipeGen() {
    // Реализация в components/recipes.js
    if (typeof showRecipeGenerator === 'function') {
        showRecipeGenerator();
    }
}

function openWeightTracker() {
    openModal('Отслеживание веса', `
        <form id="weight-form">
            <div class="form-group">
                <label class="form-label">Текущий вес (кг)</label>
                <input type="number" class="form-input" id="weight-input" 
                       step="0.1" min="20" max="300" placeholder="68.5" required>
            </div>
            <button type="submit" class="btn btn-primary">
                <i class="fas fa-save"></i>
                Записать
            </button>
        </form>
        <div id="weight-history" style="margin-top: 20px;"></div>
    `);
    
    // Обработчик формы
    document.getElementById('weight-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const weight = parseFloat(document.getElementById('weight-input').value);
        if (!weight) return;
        
        try {
            await apiPost('/weight/add', { weight });
            showSuccess(`Вес ${weight} кг записан!`);
            document.getElementById('weight-input').value = '';
            loadWeightHistory();
        } catch (error) {
            // Ошибка уже показана
        }
    });
    
    loadWeightHistory();
}

async function loadWeightHistory() {
    try {
        const history = await apiGet('/weight/history?days=30');
        
        const container = document.getElementById('weight-history');
        if (!container) return;
        
        if (!history.entries || history.entries.length === 0) {
            container.innerHTML = '<p class="text-muted text-center">Нет записей</p>';
            return;
        }
        
        container.innerHTML = `
            <h4>История за 30 дней</h4>
            <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; margin-top: 8px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span>Текущий:</span>
                    <strong>${history.current_weight} кг</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span>Начальный:</span>
                    <strong>${history.start_weight} кг</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                <span>Изменение:</span>
                    <strong style="color: ${history.change < 0 ? 'var(--success-color)' : 'var(--danger-color)'}">
                        ${history.change > 0 ? '+' : ''}${history.change} кг
                    </strong>
                </div>
            </div>
            <div style="margin-top: 16px;">
                ${history.entries.slice(0, 5).map(entry => `
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border-color);">
                        <span class="text-muted">${new Date(entry.date).toLocaleDateString('ru-RU')}</span>
                        <strong>${entry.weight} кг</strong>
                    </div>
                `).join('')}
            </div>
        `;
    } catch (error) {
        console.error('Failed to load weight history:', error);
    }
}

function openAchievements() {
    showAchievementsScreen();
}

async function showAchievementsScreen() {
    try {
        const data = await apiGet('/achievements/list');
        
        const content = data.achievements.length === 0 ? `
            <div class="text-center text-muted" style="padding: 40px;">
                <i class="fas fa-trophy" style="font-size: 48px; opacity: 0.3; margin-bottom: 16px;"></i>
                <p>Пока нет достижений</p>
                <p>Ведите дневник и участвуйте в челленджах!</p>
            </div>
        ` : `
            <div style="display: flex; flex-direction: column; gap: 12px;">
                ${data.achievements.map(ach => `
                    <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div style="font-size: 32px;">🏆</div>
                            <div style="flex: 1;">
                                <div style="font-weight: 600;">${escapeHtml(ach.badge)}</div>
                                <div class="text-muted" style="font-size: 12px;">
                                    ${new Date(ach.earned_at).toLocaleDateString('ru-RU')}
                                </div>
                                ${ach.description ? `<div style="font-size: 13px; margin-top: 4px;">${escapeHtml(ach.description)}</div>` : ''}
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        openModal(`🏆 Достижения (${data.total})`, content);
    } catch (error) {
        showError('Не удалось загрузить достижения');
    }
}

function openSubscription() {
    openModal('💎 Подписка', `
        <div style="text-align: center; padding: 20px;">
            <p class="text-muted">Подписка управляется через основного бота</p>
            <p style="margin-top: 16px;">Используйте команду /subscribe в боте для оформления или продления подписки</p>
            <button class="btn btn-primary" onclick="closeModal()" style="margin-top: 24px;">
                Понятно
            </button>
        </div>
    `);
}

// ============================================
// ЗАПУСК ПРИЛОЖЕНИЯ
// ============================================

// Запускаем инициализацию при загрузке
document.addEventListener('DOMContentLoaded', initApp);
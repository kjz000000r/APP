// ============================================
// ГЕНЕРАТОР ПЛАНА ПИТАНИЯ
// ============================================

function showPlanGenerator() {
    const content = `
        <form id="plan-form">
            <div class="form-group">
                <label class="form-label">Возраст</label>
                <input type="number" class="form-input" id="plan-age" min="1" max="120" required>
            </div>
            
            <div class="form-group">
                <label class="form-label">Пол</label>
                <select class="form-select" id="plan-sex" required>
                    <option value="">Выберите</option>
                    <option value="male">Мужской</option>
                    <option value="female">Женский</option>
                </select>
            </div>
            
            <div class="form-group">
                <label class="form-label">Вес (кг)</label>
                <input type="number" class="form-input" id="plan-weight" step="0.1" min="20" max="300" required>
            </div>
            
            <div class="form-group">
                <label class="form-label">Рост (см)</label>
                <input type="number" class="form-input" id="plan-height" min="100" max="250" required>
            </div>
            
            <div class="form-group">
                <label class="form-label">Активность</label>
                <select class="form-select" id="plan-activity" required>
                    <option value="">Выберите</option>
                    <option value="Сидячий образ жизни">Сидячий образ жизни</option>
                    <option value="Легкая активность">Легкая активность</option>
                    <option value="Умеренная активность">Умеренная активность</option>
                    <option value="Высокая активность">Высокая активность</option>
                    <option value="Экстремальная активность">Экстремальная активность</option>
                </select>
            </div>
            
            <div class="form-group">
                <label class="form-label">Цель</label>
                <select class="form-select" id="plan-goal" required>
                    <option value="">Выберите</option>
                    <option value="снижение веса">Снижение веса</option>
                    <option value="поддержание">Поддержание веса</option>
                    <option value="набор">Набор массы</option>
                </select>
            </div>
            
            <div class="form-group">
                <label class="form-label">Предпочтения (необязательно)</label>
                <textarea class="form-textarea" id="plan-preferences" 
                          placeholder="Например: предпочитаю курицу, люблю овощи"></textarea>
            </div>
            
            <div class="form-group">
                <label class="form-label">Ограничения (необязательно)</label>
                <textarea class="form-textarea" id="plan-restrictions" 
                          placeholder="Например: аллергия на орехи, не ем свинину"></textarea>
            </div>
            
            <button type="submit" class="btn btn-primary">
                <i class="fas fa-magic"></i>
                Создать план
            </button>
        </form>
    `;
    
    openModal('🥗 Генератор плана питания', content);
    
    document.getElementById('plan-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const data = {
            age: parseInt(document.getElementById('plan-age').value),
            sex: document.getElementById('plan-sex').value,
            weight: parseFloat(document.getElementById('plan-weight').value),
            height: parseFloat(document.getElementById('plan-height').value),
            activity: document.getElementById('plan-activity').value,
            goal: document.getElementById('plan-goal').value,
            preferences: document.getElementById('plan-preferences').value || '',
            restrictions: document.getElementById('plan-restrictions').value || ''
        };
        
        try {
            tg.MainButton.showProgress();
            
            const result = await apiPost('/plan/generate', data);
            
            closeModal();
            
            showPlanResult(result);
        } catch (error) {
            // Ошибка уже показана
        } finally {
            tg.MainButton.hideProgress();
        }
    });
}

function showPlanResult(result) {
    const content = `
        <div class="plan-result">
            <div style="background: linear-gradient(135deg, var(--primary-color), var(--primary-dark)); 
                        color: white; padding: 24px; border-radius: 12px; text-align: center; margin-bottom: 24px;">
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">Рекомендуемая калорийность</div>
                <div style="font-size: 48px; font-weight: 700;">${result.daily_calories}</div>
                <div style="font-size: 14px; opacity: 0.9;">ккал/день</div>
            </div>
            
            <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; white-space: pre-wrap; line-height: 1.8; font-size: 14px;">
                ${escapeHtml(result.plan)}
            </div>
        </div>
    `;
    
    openModal('✨ Ваш персональный план', content);
}
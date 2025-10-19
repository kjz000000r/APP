// ============================================
// ЧЕЛЛЕНДЖИ
// ============================================

async function showChallengesScreen() {
    try {
        const data = await apiGet('/challenges/list');
        
        const content = `
            <div class="challenges-screen">
                ${data.active.length > 0 ? `
                    <h4 style="margin-bottom: 12px;">Активные</h4>
                    <div style="display: flex; flex-direction: column; gap: 12px; margin-bottom: 24px;">
                        ${data.active.map(ch => {
                            const progress = (ch.progress / ch.total_days) * 100;
                            return `
                                <div onclick="showChallengeDetail('${ch.challenge_type}')" 
                                     style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; cursor: pointer;">
                                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                                        <div style="font-size: 32px;">${getChallengeIcon(ch.challenge_type)}</div>
                                        <div style="flex: 1;">
                                            <div style="font-weight: 600; margin-bottom: 4px;">${escapeHtml(ch.name)}</div>
                                            <div style="font-size: 14px; color: var(--primary-color); font-weight: 600;">
                                                ${ch.progress}/${ch.total_days} дней
                                            </div>
                                        </div>
                                    </div>
                                    <div class="progress-bar" style="height: 6px;">
                                        <div class="progress-fill" style="width: ${progress}%"></div>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                ` : ''}
                
                <h4 style="margin-bottom: 12px;">Доступные челленджи</h4>
                <div style="display: flex; flex-direction: column; gap: 12px;">
                    ${data.available.map(ch => `
                        <div onclick="startNewChallenge('${ch.type}')" 
                             style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; cursor: pointer; border: 2px dashed var(--border-color);">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <div style="font-size: 32px;">${getChallengeIcon(ch.type)}</div>
                                <div style="flex: 1;">
                                    <div style="font-weight: 600; margin-bottom: 4px;">${escapeHtml(ch.name)}</div>
                                    <div class="text-muted" style="font-size: 12px;">${ch.days} дней</div>
                                </div>
                                <i class="fas fa-plus" style="color: var(--primary-color);"></i>
                            </div>
                        </div>
                    `).join('')}
                </div>
                
                ${data.available.length === 0 && data.active.length === 0 ? `
                    <div class="text-center text-muted" style="padding: 40px;">
                        <p>Нет доступных челленджей</p>
                    </div>
                ` : ''}
            </div>
        `;
        
        openModal('⚡ Челленджи', content);
    } catch (error) {
        showError('Не удалось загрузить челленджи');
    }
}

async function showChallengeDetail(type) {
    closeModal();
    
    const content = `
        <div style="text-align: center; padding: 24px;">
            <div style="font-size: 64px; margin-bottom: 16px;">${getChallengeIcon(type)}</div>
            <h3 style="margin-bottom: 24px;">Отметить прогресс?</h3>
            <p class="text-muted" style="margin-bottom: 24px;">
                Подтвердите выполнение челленджа сегодня
            </p>
            <button class="btn btn-primary" onclick="logChallengeProgress('${type}')" style="margin-bottom: 12px;">
                <i class="fas fa-check"></i>
                Отметить выполнение
            </button>
            <button class="btn btn-secondary" onclick="closeModal()">
                Отмена
            </button>
        </div>
    `;
    
    openModal('Прогресс челленджа', content);
}

async function startNewChallenge(type) {
    showConfirm('Начать этот челлендж?', async (confirmed) => {
        if (!confirmed) return;
        
        try {
            await apiPost(`/challenges/start/${type}`, {});
            showSuccess('Челлендж начат! Отмечайте прогресс каждый день.');
            closeModal();
            await loadActiveChallenges();
        } catch (error) {
            // Ошибка уже показана
        }
    });
}

async function logChallengeProgress(type) {
    try {
        const result = await apiPost(`/challenges/log/${type}`, {});
        
        if (result.success) {
            showSuccess('Прогресс отмечен! Так держать! 💪');
            closeModal();
            await loadActiveChallenges();
        } else {
            showError(result.message || 'Сегодня уже отмечено');
        }
    } catch (error) {
        // Ошибка уже показана
    }
}
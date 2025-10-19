// ============================================
// ДНЕВНИК ПИТАНИЯ
// ============================================

async function showDiaryScreen() {
    try {
        const weekData = await apiGet('/diary/week');
        
        const content = `
            <div class="diary-screen">
                <div class="diary-summary" style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; margin-bottom: 16px;">
                    <div style="text-align: center;">
                        <div style="font-size: 32px; font-weight: 700; color: var(--primary-color);">
                            ${weekData.average_daily}
                        </div>
                        <div class="text-muted" style="font-size: 14px;">ккал в день в среднем</div>
                    </div>
                </div>
                
                <div class="days-list">
                    ${weekData.days.map(day => {
                        const date = new Date(day.date);
                        const dayName = date.toLocaleDateString('ru-RU', { weekday: 'short', day: 'numeric', month: 'short' });
                        
                        return `
                            <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; margin-bottom: 12px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                    <div style="font-weight: 600;">${dayName}</div>
                                    <div style="font-size: 20px; font-weight: 700; color: var(--primary-color);">
                                        ${day.total_calories} ккал
                                    </div>
                                </div>
                                <div style="display: flex; gap: 16px; font-size: 14px;">
                                    <div>
                                        <span class="text-muted">Б:</span> 
                                        <strong>${day.total_proteins}</strong>
                                    </div>
                                    <div>
                                        <span class="text-muted">Ж:</span> 
                                        <strong>${day.total_fats}</strong>
                                    </div>
                                    <div>
                                        <span class="text-muted">У:</span> 
                                        <strong>${day.total_carbs}</strong>
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
                
                ${weekData.days.length === 0 ? `
                    <div class="text-center text-muted" style="padding: 40px;">
                        <p>Нет записей за последнюю неделю</p>
                    </div>
                ` : ''}
            </div>
        `;
        
        openModal('📖 Дневник за неделю', content);
    } catch (error) {
        showError('Не удалось загрузить дневник');
    }
}
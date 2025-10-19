// ============================================
// СТАТИСТИКА
// ============================================

async function showStatsScreen() {
    try {
        const stats = await apiGet('/stats/weekly');
        
        const content = `
            <div class="stats-screen">
                <div class="stats-summary" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 24px;">
                    <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; text-align: center;">
                        <div style="font-size: 28px; font-weight: 700; color: var(--primary-color);">
                            ${stats.average_calories}
                        </div>
                        <div class="text-muted" style="font-size: 12px;">Средние ккал</div>
                    </div>
                    <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; text-align: center;">
                        <div style="font-size: 28px; font-weight: 700; color: var(--secondary-color);">
                            ${stats.total_meals}
                        </div>
                        <div class="text-muted" style="font-size: 12px;">Приёмов пищи</div>
                    </div>
                </div>
                
                <h4 style="margin-bottom: 12px;">Средние БЖУ в день</h4>
                <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; margin-bottom: 24px;">
                    <div style="display: flex; justify-content: space-around;">
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: 700; color: #3b82f6;">
                                ${stats.average_proteins}
                            </div>
                            <div class="text-muted" style="font-size: 12px;">Белки (г)</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: 700; color: #f59e0b;">
                                ${stats.average_fats}
                            </div>
                            <div class="text-muted" style="font-size: 12px;">Жиры (г)</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: 700; color: #10b981;">
                                ${stats.average_carbs}
                            </div>
                            <div class="text-muted" style="font-size: 12px;">Углеводы (г)</div>
                        </div>
                    </div>
                </div>
                
                <h4 style="margin-bottom: 12px;">По дням</h4>
                <div class="chart-container">
                    ${stats.days.map((day, index) => {
                        const date = new Date(day.date);
                        const dayName = date.toLocaleDateString('ru-RU', { weekday: 'short' });
                        const maxCal = Math.max(...stats.days.map(d => d.calories));
                        const heightPercent = (day.calories / maxCal) * 100;
                        
                        return `
                            <div style="text-align: center; flex: 1;">
                                <div style="height: 150px; display: flex; align-items: flex-end; justify-content: center; margin-bottom: 8px;">
                                    <div style="width: 40px; background: linear-gradient(to top, var(--primary-color), var(--primary-dark)); border-radius: 4px 4px 0 0; height: ${heightPercent}%; min-height: 20px; position: relative;">
                                        <div style="position: absolute; top: -20px; left: 50%; transform: translateX(-50%); font-size: 11px; font-weight: 600; white-space: nowrap;">
                                            ${day.calories}
                                        </div>
                                    </div>
                                </div>
                                <div style="font-size: 12px; font-weight: 500;">${dayName}</div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
            
            <style>
                .chart-container {
                    display: flex;
                    gap: 8px;
                    padding: 16px;
                    background: var(--bg-secondary);
                    border-radius: 12px;
                    overflow-x: auto;
                }
            </style>
        `;
        
        openModal('📊 Статистика за неделю', content);
    } catch (error) {
        showError('Не удалось загрузить статистику');
    }
}
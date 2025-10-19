// ============================================
// АНАЛИЗ ЛАБОРАТОРНЫХ ДАННЫХ
// ============================================

async function showLabsAnalysis() {
    // Проверяем кредиты
    try {
        const creditsData = await apiGet('/labs/credits');
        
        const content = `
            <div class="labs-screen">
                <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; margin-bottom: 16px; text-align: center;">
                    <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 4px;">Доступно анализов</div>
                    <div style="font-size: 32px; font-weight: 700; color: var(--primary-color);">
                        ${creditsData.credits}
                    </div>
                </div>
                
                ${creditsData.credits > 0 || userProfile?.subscription_status === 'active' ? `
                    <form id="labs-form">
                        <div class="form-group">
                            <label class="form-label">Введите данные анализов</label>
                            <textarea class="form-textarea" id="labs-text" 
                                      placeholder="Вставьте текст ваших анализов или опишите показатели" 
                                      style="min-height: 200px;" required></textarea>
                            <p class="text-muted" style="font-size: 12px; margin-top: 8px;">
                                💡 Можно скопировать текст из документа или сфотографировать анализы в боте
                            </p>
                        </div>
                        <button type="submit" class="btn btn-primary">
                            <i class="fas fa-microscope"></i>
                            Проанализировать
                        </button>
                    </form>
                ` : `
                    <div class="text-center" style="padding: 40px;">
                        <i class="fas fa-flask" style="font-size: 48px; color: var(--text-muted); margin-bottom: 16px;"></i>
                        <p class="text-muted">Нет доступных анализов</p>
                        <button class="btn btn-primary" onclick="closeModal(); openSubscription();" style="margin-top: 16px;">
                            Купить анализы
                        </button>
                    </div>
                `}
            </div>
        `;
        
        openModal('🧪 Анализ лабораторных данных', content);
        
        if (creditsData.credits > 0 || userProfile?.subscription_status === 'active') {
            document.getElementById('labs-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const text = document.getElementById('labs-text').value.trim();
                if (!text) return;
                
                try {
                    tg.MainButton.showProgress();
                    
                    const result = await apiPost('/labs/analyze', { text });
                    
                    closeModal();
                    showLabsResult(result.analysis);
                } catch (error) {
                    // Ошибка уже показана
                } finally {
                    tg.MainButton.hideProgress();
                }
            });
        }
    } catch (error) {
        showError('Не удалось загрузить информацию об анализах');
    }
}

function showLabsResult(analysis) {
    const content = `
        <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; white-space: pre-wrap; line-height: 1.8; font-size: 14px;">
            ${escapeHtml(analysis)}
        </div>
        <div style="background: #fef3c7; border: 1px solid #f59e0b; padding: 12px; border-radius: 8px; margin-top: 16px; font-size: 13px;">
            ⚠️ <strong>Важно:</strong> Это рекомендации AI. Для точной диагностики обратитесь к врачу.
        </div>
    `;
    
    openModal('📋 Результат анализа', content);
}
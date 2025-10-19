// ============================================
// ГЕНЕРАТОР РЕЦЕПТОВ
// ============================================

function showRecipeGenerator() {
    const content = `
        <form id="recipe-form">
            <div class="form-group">
                <label class="form-label">Какие продукты есть?</label>
                <textarea class="form-textarea" id="recipe-products" 
                          placeholder="Например:
курица 300г
рис 200г
помидоры
огурцы
оливковое масло" 
                          style="min-height: 150px;" required></textarea>
                <p class="text-muted" style="font-size: 12px; margin-top: 8px;">
                    💡 Укажите доступные продукты, и мы подберём рецепты
                </p>
            </div>
            <button type="submit" class="btn btn-primary">
                <i class="fas fa-hat-chef"></i>
                Создать рецепты
            </button>
        </form>
    `;
    
    openModal('🍳 Рецепты из холодильника', content);
    
    document.getElementById('recipe-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const products = document.getElementById('recipe-products').value.trim();
        if (!products) return;
        
        try {
            tg.MainButton.showProgress();
            
            const result = await apiPost('/recipe/generate', { products });
            
            closeModal();
            showRecipesResult(result.recipes);
        } catch (error) {
            // Ошибка уже показана
        } finally {
            tg.MainButton.hideProgress();
        }
    });
}

function showRecipesResult(recipes) {
    const content = `
        <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; white-space: pre-wrap; line-height: 1.8; font-size: 14px;">
            ${escapeHtml(recipes)}
        </div>
    `;
    
    openModal('🍽️ Ваши рецепты', content);
}
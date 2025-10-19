// ============================================
// ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ
// ============================================

async function showProfileScreen() {
    if (!userProfile) {
        await loadProfile();
    }
    
    const subText = userProfile.expires_at 
        ? `До ${new Date(userProfile.expires_at).toLocaleDateString('ru-RU')}`
        : userProfile.free_until
        ? `Пробная до ${new Date(userProfile.free_until).toLocaleDateString('ru-RU')}`
        : 'Не активна';
    
    const content = `
        <div class="profile-screen">
            <div style="text-align: center; padding: 24px;">
                <div class="avatar" style="width: 80px; height: 80px; margin: 0 auto 16px; font-size: 40px;">
                    <i class="fas fa-user"></i>
                </div>
                <h3>${userProfile.username || 'Пользователь'}</h3>
                <p class="text-muted">ID: ${userProfile.user_id}</p>
            </div>
            
            <div class="profile-stats" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 24px;">
                <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; text-align: center;">
                    <div style="font-size: 24px; font-weight: 700;">${userProfile.labs_credits}</div>
                    <div class="text-muted" style="font-size: 12px;">Анализов</div>
                </div>
                <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; text-align: center;">
                    <div style="font-size: 24px; font-weight: 700;">${userProfile.achievements_count}</div>
                    <div class="text-muted" style="font-size: 12px;">Достижений</div>
                </div>
                <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; text-align: center;">
                    <div style="font-size: 24px; font-weight: 700;">${userProfile.active_challenges}</div>
                    <div class="text-muted" style="font-size: 12px;">Челленджей</div>
                </div>
            </div>
            
            <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px; margin-bottom: 16px;">
                <h4 style="margin-bottom: 12px;">Подписка</h4>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600;">${getSubscriptionText(userProfile.subscription_status)}</div>
                        <div class="text-muted" style="font-size: 12px;">${subText}</div>
                    </div>
                    ${userProfile.subscription_status === 'expired' ? `
                        <button class="btn btn-primary" style="width: auto; padding: 8px 16px;" onclick="closeModal(); openSubscription();">
                            Продлить
                        </button>
                    ` : ''}
                </div>
            </div>
            
            <div style="background: var(--bg-secondary); padding: 16px; border-radius: 12px;">
                <h4 style="margin-bottom: 12px;">О приложении</h4>
                <p class="text-muted" style="font-size: 14px; line-height: 1.6;">
                    NutriCoach Mini App - современное приложение для отслеживания питания и достижения ваших целей.
                </p>
                <p class="text-muted" style="font-size: 12px; margin-top: 12px;">
                    Версия 2.0.0
                </p>
            </div>
        </div>
    `;
    
    openModal('👤 Профиль', content);
}
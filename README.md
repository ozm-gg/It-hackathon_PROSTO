# MolDQN-PyTorch: Генерация молекул с Deep Q-Network

Проект для генерации молекул с оптимизацией свойств (QED, синтетическая доступность, докинг) с использованием глубокого обучения с подкреплением (DQN). Разработан в рамках хакатона для создания молекул с улучшенными фармакологическими характеристиками.

## 📌 Основные особенности
- **Генерация молекул** через RL-агент (Deep Q-Network)
- **Оценка свойств**: 
  - Квантовая эффективность лекарства (QED)
  - Синтетическая доступность (SA-score)
  - Докинг-энергия (AutoDock Vina)
  - Предсказание раздражения/токсичности
  - Проницаемость роговицы
- **Интеграция химических инструментов**: RDKit, Open Babel, AutoDock Vina

# AML-алерты: предсказание эскалации

Итог: **CV ROC-AUC 0.6643** (OOF, stratified 5-fold × 3 повтора; бленд LightGBM + CatBoost + XGBoost).
Бейзлайн на обычных агрегатах давал 0.637.

## Что сдавать

| Deliverable | Файл | Статус |
|---|---|---|
| Предсказания | `team_F2F427DD.csv` | готово, формат проверен |
| Ноутбук | `solution.ipynb` (+ `requirements.txt`) | готово, 3 прогона дали побайтово одинаковый CSV |
| EDA-сайт | `docs/index.html` (копия `site/index.html`, один самодостаточный файл) | готово |

Ноутбук (*Restart & Run All*, ≈3 минуты) создаёт `team_F2F427DD.csv`; три независимых прогона дали побайтово одинаковый файл.

## Главная находка

У каждого клиента есть скрытый класс **A/B** (эскалаций ≈10% и ≈30%). У каждого класса в каждом сегменте
(тип × направление) свой жёсткий минимум суммы, а значения ниже минимума «прижаты» в узкую полосу над ним.
Поэтому одна транзакция у порога однозначно определяет класс: так размечается 6 045 клиентов без единого
противоречия. Остальным класс восстанавливается классификатором по форме профиля (AUC 0.955), таргет при этом
не используется. Подробности на сайте и в ноутбуке.

## Структура

```
fintech_data/          исходные данные (не в репозитории: положить локально)
solution.ipynb         финальное воспроизводимое решение (самодостаточный)
build_notebook.py      генерирует solution.ipynb (единый источник кода)
requirements.txt       точные версии библиотек
team_F2F427DD.csv      сабмит
docs/index.html        EDA-сайт для GitHub Pages
site/index.html        EDA-сайт; template.html + data.json -> build.py
src/                   исследовательский код: data, features, latent, models, cv, site_data
cache/                 кэш для src/ (создаётся локально, не в репозитории)
```

## Запуск ноутбука

```bash
brew install libomp                  # macOS: нужен LightGBM/XGBoost
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute solution.ipynb --output solution.ipynb
```

Данные должны лежать в `fintech_data/` рядом с ноутбуком.

## Публикация сайта (GitHub Pages)

**Settings → Pages** → Source: *Deploy from a branch*, Branch: `main`, папка `/docs` → Save.
Через 1–2 минуты сайт будет доступен по адресу `https://grxt777.github.io/west-test/`.
Проверить в режиме инкогнито: он должен открываться без логина. Для приватного репозитория Pages
на бесплатном плане недоступен, поэтому репозиторий нужно сделать публичным.

Альтернатива: Netlify (app.netlify.com → Add new site → Deploy manually → перетащить папку `site`).
Нужен аккаунт, иначе сайт удалится.

Сырые данные в публичный репозиторий **не выкладывать**: в `index.html` только агрегаты для графиков.

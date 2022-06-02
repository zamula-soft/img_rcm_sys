**Рекомендательная система оценки качества репортажных фотографий **

Система разработана на фреймворке Django и позволяет:
1. Зарегистрировать пользователя, в том числе и через SSO Google API
2. Создать личный кабинет пользователя
3. Загрузить фоторафию, указав основные параметры сьемки 
4. Автоматически определеить основные характеристики для оценки: 
    светотень, композиция, гистограмма, контурный анализ
5. Получить набор рекомендаций

Система разработана с помощью библиотек: OpenCV, matplotlib, numphy  
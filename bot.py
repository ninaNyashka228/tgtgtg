from threading import Thread
from flask import Flask
import asyncio
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import joblib
import os
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

flask_app = Flask(__name__)

@flask_app.route('/')
def home():
    return "Bot is running!"

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    flask_app.run(host='0.0.0.0', port=port)

Thread(target=run_web_server, daemon=True).start()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

print("=" * 60)
print("ЗАГРУЗКА ОТРАСЛЕВЫХ ДАННЫХ")
print("=" * 60)

INDUSTRY_DATA = {}
USER_HISTORY = {}
HISTORY_FILE = "user_history.json"

def load_history():
    global USER_HISTORY
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            USER_HISTORY = json.load(f)
    print("✅ История загружена")

def save_history():
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(USER_HISTORY, f, ensure_ascii=False, indent=2)

load_history()

def load_industry_data():
    industry_files = {
        'fitness': 'fitness.csv',
        'beauty': 'beauty.csv',
        'coffee': 'coffee.csv',
        'online_school': 'online_school.csv',
        'ecommerce': 'ecommerce.csv',
        'default': 'default.csv'
    }
    for key, filename in industry_files.items():
        filepath = os.path.join('industry_data', filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            data_dict = {}
            for _, row in df.iterrows():
                metric = row['metric'].strip()
                data_dict[metric] = {
                    'value': float(row['value']),
                    'source': row['source'],
                    'note': row['note'] if pd.notna(row['note']) else '',
                    'frequency': float(row['frequency']) if pd.notna(row['frequency']) else 1
                }
            INDUSTRY_DATA[key] = data_dict
            print(f"✅ Загружено: {key}")
        else:
            print(f"❌ Файл не найден: {filepath}")

load_industry_data()

print("\n" + "=" * 60)
print("ЗАГРУЗКА МОДЕЛИ")
print("=" * 60)

model = None
if os.path.exists('churn_model.pkl'):
    model = joblib.load('churn_model.pkl')
    print("✅ Модель загружена из файла")
else:
    print("ОБУЧЕНИЕ МОДЕЛИ")
    df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = df_raw.copy()
    df['tenure_days'] = df['tenure'] * 30
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df['avg_bill'] = df['TotalCharges'] / df['tenure']
    df['avg_bill'] = df['avg_bill'].fillna(df['MonthlyCharges'])
    df['churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df[['tenure_days', 'avg_bill']]
    y = df['churn']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'churn_model.pkl')
    print("✅ Модель обучена")

def calculate_ltv(industry_bill, industry_frequency, industry_tenure_days):
    tenure_months = industry_tenure_days / 30
    return industry_bill * industry_frequency * tenure_months

def get_industry_data(industry_key):
    if industry_key in INDUSTRY_DATA:
        return INDUSTRY_DATA[industry_key]
    return INDUSTRY_DATA.get('default', {})

def get_risk_level(risk):
    if risk >= 0.7:
        return 'high'
    elif risk >= 0.3:
        return 'medium'
    return 'low'

def get_recommendation(risk, ltv):
    if risk >= 0.7:
        return f"🔴 Предложите скидку 20%"
    elif risk >= 0.3:
        return f"🟡 Отправьте скидку 10%"
    return f"🟢 Всё хорошо, продолжайте"

def create_pie_chart(probs):
    counts = [sum(probs < 0.3), sum((probs >= 0.3) & (probs < 0.7)), sum(probs >= 0.7)]
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=['Низкий', 'Средний', 'Высокий'], autopct='%1.1f%%', colors=['green', 'orange', 'red'], startangle=90)
    plt.title('Распределение риска оттока')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

def create_bar_chart(probs):
    counts = [sum(probs < 0.3), sum((probs >= 0.3) & (probs < 0.7)), sum(probs >= 0.7)]
    plt.figure(figsize=(6, 4))
    plt.bar(['Низкий', 'Средний', 'Высокий'], counts, color=['green', 'orange', 'red'])
    plt.title('Количество клиентов по рискам')
    plt.ylabel('Клиенты')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

def create_scatter_chart(df_user):
    plt.figure(figsize=(8, 5))
    colors = {'high': 'red', 'medium': 'orange', 'low': 'green'}
    for level, color in colors.items():
        subset = df_user[df_user['risk_level'] == level]
        if len(subset) > 0:
            plt.scatter(subset['avg_bill'], subset['days_since_last'], c=color, label=level.upper(), alpha=0.6, s=50)
    plt.xlabel('Средний чек (₽)')
    plt.ylabel('Дней без покупки')
    plt.title('Зависимость риска от чека и давности')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

user_industry = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 Начать анализ", callback_data='select_industry')],
        [InlineKeyboardButton("📋 История", callback_data='history')],
        [InlineKeyboardButton("❓ Помощь", callback_data='help')]
    ]
    await update.message.reply_text(
        "👋 Добро пожаловать в бот анализа оттока клиентов\n\n"
        "Я помогу вам:\n"
        "• определить, какие клиенты скоро уйдут\n"
        "• оценить потенциальные потери в деньгах\n"
        "• получить конкретные рекомендации, как их удержать\n\n"
        "📊 Как это работает:\n"
        "1. Выберите вашу отрасль\n"
        "2. Загрузите файл с клиентами\n"
        "3. Получите анализ и список клиентов с риском оттока\n\n"
        "🎯 Вы сразу увидите:\n"
        "— кто в зоне риска\n"
        "— сколько денег вы можете потерять\n"
        "— что сделать, чтобы это предотвратить\n\n"
        "👇 Выберите действие:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = str(query.from_user.id)
    history_list = USER_HISTORY.get(user_id, [])
    if not history_list:
        await query.edit_message_text(
            "📋 У вас пока нет сохранённых прогнозов.\n\n"
            "Загрузите файл с клиентами, и результат автоматически сохранится в истории.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data='back_to_start')]])
        )
        return
    msg = "📋 **Ваши последние прогнозы:**\n\n"
    for i, item in enumerate(history_list[-5:][::-1], 1):
        msg += f"{i}. {item['date']} — {item['industry']}\n"
        msg += f"   🔴 Высокий риск: {item['high']}, 🟡 Средний: {item['medium']}, 🟢 Низкий: {item['low']}\n\n"
    keyboard = [[InlineKeyboardButton("🗑 Очистить историю", callback_data='clear_history')],
                [InlineKeyboardButton("🔙 Назад", callback_data='back_to_start')]]
    await query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = str(query.from_user.id)
    USER_HISTORY[user_id] = []
    save_history()
    await query.edit_message_text(
        "✅ История очищена.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data='back_to_start')]])
    )

async def select_industry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton("🏋️ Фитнес", callback_data='industry_fitness')],
        [InlineKeyboardButton("💅 Салоны красоты", callback_data='industry_beauty')],
        [InlineKeyboardButton("☕ Кофейни", callback_data='industry_coffee')],
        [InlineKeyboardButton("📚 Онлайн-школы", callback_data='industry_online_school')],
        [InlineKeyboardButton("🛒 E-commerce", callback_data='industry_ecommerce')],
        [InlineKeyboardButton("❓ Другая отрасль", callback_data='industry_default')]
    ]
    await query.edit_message_text(
        "👇 **Выберите тип бизнеса:**",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def set_industry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    industry_key = query.data.replace('industry_', '')
    user_industry[query.from_user.id] = industry_key
    industry_names = {
        'fitness': 'Фитнес-клубы',
        'beauty': 'Салоны красоты',
        'coffee': 'Кофейни',
        'online_school': 'Онлайн-школы',
        'ecommerce': 'E-commerce',
        'default': 'другую отрасль'
    }
    keyboard = [
        [InlineKeyboardButton("📁 Пример файла", callback_data='example_file')],
        [InlineKeyboardButton("🔙 Выбрать другую отрасль", callback_data='select_industry')]
    ]
    await query.edit_message_text(
        f"✅ **Вы выбрали:** {industry_names.get(industry_key, industry_key)}\n\n"
        "📥 **Теперь загрузите CSV-файл с данными клиентов**\n\n"
        "Файл должен содержать следующие поля:\n"
        "• `customer_id` — ID или имя клиента\n"
        "• `tenure_days` — сколько дней клиент с вами\n"
        "• `avg_bill` — средний чек (₽)\n"
        "• `days_since_last` — дней с последней покупки\n\n"
        "📌 **Пример формата:**\n"
        "```\n"
        "customer_id,tenure_days,avg_bill,days_since_last\n"
        "Иван Петров,540,3500,21\n"
        "Ольга Смирнова,365,4200,7\n"
        "```\n\n"
        "⚠️ **Важно:**\n"
        "— используйте разделитель «запятая»\n"
        "— не добавляйте лишние столбцы\n\n"
        "⬇️ **Отправьте файл, и я сразу сделаю анализ**",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "❓ Что такое риск оттока?\n"
        "Это вероятность того, что клиент перестанет покупать у вас.\n\n"
        "💰 Как считается LTV?\n"
        "LTV = средний чек × частота покупок × срок жизни клиента\n\n"
        "📊 Зачем сравнение с рынком?\n"
        "Чтобы понять, где вы теряете деньги относительно конкурентов.\n\n"
        "📁 Как подготовить файл?\n"
        "Нажмите «Пример файла» и используйте шаблон.\n\n"
        "👇 Вернуться в меню",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data='back_to_start')]])
    )

async def example_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    example_content = "customer_id,tenure_days,avg_bill,days_since_last\nИван Петров,540,3500,21\nОльга Смирнова,365,4200,7\nСергей Иванов,180,800,90"
    await query.edit_message_text(
        f"📁 **Пример файла**\n\n"
        f"```\n{example_content}\n```\n\n"
        "1. Скопируйте этот текст\n"
        "2. Вставьте в Блокнот\n"
        "3. Сохраните как `clients.csv`\n"
        "4. Отправьте файл боту\n\n"
        "⬇️ **После загрузки файла я сразу сделаю анализ**",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data='back_to_industry')]]),
        parse_mode='Markdown'
    )

async def back_to_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await start(update, context)

async def back_to_industry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    industry_key = user_industry.get(user_id, 'default')
    industry_names = {
        'fitness': 'Фитнес-клубы',
        'beauty': 'Салоны красоты',
        'coffee': 'Кофейни',
        'online_school': 'Онлайн-школы',
        'ecommerce': 'E-commerce',
        'default': 'другую отрасль'
    }
    keyboard = [
        [InlineKeyboardButton("📁 Пример файла", callback_data='example_file')],
        [InlineKeyboardButton("🔙 Выбрать другую отрасль", callback_data='select_industry')]
    ]
    await query.edit_message_text(
        f"✅ **Вы выбрали:** {industry_names.get(industry_key, industry_key)}\n\n"
        "📥 **Загрузите CSV-файл с данными клиентов**\n\n"
        "Файл должен содержать: `customer_id, tenure_days, avg_bill, days_since_last`\n\n"
        "📌 Пример:\n"
        "```\n"
        "customer_id,tenure_days,avg_bill,days_since_last\n"
        "Иван Петров,540,3500,21\n"
        "```\n\n"
        "⬇️ **Отправьте файл**",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    industry_key = user_industry.get(user_id, 'default')
    industry_data = get_industry_data(industry_key)
    industry_names = {
        'fitness': 'Фитнес-клубы',
        'beauty': 'Салоны красоты',
        'coffee': 'Кофейни',
        'online_school': 'Онлайн-школы',
        'ecommerce': 'E-commerce',
        'default': 'другая отрасль'
    }
    industry_bill = industry_data.get('avg_bill', {}).get('value', 2000)
    industry_frequency = industry_data.get('avg_payment_frequency', {}).get('value', 1)
    industry_tenure_days = industry_data.get('avg_tenure_days', {}).get('value', 400)
    industry_ltv = calculate_ltv(industry_bill, industry_frequency, industry_tenure_days)
    file = await update.message.document.get_file()
    csv_file = io.BytesIO()
    await file.download_to_memory(csv_file)
    csv_file.seek(0)
    try:
        df_user = pd.read_csv(csv_file)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}\nУбедитесь, что файл в формате CSV")
        return
    required = ['customer_id', 'tenure_days', 'avg_bill', 'days_since_last']
    missing = [col for col in required if col not in df_user.columns]
    if missing:
        await update.message.reply_text(f"❌ Ошибка: нет колонок {missing}\n\nНужны: customer_id, tenure_days, avg_bill, days_since_last")
        return
    X_user = df_user[['tenure_days', 'avg_bill']]
    probs = model.predict_proba(X_user)[:, 1]
    df_user['risk'] = probs
    df_user['risk_level'] = df_user['risk'].apply(get_risk_level)
    df_user['ltv'] = industry_ltv
    df_user['recommendation'] = df_user.apply(lambda r: get_recommendation(r['risk'], r['ltv']), axis=1)
    high_count = sum(df_user['risk_level'] == 'high')
    medium_count = sum(df_user['risk_level'] == 'medium')
    low_count = sum(df_user['risk_level'] == 'low')
    stats = f"📊 **Общая статистика по клиентам:**\n\n"
    stats += f"🔴 Высокий риск: {high_count}\n"
    stats += f"🟡 Средний риск: {medium_count}\n"
    stats += f"🟢 Низкий риск: {low_count}\n\n"
    stats += f"💡 Клиенты со средним риском — ваша зона роста"
    await update.message.reply_text(stats, parse_mode='Markdown')
    avg_user_bill = df_user['avg_bill'].mean()
    comparison = f"📊 **Сравнение с отраслью:**\n\n"
    comparison += f"Ваш средний чек: {avg_user_bill:,.0f} ₽\n"
    comparison += f"Средний чек по отрасли: {industry_bill:,.0f} ₽\n\n"
    if avg_user_bill < industry_bill:
        percent = (industry_bill - avg_user_bill) / industry_bill * 100
        comparison += f"⬇️ Вы ниже рынка на {percent:.0f}%\n\n"
        comparison += f"⚠️ Это увеличивает риск оттока клиентов\n"
        comparison += f"💡 **Рекомендуем:**\n"
        comparison += f"— улучшить ценность предложения\n"
        comparison += f"— добавить бонусы или подарки\n"
        comparison += f"— усилить коммуникацию с клиентами"
    else:
        comparison += f"✅ Выше рынка — хороший знак!"
    await update.message.reply_text(comparison, parse_mode='Markdown')
    top5 = df_user.nlargest(5, 'risk')[['customer_id', 'risk', 'ltv', 'recommendation']]
    if len(top5) > 0:
        msg = "🚨 **Клиенты с наибольшим риском оттока:**\n\n"
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            msg += f"{i}. {row['customer_id']}\n"
            msg += f"Риск: {row['risk']:.0%}\n"
            msg += f"Потенциальная потеря: {row['ltv']:,.0f} ₽\n"
            msg += f"👉 {row['recommendation']}\n\n"
        await update.message.reply_text(msg, parse_mode='Markdown')
    pie_img = create_pie_chart(probs)
    bar_img = create_bar_chart(probs)
    scatter_img = create_scatter_chart(df_user)
    await update.message.reply_photo(photo=pie_img, caption="🥧 Распределение рисков")
    await update.message.reply_photo(photo=bar_img, caption="📊 Количество клиентов по рискам")
    await update.message.reply_photo(photo=scatter_img, caption="📈 Зависимость риска от чека и давности")
    user_id_str = str(user_id)
    if user_id_str not in USER_HISTORY:
        USER_HISTORY[user_id_str] = []
    USER_HISTORY[user_id_str].append({
        'date': datetime.now().strftime("%d.%m.%Y %H:%M"),
        'industry': industry_names.get(industry_key, industry_key),
        'high': high_count,
        'medium': medium_count,
        'low': low_count
    })
    save_history()
    keyboard = [[InlineKeyboardButton("📋 История", callback_data='history')],
                [InlineKeyboardButton("🔄 Новый анализ", callback_data='select_industry')],
                [InlineKeyboardButton("🏠 Главное меню", callback_data='back_to_start')]]
    await update.message.reply_text(
        "✅ Прогноз сохранён в историю.\n\n"
        "Выберите следующее действие:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Отправьте CSV-файл или нажмите /start")

TOKEN = os.environ.get("BOT_TOKEN")
if not TOKEN:
    raise ValueError("BOT_TOKEN не задан в переменных окружения")

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(select_industry, pattern='select_industry'))
    app.add_handler(CallbackQueryHandler(set_industry, pattern='industry_'))
    app.add_handler(CallbackQueryHandler(history, pattern='history'))
    app.add_handler(CallbackQueryHandler(clear_history, pattern='clear_history'))
    app.add_handler(CallbackQueryHandler(show_help, pattern='help'))
    app.add_handler(CallbackQueryHandler(example_file, pattern='example_file'))
    app.add_handler(CallbackQueryHandler(back_to_start, pattern='back_to_start'))
    app.add_handler(CallbackQueryHandler(back_to_industry, pattern='back_to_industry'))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, unknown))
    print("=" * 60)
    print("🚀 БОТ ЗАПУЩЕН!")
    print("=" * 60)
    print("Теперь откройте Telegram и отправьте /start")
    print("Для остановки нажмите Ctrl+C\n")
    app.run_polling()

if __name__ == "__main__":
    main()
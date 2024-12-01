import os
import logging
import requests

from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, \
                         filters, CallbackQueryHandler, ContextTypes
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
load_dotenv()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info('start')
    await update.message.reply_text('Привет, я ваш телеграм-бот!')


async def send_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text == 'root':
        return await send_root(update, context)
    if update.message.text == 'post':
        return await send_post(update, context)
    logger.info('send_message')
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="Привет, мир!"
    )


async def send_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    logger.info(f'send_image {photo_file}')
    await context.bot.send_photo(
        chat_id=update.effective_chat.id, photo=photo_file.file_id
    )


def build_menu(buttons, n_cols,
               header_buttons=None,
               footer_buttons=None):
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if header_buttons:
        menu.insert(0, [header_buttons])
    if footer_buttons:
        menu.append([footer_buttons])
    return menu


async def send_keyboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info('send_keyboard')
    keyboard = [
        InlineKeyboardButton("Кнопка 1", callback_data='1'),
        InlineKeyboardButton("Кнопка 2", callback_data='2')
    ]
    reply_markup = InlineKeyboardMarkup(build_menu(keyboard, n_cols=2))
    text = 'Выберите кнопку:'
    await update.message.reply_text(text, reply_markup=reply_markup)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info('button')
    query = update.callback_query
    await query.answer()
    logger.info(f'Пользователь нажал кнопку: {query.data}')
    await query.edit_message_text(text=f"Вы выбрали опцию {query.data}")


async def send_root(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info('send_root')
    response = requests.get('http://fastapi:8000/items/')

    # Проверка ответа
    if response.status_code == 200:
        logger.info('Сообщение успешно получено!')
    else:
        logger.info('Ошибка при получении сообщения:', response.text)

    await context.bot.send_message(update.effective_chat.id, response.text)


async def send_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info('send_post')
    json = {'name': 'name', 'description': 'description'}
    response = requests.post('http://fastapi:8000/items/',  json=json)

    # Проверка ответа
    if response.status_code == 200:
        logger.info('Сообщение успешно отправлено!')

    await context.bot.send_message(update.effective_chat.id, response.text)


def main() -> None:
    application = ApplicationBuilder().token(os.getenv('BOT_TOKEN')).build()

    # Регистрация обработчиков
    application.add_handler(CommandHandler('start', start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, send_message)
    )
    application.add_handler(MessageHandler(filters.PHOTO, send_image))
    application.add_handler(CommandHandler('keyboard', send_keyboard))
    application.add_handler(CallbackQueryHandler(button))

    # Запуск бота
    application.run_polling()


if __name__ == '__main__':
    main()

import os


class Config:
    # Sử dụng SQLite, có thể thay đổi thành MySQL hoặc PostgreSQL
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

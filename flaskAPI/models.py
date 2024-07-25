from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Door(db.Model):
    __tablename__ = 'door'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # Tự động tăng
    door_name = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f'<Door {self.door_name}>'


class Member(db.Model):
    __tablename__ = 'member'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # Tự động tăng
    fingerprint = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f'<Member {self.name}>'


class History(db.Model):
    __tablename__ = 'history'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # Tự động tăng
    detail_verify_id = db.Column(db.Integer, db.ForeignKey('detail_verify.id'), nullable=False)
    time = db.Column(db.DateTime, nullable=False)

    def __repr__(self):
        return f'<History {self.id}>'


class DetailVerify(db.Model):
    __tablename__ = 'detail_verify'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # Tự động tăng
    door_id = db.Column(db.Integer, db.ForeignKey('door.id'), nullable=False)
    member_id = db.Column(db.Integer, db.ForeignKey('member.id'), nullable=False)

    __table_args__ = (db.UniqueConstraint('door_id', 'member_id', name='uix_door_member'),)

    def __repr__(self):
        return f'<DetailVerify Door ID: {self.door_id}, Member ID: {self.member_id}>'
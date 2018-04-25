from walrus import Database

db = Database(host='localhost', port=6379, db=0)
cache = db.cache()

import mysql.connector

class DatabaseManager:
    def __init__(self, config):
        self.config = config

    def get_connection(self):
        return mysql.connector.connect(
            host=self.config["host"],
            port=self.config["port"],
            user=self.config["user"],
            password=self.config["password"],
            database=self.config["database"],
            ssl_ca=self.config["ssl_ca"],
            ssl_verify_cert=self.config["ssl_verify_cert"],
            ssl_verify_identity=self.config["ssl_verify_identity"]
    )

    def fetch_species_map(self):
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("SELECT species_id, scientific_name FROM species_info")
            mapping = {
                row['scientific_name']: row['species_id']
                for row in cursor.fetchall()
            }

            conn.close()
            return mapping

        except Exception as e:
            print(f"DB Mapping Error: {e}")
            return {}

    def fetch_all_species(self):
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("SELECT * FROM species_info")
            data = cursor.fetchall()

            conn.close()
            return data

        except Exception as e:
            print(f" DB Fetch Error: {e}")
            return []

    def save_detection(self, species_id, confidence, lat, lng):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            query = """
                INSERT INTO plant_detections 
                (species_id, confidence, latitude, longitude) 
                VALUES (%s, %s, %s, %s)
            """

            cursor.execute(query, (species_id, confidence, lat, lng))
            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f" DB Save Error: {e}")
            return False
        
    def get_detections(self, page=1, limit=50):
        try:
            offset = (page - 1) * limit

            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("SELECT COUNT(*) AS total FROM plant_detections")
            total = cursor.fetchone()["total"]

            query = """
                SELECT
                    pd.id,
                    pd.species_id,
                    si.name,
                    si.scientific_name,
                    pd.confidence,
                    pd.detected_at,
                    pd.latitude,
                    pd.longitude
                FROM plant_detections pd
                JOIN species_info si
                    ON pd.species_id = si.species_id
                ORDER BY pd.detected_at DESC
                LIMIT %s OFFSET %s
            """

            cursor.execute(query, (limit, offset))
            records = cursor.fetchall()

            conn.close()

            return {
                "records": records,
                "page": page,
                "pages": (total + limit - 1) // limit,
                "total": total
            }

        except Exception as e:
            print(f" DB Fetch Error: {e}")

            return {
                "records": [],
                "page": 1,
                "pages": 1,
                "total": 0
            }
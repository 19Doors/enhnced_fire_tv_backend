from kafka import KafkaProducer
import json

class RecommendationProducer:
    def __init__(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=['kafka:9092'],  # Kafka server address
                # bootstrap_servers=['localhost:9092'],  # Kafka server address
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),  # Convert dict to JSON
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                # request_timeout_ms=30000,  # 30 second timeout
                # retry_backoff_ms=500,
                # retries=5,
                # acks='all'  # Wait for all replicas to acknowledge
            )
            print("CONNECTED TO KAFKA!")
            
        except Exception as e:
            print(f"❌ Failed to initialize Kafka producer: {e}")
            raise

    def send_for_recommendation(self, user_id):
        print("TRYING")
        try:
            message = {
                    'user_id': user_id,
                    }
            # self.producer.send('user_interaction_ai', key=)
            future = self.producer.send('user_interaction_ai', key=user_id, value=message)
            self.producer.flush(timeout=10)

            # result = future.get(timeout=10)
            print("message sent successfully")
            return True

        except Exception as e:
            print(f"Error sending message: {e}")
            return False

    def close(self):
        self.producer.close()
